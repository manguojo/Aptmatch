from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EVENT_WEIGHTS = {
    "listing_expand": 1,
    "recommendation_expand": 1,
    "listing_select": 2,
    "marker_select": 3,
    "recommendation_select": 3,
}

DEFAULT_INPUT = Path(__file__).resolve().parent / "interaction_events.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lightgbm_training_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AptMatch interaction_events.jsonl into a LightGBM-ready tabular dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the analytics JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["_sequence"] = idx
            events.append(payload)
    events.sort(key=lambda item: (_parse_timestamp(item.get("timestamp")), item["_sequence"]))
    return events


def _build_row(
    *,
    query_id: str,
    session_id: str,
    search_timestamp: str | None,
    position: int,
    returned_count: int,
    listing: dict[str, Any],
    filters: dict[str, Any],
    user_profile: dict[str, Any],
    query: str | None,
    recommendation_ids: set[str],
) -> dict[str, Any]:
    listing_id = str(listing.get("listingId") or "")
    return {
        "query_id": query_id,
        "session_id": session_id,
        "search_timestamp": search_timestamp,
        "query_text": query,
        "query_group_size": returned_count,
        "position": position,
        "position_pct": round(position / returned_count, 6) if returned_count else None,
        "recommended_top_pick": int(listing_id in recommendation_ids),
        "strict_location": int(bool(filters.get("strictLocation"))),
        "query_location": filters.get("location"),
        "query_landmark": filters.get("landmark"),
        "query_geo_target_type": filters.get("geoTargetType"),
        "query_max_distance": filters.get("maxDistance"),
        "query_min_price": filters.get("minPrice"),
        "query_max_price": filters.get("maxPrice"),
        "query_beds": filters.get("beds"),
        "query_max_beds": filters.get("maxBeds"),
        "query_baths": filters.get("baths"),
        "query_max_baths": filters.get("maxBaths"),
        "query_min_walk_score": filters.get("minWalkScore"),
        "query_max_walk_score": filters.get("maxWalkScore"),
        "query_min_bike_score": filters.get("minBikeScore"),
        "query_max_bike_score": filters.get("maxBikeScore"),
        "query_near_metro": int(bool(filters.get("nearMetro"))),
        "query_quiet": int(bool(filters.get("quiet"))),
        "query_safe": int(bool(filters.get("safe"))),
        "query_neighborhood_count": len(filters.get("neighborhoods") or []),
        "query_zip_count": len(filters.get("zipCodes") or []),
        "query_city_count": len(filters.get("cities") or []),
        "user_budget_max": user_profile.get("budget_max"),
        "user_budget_min": user_profile.get("budget_min"),
        "user_bedrooms": user_profile.get("bedrooms"),
        "user_bathrooms": user_profile.get("bathrooms"),
        "user_preferred_neighborhood_count": len(user_profile.get("preferred_neighborhoods") or []),
        "user_preferred_landmark_count": len(user_profile.get("preferred_landmarks") or []),
        "user_lifestyle_pref_count": len(user_profile.get("lifestyle_preferences") or []),
        "listing_id": listing_id,
        "listing_score": listing.get("score"),
        "listing_price": listing.get("price"),
        "listing_beds": listing.get("beds"),
        "listing_baths": listing.get("baths"),
        "listing_walk_score": listing.get("walkScore"),
        "listing_bike_score": listing.get("bikeScore"),
        "listing_bus_access_score": listing.get("busAccessScore"),
        "listing_commute_minutes": listing.get("commuteMinutes"),
        "listing_value_score": listing.get("valueScore"),
        "listing_neighborhood": listing.get("neighborhood"),
        "listing_zip_code": listing.get("zipCode"),
        "listing_nearest_metro_stop": listing.get("nearestMetroStop"),
        "listing_select_count": 0,
        "listing_expand_count": 0,
        "marker_select_count": 0,
        "recommendation_select_count": 0,
        "recommendation_expand_count": 0,
        "engagement_event_count": 0,
        "distinct_engagement_types": 0,
        "engagement_label": 0,
        "any_engagement": 0,
    }


def _finalize_rows(rows_by_listing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for row in rows_by_listing.values():
        active_events = {
            event_type: row[f"{event_type}_count"]
            for event_type in EVENT_WEIGHTS
            if row[f"{event_type}_count"] > 0
        }
        row["engagement_event_count"] = sum(active_events.values())
        row["distinct_engagement_types"] = len(active_events)
        max_weight = max((EVENT_WEIGHTS[event_type] for event_type in active_events), default=0)
        bonus = 1 if len(active_events) >= 2 and max_weight > 0 else 0
        label = min(4, max_weight + bonus)
        row["engagement_label"] = label
        row["any_engagement"] = int(label > 0)
        finalized.append(row)
    return finalized


def export_training_rows(events: list[dict[str, Any]]) -> pd.DataFrame:
    exported_rows: list[dict[str, Any]] = []
    active_searches: dict[str, dict[str, Any]] = {}
    search_counter = 0

    for event in events:
        session_id = str(event.get("sessionId") or "")
        event_type = str(event.get("eventType") or "")
        metadata = event.get("metadata") or {}

        if event_type == "search_impression":
            if session_id in active_searches:
                exported_rows.extend(_finalize_rows(active_searches.pop(session_id)["rows"]))

            search_counter += 1
            results = metadata.get("results") or []
            recommendation_ids = {
                str(item)
                for item in (metadata.get("recommendationListingIds") or [])
                if item is not None
            }
            query_id = f"{session_id}:{search_counter}"
            rows_by_listing: dict[str, dict[str, Any]] = {}
            for listing in results:
                listing_id = str(listing.get("listingId") or "")
                if not listing_id:
                    continue
                position = _as_count(listing.get("position")) or (len(rows_by_listing) + 1)
                rows_by_listing[listing_id] = _build_row(
                    query_id=query_id,
                    session_id=session_id,
                    search_timestamp=event.get("timestamp"),
                    position=position,
                    returned_count=_as_count(metadata.get("returnedCount")) or len(results),
                    listing=listing,
                    filters=metadata.get("filters") or {},
                    user_profile=metadata.get("userProfile") or {},
                    query=metadata.get("query"),
                    recommendation_ids=recommendation_ids,
                )

            active_searches[session_id] = {
                "query_id": query_id,
                "rows": rows_by_listing,
            }
            continue

        if event_type == "start_over":
            if session_id in active_searches:
                exported_rows.extend(_finalize_rows(active_searches.pop(session_id)["rows"]))
            continue

        if event_type not in EVENT_WEIGHTS:
            continue

        current = active_searches.get(session_id)
        if not current:
            continue

        listing_id = str(event.get("listingId") or "")
        if not listing_id:
            continue

        row = current["rows"].get(listing_id)
        if not row:
            continue

        counter_key = f"{event_type}_count"
        row[counter_key] += 1

    for state in active_searches.values():
        exported_rows.extend(_finalize_rows(state["rows"]))

    return pd.DataFrame(exported_rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    events = _load_events(args.input)
    frame = export_training_rows(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    label_counts = Counter(frame["engagement_label"].tolist()) if not frame.empty else Counter()
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "events_loaded": len(events),
                "rows_exported": len(frame),
                "queries_exported": int(frame["query_id"].nunique()) if not frame.empty else 0,
                "label_distribution": dict(sorted(label_counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
