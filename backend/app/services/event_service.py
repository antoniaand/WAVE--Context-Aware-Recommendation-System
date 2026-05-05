# WAVE Production API - Context-Aware Event Recommender
"""
app/services/event_service.py
------------------------------
Fetches events from the Supabase `events` table for a given city and date range.
Falls back to generated synthetic events if fewer than MIN_EVENTS real events exist,
ensuring the recommendation engine always has enough candidates to score.
"""

import logging
from datetime import date as DateType, timedelta
from typing import List

from app.core.database import get_supabase_admin_client

logger = logging.getLogger(__name__)

MIN_EVENTS = 5

CITY_CLIMATE: dict[str, str] = {
    "Bucharest": "Moderate", "Cluj-Napoca": "Moderate",
    "Timisoara": "Moderate", "Iasi": "Moderate",
    "Constanta": "Moderate", "Brasov": "Moderate",
    "Oslo": "Cold", "Helsinki": "Cold", "Quebec": "Cold",
    "Dubai": "Hot", "Phoenix": "Hot", "Seville": "Hot",
    "London": "Rainy", "Bergen": "Rainy", "Seattle": "Rainy",
}

# Canonical event types with indoor/outdoor flag
_EVENT_DEFAULTS = [
    ("Concert", True),
    ("Festival", True),
    ("Sports", True),
    ("Theatre", False),
    ("Conference", False),
]


def _normalize(event: dict, fallback_hour: int = 12) -> dict:
    """Coerce a Supabase row into the format expected by ml_service."""
    return {
        "event_type":   event["event_type"],
        "event_name":   event.get("event_name"),
        "location":     event["location"],
        "venue":        event.get("venue"),
        "event_date":   str(event["event_date"]),
        "event_hour":   event.get("event_hour") or fallback_hour,
        "climate_zone": event.get("climate_zone") or CITY_CLIMATE.get(event["location"], "Moderate"),
        "is_outdoor":   int(bool(event.get("is_outdoor", False))),
        "source":       event.get("source", "generated"),
        "is_generated": bool(event.get("is_generated", False)),
        "url":          event.get("url"),
        "image_url":    event.get("image_url"),
        "description":  event.get("description"),
    }


def _generate_fallback(city: str, date_str: str, hour: int, needed: int, existing_types: set) -> List[dict]:
    """Synthetic placeholder events so the ML engine always has candidates."""
    missing = [(t, o) for t, o in _EVENT_DEFAULTS if t not in existing_types]
    pool = missing + _EVENT_DEFAULTS  # fall back to repeating if all types present

    climate = CITY_CLIMATE.get(city, "Moderate")
    results = []
    for i in range(needed):
        event_type, is_outdoor = pool[i % len(pool)]
        results.append({
            "event_type":   event_type,
            "event_name":   f"{event_type} in {city}",
            "location":     city,
            "venue":        None,
            "event_date":   date_str,
            "event_hour":   hour,
            "climate_zone": climate,
            "is_outdoor":   int(is_outdoor),
            "source":       "generated",
            "is_generated": True,
            "url":          None,
            "image_url":    None,
            "description":  None,
        })
    return results


async def get_events_for_date(city: str, date_str: str, hour: int = 12) -> List[dict]:
    """Events for a single city + date, with generated fallback."""
    client = get_supabase_admin_client()
    try:
        resp = client.table("events").select("*").eq("location", city).eq("event_date", date_str).execute()
        rows = [_normalize(r, hour) for r in (resp.data or [])]
    except Exception as exc:
        logger.error("Event fetch failed for %s %s: %s", city, date_str, exc)
        rows = []

    if len(rows) < MIN_EVENTS:
        existing = {r["event_type"] for r in rows}
        rows += _generate_fallback(city, date_str, hour, MIN_EVENTS - len(rows), existing)

    return rows


async def get_events_for_range(city: str, start_date: str, end_date: str, hour: int = 12) -> List[dict]:
    """Events for a city within a date range, with generated fallback."""
    client = get_supabase_admin_client()
    try:
        resp = (
            client.table("events")
            .select("*")
            .eq("location", city)
            .gte("event_date", start_date)
            .lte("event_date", end_date)
            .execute()
        )
        rows = [_normalize(r, hour) for r in (resp.data or [])]
    except Exception as exc:
        logger.error("Event range fetch failed for %s %s-%s: %s", city, start_date, end_date, exc)
        rows = []

    if len(rows) < MIN_EVENTS:
        existing = {r["event_type"] for r in rows}
        rows += _generate_fallback(city, start_date, hour, MIN_EVENTS - len(rows), existing)

    return rows
