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
from scripts.common import coords_to_climate_zone, nearest_canonical_city

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

# Canonical event types (order defines generation priority)
_EVENT_TYPES = ["Concert", "Festival", "Sports", "Theatre", "Conference"]

# Named variants per type — cycled so each slot gets a different label
_EVENT_NAMES: dict[str, list[str]] = {
    "Concert":    ["Live Concert", "Music Night", "Acoustic Session", "Rock Concert", "Pop Evening"],
    "Festival":   ["Cultural Festival", "Arts Festival", "Music Festival", "Food Festival", "Street Festival"],
    "Sports":     ["Championship Match", "Sports Tournament", "League Game", "Athletic Event", "Sports Competition"],
    "Theatre":    ["Theatre Performance", "Drama Show", "Stage Play", "Live Theatre", "Cabaret Night"],
    "Conference": ["Professional Conference", "Industry Summit", "Business Workshop", "Tech Meetup", "Academic Forum"],
}

# Three time slots generated per event type
_SLOTS: list[int] = [10, 15, 20]  # morning, afternoon, evening

# Types that are always indoor / always outdoor regardless of season
_ALWAYS_INDOOR  = {"Theatre", "Conference"}
_ALWAYS_OUTDOOR = {"Festival", "Sports"}


def _outdoor_for_season(event_type: str, climate_zone: str, month: int) -> bool:
    """Season- and climate-aware is_outdoor for generated events."""
    if event_type in _ALWAYS_INDOOR:
        return False
    if event_type in _ALWAYS_OUTDOOR:
        return True
    # Concert: depends on climate + season
    summer = 5 <= month <= 9
    if climate_zone == "Rainy":
        return False                        # almost always indoors in maritime climates
    if climate_zone == "Cold":
        return summer                       # outdoor only May-Sep
    if climate_zone == "Hot":
        return month not in (6, 7, 8)      # avoid peak heat months
    return summer                           # Moderate: outdoor in summer


def _normalize(event: dict, fallback_hour: int = 12) -> dict:
    """Coerce a Supabase row into the format expected by ml_service."""
    return {
        "event_type":   event["event_type"],
        "event_name":   event.get("event_name"),
        "location":     event["location"],
        "venue":        event.get("venue_name"),
        "event_date":   str(event["event_date"]),
        "event_hour":   event.get("event_hour") or fallback_hour,
        "climate_zone": event.get("climate_zone") or CITY_CLIMATE.get(event["location"], "Moderate"),
        "is_outdoor":   int(bool(event.get("is_outdoor", False))),
        "source":       event.get("source", "generated"),
        "is_generated": bool(event.get("is_generated", False)),
        "url":          event.get("ticket_url"),
        "image_url":    event.get("image_url"),
        "description":  event.get("description"),
    }


def _generate_fallback(
    city: str,
    date_str: str,
    *,
    display_name: str | None = None,
    climate_zone: str | None = None,
) -> List[dict]:
    """Return a rich pool of 15 synthetic events (5 types × 3 time slots).

    Args:
        city:         Canonical city name stored on the event row (used as ``location``).
        date_str:     ISO-8601 date string for the events (``YYYY-MM-DD``).
        display_name: Human-readable label used in event names (e.g. "Lyon").
                      Falls back to ``city`` when omitted.
        climate_zone: Override the climate zone lookup (used when coords are
                      known but ``city`` is not a canonical city).
                      Falls back to ``CITY_CLIMATE.get(city, "Moderate")``.
    """
    label   = display_name or city
    climate = climate_zone or CITY_CLIMATE.get(city, "Moderate")

    try:
        month = int(date_str[5:7])
    except (ValueError, IndexError):
        month = 6  # default to summer if date is malformed

    events: List[dict] = []
    for slot_idx, slot_hour in enumerate(_SLOTS):
        for type_idx, event_type in enumerate(_EVENT_TYPES):
            name_variants = _EVENT_NAMES[event_type]
            name = f"{name_variants[slot_idx % len(name_variants)]} in {label}"
            events.append({
                "event_type":      event_type,
                "event_name":      name,
                "location":        city,          # canonical city — used for ML encoding
                "display_location": label,         # human-readable — used in API response
                "venue":           None,
                "event_date":      date_str,
                "event_hour":      slot_hour,
                "climate_zone":    climate,
                "is_outdoor":      int(_outdoor_for_season(event_type, climate, month)),
                "source":          "generated",
                "is_generated":    True,
                "url":             None,
                "image_url":       None,
                "description":     None,
            })
    return events  # always 15 events


def _resolve_location(lat: float, lng: float) -> tuple[str | None, str]:
    """Return (canonical_city, climate_zone) for a coordinate pair.

    canonical_city — canonical city within 100 km, or None (skip DB query).
    climate_zone   — WAVE climate zone derived from coordinates.
    """
    return nearest_canonical_city(lat, lng), coords_to_climate_zone(lat, lng)


async def get_events_for_date(
    lat: float, lng: float, display_name: str, date_str: str, hour: int = 12
) -> List[dict]:
    """Events for a location + date, with generated fallback for unknown locations.

    If coordinates fall within 100 km of a canonical city, real events are fetched
    from Supabase.  Otherwise (or when real events are sparse) a rich pool of 15
    synthetic events is returned, using *display_name* for readable titles and
    *climate_zone* derived from the coordinates.
    """
    canonical, climate = _resolve_location(lat, lng)
    # For arbitrary locations canonical is None; display_name becomes the location
    # value in generated events (encodes as -1 in ML — fine, climate_zone +
    # weather already carry the real local signal).
    city = canonical or display_name

    rows: List[dict] = []
    if canonical:
        client = get_supabase_admin_client()
        try:
            resp = (
                client.table("events")
                .select("*")
                .eq("location", canonical)
                .eq("event_date", date_str)
                .execute()
            )
            rows = [_normalize(r, hour) for r in (resp.data or [])]
        except Exception as exc:
            logger.error("Event fetch failed for %s %s: %s", canonical, date_str, exc)

    if len(rows) < MIN_EVENTS:
        rows += _generate_fallback(city, date_str, display_name=display_name, climate_zone=climate)

    return rows


async def get_events_for_range(
    lat: float, lng: float, display_name: str, start_date: str, end_date: str, hour: int = 12
) -> List[dict]:
    """Events for a location within a date range, with generated fallback.

    Same location-resolution logic as get_events_for_date.  Generated fallback
    uses *start_date* as the reference date for season/climate calculations.
    """
    canonical, climate = _resolve_location(lat, lng)
    city = canonical or display_name

    rows: List[dict] = []
    if canonical:
        client = get_supabase_admin_client()
        try:
            resp = (
                client.table("events")
                .select("*")
                .eq("location", canonical)
                .gte("event_date", start_date)
                .lte("event_date", end_date)
                .execute()
            )
            rows = [_normalize(r, hour) for r in (resp.data or [])]
        except Exception as exc:
            logger.error("Event range fetch failed for %s %s-%s: %s", canonical, start_date, end_date, exc)

    if len(rows) < MIN_EVENTS:
        rows += _generate_fallback(city, start_date, display_name=display_name, climate_zone=climate)

    return rows
