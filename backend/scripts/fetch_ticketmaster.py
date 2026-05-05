"""
backend/scripts/fetch_ticketmaster.py
--------------------------------------
Fetches real events from the Ticketmaster Discovery API v2 for all
international cities in the WAVE project and upserts them into Supabase.

Run from backend/ directory:
    python -m scripts.fetch_ticketmaster
"""

import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from scripts.common import (
    CANONICAL_CITIES,
    CANONICAL_EVENT_TYPES,
    CITY_CLIMATE,
    TICKETMASTER_CATEGORY_MAP,
    TICKETMASTER_CITY_KEYWORDS,
    infer_is_outdoor,
    make_source_key,
    upsert_events,
)

_BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
_PAGE_DELAY = 0.5       # seconds between page requests
_RATE_LIMIT_WAIT = 10   # seconds to wait after HTTP 429
_MAX_PAGES = 5
_PAGE_SIZE = 50


def _get_image_url(images: list[dict]) -> Optional[str]:
    """
    Return the URL of the 16_9-ratio image from a Ticketmaster images list.

    Falls back to the first image if no 16_9 entry exists.

    Args:
        images: List of image dicts from the Ticketmaster event response.

    Returns:
        Image URL string, or None if images is empty.
    """
    if not images:
        return None
    for img in images:
        if img.get("ratio") == "16_9":
            return img.get("url")
    return images[0].get("url")


def _parse_event(raw: dict, canonical_city: str) -> Optional[dict]:
    """
    Parse a single Ticketmaster event object into a WAVE-normalized event dict.

    Returns None (with a warning printed) if the event has no recognizable
    event_type, a missing or unparseable date, or a past event_date.

    Args:
        raw:            Raw event dict from the Ticketmaster API response.
        canonical_city: The WAVE canonical city name (e.g. "London").

    Returns:
        Normalized event dict, or None if the event should be skipped.
    """
    today = date.today()
    name = raw.get("name", "Unknown Event")

    # ── Date ──────────────────────────────────────────────────────────────────
    dates_start = raw.get("dates", {}).get("start", {})
    event_date_str = dates_start.get("localDate")
    if not event_date_str:
        print(f"[ticketmaster] WARN: missing localDate for '{name}', skipping.")
        return None

    try:
        event_date = date.fromisoformat(event_date_str)
    except ValueError:
        print(f"[ticketmaster] WARN: unparseable date '{event_date_str}' for '{name}', skipping.")
        return None

    if event_date < today:
        return None  # silently drop past events

    # ── Hour ──────────────────────────────────────────────────────────────────
    local_time = dates_start.get("localTime", "20:00:00")
    try:
        event_hour = int(local_time.split(":")[0])
    except (ValueError, IndexError):
        event_hour = 20

    # ── Event type via segment → genre fallback ────────────────────────────────
    classifications = raw.get("classifications", [])
    event_type: Optional[str] = None
    if classifications:
        segment_name = classifications[0].get("segment", {}).get("name", "")
        event_type = TICKETMASTER_CATEGORY_MAP.get(segment_name)
        if event_type is None:
            genre_name = classifications[0].get("genre", {}).get("name", "")
            event_type = TICKETMASTER_CATEGORY_MAP.get(genre_name)
            if event_type is None:
                print(
                    f"[ticketmaster] WARN: unknown segment '{segment_name}' / "
                    f"genre '{genre_name}' for '{name}', skipping."
                )
                return None

    if event_type is None:
        print(f"[ticketmaster] WARN: no classification found for '{name}', skipping.")
        return None

    # ── Venue ─────────────────────────────────────────────────────────────────
    venues = raw.get("_embedded", {}).get("venues", [])
    venue_name: Optional[str] = None
    indoor_outdoor: Optional[str] = None
    if venues:
        venue_name = venues[0].get("name")
        indoor_outdoor = venues[0].get("indoorOutdoor")

    # ── is_outdoor ────────────────────────────────────────────────────────────
    if indoor_outdoor == "outdoor":
        is_outdoor = True
    elif indoor_outdoor == "indoor":
        is_outdoor = False
    else:
        is_outdoor = infer_is_outdoor(event_type, venue_name or "")

    # ── Image ─────────────────────────────────────────────────────────────────
    image_url = _get_image_url(raw.get("images", []))

    return {
        "event_name":   name,
        "event_type":   event_type,
        "location":     canonical_city,
        "climate_zone": CITY_CLIMATE.get(canonical_city, "Moderate"),
        "is_outdoor":   is_outdoor,
        "event_date":   event_date_str,
        "event_hour":   event_hour,
        "source":       "ticketmaster",
        "is_generated": False,
        "source_key":   make_source_key(name, event_date_str, canonical_city),
        "ticket_url":   raw.get("url"),
        "image_url":    image_url,
        "venue_name":   venue_name,
    }


def _validate_event(event: dict) -> bool:
    """
    Validate that a normalized event dict meets WAVE schema constraints.

    Checks that event_type and location are in their respective canonical sets
    and that event_date is a valid future date.

    Args:
        event: Normalized event dict produced by _parse_event().

    Returns:
        True if all constraints pass; False if the event should be skipped.
    """
    today = date.today()

    if event["event_type"] not in CANONICAL_EVENT_TYPES:
        print(
            f"[ticketmaster] WARN: invalid event_type '{event['event_type']}' "
            f"for '{event['event_name']}', skipping."
        )
        return False

    if event["location"] not in CANONICAL_CITIES:
        print(
            f"[ticketmaster] WARN: invalid location '{event['location']}' "
            f"for '{event['event_name']}', skipping."
        )
        return False

    try:
        event_date = date.fromisoformat(event["event_date"])
    except ValueError:
        print(
            f"[ticketmaster] WARN: invalid event_date '{event['event_date']}' "
            f"for '{event['event_name']}', skipping."
        )
        return False

    if event_date < today:
        print(
            f"[ticketmaster] WARN: past event_date '{event['event_date']}' "
            f"for '{event['event_name']}', skipping."
        )
        return False

    return True


def _fetch_city(
    city: str,
    api_key: str,
    start_dt: str,
    end_dt: str,
) -> list[dict]:
    """
    Fetch and parse all Ticketmaster events for one canonical city.

    Paginates up to _MAX_PAGES pages (0-indexed), inserting a short delay
    between requests.  On HTTP 429 waits _RATE_LIMIT_WAIT seconds and retries
    that page once.  Stops early if a page returns no _embedded key.

    Args:
        city:     WAVE canonical city name — must be a key in TICKETMASTER_CITY_KEYWORDS.
        api_key:  Ticketmaster Discovery API key.
        start_dt: ISO-8601 UTC datetime string for the start of the search window.
        end_dt:   ISO-8601 UTC datetime string for the end of the search window.

    Returns:
        List of normalized, validated event dicts ready for upsertion.
    """
    city_params = TICKETMASTER_CITY_KEYWORDS[city]
    events: list[dict] = []
    total_pages = 1  # updated after the first successful response

    for page in range(_MAX_PAGES):
        if page >= total_pages:
            break

        params: dict = {
            "apikey":        api_key,
            "city":          city_params["city"],
            "countryCode":   city_params["countryCode"],
            "size":          _PAGE_SIZE,
            "page":          page,
            "startDateTime": start_dt,
            "endDateTime":   end_dt,
            "sort":          "date,asc",
        }

        try:
            response = requests.get(_BASE_URL, params=params, timeout=15)

            if response.status_code == 429:
                print(
                    f"[ticketmaster] Rate limited on {city} page {page}. "
                    f"Waiting {_RATE_LIMIT_WAIT}s…"
                )
                time.sleep(_RATE_LIMIT_WAIT)
                response = requests.get(_BASE_URL, params=params, timeout=15)

            response.raise_for_status()
            data: dict = response.json()

        except requests.RequestException as exc:
            print(f"[ticketmaster] HTTP error for {city} page {page}: {exc}")
            break

        # Update total page count from first response
        if page == 0:
            api_total = data.get("page", {}).get("totalPages", 1)
            total_pages = min(api_total, _MAX_PAGES)

        if "_embedded" not in data:
            print(f"[ticketmaster] No events on {city} page {page}, stopping.")
            break

        raw_events: list[dict] = data["_embedded"].get("events", [])
        page_valid = 0
        for raw in raw_events:
            parsed = _parse_event(raw, city)
            if parsed is None:
                continue
            if _validate_event(parsed):
                events.append(parsed)
                page_valid += 1

        print(
            f"[ticketmaster] {city} page {page}/{total_pages - 1}: "
            f"{len(raw_events)} raw → {page_valid} valid (running total: {len(events)})"
        )

        if page < total_pages - 1:
            time.sleep(_PAGE_DELAY)

    return events


def fetch_ticketmaster(cities: list[str] | None = None) -> dict:
    """
    Fetch events from Ticketmaster API for the given cities.

    If cities is None, fetches all cities in TICKETMASTER_CITY_KEYWORDS.
    Events are upserted into Supabase city-by-city so a failure in one city
    does not prevent others from being persisted.

    Args:
        cities: Optional list of canonical WAVE city names to fetch.
                Each entry must be a key in TICKETMASTER_CITY_KEYWORDS.
                Unknown city names are warned about and skipped.

    Returns:
        Aggregate result dict with keys:
            "fetched"  — total valid events parsed across all cities
            "inserted" — events successfully upserted into Supabase
            "skipped"  — events that failed to upsert
    """
    api_key = os.environ.get("TICKETMASTER_API_KEY", "")
    if not api_key:
        print(
            "[ticketmaster] ERROR: TICKETMASTER_API_KEY environment variable is not set.\n"
            "  Get a free key (5 000 req/day) at https://developer.ticketmaster.com/\n"
            "  then add it to backend/.env:  TICKETMASTER_API_KEY=your_key_here"
        )
        return {"fetched": 0, "inserted": 0, "skipped": 0}

    target_cities: list[str] = (
        list(TICKETMASTER_CITY_KEYWORDS.keys()) if cities is None else list(cities)
    )

    unknown = [c for c in target_cities if c not in TICKETMASTER_CITY_KEYWORDS]
    if unknown:
        print(f"[ticketmaster] WARN: unknown cities will be skipped: {unknown}")
        target_cities = [c for c in target_cities if c in TICKETMASTER_CITY_KEYWORDS]

    now = datetime.utcnow()
    start_dt = now.strftime("%Y-%m-%dT00:00:00Z")
    end_dt   = (now + timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z")

    totals: dict[str, int] = {"fetched": 0, "inserted": 0, "skipped": 0}

    for city in target_cities:
        print(f"\n[ticketmaster] ── Fetching {city} ({start_dt} → {end_dt}) ──")
        try:
            batch = _fetch_city(city, api_key, start_dt, end_dt)
        except Exception as exc:
            print(f"[ticketmaster] Unexpected error for {city}: {exc}")
            batch = []

        totals["fetched"] += len(batch)

        if batch:
            result = upsert_events(batch, "ticketmaster")
            totals["inserted"] += result.get("inserted", 0)
            totals["skipped"]  += result.get("skipped", 0)
        else:
            print(f"[ticketmaster] No valid events found for {city}.")

    print(
        f"\n[ticketmaster] Done. "
        f"fetched={totals['fetched']} "
        f"inserted={totals['inserted']} "
        f"skipped={totals['skipped']}"
    )
    return totals


if __name__ == "__main__":
    result = fetch_ticketmaster()
    print(result)
