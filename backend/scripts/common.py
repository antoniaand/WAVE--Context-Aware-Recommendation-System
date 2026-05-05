"""
backend/scripts/common.py
--------------------------
Shared constants, lookup tables, and utility functions used by all WAVE
data-pipeline scripts (scrapers, importers, etc.).

Run from backend/ directory:
    python -m scripts.scrape_iabilet
    python -m scripts.fetch_ticketmaster
"""

import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add backend/ to sys.path so "from app.core.database import ..." works when
# scripts are executed as modules from the backend/ directory.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Load .env from backend/.env before any app imports resolve settings
load_dotenv(dotenv_path=_BACKEND_DIR / ".env")


# ── Canonical cities ───────────────────────────────────────────────────────────
# Must match the keys in ml_service.py LABEL_MAPS["location"] exactly.

CANONICAL_CITIES: set[str] = {
    # Romanian (moderate climate)
    "Bucharest", "Cluj-Napoca", "Timisoara", "Iasi", "Constanta", "Brasov",
    # Rainy hotspots
    "London", "Bergen", "Seattle",
    # Cold hotspots
    "Oslo", "Helsinki", "Quebec",
    # Hot hotspots
    "Dubai", "Phoenix", "Seville",
}

ROMANIAN_CITIES: set[str] = {
    "Bucharest", "Cluj-Napoca", "Timisoara", "Iasi", "Constanta", "Brasov",
}

INTERNATIONAL_CITIES: set[str] = CANONICAL_CITIES - ROMANIAN_CITIES


# ── Canonical event types ──────────────────────────────────────────────────────
# Must match ml_service.py LABEL_MAPS["event_type"] and LABEL_MAPS["top_event"].

CANONICAL_EVENT_TYPES: set[str] = {
    "Concert", "Festival", "Sports", "Theatre", "Conference",
}


# ── Climate zone mapping ───────────────────────────────────────────────────────
# Values must match ml_service.py LABEL_MAPS["climate_zone"]:
#   {"Cold": 0, "Hot": 1, "Moderate": 2, "Rainy": 3}

CITY_CLIMATE: dict[str, str] = {
    # Romanian cities — temperate continental
    "Bucharest":   "Moderate",
    "Cluj-Napoca": "Moderate",
    "Timisoara":   "Moderate",
    "Iasi":        "Moderate",
    "Constanta":   "Moderate",
    "Brasov":      "Moderate",
    # Rainy hotspots — Atlantic / Pacific maritime
    "London":      "Rainy",
    "Bergen":      "Rainy",
    "Seattle":     "Rainy",
    # Cold hotspots — sub-arctic / continental cold
    "Oslo":        "Cold",
    "Helsinki":    "Cold",
    "Quebec":      "Cold",
    # Hot hotspots — desert / semi-arid
    "Dubai":       "Hot",
    "Phoenix":     "Hot",
    "Seville":     "Hot",
}


# ── Ticketmaster API city params ───────────────────────────────────────────────
# Maps each WAVE international city to the keyword params sent to the
# Ticketmaster Discovery API (/events.json?city=...&countryCode=...).
# "Seville" uses the Spanish spelling "Sevilla" as required by the API.

TICKETMASTER_CITY_KEYWORDS: dict[str, dict[str, str]] = {
    "London":   {"city": "London",   "countryCode": "GB"},
    "Oslo":     {"city": "Oslo",     "countryCode": "NO"},
    "Helsinki": {"city": "Helsinki", "countryCode": "FI"},
    "Bergen":   {"city": "Bergen",   "countryCode": "NO"},
    "Seattle":  {"city": "Seattle",  "countryCode": "US"},
    "Dubai":    {"city": "Dubai",    "countryCode": "AE"},
    "Phoenix":  {"city": "Phoenix",  "countryCode": "US"},
    "Seville":  {"city": "Sevilla",  "countryCode": "ES"},
    "Quebec":   {"city": "Quebec",   "countryCode": "CA"},
}


# ── Ticketmaster segment → WAVE event type ─────────────────────────────────────
# Ticketmaster returns a "segment.name" field for each event.
# Map it to one of the five CANONICAL_EVENT_TYPES.

TICKETMASTER_CATEGORY_MAP: dict[str, str] = {
    "Music":          "Concert",
    "Sports":         "Sports",
    "Arts & Theatre": "Theatre",
    "Film":           "Theatre",
    "Miscellaneous":  "Conference",
    "Family":         "Festival",
    "Festival":       "Festival",
    "Conference":     "Conference",
    "Exhibition":     "Conference",
}


# ── iabilet.ro category slug → WAVE event type ────────────────────────────────
# iabilet URL slugs appear in category breadcrumbs and filter params.
# Comedy and stand-up map to Concert as the closest energetic live-performance type.

IABILET_CATEGORY_MAP: dict[str, str] = {
    "concerte":   "Concert",
    "muzica":     "Concert",
    "stand-up":   "Concert",
    "comedy":     "Concert",
    "festivaluri":"Festival",
    "festival":   "Festival",
    "sport":      "Sports",
    "sporturi":   "Sports",
    "teatru":     "Theatre",
    "teatru-dans":"Theatre",
    "opera":      "Theatre",
    "balet":      "Theatre",
    "spectacole": "Theatre",
    "conferinte": "Conference",
    "conferinta": "Conference",
    "business":   "Conference",
}


# ── iabilet.ro city name → canonical city ─────────────────────────────────────
# iabilet uses lowercase, diacritic-free or diacritic-bearing Romanian spellings.
# All variants map to the CANONICAL_CITIES spelling used in LABEL_MAPS.

IABILET_CITY_MAP: dict[str, str] = {
    "bucuresti":   "Bucharest",
    "bucurești":   "Bucharest",
    "cluj-napoca": "Cluj-Napoca",
    "cluj":        "Cluj-Napoca",
    "timisoara":   "Timisoara",
    "timișoara":   "Timisoara",
    "iasi":        "Iasi",
    "iași":        "Iasi",
    "constanta":   "Constanta",
    "constanța":   "Constanta",
    "brasov":      "Brasov",
    "brașov":      "Brasov",
}


# ── Romanian + English month names → zero-padded month number ─────────────────
# Used by scrapers that parse human-readable date strings from event pages.
# Includes both full names and common abbreviations, all lowercased.

ROMANIAN_MONTHS: dict[str, str] = {
    # Romanian full names
    "ianuarie":    "01",
    "februarie":   "02",
    "martie":      "03",
    "aprilie":     "04",
    "mai":         "05",
    "iunie":       "06",
    "iulie":       "07",
    "august":      "08",
    "septembrie":  "09",
    "octombrie":   "10",
    "noiembrie":   "11",
    "decembrie":   "12",
    # English full names
    "january":     "01",
    "february":    "02",
    "march":       "03",
    "april":       "04",
    # "may" already covered above (same spelling)
    "june":        "06",
    "july":        "07",
    # "august" already covered above
    "september":   "09",
    "october":     "10",
    "november":    "11",
    "december":    "12",
    # English abbreviations
    "jan":         "01",
    "feb":         "02",
    "mar":         "03",
    "apr":         "04",
    "jun":         "06",
    "jul":         "07",
    "aug":         "08",
    "sep":         "09",
    "sept":        "09",
    "oct":         "10",
    "nov":         "11",
    "dec":         "12",
}


# ── Outdoor venue keyword detection ───────────────────────────────────────────
# Substrings searched (case-insensitive) in venue names to infer is_outdoor.
# Used by infer_is_outdoor() as a fallback when the event type is ambiguous.

OUTDOOR_VENUE_KEYWORDS: list[str] = [
    "parc", "arena", "arenă", "stadion", "piața", "piata", "amfiteatru",
    "lac", "outdoor", "open air", "open-air", "gradina", "grădină",
    "park", "stadium", "square", "field", "beach", "plaja", "plajă",
]

# Event types that are always indoor or always outdoor regardless of venue name.
ALWAYS_OUTDOOR_TYPES: set[str] = {"Festival", "Sports"}
ALWAYS_INDOOR_TYPES:  set[str] = {"Theatre", "Conference"}


# ── Utility functions ──────────────────────────────────────────────────────────

def make_source_key(event_name: str, event_date: str, location: str) -> str:
    """
    Return a 16-character SHA-256 hex fingerprint for deduplication.

    The key is derived from the tuple (event_name, event_date, location),
    each normalised to lowercase and stripped of leading/trailing whitespace
    before hashing, so minor formatting differences produce the same key.

    Used as the `source_key` column value and as the ON CONFLICT target when
    upserting into the Supabase `events` table.

    Args:
        event_name: Human-readable event title (e.g. "Untold Festival 2025").
        event_date: ISO-8601 date string (e.g. "2025-07-04").
        location:   Canonical city name (e.g. "Cluj-Napoca").

    Returns:
        16-character lowercase hexadecimal string.
    """
    fingerprint = "|".join([
        event_name.strip().lower(),
        event_date.strip().lower(),
        location.strip().lower(),
    ])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def infer_is_outdoor(event_type: str, venue_name: str = "") -> bool:
    """
    Infer whether an event is held outdoors.

    Priority order:
      1. If event_type is in ALWAYS_INDOOR_TYPES  → False (Theatre, Conference).
      2. If event_type is in ALWAYS_OUTDOOR_TYPES → True  (Festival, Sports).
      3. If any OUTDOOR_VENUE_KEYWORDS found in venue_name (case-insensitive) → True.
      4. Default → False (assume indoor when uncertain).

    Args:
        event_type: Canonical event type string (must be in CANONICAL_EVENT_TYPES).
        venue_name: Raw venue string from the scraper (may be empty).

    Returns:
        True if the event is considered outdoor, False otherwise.
    """
    if event_type in ALWAYS_INDOOR_TYPES:
        return False
    if event_type in ALWAYS_OUTDOOR_TYPES:
        return True
    venue_lower = venue_name.lower()
    return any(kw in venue_lower for kw in OUTDOOR_VENUE_KEYWORDS)


def upsert_events(events: list[dict], source_label: str) -> dict:
    """
    Upsert a list of event dicts into the Supabase ``events`` table.

    Uses ``on_conflict="source_key"`` so repeated scraper runs update existing
    rows rather than raising duplicate-key errors.  Each dict in *events* must
    contain at minimum: ``event_name``, ``event_type``, ``location``,
    ``event_date``, and ``source_key``.

    The Supabase admin client is imported lazily (inside this function) so the
    module can be imported without triggering settings validation — useful for
    unit tests that only need the constants.

    Args:
        events:       List of row dicts ready for insertion.
        source_label: Short identifier printed in log messages (e.g. "iabilet",
                      "ticketmaster").

    Returns:
        ``{"inserted": len(events), "skipped": 0}`` on success.
        ``{"inserted": 0, "skipped": len(events)}``  on any exception.
    """
    if not events:
        print(f"[{source_label}] No events to upsert.")
        return {"inserted": 0, "skipped": 0}

    try:
        from app.core.database import get_supabase_admin_client  # noqa: PLC0415
        client = get_supabase_admin_client()
        client.table("events").upsert(events, on_conflict="source_key").execute()
        print(f"[{source_label}] ✓ Upserted {len(events)} event(s) into Supabase.")
        return {"inserted": len(events), "skipped": 0}
    except Exception as exc:
        print(f"[{source_label}] ✗ Upsert failed: {exc}")
        return {"inserted": 0, "skipped": len(events)}
