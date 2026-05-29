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
import math
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
# Values must match ml_service.py LABEL_MAPS["climate_zone"].
# Capitalized form ("Cold", "Hot", "Moderate", "Rainy") is used here and in
# event_service.py. The LABEL_MAPS also accepts lowercase training-time aliases
# ("cold", "heat", "moderate", "rain") so old Supabase rows are handled safely.

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


# ── Canonical city coordinates ────────────────────────────────────────────────
# (lat, lng) centre-point for each canonical city.
# Used by nearest_canonical_city() to match arbitrary coordinates.

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Bucharest":   (44.4268,  26.1025),
    "Cluj-Napoca": (46.7712,  23.6236),
    "Timisoara":   (45.7489,  21.2087),
    "Iasi":        (47.1585,  27.6014),
    "Constanta":   (44.1598,  28.6348),
    "Brasov":      (45.6427,  25.5887),
    "London":      (51.5074,  -0.1278),
    "Bergen":      (60.3913,   5.3221),
    "Seattle":     (47.6062, -122.3321),
    "Oslo":        (59.9139,  10.7522),
    "Helsinki":    (60.1699,  25.0000),
    "Quebec":      (46.8139,  -71.2082),
    "Dubai":       (25.2048,  55.2708),
    "Phoenix":     (33.4484, -112.0740),
    "Seville":     (37.3891,  -5.9845),
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

def coords_to_climate_zone(lat: float, lng: float) -> str:
    """Return a WAVE climate zone ("Hot", "Rainy", "Cold", "Moderate") for any coordinates.

    Rules are latitude-band heuristics cross-checked against all 15 canonical
    cities so that known cities produce the same zone as CITY_CLIMATE.

    Verification against canonical cities:
        Bucharest 44.4N 26.1E  → Moderate ✓   Dubai   25.2N 55.3E → Hot  ✓
        Cluj-N.   46.8N 23.6E  → Moderate ✓   Phoenix 33.4N 112.1W → Hot  ✓
        London    51.5N  0.1W  → Rainy    ✓   Seville 37.4N  5.9W → Hot  ✓
        Bergen    60.4N  5.3E  → Rainy    ✓   Oslo    59.9N 10.7E → Cold ✓
        Seattle   47.6N 122.3W → Rainy    ✓   Helsinki 60.2N 25.0E → Cold ✓
                                               Quebec  46.8N 71.2W → Cold ✓
    """
    abs_lat = abs(lat)

    # Polar / sub-arctic
    if abs_lat >= 63:
        return "Cold"

    # Tropical belt
    if abs_lat < 23.5:
        return "Hot"

    # Subtropical (23.5–40°): hot/arid across all longitudes
    if abs_lat < 40:
        return "Hot"

    # Temperate zone (40–63°) — distinguish maritime, cold-continental, moderate
    # Atlantic maritime: British Isles, Low Countries, France, coastal Norway
    if -15 <= lng <= 10:
        return "Rainy"

    # Pacific maritime: US/Canada Pacific coast (Seattle, Portland, Vancouver)
    if -130 <= lng <= -114 and 42 <= lat <= 56:
        return "Rainy"

    # Sub-arctic / cold continental
    if abs_lat >= 55:
        return "Cold"
    if lat > 42 and lng < -60:  # N American interior/east (Quebec, Montreal, etc.)
        return "Cold"

    return "Moderate"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres between two coordinate pairs."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def nearest_canonical_city(lat: float, lng: float, radius_km: float = 100.0) -> str | None:
    """Return the nearest canonical city name if it is within *radius_km*, else None.

    Iterates over all entries in CITY_COORDS, computes the Haversine distance,
    and returns the city whose centre is closest — provided that distance is
    within the supplied threshold (default 100 km).

    Args:
        lat:       Latitude of the query point in decimal degrees.
        lng:       Longitude of the query point in decimal degrees.
        radius_km: Maximum distance (km) to consider a city "nearby".

    Returns:
        A canonical city name string (key in CITY_CLIMATE) or None.

    Examples:
        >>> nearest_canonical_city(44.5, 26.0)   # just outside Bucharest
        'Bucharest'
        >>> nearest_canonical_city(48.8, 2.35)   # Paris — no canonical city within 100 km
        None
    """
    best_city: str | None = None
    best_dist = float("inf")

    for city, (clat, clng) in CITY_COORDS.items():
        d = _haversine_km(lat, lng, clat, clng)
        if d < best_dist:
            best_dist = d
            best_city = city

    return best_city if best_dist <= radius_km else None


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

    # Deduplicate within the batch — same event can appear on multiple API pages
    seen: dict[str, dict] = {}
    for e in events:
        seen[e["source_key"]] = e
    deduped = list(seen.values())
    if len(deduped) < len(events):
        print(f"[{source_label}] Removed {len(events) - len(deduped)} duplicate(s) before upsert.")

    try:
        from app.core.database import get_supabase_admin_client  # noqa: PLC0415
        client = get_supabase_admin_client()
        client.table("events").upsert(deduped, on_conflict="source_key").execute()
        print(f"[{source_label}] ✓ Upserted {len(deduped)} event(s) into Supabase.")
        return {"inserted": len(deduped), "skipped": 0}
    except Exception as exc:
        print(f"[{source_label}] ✗ Upsert failed: {exc}")
        return {"inserted": 0, "skipped": len(deduped)}
