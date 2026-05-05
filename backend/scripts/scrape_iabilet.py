"""
backend/scripts/scrape_iabilet.py
----------------------------------
Scrapes real events from iabilet.ro for all Romanian cities and upserts
them into Supabase.

Two-phase approach (required because the listing page only shows title + link):
  Phase 1 — listing pages: extract event stubs (name, href, event_id, image).
  Phase 2 — per-event enrichment (tried in order):
              a) iabilet internal API  → structured JSON
              b) detail page JSON-LD   → schema.org/Event structured data
              c) detail page HTML      → broad regex / breadcrumb search

Run from backend/ directory:
    python -m scripts.scrape_iabilet
"""

import json
import logging
import random
import re
import time
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scripts.common import (
    CITY_CLIMATE,
    IABILET_CATEGORY_MAP,
    IABILET_CITY_MAP,
    ROMANIAN_CITIES,
    ROMANIAN_MONTHS,
    infer_is_outdoor,
    make_source_key,
    upsert_events,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════
# DOM SELECTORS — INSPECT iabilet.ro AND UPDATE THESE
# Open https://www.iabilet.ro/bilete/ in Chrome
# Right-click any event card → Inspect
# Find the selectors for each element below
# ══════════════════════════════════════════════════════
# — Listing page (https://www.iabilet.ro/bilete/) —
SEL_EVENT_CARD = "div.poster-box"      # container of ONE event card
SEL_TITLE      = "div.title"           # event name inside div.event-info
SEL_LINK       = "a"                   # first <a> wrapping the card; href → detail URL
SEL_IMAGE      = "div.image"           # container with background-image or child <img>

# — Detail page — update if the site's breadcrumb or meta structure differs
SEL_BREADCRUMB = "nav.breadcrumb a, .breadcrumb a, ol.breadcrumb li a"

BASE_URL        = "https://www.iabilet.ro"
LISTING_URL     = "https://www.iabilet.ro/bilete/"
# NOTE: API endpoint is speculative — verified by opening DevTools Network tab
# while visiting an event page and watching for XHR calls to /api/event/*
API_EVENT_URL   = "https://www.iabilet.ro/api/event/{event_id}"
MAX_PAGES       = 10
REQUEST_DELAY   = (1.5, 3.0)           # random delay range in seconds
REQUEST_TIMEOUT = 15
MAX_RETRIES     = 3

# ── City URL slugs ─────────────────────────────────────────────────────────────
# NOTE: Verify by visiting https://www.iabilet.ro/bilete/ and filtering by city
# in the browser. Inspect the resulting URL to confirm the param name (?oras=)
# and slug values match what the site actually uses.
CITY_URL_SLUGS: dict[str, str] = {
    "Bucharest":   "bucuresti",
    "Cluj-Napoca": "cluj-napoca",
    "Timisoara":   "timisoara",
    "Iasi":        "iasi",
    "Constanta":   "constanta",
    "Brasov":      "brasov",
}

# Module-level UserAgent instance — initialised once; .random is cheap
_ua = UserAgent()

# Date format patterns tried in order by parse_romanian_date
_DATE_FORMATS: list[str] = ["%d %m %Y", "%d.%m.%Y", "%Y-%m-%d"]

# Regex patterns reused across detail parsers
_RE_DATE = re.compile(r"\d{1,2}\s+\w+\s+\d{4}")
_RE_TIME = re.compile(r"\b(\d{1,2}:\d{2})\b")
_RE_LIKABLE = re.compile(r"event/(\d+)")
_RE_BG_URL = re.compile(r"url\(['\"]?([^'\")\s]+)['\"]?\)")


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """
    Create an HTTP session pre-configured for iabilet.ro scraping.

    The User-Agent is deliberately not set here — _fetch_page rotates it to a
    fresh random value before each request so the server sees different clients.

    Returns:
        requests.Session with Accept-Language and Referer headers preset.
    """
    session = requests.Session()
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
        "Referer":         "https://www.iabilet.ro/",
    })
    return session


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
)
def _fetch_page(session: requests.Session, url: str) -> requests.Response:
    """
    Fetch a URL with a random delay and a rotated User-Agent.

    Decorated with tenacity: up to MAX_RETRIES attempts with exponential
    back-off, retrying only on network-level errors (Timeout, ConnectionError).
    HTTP 4xx/5xx responses are returned as-is — the caller is responsible for
    calling response.raise_for_status() when needed.

    Args:
        session: Active requests.Session whose headers are mutated in-place.
        url:     Absolute URL to fetch.

    Returns:
        requests.Response object.
    """
    session.headers["User-Agent"] = _ua.random
    time.sleep(random.uniform(*REQUEST_DELAY))
    return session.get(url, timeout=REQUEST_TIMEOUT)


# ── Date / time parsing ────────────────────────────────────────────────────────

def parse_romanian_date(date_str: str, time_str: str) -> Optional[tuple[str, int]]:
    """
    Parse a human-readable Romanian date/time pair into (YYYY-MM-DD, hour).

    Steps:
      1. Lowercase the date string.
      2. Replace month names (longest first to avoid partial matches) with
         their zero-padded numeric equivalents from ROMANIAN_MONTHS.
      3. Collapse extra whitespace.
      4. Try each pattern in _DATE_FORMATS until one succeeds.
      5. Parse the hour from time_str (HH:MM or HH:MM:SS), default 20.

    Args:
        date_str: Raw date string (e.g. "14 Iulie 2025" or "2025-07-14").
        time_str: Raw time string (e.g. "20:00" or "20:00:00").  May be empty.

    Returns:
        Tuple of (ISO date string "YYYY-MM-DD", hour int), or None on failure.
    """
    hour = 20
    if time_str:
        try:
            hour = int(time_str.strip().split(":")[0])
        except (ValueError, IndexError):
            hour = 20

    normalised = date_str.strip().lower()
    for month_name in sorted(ROMANIAN_MONTHS, key=len, reverse=True):
        normalised = normalised.replace(month_name, ROMANIAN_MONTHS[month_name])
    normalised = re.sub(r"\s+", " ", normalised).strip()

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(normalised, fmt)
            return parsed.strftime("%Y-%m-%d"), hour
        except ValueError:
            continue

    return None


# ── Phase 1 — listing stub extraction ─────────────────────────────────────────

def _parse_listing_stub(card: BeautifulSoup) -> Optional[dict]:
    """
    Phase 1: extract the minimal fields available on the listing page.

    The iabilet listing page exposes only: event title, detail-page href,
    event_id (from data-likable-item), and an optional image.  All other
    fields require fetching the detail page (Phase 2).

    Args:
        card: BeautifulSoup Tag matched by SEL_EVENT_CARD ("div.poster-box").

    Returns:
        Dict with keys {event_name, href, event_id, image_url}, or None if
        the card is missing the required title or link.
    """
    # ── Title ──────────────────────────────────────────────────────────────────
    title_el = card.select_one(SEL_TITLE)
    if not title_el:
        logger.warning("Listing card missing title element, skipping.")
        return None
    raw_title = title_el.get_text(strip=True)
    if not raw_title:
        logger.warning("Listing card has empty title, skipping.")
        return None

    # ── Link ───────────────────────────────────────────────────────────────────
    link_el = card.select_one(SEL_LINK)
    if not link_el:
        logger.warning("Listing card missing <a> for '%s', skipping.", raw_title)
        return None
    href = link_el.get("href", "").strip()
    if not href:
        logger.warning("Listing card <a> has no href for '%s', skipping.", raw_title)
        return None

    # ── event_id from data-likable-item ───────────────────────────────────────
    likable = card.get("data-likable-item", "")
    m = _RE_LIKABLE.search(likable)
    event_id: Optional[str] = m.group(1) if m else None

    # ── Image ──────────────────────────────────────────────────────────────────
    image_url: Optional[str] = None
    image_container = card.select_one(SEL_IMAGE)
    if image_container:
        # Prefer a child <img> tag
        img_tag = image_container.find("img")
        if img_tag:
            src = img_tag.get("src") or img_tag.get("data-src") or ""
            if src:
                image_url = src if src.startswith("http") else BASE_URL + src

        # Fall back to CSS background-image on the container itself
        if not image_url:
            style = image_container.get("style", "")
            bg_match = _RE_BG_URL.search(style)
            if bg_match:
                src = bg_match.group(1)
                image_url = src if src.startswith("http") else BASE_URL + src

    return {
        "event_name": raw_title[:255],
        "href":       href,
        "event_id":   event_id,
        "image_url":  image_url,
    }


# ── Phase 2a — internal API ───────────────────────────────────────────────────

def _try_api_detail(session: requests.Session, event_id: str) -> Optional[dict]:
    """
    Phase 2a: attempt to retrieve structured event data from the iabilet API.

    The endpoint URL and response schema are speculative.  This function handles
    every failure mode silently and returns None rather than raising, so callers
    can fall through to the HTML-based parsers.

    Tries several candidate field names for each datum (date, city, venue,
    category) because the schema is unknown until a real response is observed.

    Args:
        session:  Active requests.Session (UA not rotated here — no delay needed
                  for an API call that follows a listing fetch).
        event_id: Numeric event identifier extracted from data-likable-item.

    Returns:
        Enrichment dict {event_date_str, event_hour, raw_city, venue_name,
        raw_category} if a parseable date was found, otherwise None.
    """
    url = API_EVENT_URL.format(event_id=event_id)
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("API returned HTTP %d for event %s.", resp.status_code, event_id)
            return None
        data = resp.json()
    except Exception as exc:
        logger.debug("API fetch failed for event %s: %s", event_id, exc)
        return None

    if not isinstance(data, dict):
        return None

    # ── Date ──────────────────────────────────────────────────────────────────
    raw_date: str = (
        data.get("startDate") or data.get("start_date") or
        data.get("date") or data.get("eventDate") or ""
    )
    if not raw_date:
        return None  # without a date this response is useless

    event_date_str: Optional[str] = None
    event_hour = 20
    try:
        if "T" in raw_date:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            event_date_str = dt.strftime("%Y-%m-%d")
            event_hour = dt.hour
        else:
            event_date_str = raw_date[:10]
    except ValueError:
        result = parse_romanian_date(raw_date, "")
        if result:
            event_date_str, event_hour = result

    if not event_date_str:
        return None

    # ── Time (if not embedded in the date field) ──────────────────────────────
    if event_hour == 20:
        raw_time: str = (
            data.get("startTime") or data.get("start_time") or data.get("time") or ""
        )
        if raw_time:
            try:
                event_hour = int(raw_time.split(":")[0])
            except (ValueError, IndexError):
                pass

    # ── City / venue / category ────────────────────────────────────────────────
    raw_city: str = (
        data.get("city") or data.get("cityName") or data.get("city_name") or
        data.get("location") or ""
    )
    venue_name: str = (
        data.get("venue") or data.get("venueName") or data.get("venue_name") or
        data.get("location_name") or ""
    )
    raw_category: str = (
        data.get("category") or data.get("categoryName") or
        data.get("type") or data.get("genre") or ""
    )

    return {
        "event_date_str": event_date_str,
        "event_hour":     event_hour,
        "raw_city":       raw_city if isinstance(raw_city, str) else "",
        "venue_name":     venue_name if isinstance(venue_name, str) else "",
        "raw_category":   raw_category if isinstance(raw_category, str) else "",
    }


# ── Phase 2b — JSON-LD structured data ────────────────────────────────────────

def _parse_jsonld(soup: BeautifulSoup) -> Optional[dict]:
    """
    Phase 2b: extract event fields from a JSON-LD <script> block (schema.org).

    Event sites commonly embed a complete schema.org/Event object in JSON-LD,
    including startDate, location (name + addressLocality), and performer.
    This is the most reliable HTML-level source when available.

    Handles both a single object and an array root; also handles @graph wrappers.

    Args:
        soup: Parsed BeautifulSoup of the event detail page.

    Returns:
        Enrichment dict or None if no parseable JSON-LD Event was found.
    """
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        # Unwrap array or @graph
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict) and "@graph" in data:
            candidates = data["@graph"]
        else:
            candidates = [data]

        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("@type", "")
            if not (isinstance(obj_type, str) and "Event" in obj_type):
                continue

            # ── startDate ────────────────────────────────────────────────────
            start = obj.get("startDate", "")
            event_date_str: Optional[str] = None
            event_hour = 20
            if start:
                try:
                    if "T" in start:
                        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        event_date_str = dt.strftime("%Y-%m-%d")
                        event_hour = dt.hour
                    else:
                        event_date_str = start[:10]
                except ValueError:
                    result = parse_romanian_date(start, "")
                    if result:
                        event_date_str, event_hour = result

            if not event_date_str:
                continue  # try next candidate

            # ── Location ──────────────────────────────────────────────────────
            location = obj.get("location", {})
            if not isinstance(location, dict):
                location = {}
            venue_name: str = location.get("name", "") or ""
            address = location.get("address", {})
            if isinstance(address, dict):
                raw_city: str = address.get("addressLocality", "") or ""
            elif isinstance(address, str):
                raw_city = address
            else:
                raw_city = ""

            return {
                "event_date_str": event_date_str,
                "event_hour":     event_hour,
                "raw_city":       raw_city,
                "venue_name":     venue_name,
                "raw_category":   None,  # not standardised in schema.org/Event
            }

    return None


# ── Phase 2c — broad HTML search ──────────────────────────────────────────────

def _parse_detail_html(soup: BeautifulSoup) -> dict:
    """
    Phase 2c: fallback — broad regex and breadcrumb search on the detail page.

    Used when neither the API nor JSON-LD returned usable data.  Searches the
    full page text for date and time patterns, and scans breadcrumb links to
    infer category and city.

    Never raises; always returns a dict (values may be None / empty).

    Args:
        soup: Parsed BeautifulSoup of the event detail page.

    Returns:
        Enrichment dict {event_date_str, event_hour, raw_city, venue_name,
        raw_category} — any field may be None/"" if not found.
    """
    result: dict = {
        "event_date_str": None,
        "event_hour":     20,
        "raw_city":       None,
        "venue_name":     "",
        "raw_category":   None,
    }

    # ── Date via full-text regex scan ──────────────────────────────────────────
    date_node = soup.find(string=_RE_DATE)
    if date_node:
        m = _RE_DATE.search(date_node)
        if m:
            parsed = parse_romanian_date(m.group(), "")
            if parsed:
                result["event_date_str"], result["event_hour"] = parsed

    # ── Time via full-text regex scan (only if date was found) ────────────────
    if result["event_date_str"]:
        time_node = soup.find(string=_RE_TIME)
        if time_node:
            m = _RE_TIME.search(time_node)
            if m:
                try:
                    result["event_hour"] = int(m.group(1).split(":")[0])
                except (ValueError, IndexError):
                    pass

    # ── Breadcrumbs: city and category ────────────────────────────────────────
    for link in soup.select(SEL_BREADCRUMB):
        text = link.get_text(strip=True).lower()
        if result["raw_category"] is None and IABILET_CATEGORY_MAP.get(text):
            result["raw_category"] = text
        if result["raw_city"] is None and IABILET_CITY_MAP.get(text):
            result["raw_city"] = text

    return result


# ── Phase 2 orchestrator ───────────────────────────────────────────────────────

def _fetch_detail(
    session: requests.Session,
    href: str,
    event_id: Optional[str],
) -> dict:
    """
    Phase 2 orchestrator: enrich a listing stub with date, city, category, venue.

    Tries data sources in priority order and returns the first one that yields a
    parseable date.  If all sources fail the returned dict has event_date_str=None.

    Priority:
      1. iabilet internal API  (fast, no HTML parsing, if event_id available)
      2. JSON-LD on detail page (reliable when present — schema.org standard)
      3. Broad HTML search      (last resort — least reliable)

    Args:
        session:  Active requests.Session.
        href:     Relative or absolute URL to the event detail page.
        event_id: Numeric event id from data-likable-item, or None.

    Returns:
        Enrichment dict with keys: event_date_str, event_hour, raw_city,
        venue_name, raw_category.
    """
    empty: dict = {
        "event_date_str": None,
        "event_hour":     20,
        "raw_city":       None,
        "venue_name":     "",
        "raw_category":   None,
    }

    # 1. API
    if event_id:
        api_data = _try_api_detail(session, event_id)
        if api_data and api_data.get("event_date_str"):
            logger.debug("Detail sourced from API for event %s.", event_id)
            return api_data

    # 2 + 3. Fetch detail page once, try JSON-LD then HTML
    detail_url = href if href.startswith("http") else BASE_URL + href
    try:
        response = _fetch_page(session, detail_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
    except Exception as exc:
        logger.warning("Failed to fetch detail page %s: %s", detail_url, exc)
        return empty

    jsonld = _parse_jsonld(soup)
    if jsonld and jsonld.get("event_date_str"):
        logger.debug("Detail sourced from JSON-LD for %s.", detail_url)
        return jsonld

    logger.debug("Falling back to HTML search for %s.", detail_url)
    return _parse_detail_html(soup)


# ── Final event assembly ───────────────────────────────────────────────────────

def _build_event(
    stub: dict,
    detail: dict,
    canonical_city: str,
) -> Optional[dict]:
    """
    Assemble a complete WAVE event dict from a listing stub and enrichment data.

    canonical_city (from the scrape loop's listing-URL filter) is the trusted
    city value.  The city parsed from the detail page is used only as a debug
    hint — discarded if it maps to a different canonical city, to avoid
    cross-city contamination.

    Args:
        stub:           Dict from _parse_listing_stub (event_name, href, image_url).
        detail:         Dict from _fetch_detail (date, hour, city, venue, category).
        canonical_city: Canonical WAVE city name for this scrape iteration.

    Returns:
        Normalized event dict with all required WAVE keys, or None if a required
        field (date, event_type) could not be resolved.
    """
    today = date.today()

    event_date_str = detail.get("event_date_str")
    if not event_date_str:
        logger.warning("No date found for '%s', skipping.", stub["event_name"])
        return None

    try:
        event_date = date.fromisoformat(event_date_str)
    except ValueError:
        logger.warning(
            "Invalid date '%s' for '%s', skipping.", event_date_str, stub["event_name"]
        )
        return None

    if event_date < today:
        return None  # silently drop past events

    event_hour: int = detail.get("event_hour") or 20

    # ── Category ───────────────────────────────────────────────────────────────
    raw_category = (detail.get("raw_category") or "").strip().lower()
    event_type = IABILET_CATEGORY_MAP.get(raw_category)
    if event_type is None:
        logger.warning(
            "Unmapped category '%s' for '%s', skipping.", raw_category, stub["event_name"]
        )
        return None

    # ── Venue and outdoor flag ─────────────────────────────────────────────────
    venue_name: str = (detail.get("venue_name") or "").strip()
    is_outdoor = infer_is_outdoor(event_type, venue_name)

    # ── Build final dict ───────────────────────────────────────────────────────
    event_name   = stub["event_name"]
    climate_zone = CITY_CLIMATE.get(canonical_city, "Moderate")
    source_key   = make_source_key(event_name, event_date_str, canonical_city)
    href         = stub["href"]
    ticket_url   = href if href.startswith("http") else BASE_URL + href

    return {
        "event_name":   event_name,
        "event_type":   event_type,
        "location":     canonical_city,
        "climate_zone": climate_zone,
        "is_outdoor":   is_outdoor,
        "event_date":   event_date_str,
        "event_hour":   event_hour,
        "source":       "iabilet",
        "is_generated": False,
        "source_key":   source_key,
        "ticket_url":   ticket_url,
        "image_url":    stub.get("image_url"),
        "venue_name":   venue_name,
    }


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_iabilet(cities: list[str] | None = None) -> dict:
    """
    Scrape iabilet.ro for events in the given Romanian cities.

    For each city, paginates through up to MAX_PAGES listing pages (stopping
    early when a page yields zero cards), then enriches each stub via a
    two-phase detail fetch.  Events are deduplicated within the current run
    by source_key and upserted city-by-city so a failure in one city does not
    prevent others from being persisted.

    Args:
        cities: Optional list of canonical Romanian city names to scrape.
                Each entry must be a key in CITY_URL_SLUGS.
                Unknown names are warned about and skipped.

    Returns:
        Aggregate result dict with keys:
            "scraped"  — total valid events assembled across all cities
            "inserted" — events successfully upserted into Supabase
            "skipped"  — events that failed to upsert
    """
    target_cities: list[str] = (
        list(ROMANIAN_CITIES) if cities is None else list(cities)
    )

    unknown = [c for c in target_cities if c not in CITY_URL_SLUGS]
    if unknown:
        logger.warning("Unknown cities will be skipped: %s", unknown)
        target_cities = [c for c in target_cities if c in CITY_URL_SLUGS]

    session = _build_session()
    totals: dict[str, int] = {"scraped": 0, "inserted": 0, "skipped": 0}

    for city in target_cities:
        logger.info("── Scraping %s ──", city)
        city_slug  = CITY_URL_SLUGS[city]
        batch:     list[dict] = []
        seen_keys: set[str]   = set()

        for page in range(1, MAX_PAGES + 1):
            url = f"{LISTING_URL}?oras={city_slug}&page={page}"
            logger.info("%s page %d → %s", city, page, url)

            # ── Phase 1: fetch listing page ────────────────────────────────────
            try:
                response = _fetch_page(session, url)
                response.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch %s page %d: %s", city, page, exc)
                break

            try:
                soup  = BeautifulSoup(response.text, "lxml")
                cards = soup.select(SEL_EVENT_CARD)
            except Exception as exc:
                logger.warning("Failed to parse HTML for %s page %d: %s", city, page, exc)
                break

            if not cards:
                logger.info("%s page %d: no event cards — stopping pagination.", city, page)
                break

            # ── Phase 2: enrich each stub ──────────────────────────────────────
            page_valid = 0
            for card in cards:
                try:
                    stub = _parse_listing_stub(card)
                except Exception as exc:
                    logger.warning(
                        "Unexpected error parsing listing card on %s page %d: %s",
                        city, page, exc,
                    )
                    continue

                if stub is None:
                    continue

                try:
                    detail = _fetch_detail(session, stub["href"], stub["event_id"])
                    event  = _build_event(stub, detail, city)
                except Exception as exc:
                    logger.warning(
                        "Unexpected error enriching '%s': %s", stub["event_name"], exc
                    )
                    continue

                if event is None:
                    continue

                sk = event["source_key"]
                if sk in seen_keys:
                    continue
                seen_keys.add(sk)
                batch.append(event)
                page_valid += 1

            logger.info(
                "%s page %d: %d cards → %d new valid events (batch total: %d)",
                city, page, len(cards), page_valid, len(batch),
            )

        totals["scraped"] += len(batch)

        if batch:
            result = upsert_events(batch, "iabilet")
            totals["inserted"] += result.get("inserted", 0)
            totals["skipped"]  += result.get("skipped", 0)
        else:
            logger.info("No valid events found for %s.", city)

    logger.info(
        "Done. scraped=%d inserted=%d skipped=%d",
        totals["scraped"], totals["inserted"], totals["skipped"],
    )
    return totals


if __name__ == "__main__":
    result = scrape_iabilet()
    print(result)
