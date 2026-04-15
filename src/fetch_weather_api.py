"""
fetch_weather_api.py
--------------------
Step 2 of the modular WAVE dataset rebuild.

Fetches real historical weather from the Open-Meteo Historical Weather API
for every unique (city, date, hour) combination present in
interaction_foundation.csv.

Strategy: one API request per city covering the full event date range
(Jun 2024 - Jun 2025) rather than one call per event, then a local
hour-level lookup.  This reduces 300 potential calls to exactly 6.

Outputs:
  data/raw/weather_archive_cache.csv        -- raw hourly weather per city
  data/processed/interaction_with_weather.csv -- enriched 33,000-row dataset
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[1]
FOUNDATION_CSV = ROOT / "data" / "processed" / "interaction_foundation.csv"
CACHE_CSV      = ROOT / "data" / "raw" / "weather_archive_cache.csv"
OUTPUT_CSV     = ROOT / "data" / "processed" / "interaction_with_weather.csv"

# ── City coordinates (WGS-84) ─────────────────────────────────────────────────
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Moderate — Romanian cities
    "Bucharest":   (44.43,  26.10),
    "Cluj-Napoca": (46.77,  23.62),
    "Timisoara":   (45.75,  21.23),
    "Iasi":        (47.16,  27.58),
    "Constanta":   (44.17,  28.63),
    "Brasov":      (45.65,  25.60),
    # Cold hotspots — Dec-Feb window
    "Oslo":        (59.91,  10.75),
    "Helsinki":    (60.17,  24.94),
    "Quebec":      (46.81, -71.21),
    # Heat hotspots — Jun-Aug window
    "Dubai":       (25.20,  55.27),
    "Phoenix":     (33.45, -112.07),
    "Seville":     (37.39,  -5.99),
    # Rain hotspots — Oct-Nov window
    "London":      (51.51,  -0.13),
    "Bergen":      (60.39,   5.33),
    "Seattle":     (47.61, -122.33),
}

# ── Open-Meteo API settings ───────────────────────────────────────────────────
API_URL  = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Bucharest"
RETRY_DELAY_S = 5      # seconds between retries on failure
MAX_RETRIES   = 3


# ── Step 1 – Extract unique lookup keys ───────────────────────────────────────

def load_unique_pairs(path: Path) -> pd.DataFrame:
    """Return deduplicated (location, event_date, event_hour) rows."""
    df = pd.read_csv(path, usecols=["location", "event_date", "event_hour"])
    pairs = df.drop_duplicates().reset_index(drop=True)
    print(f"Unique (location, date, hour) combinations: {len(pairs)}")
    return pairs


# ── Step 2 – Fetch hourly weather from Open-Meteo ────────────────────────────

def _fetch_city(city: str, lat: float, lon: float,
                start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch full hourly weather for one city over [start, end].
    Returns a DataFrame with columns:
        city, date, hour, weather_temp_C, weather_precip_mm, weather_wind_speed_kmh
    or None on permanent failure.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m",
        "timezone": TIMEZONE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            print(f"  [{city}] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
            else:
                print(f"  [{city}] Giving up after {MAX_RETRIES} attempts.")
                return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        print(f"  [{city}] Empty response payload.")
        return None

    dt_series  = pd.to_datetime(times)
    city_df = pd.DataFrame({
        "city":                    city,
        "date":                    dt_series.strftime("%Y-%m-%d"),
        "hour":                    dt_series.hour,
        "weather_temp_C":          hourly.get("temperature_2m", [None] * len(times)),
        "weather_humidity":        hourly.get("relative_humidity_2m", [None] * len(times)),
        "weather_precip_mm":       hourly.get("precipitation",  [None] * len(times)),
        "weather_wind_speed_kmh":  hourly.get("windspeed_10m",  [None] * len(times)),
    })
    return city_df


def fetch_all_cities(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Make one API call per city using each city's own event date window.
    Cold/heat/rain cities have narrow windows (2-3 months); Romanian cities
    span the full year.  This avoids fetching irrelevant hourly records.
    """
    # Per-city date range derived directly from the events in that city
    city_ranges = (
        pairs.groupby("location")["event_date"]
        .agg(["min", "max"])
        .rename(columns={"min": "start", "max": "end"})
    )

    cities_needed = city_ranges.index.tolist()
    print(f"\nCities to fetch ({len(cities_needed)}): {cities_needed}\n")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for city in cities_needed:
        if city not in CITY_COORDS:
            print(f"  [{city}] WARNING: no coordinates defined, skipping.")
            failed.append(city)
            continue

        lat, lon  = CITY_COORDS[city]
        city_start = city_ranges.loc[city, "start"]
        city_end   = city_ranges.loc[city, "end"]
        print(f"  Fetching {city} ({lat}, {lon})  "
              f"{city_start} -> {city_end} ...", end=" ", flush=True)

        city_df = _fetch_city(city, lat, lon, city_start, city_end)

        if city_df is not None:
            frames.append(city_df)
            print(f"OK  ({len(city_df):,} hourly records)")
        else:
            failed.append(city)

        time.sleep(0.5)

    if not frames:
        print("\nERROR: No weather data could be fetched.")
        sys.exit(1)

    cache = pd.concat(frames, ignore_index=True)

    if failed:
        print(f"\nWARNING: These cities failed entirely: {failed}")

    return cache, failed


# ── Step 3 – Save cache ───────────────────────────────────────────────────────

def save_cache(cache: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)
    print(f"\nWeather cache saved -> {path}  ({len(cache):,} rows)")


# ── Step 4 – Merge weather into foundation ────────────────────────────────────

def merge_weather(foundation_path: Path, cache: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(foundation_path)

    # Build a lookup keyed on (city, date, hour) with only the columns we need
    weather_lookup = (
        cache[["city", "date", "hour",
               "weather_temp_C", "weather_humidity",
               "weather_precip_mm", "weather_wind_speed_kmh"]]
        .drop_duplicates(subset=["city", "date", "hour"])
        .rename(columns={"city": "location", "date": "event_date", "hour": "event_hour"})
    )

    enriched = df.merge(weather_lookup, on=["location", "event_date", "event_hour"],
                        how="left")

    missing = enriched["weather_temp_C"].isna().sum()
    if missing > 0:
        print(f"\nWARNING: {missing:,} rows have no weather match after merge.")
    else:
        print("\nAll rows successfully matched with weather data.")

    return enriched


# ── Step 5 – Verification report ─────────────────────────────────────────────

def print_verification(pairs: pd.DataFrame, cache: pd.DataFrame,
                       enriched: pd.DataFrame, failed: list[str]) -> None:
    sep = "=" * 65
    print("\n" + sep)
    print("  WEATHER ENRICHMENT - VERIFICATION REPORT")
    print(sep)

    unique_calls = len(pairs["location"].unique())
    print(f"\n  Unique API calls made          : {unique_calls}")
    print(f"  Hourly records in cache        : {len(cache):,}")
    print(f"  Cities that failed             : {len(failed)}  {failed or '(none)'}")

    event_weather = enriched.drop_duplicates("event_id")
    matched   = event_weather["weather_temp_C"].notna().sum()
    unmatched = event_weather["weather_temp_C"].isna().sum()
    print(f"\n  Events with weather matched    : {matched} / {len(event_weather)}")
    if unmatched:
        missing_events = event_weather[event_weather["weather_temp_C"].isna()]
        print(f"  Events WITHOUT weather         : {unmatched}")
        print(missing_events[["event_id","event_type","location","event_date"]].to_string())

    print("\n-- Temperature by climate zone --------------------------------")
    if "climate_zone" in enriched.columns:
        zone_stats = (
            enriched.groupby("climate_zone")["weather_temp_C"]
            .agg(["min", "mean", "max"])
            .round(1)
        )
        print(zone_stats.to_string())

    print("\n-- Extreme weather event counts -------------------------------")
    extreme_heat  = (event_weather["weather_temp_C"] > 35).sum()
    extreme_cold  = (event_weather["weather_temp_C"] < 0).sum()
    heavy_rain    = (event_weather["weather_precip_mm"] > 5).sum()
    any_precip    = (event_weather["weather_precip_mm"] > 0).sum()
    total_ev      = len(event_weather)
    print(f"  Temp > 35 C (extreme heat)     : {extreme_heat:>4} / {total_ev}  "
          f"({extreme_heat/total_ev*100:.1f}%)")
    print(f"  Temp < 0 C  (extreme cold)     : {extreme_cold:>4} / {total_ev}  "
          f"({extreme_cold/total_ev*100:.1f}%)")
    print(f"  Precip > 5mm (heavy rain)      : {heavy_rain:>4} / {total_ev}  "
          f"({heavy_rain/total_ev*100:.1f}%)")
    print(f"  Any precip > 0                 : {any_precip:>4} / {total_ev}  "
          f"({any_precip/total_ev*100:.1f}%)")
    bad_weather = extreme_heat + extreme_cold + heavy_rain
    print(f"  Total bad-weather events       : {bad_weather:>4} / {total_ev}  "
          f"({bad_weather/total_ev*100:.1f}%)")

    total_rows   = len(enriched)
    matched_rows = enriched["weather_temp_C"].notna().sum()
    status = "OK" if matched_rows == total_rows else "WARNING"
    print("\n" + sep)
    print(f"  Total rows: {total_rows:,}  |  Matched: {matched_rows:,}  [{status}]")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  WAVE - fetch_weather_api.py")
    print("=" * 60)

    print("\n[1/5] Loading unique event pairs from foundation ...")
    pairs = load_unique_pairs(FOUNDATION_CSV)

    print("\n[2/5] Fetching historical weather from Open-Meteo ...")
    cache, failed = fetch_all_cities(pairs)

    print("\n[3/5] Saving weather cache ...")
    save_cache(cache, CACHE_CSV)

    print("\n[4/5] Merging weather into interaction foundation ...")
    enriched = merge_weather(FOUNDATION_CSV, cache)

    print("\n[5/5] Saving enriched dataset ...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}  ({len(enriched):,} rows x {len(enriched.columns)} cols)")

    print_verification(pairs, cache, enriched, failed)


if __name__ == "__main__":
    main()
