#!/usr/bin/env python3
"""
WAVE – Fetch real historical weather data from Open-Meteo API (DAILY granularity).

Strategy:
  - One API call per US state covering the full date range in train_ready.csv
  - Returns daily temperature_2m_mean and precipitation_sum
  - Joins each event row to its EXACT event date (not monthly average)
  - This gives row-level weather variation: events on different days
    in the same location get different weather values
  - Caches daily data to data/raw/weather_cache.csv

Why daily (not monthly)?
  Monthly averages make weather features collinear with event_month/season
  (same value for every event in AZ in January → zero new information).
  Daily data captures actual day-to-day variation: a cold snap, a rainy
  week, a heatwave — the kind of signal that actually influences attendance.

Why not hourly?
  Daily weather (was it cold/rainy on that day?) is the relevant signal
  for attendance decisions. Hourly resolution would add marginal value
  at much higher data volume and API complexity.

Usage:
    python src/fetch_weather.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train_ready.csv"
RAW_DIR    = ROOT / "data" / "raw"
CACHE_PATH = RAW_DIR / "weather_cache.csv"
API_URL    = "https://archive-api.open-meteo.com/v1/archive"

# ─── State capital coordinates ───────────────────────────────────────────────
STATE_COORDS: dict[str, tuple[float, float]] = {
    "AL": (32.3617, -86.2792),  "AK": (58.3005, -134.4197),
    "AZ": (33.4484, -112.0740), "AR": (34.7465, -92.2896),
    "CA": (38.5767, -121.4933), "CO": (39.7392, -104.9847),
    "CT": (41.7637, -72.6851),  "DE": (39.1582, -75.5244),
    "FL": (30.4383, -84.2807),  "GA": (33.7490, -84.3880),
    "HI": (21.3069, -157.8583), "ID": (43.6150, -116.2023),
    "IL": (39.7983, -89.6544),  "IN": (39.7684, -86.1581),
    "IA": (41.5868, -93.6250),  "KS": (39.0473, -95.6752),
    "KY": (38.1867, -84.8753),  "LA": (30.4515, -91.1871),
    "ME": (44.3106, -69.7795),  "MD": (38.9784, -76.4922),
    "MA": (42.3601, -71.0589),  "MI": (42.7335, -84.5467),
    "MN": (44.9537, -93.0900),  "MS": (32.2988, -90.1848),
    "MO": (38.5767, -92.1736),  "MT": (46.5958, -112.0270),
    "NE": (40.8136, -96.7026),  "NV": (39.1638, -119.7674),
    "NH": (43.2081, -71.5376),  "NJ": (40.2206, -74.7697),
    "NM": (35.6870, -105.9378), "NY": (42.6526, -73.7562),
    "NC": (35.7796, -78.6382),  "ND": (46.8083, -100.7837),
    "OH": (39.9612, -82.9988),  "OK": (35.4676, -97.5164),
    "OR": (44.9429, -123.0351), "PA": (40.2732, -76.8867),
    "RI": (41.8240, -71.4128),  "SC": (34.0007, -81.0348),
    "SD": (44.3668, -100.3538), "TN": (36.1627, -86.7816),
    "TX": (30.2672, -97.7431),  "UT": (40.7608, -111.8910),
    "VT": (44.2601, -72.5754),  "VA": (37.5407, -77.4360),
    "WA": (47.0379, -122.9007), "WV": (38.3498, -81.6326),
    "WI": (43.0731, -89.4012),  "WY": (41.1400, -104.8202),
    "DC": (38.9072, -77.0369),
    "PR": (18.2208, -66.5901),
    "GU": (13.4443, 144.7937),
    "VI": (18.3358, -64.8963),
    "AS": (-14.2756, -170.7020),
    "MP": (15.1850, 145.7470),
    "MH": (7.0736,  171.2327),
    "PW": (7.5004,  134.6241),
}

STATE_RE = re.compile(r',\s*([A-Z]{2})\s+\d{5}')


def extract_state(location: str) -> str | None:
    m = STATE_RE.search(str(location))
    return m.group(1) if m else None


# ─── API fetch (returns daily rows) ──────────────────────────────────────────
def fetch_daily_weather(state: str, lat: float, lon: float,
                        start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch daily temperature_2m_mean and precipitation_sum for one state.
    Returns DataFrame with columns: state, date, weather_temp_C, weather_precip_mm.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      "temperature_2m_mean,precipitation_sum",
        "timezone":   "UTC",
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        daily = data.get("daily", {})
        if not daily.get("time"):
            return None
        n = len(daily["time"])
        df = pd.DataFrame({
            "state":             state,
            "date":              pd.to_datetime(daily["time"]),
            "weather_temp_C":    daily.get("temperature_2m_mean", [None] * n),
            "weather_precip_mm": daily.get("precipitation_sum",   [None] * n),
        })
        return df
    except Exception as exc:
        print(f"  [WARN] {state}: API error - {exc}")
        return None


# ─── Build/load daily cache ───────────────────────────────────────────────────
def build_daily_cache(states: list[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily data for every state and cache to weather_cache.csv.
    Skips states already present in cache.
    """
    if CACHE_PATH.exists():
        cache = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        # Detect if this is the old monthly cache (has year/month cols, no date col)
        if "date" not in cache.columns:
            print("  Old monthly cache detected - deleting and re-fetching...")
            cache = pd.DataFrame(
                columns=["state", "date", "weather_temp_C", "weather_precip_mm"]
            )
        else:
            print(f"Loaded existing daily cache: {len(cache):,} rows, "
                  f"{cache['state'].nunique()} states already cached.")
    else:
        cache = pd.DataFrame(
            columns=["state", "date", "weather_temp_C", "weather_precip_mm"]
        )

    cached_states = set(cache["state"].unique())
    to_fetch = [s for s in states if s not in cached_states]
    print(f"States to fetch: {len(to_fetch)}  (already cached: {len(cached_states)})")

    new_rows = []
    for i, state in enumerate(to_fetch, 1):
        coords = STATE_COORDS.get(state)
        if coords is None:
            print(f"  [{i}/{len(to_fetch)}] {state}: no coordinates - skipping.")
            continue
        lat, lon = coords
        print(f"  [{i}/{len(to_fetch)}] Fetching {state} ({lat:.4f}, {lon:.4f})...",
              end=" ", flush=True)
        daily = fetch_daily_weather(state, lat, lon, start_date, end_date)
        if daily is None or daily.empty:
            print("FAILED")
            continue
        new_rows.append(daily)
        print(f"OK ({len(daily)} days)")
        time.sleep(0.3)

    if new_rows:
        fresh = pd.concat(new_rows, ignore_index=True)
        cache = pd.concat([cache, fresh], ignore_index=True)
        cache.to_csv(CACHE_PATH, index=False)
        print(f"\nCache saved: {len(cache):,} total rows -> {CACHE_PATH}")

    return cache


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("WAVE - Fetch Real Historical Weather (DAILY granularity)")
    print("=" * 60)

    # 1. Load train_ready, extract state and event date
    print("\n[1] Loading train_ready.csv...")
    df = pd.read_csv(TRAIN_PATH)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["_state"]      = df["location"].apply(extract_state)
    df["_event_date"] = df["event_datetime"].dt.normalize()   # midnight = date only

    all_states = sorted(df["_state"].dropna().unique().tolist())
    print(f"  Unique states: {len(all_states)}")
    print(f"  Unique event dates: {df['_event_date'].nunique()}")

    min_date   = df["event_datetime"].min()
    max_date   = df["event_datetime"].max()
    start_date = min_date.strftime("%Y-%m-%d")
    end_date   = max_date.strftime("%Y-%m-%d")
    print(f"  Date range: {start_date} to {end_date}")

    # 2. Build or load daily weather cache
    print("\n[2] Building daily weather cache...")
    daily_cache = build_daily_cache(all_states, start_date, end_date)
    daily_cache["date"] = pd.to_datetime(daily_cache["date"])

    # 3. Merge on (state, exact date)
    print("\n[3] Merging real daily weather into train_ready.csv...")
    df = df.merge(
        daily_cache.rename(columns={
            "weather_temp_C":    "weather_temp_C_new",
            "weather_precip_mm": "weather_precip_mm_new",
        }),
        left_on=["_state", "_event_date"],
        right_on=["state", "date"],
        how="left",
    )

    filled   = df["weather_temp_C_new"].notna().sum()
    missing  = df["weather_temp_C_new"].isna().sum()
    fill_pct = filled / len(df) * 100
    print(f"  Rows with real daily weather : {filled:,} ({fill_pct:.1f}%)")
    print(f"  Rows still missing           : {missing:,}")

    if missing > 0:
        print(f"  [WARN] Filling {missing} missing rows with column medians.")
        med_t = df["weather_temp_C_new"].median()
        med_p = df["weather_precip_mm_new"].median()
        df["weather_temp_C_new"]    = df["weather_temp_C_new"].fillna(med_t)
        df["weather_precip_mm_new"] = df["weather_precip_mm_new"].fillna(med_p)

    # Replace old weather columns with daily values
    df["weather_temp_C"]    = df["weather_temp_C_new"]
    df["weather_precip_mm"] = df["weather_precip_mm_new"]

    # Drop all helper columns
    drop_cols = [c for c in [
        "_state", "_event_date", "state", "date",
        "weather_temp_C_new", "weather_precip_mm_new",
    ] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # 4. Sanity check: seasonal mean temperature
    print("\n[4] Sanity check - mean temp by season (should be: winter < spring < summer):")
    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    df["_sname"] = df["season"].map(season_map)
    seasonal = df.groupby("_sname")["weather_temp_C"].agg(["mean", "std"]).round(2)
    print(seasonal.to_string())
    df.drop(columns=["_sname"], inplace=True)

    # 5. Variance check: confirm daily values vary within same (state, month)
    print("\n[5] Variance check - daily variation WITHIN same state+month:")
    df["_state2"] = df["location"].apply(extract_state)
    sample = df[df["_state2"] == "AZ"]
    jan_az = sample[sample["event_month"] == 1]
    if len(jan_az) > 1:
        print(f"  AZ, January events: n={len(jan_az)}, "
              f"temp unique values={jan_az['weather_temp_C'].nunique()}, "
              f"temp range=[{jan_az['weather_temp_C'].min():.1f}, "
              f"{jan_az['weather_temp_C'].max():.1f}]")
    df.drop(columns=["_state2"], inplace=True)

    # 6. Save
    print(f"\n[6] Saving updated train_ready.csv...")
    df.to_csv(TRAIN_PATH, index=False)
    print(f"  Saved: {TRAIN_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    att_dist = df["attended"].value_counts(normalize=True).mul(100).round(1)
    print(f"\n  Class distribution: attended=0: {att_dist.get(0,0):.1f}%  "
          f"attended=1: {att_dist.get(1,0):.1f}%")

    print("\nDone. Daily weather successfully integrated into train_ready.csv.")
    print("NOTE: Values are RAW (unscaled). StandardScaler applied in train_models.py.")
    print("NOTE: Do NOT run generate_weather.py - it would overwrite real data.")


if __name__ == "__main__":
    main()
