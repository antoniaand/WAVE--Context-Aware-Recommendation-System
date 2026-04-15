"""
generate_foundation.py  --  GLOBAL REBUILD
------------------------------------------
Step 1 of the modular WAVE dataset rebuild.

Creates 450 events across 18 cities in 4 climate zones:
  moderate  : 6 Romanian cities, full year (Jun 2024 - Jun 2025)   -> 270 events
  cold      : Oslo, Helsinki, Quebec, Dec 2024 - Feb 2025           ->  60 events
  heat      : Dubai, Phoenix, Seville, Jun 2024 - Aug 2024          ->  60 events
  rain      : London, Bergen, Seattle, Oct 2024 - Nov 2024          ->  60 events

Extreme zones = 180 / 450 = 40 % of all events.

110 users x 450 events = 49,500 rows  (~50,000 target)

No weather data or attendance labels are generated here.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
USERS_CSV  = ROOT / "data" / "processed" / "app_users.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "interaction_foundation.csv"

# ── Event categories ──────────────────────────────────────────────────────────
CATEGORIES = ["Concert", "Festival", "Sports", "Theatre", "Conference", "Exhibition"]
HOUR_MIN, HOUR_MAX = 18, 21

# ── City groups with climate season windows ───────────────────────────────────
#
# events_per_cat: how many events of each category are generated for this group
# Total per group = events_per_cat × len(CATEGORIES)
#
CITY_GROUPS: dict[str, dict] = {
    "moderate": {
        "cities":        ["Bucharest", "Cluj-Napoca", "Timisoara",
                          "Iasi", "Constanta", "Brasov"],
        "date_start":    pd.Timestamp("2024-06-01"),
        "date_end":      pd.Timestamp("2025-06-30"),
        "events_per_cat": 45,   # 45 × 6 = 270 events
    },
    "cold": {
        "cities":        ["Oslo", "Helsinki", "Quebec"],
        "date_start":    pd.Timestamp("2024-12-01"),
        "date_end":      pd.Timestamp("2025-02-28"),
        "events_per_cat": 10,   # 10 × 6 = 60 events
    },
    "heat": {
        "cities":        ["Dubai", "Phoenix", "Seville"],
        "date_start":    pd.Timestamp("2024-06-01"),
        "date_end":      pd.Timestamp("2024-08-31"),
        "events_per_cat": 10,   # 10 × 6 = 60 events
    },
    "rain": {
        "cities":        ["London", "Bergen", "Seattle"],
        "date_start":    pd.Timestamp("2024-10-01"),
        "date_end":      pd.Timestamp("2024-11-30"),
        "events_per_cat": 10,   # 10 × 6 = 60 events
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def outdoor_flags(category: str, n: int) -> np.ndarray:
    """is_outdoor assignment rules (same across all city groups)."""
    if category in ("Theatre", "Conference"):
        return np.zeros(n, dtype=int)
    if category in ("Festival", "Sports"):
        return np.ones(n, dtype=int)
    # Concert, Exhibition: 50 % indoor / 50 % outdoor
    flags = np.zeros(n, dtype=int)
    flags[: n // 2] = 1
    rng.shuffle(flags)
    return flags


# ── Step 1 – Load users ───────────────────────────────────────────────────────

def load_users(path: Path) -> pd.DataFrame:
    users = pd.read_csv(path).dropna(subset=["user_id"])
    users["user_id"] = users["user_id"].astype(int)
    assert len(users) == 110, f"Expected 110 users, got {len(users)}"
    return users


# ── Step 2 – Generate events ──────────────────────────────────────────────────

def generate_events() -> pd.DataFrame:
    blocks = []
    event_id = 1

    for zone_name, zone in CITY_GROUPS.items():
        cities      = zone["cities"]
        date_start  = zone["date_start"]
        date_end    = zone["date_end"]
        n_per_cat   = zone["events_per_cat"]
        date_range  = (date_end - date_start).days

        for cat in CATEGORIES:
            n       = n_per_cat
            outdoor = outdoor_flags(cat, n)

            # Spread dates evenly across the zone's seasonal window
            offsets = np.linspace(0, date_range, n, dtype=int)
            dates   = [date_start + pd.Timedelta(days=int(d)) for d in offsets]

            hours     = rng.integers(HOUR_MIN, HOUR_MAX + 1, size=n)
            locations = rng.choice(cities, size=n)

            block = pd.DataFrame({
                "event_id":     range(event_id, event_id + n),
                "event_type":   cat,
                "climate_zone": zone_name,
                "is_outdoor":   outdoor,
                "location":     locations,
                "event_date":   dates,
                "event_hour":   hours,
            })
            blocks.append(block)
            event_id += n

    events = pd.concat(blocks, ignore_index=True)
    return events


# ── Step 3 – Cartesian product ────────────────────────────────────────────────

USER_PROFILE_COLS = [
    "user_id", "gender", "age_range", "attendance_freq", "indoor_outdoor",
    "top_event", "rain_avoid", "cold_tolerance", "heat_sensitivity",
    "wind_sensitivity", "override_weather", "scenario_concert",
    "scenario_festival", "scenario_sports", "scenario_theatre",
    "scenario_conference", "preferred_event_types",
]

EVENT_COLS = [
    "event_id", "event_type", "climate_zone", "is_outdoor",
    "location", "event_date", "event_hour",
]


def build_grid(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    u = users[[c for c in USER_PROFILE_COLS if c in users.columns]].copy()
    e = events[EVENT_COLS].copy()

    u["_key"] = 1
    e["_key"] = 1
    grid = pd.merge(u, e, on="_key").drop(columns="_key")
    grid.insert(0, "interaction_id", range(1, len(grid) + 1))
    return grid


# ── Step 4 – Report ───────────────────────────────────────────────────────────

def print_report(events: pd.DataFrame, grid: pd.DataFrame) -> None:
    sep = "=" * 65
    total_events   = len(events)
    extreme_events = (events["climate_zone"] != "moderate").sum()
    total_rows     = len(grid)

    print("\n" + sep)
    print("  WAVE GLOBAL FOUNDATION -- GENERATION REPORT")
    print(sep)
    print(f"\n  {'Users:':<35} {grid['user_id'].nunique():>6}")
    print(f"  {'Events total:':<35} {total_events:>6}")
    print(f"  {'  Moderate (Romanian):':<35} {(events['climate_zone']=='moderate').sum():>6}")
    print(f"  {'  Extreme (cold+heat+rain):':<35} {extreme_events:>6}  "
          f"({extreme_events/total_events*100:.1f}%)")
    print(f"  {'Interaction rows:':<35} {total_rows:>6,}")

    print("\n-- Events by climate zone & environment ---------------------")
    summary = (
        events.groupby(["climate_zone", "is_outdoor"])
        .size().rename("count").reset_index()
    )
    summary["environment"] = summary["is_outdoor"].map({1: "Outdoor", 0: "Indoor"})
    print(summary[["climate_zone", "environment", "count"]].to_string(index=False))

    print("\n-- Events by zone (with date window) ------------------------")
    for zone_name, zone in CITY_GROUPS.items():
        z_events = events[events["climate_zone"] == zone_name]
        print(f"  {zone_name:<10} {len(z_events):>4} events | "
              f"{zone['date_start'].date()} -> {zone['date_end'].date()} | "
              f"cities: {', '.join(zone['cities'])}")

    print("\n-- Outdoor breakdown ----------------------------------------")
    out_pct = events["is_outdoor"].mean() * 100
    print(f"  Overall outdoor events: {out_pct:.1f}%")
    extreme_out = events[events["climate_zone"] != "moderate"]["is_outdoor"].mean() * 100
    print(f"  Extreme-zone outdoor  : {extreme_out:.1f}%")

    status = "OK" if total_rows == 49_500 else f"NOTE: expected 49,500"
    print(f"\n  Total row count: {total_rows:,}  [{status}]")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading users ...")
    users = load_users(USERS_CSV)

    print("Generating 450 events across 4 climate zones ...")
    events = generate_events()
    print(f"  {len(events)} events generated  "
          f"(extreme: {(events['climate_zone']!='moderate').sum()} / "
          f"{len(events)} = "
          f"{(events['climate_zone']!='moderate').sum()/len(events)*100:.1f}%)")

    print("Building 110 x 450 interaction grid ...")
    grid = build_grid(users, events)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    print_report(events, grid)


if __name__ == "__main__":
    main()
