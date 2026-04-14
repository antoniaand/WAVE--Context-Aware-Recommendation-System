#!/usr/bin/env python3
"""
WAVE – Build user × event × weather interaction dataset.

WHY THIS SCRIPT EXISTS (see docs/DATASET_METHODOLOGY.md Section 3):
  The original train_ready.csv had 200 events × 1000 identical rows each.
  Every row for the same event shared the exact same feature vector, with
  attended=0 and attended=1 mixed randomly. No model can learn from that.

WHAT THIS SCRIPT DOES:
  Builds a new dataset where each row = one user deciding whether to attend
  one event on a specific real weather day. Features differ per row because:
    - each user has a unique profile (tolerances, preferences)
    - each weather day is a real historical observation from Open-Meteo
  The attendance label is generated transparently from the user's own survey
  answers — the formula is documented here and in docs/DATASET_METHODOLOGY.md.

OUTPUT: data/processed/train_ready_interactions.csv
  ~88 000 rows (110 users × 200 events × 4 weather scenarios)
  ready for train_models.py with GroupShuffleSplit on user_id.

Usage:
    python src/build_interaction_dataset.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT         = Path(__file__).resolve().parents[1]
PROCESSED    = ROOT / "data" / "processed"
RAW          = ROOT / "data" / "raw"
TRAIN_PATH   = PROCESSED / "train_ready.csv"
USERS_PATH   = PROCESSED / "app_users.csv"
CACHE_PATH   = RAW / "weather_cache.csv"
OUT_PATH     = PROCESSED / "train_ready_interactions.csv"

STATE_RE = re.compile(r',\s*([A-Z]{2})\s+\d{5}')

# How many real weather days to sample per (user, event) pair.
# 4 gives 110 × 200 × 4 = 88 000 rows — enough for RF/LGBM/XGB.
WEATHER_SCENARIOS_PER_PAIR = 4

# Map event_name keywords to canonical type (Concert, Festival, etc.)
EVENT_TYPE_KEYWORDS = {
    "Concert":    ["concert", "music", "band", "orchestra", "recital", "gig"],
    "Festival":   ["festival", "fest", "fair", "carnival", "fiesta"],
    "Theatre":    ["theatre", "theater", "play", "drama", "opera", "comedy", "performance"],
    "Sports":     ["sport", "game", "match", "tournament", "race", "championship"],
    "Conference": ["conference", "summit", "seminar", "symposium", "workshop",
                   "forum", "congress", "convention"],
    "Exhibition": ["exhibition", "exhibit", "expo", "gallery", "museum", "show"],
    # Technology/software event names from Kaggle → Conference
    "Tech":       ["interface", "platform", "software", "system", "network",
                   "database", "graphical", "api", "web", "digital", "mobile",
                   "cloud", "data", "application", "solution", "algorithm",
                   "framework", "infrastructure", "server", "security"],
}

# Outdoor event types (weather penalty applies more strongly)
OUTDOOR_TYPES = {"Concert", "Festival", "Sports"}

# Map canonical event type to which scenario column to use from app_users.csv
SCENARIO_COL = {
    "Concert":    "scenario_concert",
    "Festival":   "scenario_festival",
    "Sports":     "scenario_sports",
    "Theatre":    "scenario_theatre",
    "Conference": "scenario_conference",
    "Exhibition": "scenario_conference",  # closest proxy
    "Tech":       "scenario_conference",  # tech events → conference proxy
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_state(location: str) -> str | None:
    m = STATE_RE.search(str(location))
    return m.group(1) if m else None


def classify_event_type(name: str) -> str:
    name_lower = name.lower()
    for etype, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return etype
    return "Conference"  # safe default for unmatched names


def attendance_probability(
    scenario_score: float,   # 0–3 from survey, normalised to 0–1
    weather_temp: float,     # degrees Celsius (real daily value)
    weather_precip: float,   # mm precipitation (real daily value)
    rain_avoid: float,       # 1–5
    cold_tolerance: float,   # 1–5
    heat_sensitivity: float, # 1–5
    override_weather: float, # 1–5
    is_outdoor: bool,
) -> float:
    """
    Transparent formula for P(attend | user, event, weather).

    Components:
      scenario_score  — user's declared likelihood of attending this event type
      weather_penalty — how much bad weather deters this user (modulated by
                        their declared tolerances and whether event is outdoor)
      override_boost  — user's self-declared tendency to show up regardless

    All weights come directly from the user's survey answers.
    Formula documented in docs/DATASET_METHODOLOGY.md Section 4.
    """
    # Normalise scenario score 0–3 → 0.35–0.75 range
    # (avoids extreme base probabilities that cause heavy imbalance)
    base = 0.35 + (scenario_score / 3.0) * 0.40

    # Weather components (each clipped to [0, 1])
    rain_factor  = float(np.clip(weather_precip / 15.0, 0.0, 1.0))
    cold_factor  = float(np.clip((5.0 - weather_temp) / 15.0, 0.0, 1.0))
    heat_factor  = float(np.clip((weather_temp - 28.0) / 10.0, 0.0, 1.0))

    # User sensitivity scores (normalise 1–5 → 0–1)
    rain_sens  = rain_avoid      / 5.0
    cold_sens  = 1.0 - cold_tolerance / 5.0   # high tolerance → low penalty
    heat_sens  = heat_sensitivity / 5.0
    override   = override_weather / 5.0

    # Outdoor multiplier: weather matters more for outdoor events
    outdoor_mult = 1.0 if is_outdoor else 0.40

    weather_penalty = outdoor_mult * (
        rain_sens  * rain_factor +
        cold_sens  * cold_factor +
        heat_sens  * heat_factor
    )
    # Override partially cancels the weather penalty
    override_reduction = override * weather_penalty

    final_prob = base * (1.0 - weather_penalty + override_reduction)
    return float(np.clip(final_prob, 0.05, 0.95))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("WAVE – Build Interaction Dataset")
    print("=" * 60)

    # ── 1. Load events from train_ready.csv (unique events only) ──────────────
    print("\n[1] Loading events from train_ready.csv...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["_state"] = train_df["location"].apply(extract_state)

    # One row per event (all feature columns are identical within an event_id)
    events = (
        train_df
        .groupby("event_id")[
            ["event_id", "event_name", "location", "event_datetime",
             "event_hour", "event_weekday", "event_month", "season",
             "event_name_enc", "location_enc", "_state"]
        ]
        .first()
        .reset_index(drop=True)
    )
    events["event_type"]   = events["event_name"].apply(classify_event_type)
    events["is_outdoor"]   = events["event_type"].isin(OUTDOOR_TYPES).astype(int)
    print(f"  Events loaded : {len(events)}")
    print(f"  Event types   : {events['event_type'].value_counts().to_dict()}")

    # ── 2. Load user profiles ─────────────────────────────────────────────────
    print("\n[2] Loading user profiles from app_users.csv...")
    users = pd.read_csv(USERS_PATH)

    # Ensure scenario columns are numeric
    for col in ["scenario_concert", "scenario_festival", "scenario_sports",
                "scenario_theatre", "scenario_conference"]:
        users[col] = pd.to_numeric(users[col], errors="coerce").fillna(1)

    for col in ["rain_avoid", "cold_tolerance", "heat_sensitivity",
                "wind_sensitivity", "override_weather"]:
        users[col] = pd.to_numeric(users[col], errors="coerce").fillna(3)

    print(f"  Users loaded  : {len(users)}")

    # ── 3. Load weather cache (daily data) ────────────────────────────────────
    print("\n[3] Loading weather cache...")
    weather_cache = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    weather_cache["month"] = weather_cache["date"].dt.month
    weather_cache["season"] = weather_cache["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    print(f"  Cache rows    : {len(weather_cache):,}")
    print(f"  States in cache: {weather_cache['state'].nunique()}")

    # ── 4. Build cross-product and sample weather scenarios ───────────────────
    print(f"\n[4] Building interaction rows "
          f"(K={WEATHER_SCENARIOS_PER_PAIR} weather scenarios per user×event)...")

    rows = []
    total_pairs = len(users) * len(events)

    for u_idx, user in users.iterrows():
        for e_idx, event in events.iterrows():

            state    = event["_state"]
            season   = int(event["season"])
            etype    = event["event_type"]
            scenario_col = SCENARIO_COL[etype]
            scenario_score = float(user[scenario_col])
            is_outdoor     = bool(event["is_outdoor"])

            # Sample K real weather days from same state + same season
            pool = weather_cache[
                (weather_cache["state"] == state) &
                (weather_cache["season"] == season)
            ]

            # Fall back to same state (any season) if pool too small
            if len(pool) < WEATHER_SCENARIOS_PER_PAIR:
                pool = weather_cache[weather_cache["state"] == state]

            # Fall back to any state with same season if state not in cache
            if len(pool) < WEATHER_SCENARIOS_PER_PAIR:
                pool = weather_cache[weather_cache["season"] == season]

            if pool.empty:
                continue

            # Stratified sample: roughly equal good/bad weather days
            k = min(WEATHER_SCENARIOS_PER_PAIR, len(pool))
            # Sort by precip and take spread
            pool_sorted = pool.sort_values("weather_precip_mm").reset_index(drop=True)
            step = max(1, len(pool_sorted) // k)
            chosen_idxs = list(range(0, len(pool_sorted), step))[:k]
            # Add random jitter for diversity
            jitter = rng.integers(0, max(step, 1), size=k)
            chosen_idxs = [
                min(i + int(j), len(pool_sorted) - 1)
                for i, j in zip(chosen_idxs, jitter)
            ]
            selected_weather = pool_sorted.iloc[chosen_idxs]

            for _, w_row in selected_weather.iterrows():
                temp   = float(w_row["weather_temp_C"])
                precip = float(w_row["weather_precip_mm"])

                p_attend = attendance_probability(
                    scenario_score   = scenario_score,
                    weather_temp     = temp,
                    weather_precip   = precip,
                    rain_avoid       = float(user["rain_avoid"]),
                    cold_tolerance   = float(user["cold_tolerance"]),
                    heat_sensitivity = float(user["heat_sensitivity"]),
                    override_weather = float(user["override_weather"]),
                    is_outdoor       = is_outdoor,
                )
                attended = int(rng.random() < p_attend)

                rows.append({
                    # Identifiers (dropped before training)
                    "user_id":          int(user["user_id"]),
                    "event_id":         int(event["event_id"]),
                    # Event features
                    "event_type_enc":   int(event["event_name_enc"]),
                    "event_hour":       int(event["event_hour"]),
                    "event_weekday":    int(event["event_weekday"]),
                    "event_month":      int(event["event_month"]),
                    "season":           int(season),
                    "location_enc":     int(event["location_enc"]),
                    "is_outdoor":       int(is_outdoor),
                    # Real weather (unscaled — StandardScaler applied in train_models.py)
                    "weather_temp_C":   round(temp, 3),
                    "weather_precip_mm": round(precip, 3),
                    # User profile features
                    "user_rain_avoid":      int(user["rain_avoid"]),
                    "user_cold_tolerance":  int(user["cold_tolerance"]),
                    "user_heat_sensitivity": int(user["heat_sensitivity"]),
                    "user_wind_sensitivity": int(user["wind_sensitivity"]),
                    "user_override_weather": int(user["override_weather"]),
                    "user_type_preference": int(
                        etype in str(user.get("preferred_event_types", ""))
                    ),
                    # Target
                    "attended": attended,
                })

    df = pd.DataFrame(rows)
    print(f"  Total interaction rows: {len(df):,}")

    # ── 5. Verification ───────────────────────────────────────────────────────
    print("\n[5] Dataset verification:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique users : {df['user_id'].nunique()}")
    print(f"  Unique events: {df['event_id'].nunique()}")
    print(f"  Unique feature vectors: {df.drop(columns=['user_id','event_id','attended']).drop_duplicates().shape[0]:,}")

    # Class balance
    n0 = (df["attended"] == 0).sum()
    n1 = (df["attended"] == 1).sum()
    print(f"  attended=0: {n0:,} ({n0/len(df)*100:.1f}%)")
    print(f"  attended=1: {n1:,} ({n1/len(df)*100:.1f}%)")

    # Weather correlation with attended
    r_temp   = df["weather_temp_C"].corr(df["attended"])
    r_precip = df["weather_precip_mm"].corr(df["attended"])
    print(f"  Pearson r (temp vs attended)  : {r_temp:+.4f}")
    print(f"  Pearson r (precip vs attended): {r_precip:+.4f}")

    # Attendance rate: outdoor + bad weather vs good weather (sanity check)
    outdoor_bad = df[
        (df["is_outdoor"] == 1) &
        ((df["weather_precip_mm"] > 10) | (df["weather_temp_C"] < 2))
    ]
    outdoor_good = df[
        (df["is_outdoor"] == 1) &
        (df["weather_temp_C"].between(12, 26)) &
        (df["weather_precip_mm"] < 5)
    ]
    if len(outdoor_bad) > 0 and len(outdoor_good) > 0:
        print(f"  Outdoor events, BAD weather  attendance: {outdoor_bad['attended'].mean()*100:.1f}%")
        print(f"  Outdoor events, GOOD weather attendance: {outdoor_good['attended'].mean()*100:.1f}%")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[6] Saved: {OUT_PATH}")
    print(f"  Columns: {df.columns.tolist()}")
    print("\nNext step: python src/train_models.py")


if __name__ == "__main__":
    main()
