#!/usr/bin/env python3
"""
scenario_validation.py
---------------------
Cheap calibration sanity-check: run the simulator on the 110 real users against
4 survey scenario questions and measure agreement.

We map each scenario to a plausible event + weather snapshot, then use the same
simulation functions as `simulate_labels.py` (DI comfort index, override logic,
affinity boost, extreme multiplier) to get a deterministic probability.

Ground truth: scenario score in {0,1,2,3}. We interpret:
  attend_true = 1 if score >= 2 else 0

Prediction:
  attend_hat = 1 if final_prob >= 0.50 else 0

Outputs:
  - prints per-scenario and overall agreement (%)
  - saves results/scenario_validation.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from simulate_labels import (
    compute_affinity_boost,
    compute_base_prob,
    compute_extreme_multiplier,
    compute_raw_weather,
    compute_weather_adjust,
    apply_override,
)

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USERS_CSV = ROOT / "data" / "processed" / "app_users.csv"
OUT_CSV = ROOT / "results" / "scenario_validation.csv"


SCENARIOS = [
    # (name, event_type, is_outdoor, weather_temp_C, humidity%, precip mm/hr, wind km/h)
    ("Concert",  "Concert",   0, 21.0, 45.0, 0.0,  8.0),
    ("Festival", "Festival",  1, 24.0, 50.0, 0.0, 10.0),
    ("Sports", "Sports", 1, 18.0, 50.0, 0.0, 15.0),  
    ("Theatre",  "Theatre",   0,  5.0, 65.0, 1.0, 15.0),
]


def build_rows(users: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for scen_name, event_type, is_outdoor, t, hum, p, w in SCENARIOS:
        df = users.copy()
        df["scenario_name"] = scen_name
        df["event_type"] = event_type
        df["is_outdoor"] = int(is_outdoor)
        df["weather_temp_C"] = float(t)
        df["weather_humidity"] = float(hum)
        df["weather_precip_mm"] = float(p)
        df["weather_wind_speed_kmh"] = float(w)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def scenario_truth(df: pd.DataFrame) -> pd.Series:
    col_map = {
        "Concert": "scenario_concert",
        "Festival": "scenario_festival",
        "Sports": "scenario_sports",
        "Theatre": "scenario_theatre",
    }
    scores = df.apply(lambda r: r[col_map[r["scenario_name"]]], axis=1).astype(int)
    return (scores >= 2).astype(int)


def main() -> None:
    users = pd.read_csv(USERS_CSV).dropna(subset=["user_id"]).copy()
    df = build_rows(users)

    # Deterministic probability (no jitter) using same components as main simulator
    base = compute_base_prob(df)
    raw = compute_raw_weather(df)
    adjust = compute_weather_adjust(df, raw)
    adjust = apply_override(df, adjust)
    boost = compute_affinity_boost(df)
    mult = compute_extreme_multiplier(df)
    final_prob = ((base + adjust + boost) * mult).clip(0.0, 1.0)

    y_true = scenario_truth(df)
    y_hat = (final_prob >= 0.50).astype(int)

    out = df[["user_id", "scenario_name"]].copy()
    out["final_prob"] = final_prob
    out["attend_true"] = y_true
    out["attend_hat"] = y_hat

    # Report
    print("=" * 60)
    print("SCENARIO VALIDATION (110 users x 4 scenarios)")
    print("=" * 60)
    overall = (out["attend_true"] == out["attend_hat"]).mean() * 100
    print(f"\nOverall agreement: {overall:.1f}%")

    print("\nPer-scenario agreement:")
    rows = []
    for scen, g in out.groupby("scenario_name"):
        acc = (g["attend_true"] == g["attend_hat"]).mean() * 100
        pos_rate = g["attend_true"].mean() * 100
        pred_rate = g["attend_hat"].mean() * 100
        rows.append((scen, acc, pos_rate, pred_rate))
    rep = pd.DataFrame(rows, columns=["scenario", "agreement_pct", "true_yes_pct", "pred_yes_pct"])
    print(rep.to_string(index=False, formatters={
        "agreement_pct": lambda x: f"{x:.1f}",
        "true_yes_pct": lambda x: f"{x:.1f}",
        "pred_yes_pct": lambda x: f"{x:.1f}",
    }))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()

