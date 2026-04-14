#!/usr/bin/env python3
"""
WAVE – Build event-level dataset from train_ready.csv.

PROBLEMA IDENTIFICATA (debug):
  train_ready.csv are 200 evenimente × 1000 randuri identice = 200,000 randuri.
  Toate randurile aceluiasi event au features identice (loc, data, vreme).
  Labels 0/1 sunt distribuite aleatoriu per rand → modelul nu poate distinge.

SOLUTIA:
  Collapse la nivel de event (200 randuri, unu per event).
  Target: attended_rate = proportia participantilor per event (0.0 – 1.0)
  + target binar: high_attendance = 1 daca attended_rate > median

La nivel de event:
  - Fiecare event are o singura valoare de weather (zilnica, reala)
  - weather_temp_C variaza intre events → semnal real
  - Contextual model POATE depasi baseline

Output: data/processed/train_ready_event_level.csv (200 randuri)

Usage:
    python src/build_event_dataset.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT        = Path(__file__).resolve().parents[1]
TRAIN_PATH  = ROOT / "data" / "processed" / "train_ready.csv"
OUT_PATH    = ROOT / "data" / "processed" / "train_ready_event_level.csv"


def main():
    print("=" * 60)
    print("WAVE – Build Event-Level Dataset")
    print("=" * 60)

    df = pd.read_csv(TRAIN_PATH)
    print(f"\nLoaded train_ready.csv: {len(df):,} rows, {df['event_id'].nunique()} unique events")

    # ─── Aggregate to event level ───────────────────────────────────────────
    # All feature columns are identical within each event_id group,
    # so we take the first value. Only 'attended' varies → take mean = rate.
    feature_cols = [
        "event_id", "event_name", "location", "event_datetime",
        "event_hour", "event_weekday", "event_month", "season",
        "event_name_enc", "location_enc",
        "weather_temp_C", "weather_precip_mm",
    ]

    # First row of each event for all features (they're identical within group)
    event_df = df.groupby("event_id")[feature_cols].first().reset_index(drop=True)

    # Attendance rate per event
    attended_rate = df.groupby("event_id")["attended"].mean().reset_index()
    attended_rate.columns = ["event_id", "attended_rate"]

    event_df = event_df.merge(attended_rate, on="event_id")

    # Binary target: high_attendance = 1 if rate > median
    # Using median ensures balanced classes (50/50 split)
    median_rate = event_df["attended_rate"].median()
    event_df["attended"] = (event_df["attended_rate"] > median_rate).astype(int)

    print(f"\nEvent-level dataset built:")
    print(f"  Total events        : {len(event_df)}")
    print(f"  Median attended_rate: {median_rate:.3f}")
    print(f"  high_attendance=1   : {event_df['attended'].sum()} ({event_df['attended'].mean()*100:.1f}%)")
    print(f"  high_attendance=0   : {(event_df['attended']==0).sum()} ({(event_df['attended']==0).mean()*100:.1f}%)")
    print(f"\nAttended rate distribution:")
    print(f"  min   : {event_df['attended_rate'].min():.3f}")
    print(f"  25%   : {event_df['attended_rate'].quantile(0.25):.3f}")
    print(f"  median: {median_rate:.3f}")
    print(f"  75%   : {event_df['attended_rate'].quantile(0.75):.3f}")
    print(f"  max   : {event_df['attended_rate'].max():.3f}")

    print(f"\nWeather variation across events:")
    print(f"  weather_temp_C    unique: {event_df['weather_temp_C'].nunique()}")
    print(f"  weather_precip_mm unique: {event_df['weather_precip_mm'].nunique()}")
    print(f"  Temp range: {event_df['weather_temp_C'].min():.1f}°C to {event_df['weather_temp_C'].max():.1f}°C")

    # Correlation check at event level
    corr_temp  = event_df["weather_temp_C"].corr(event_df["attended_rate"])
    corr_precip = event_df["weather_precip_mm"].corr(event_df["attended_rate"])
    print(f"\nCorrelation weather vs attended_rate (event level):")
    print(f"  weather_temp_C    r = {corr_temp:+.4f}")
    print(f"  weather_precip_mm r = {corr_precip:+.4f}")
    print(f"  (previously at row level: temp r=+0.1686, precip r=-0.0108)")

    event_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Shape: {event_df.shape}")
    print(f"Columns: {event_df.columns.tolist()}")


if __name__ == "__main__":
    main()
