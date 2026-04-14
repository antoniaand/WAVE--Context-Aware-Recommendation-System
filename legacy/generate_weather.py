#!/usr/bin/env python3
"""
Populate weather_temp_C and weather_precip_mm with synthetic, realistic data.
Correlates with season/month and introduces impact on attended (bad weather → lower attendance).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train_ready.csv"

# Temperature by month (mean °C, std) – climat temperate
TEMP_BY_MONTH = {
    1: (2, 4), 2: (3, 4), 3: (8, 5), 4: (13, 5), 5: (18, 4), 6: (22, 4),
    7: (25, 4), 8: (24, 4), 9: (19, 5), 10: (13, 5), 11: (7, 5), 12: (3, 4),
}

# Precipitation by season (mean mm, std) – mai mult toamna/iarna
# season: 0=winter, 1=spring, 2=summer, 3=autumn
PRECIP_BY_SEASON = {
    0: (3.5, 2.5),   # winter – mai mult
    1: (2.0, 1.8),   # spring
    2: (1.2, 1.5),   # summer – mai puțin
    3: (4.0, 3.0),   # autumn – cel mai mult
}


def _get_season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def main():
    df = pd.read_csv(TRAIN_PATH)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    n = len(df)
    rng = np.random.default_rng(42)

    # --- Extract month for weather logic (from datetime) ---
    month = df["event_datetime"].dt.month.fillna(6).astype(int)
    season = month.apply(_get_season)

    # --- Generate realistic temperature (normal by month) ---
    temp_raw = np.zeros(n)
    for i in range(n):
        m = month.iloc[i]
        mean_t, std_t = TEMP_BY_MONTH.get(m, (15, 5))
        temp_raw[i] = rng.normal(mean_t, std_t)
    df["weather_temp_C"] = np.clip(temp_raw, -15, 40)

    # --- Generate realistic precipitation (lognormal by season, > 0) ---
    precip_raw = np.zeros(n)
    for i in range(n):
        s = season.iloc[i]
        mean_p, std_p = PRECIP_BY_SEASON[s]
        val = rng.lognormal(np.log(mean_p + 0.1), 0.8)
        precip_raw[i] = min(val, 25)
    df["weather_precip_mm"] = precip_raw

    # --- Outdoor sensitivity: mark ~40% of events as outdoor-sensitive deterministically ---
    # Use event_id modulo rule to select approximately 40% (event_id % 5 in {0,1} => 2/5 = 40%)
    event_id = df["event_id"].fillna(1).astype(int)
    outdoor_flag = event_id % 5 <= 1
    # sensitivity factor range [0.5, 1.0] for outdoor events, lower for indoor
    outdoor_sensitivity = outdoor_flag.astype(float) * 0.5 + 0.5  # outdoor -> 1.0, indoor -> 0.5

    # --- Broaden definition of 'bad weather' (more aggressive) ---
    # Temperatures considered uncomfortable: <5°C or >28°C
    extreme_cold = df["weather_temp_C"] < 5
    extreme_hot = df["weather_temp_C"] > 28
    # Precipitation threshold lowered to >3 mm
    heavy_precip = df["weather_precip_mm"] > 3
    bad_weather = extreme_cold | extreme_hot | heavy_precip

    # --- Increase base flip probability when bad weather occurs ---
    # Use a stronger base factor to push class balance toward more 0 labels.
    base_factor = 0.9
    # Add a small extra penalty for very heavy precip or extreme temperatures
    extra = ((df["weather_precip_mm"] > 6).astype(float) + (df["weather_temp_C"] < -2).astype(float) + (df["weather_temp_C"] > 34).astype(float)) * 0.05

    flip_prob = bad_weather.astype(float) * outdoor_sensitivity * base_factor + extra
    # cap probabilities to [0, 0.95]
    flip_prob = np.minimum(flip_prob, 0.95)
    flip_mask = rng.random(n) < flip_prob
    df.loc[flip_mask, "attended"] = 0

    # --- Standardization (StandardScaler) pe weather ---
    scaler = StandardScaler()
    df[["weather_temp_C", "weather_precip_mm"]] = scaler.fit_transform(
        df[["weather_temp_C", "weather_precip_mm"]]
    )

    df.to_csv(TRAIN_PATH, index=False)
    print(f"Updated {TRAIN_PATH}: weather populated, attended 0: {(df['attended'] == 0).sum()}")


if __name__ == "__main__":
    main()
