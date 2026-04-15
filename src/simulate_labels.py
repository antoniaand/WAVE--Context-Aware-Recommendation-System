"""
simulate_labels.py
------------------
Step 3 of the modular WAVE dataset rebuild.

Generates the binary `attended` label (0/1) for every row in
interaction_with_weather.csv using a high-fidelity behavioural simulation.

Design goals
------------
* Weather must be a PRIMARY driver for outdoor events (not a minor nudge).
* Indoor events are almost fully immune to weather.
* The blind model (no weather / no sensitivity features) should score
  ~10+ pp lower than the full contextual model on this data.

base_prob is rescaled to [0.10, 0.65] so the decision threshold sits near
the middle of the distribution — making it possible for a realistic weather
penalty to flip individual decisions without needing extreme conditions.

Pipeline
--------
1. Load data/processed/interaction_with_weather.csv
2. Compute base_prob  from scenario columns → [0.10, 0.65]
3. Compute weather_adjust: strong negative for bad outdoor weather,
   small positive bonus for ideal outdoor weather
4. Apply motivational override multiplier
5. Combine into final_prob + small Gaussian jitter
6. Threshold at the median → balanced 50/50 attended label
7. Export data/processed/train_ready_interactions.csv
8. Print a validation report
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Comfort index (Zhang et al., 2020) ────────────────────────────────────────
#
# We implement the Discomfort Index (DI) form used in the key paper, which
# combines temperature, humidity, and wind into a single comfort signal.
# Humidity is fetched from Open-Meteo as `relative_humidity_2m` and stored as
# `weather_humidity` (0–100).
#
# DI = 1.8*T - 0.55*(1.8*T - 26)*(1 - F) - 3.2*sqrt(Ws) + 32
#   T  = temperature in °C
#   F  = relative humidity fraction (0–1)
#   Ws = wind speed in m/s
#
# Higher/lower DI away from an "ideal" comfort point reduces outdoor attendance.

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "data" / "processed" / "interaction_with_weather.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "train_ready_interactions.csv"

SEED = 42
rng  = np.random.default_rng(SEED)

# ── Scenario column map ───────────────────────────────────────────────────────
# Maps each event_type to the relevant survey column(s).
# Exhibition has no direct column → we proxy it as the mean of theatre + concert
# (both are cultural, often indoor events with similar audience).
SCENARIO_COL: dict[str, list[str]] = {
    "Concert":    ["scenario_concert"],
    "Festival":   ["scenario_festival"],
    "Sports":     ["scenario_sports"],
    "Theatre":    ["scenario_theatre"],
    "Conference": ["scenario_conference"],
    "Exhibition": ["scenario_theatre", "scenario_concert"],   # proxy
}


# ── Step 1 – Base probability ─────────────────────────────────────────────────
#
# Rescaled to [0.10, 0.65] so the decision threshold sits near 0.45–0.52,
# which is within reach of realistic weather penalties.
# Using [0, 1] made the threshold ~0.93 and weather penalties could not
# meaningfully flip decisions.

def compute_base_prob(df: pd.DataFrame) -> pd.Series:
    base = pd.Series(np.nan, index=df.index)
    for event_type, cols in SCENARIO_COL.items():
        mask = df["event_type"] == event_type
        base.loc[mask] = df.loc[mask, cols].mean(axis=1) / 3.0
    base = base.fillna(0.5)
    # Rescale from [0, 1] → [0.10, 0.65]
    return 0.10 + base * 0.55


# ── Step 2 – Raw weather components ──────────────────────────────────────────
#
#  rain_raw  : 0 mm = 0.0  |  >= 4 mm  = 1.0
#  cold_raw  : >= 8°C = 0  |  <= -12°C = 1.0
#  heat_raw  : <= 27°C = 0 |  >= 38°C  = 1.0
#  wind_raw  : 0 km/h = 0  |  >= 45 km/h = 1.0

def compute_raw_weather(df: pd.DataFrame) -> pd.DataFrame:
    rain_raw = (df["weather_precip_mm"] / 4.0).clip(0, 1)
    cold_raw = ((8  - df["weather_temp_C"])  / 20.0).clip(0, 1)
    heat_raw = ((df["weather_temp_C"] - 27)  / 11.0).clip(0, 1)
    wind_raw = (df["weather_wind_speed_kmh"] / 45.0).clip(0, 1)
    return pd.DataFrame({
        "rain_raw": rain_raw,
        "cold_raw": cold_raw,
        "heat_raw": heat_raw,
        "wind_raw": wind_raw,
    })

def compute_comfort_index(df: pd.DataFrame) -> pd.Series:
    T = df["weather_temp_C"]
    F = df["weather_humidity"] / 100.0
    Ws = df["weather_wind_speed_kmh"] / 3.6  # km/h -> m/s
    DI = 1.8 * T - 0.55 * (1.8 * T - 26) * (1 - F) - 3.2 * np.sqrt(Ws.clip(lower=0)) + 32
    return DI.clip(0, 90)


# ── Step 3 – Weather adjustment (signed) ─────────────────────────────────────
#
# Separate logic for outdoor vs. indoor:
#
#   OUTDOOR  — weather is a PRIMARY driver.
#     Penalty weights are high enough that bad weather + sensitive user
#     flips the decision even for someone who loves that event type.
#     Nice weather adds a small positive bonus.
#
#   INDOOR   — weather is almost irrelevant; only mild cold/heat effect.
#
# User sensitivity normalisation (1–5 scale → [0, 1]):
#   rain_intol  = (rain_avoid - 1) / 4          higher = more rain-averse
#   cold_intol  = (5 - cold_tolerance) / 4      higher = less cold-tolerant
#   heat_intol  = (heat_sensitivity - 1) / 4    higher = more heat-sensitive
#   wind_intol  = (wind_sensitivity - 1) / 4    higher = more wind-sensitive
#
# Literature alignment (why these weights look like this)
# -------------------------------------------------------
# - Zhang et al. (2020) (real Meetup check-ins + real weather) report that:
#   - Temperature has the strongest *direct* effect on outdoor attendance.
#   - Rain/snow reduces attendance (secondary but visible), and wind contributes via a
#     "human comfort index" (temperature + humidity + wind) that suppresses attendance
#     when too high/low. Indoor events show mostly indirect / damped effects.
# - Multiple extreme-weather attendance analyses report substantial drops under
#   extreme cold/heat/heavy precipitation (often tens of percent for outdoor festivals).
#
# Implementation notes:
# - We approximate the comfort effect using temperature deviation (cold_raw/heat_raw)
#   plus wind (wind_raw) because humidity is not available in our archive features.
# - We intentionally make temperature-related penalties (cold + heat) the largest
#   contributors for outdoor events, while keeping rain stronger than heat (rain > heat).

def compute_weather_adjust(df: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    rain_intol = (df["rain_avoid"]        - 1) / 4
    cold_intol = (5 - df["cold_tolerance"]) / 4
    heat_intol = (df["heat_sensitivity"]  - 1) / 4
    wind_intol = (df["wind_sensitivity"]  - 1) / 4

    # --- Outdoor penalty (strong) ---
    # Discomfort from DI: distance from an ideal comfort point (≈69 in this DI scale).
    # Scale factor chosen so a 20–25 point deviation saturates to 1.0 discomfort.
    DI = compute_comfort_index(df)
    ideal = 69.0
    di_discomfort = (DI - ideal).abs() / 25.0
    di_discomfort = di_discomfort.clip(0, 1)

    # Temperature is the strongest direct driver (via DI), rain is secondary but strong,
    # and we keep a small explicit wind term to reflect direct wind aversion beyond DI.
    outdoor_penalty = (
        ((cold_intol + heat_intol) / 2.0) * di_discomfort * 0.80
        + rain_intol * raw["rain_raw"] * 0.55
        + wind_intol * raw["wind_raw"] * 0.10
    ).clip(0, 0.80)

    # Nice-weather outdoor bonus: clear + 13–24 °C + low wind
    nice = (
        (raw["rain_raw"] == 0)
        & df["weather_temp_C"].between(13, 24)
        & (raw["wind_raw"] < 0.25)
    )
    outdoor_bonus = nice.astype(float) * 0.10

    outdoor_adjust = outdoor_bonus - outdoor_penalty

    # --- Indoor adjustment (minimal) ---
    indoor_adjust = -(cold_intol * raw["cold_raw"] * 0.05
                      + heat_intol * raw["heat_raw"] * 0.03).clip(0, 0.12)

    adjust = pd.Series(np.where(df["is_outdoor"] == 1,
                                outdoor_adjust, indoor_adjust),
                       index=df.index)
    return adjust


# ── Step 4 – Motivational override multiplier ─────────────────────────────────
#
# A highly motivated user (override_weather >= 4) who specifically prefers
# this event type resists weather pressure — their penalty is reduced to 40 %.
# They still feel *some* weather effect (not 0), making the signal learnable.

def apply_override(df: pd.DataFrame, adjust: pd.Series) -> pd.Series:
    adjust = adjust.copy()
    high_override = df["override_weather"] >= 4
    in_preferred  = df.apply(
        lambda row: str(row["event_type"]) in str(row["preferred_event_types"]),
        axis=1,
    )
    # Only dampen the negative part; leave positive bonus untouched
    penalty_mask = adjust < 0
    override_mask = high_override & in_preferred & penalty_mask
    adjust.loc[override_mask] *= 0.40
    return adjust


# ── Step 5 – Affinity boost ───────────────────────────────────────────────────

def compute_affinity_boost(df: pd.DataFrame) -> pd.Series:
    top_match  = df["event_type"] == df["top_event"]
    pref_match = df.apply(
        lambda row: str(row["event_type"]) in str(row["preferred_event_types"]),
        axis=1,
    )
    boost = pd.Series(0.0, index=df.index)
    boost.loc[top_match]              = 0.10
    boost.loc[pref_match & ~top_match] = 0.04
    return boost


# ── Step 6 – Extreme-weather hard multiplier ─────────────────────────────────
#
# For truly extreme conditions, this overrides the smooth penalty and
# drives the probability toward 0 for sensitive users.  This forces
# the model to USE weather features — without them it cannot recover.
#
# Rules (outdoor events only):
#   Extreme heat  (temp > 35 C):  heat_sensitivity 5 -> factor 0.05
#   Extreme cold  (temp < 0 C):   cold_tolerance  1 -> factor 0.08
#                                  override_weather 5 -> factor 1.0 (immune)
#   Heavy rain    (precip > 5mm): rain_avoid      5 -> factor 0.10

def compute_extreme_multiplier(df: pd.DataFrame) -> pd.Series:
    mult       = pd.Series(1.0, index=df.index)
    is_outdoor = df["is_outdoor"] == 1

    # ── Extreme heat ──────────────────────────────────────────────────────────
    xheat      = is_outdoor & (df["weather_temp_C"] > 35)
    heat_intol = (df["heat_sensitivity"] - 1) / 4.0          # [0, 1]
    # factor: 0.05 (fully sensitive) … 1.0 (fully tolerant)
    heat_factor = (1.0 - heat_intol * 0.95).clip(0.05, 1.0)
    mult = mult.where(~xheat, mult * heat_factor)

    # ── Extreme cold ──────────────────────────────────────────────────────────
    xcold      = is_outdoor & (df["weather_temp_C"] < 0)
    cold_intol = (5 - df["cold_tolerance"]) / 4.0             # [0, 1]
    cold_factor = (1.0 - cold_intol * 0.92).clip(0.08, 1.0)
    # override_weather = 5 makes user immune to cold deterrent
    cold_factor_final = cold_factor.where(df["override_weather"] < 5, 1.0)
    mult = mult.where(~xcold, mult * cold_factor_final)

    # ── Heavy rain (>= 2 mm/hr is significant outdoor deterrent) ─────────────
    xrain      = is_outdoor & (df["weather_precip_mm"] >= 2)
    rain_intol = (df["rain_avoid"] - 1) / 4.0                 # [0, 1]
    rain_factor = (1.0 - rain_intol * 0.90).clip(0.10, 1.0)
    mult = mult.where(~xrain, mult * rain_factor)

    return mult


# ── Step 7 – Final probability & balanced label ───────────────────────────────

def compute_final_prob(base: pd.Series, adjust: pd.Series,
                       boost: pd.Series, extreme_mult: pd.Series) -> pd.Series:
    prob = (base + adjust + boost) * extreme_mult
    jitter = rng.normal(loc=0.0, scale=0.025, size=len(prob))
    return (prob + jitter).clip(0.0, 1.0)


def threshold_balanced(prob: pd.Series) -> pd.Series:
    """
    Threshold at the median to guarantee a 50/50 class split.
    Ties at the median are deterministically resolved by the sign of
    (prob - median) so the label is reproducible.
    """
    median = prob.median()
    attended = (prob >= median).astype(int)
    # Verify balance (should be ±1 row at most)
    rate = attended.mean()
    if not (0.49 <= rate <= 0.51):
        print(f"  WARNING: attendance rate {rate:.4f} is outside [0.49, 0.51]")
    return attended


# ── Step 7 – Validation report ────────────────────────────────────────────────

def print_validation(df: pd.DataFrame) -> None:
    sep = "=" * 60
    print("\n" + sep)
    print("  LABEL SIMULATION - VALIDATION REPORT")
    print(sep)

    total  = len(df)
    attend = df["attended"].sum()
    print(f"\n  Total rows          : {total:,}")
    print(f"  Attended = 1        : {attend:,}  ({attend/total*100:.1f}%)")
    print(f"  Attended = 0        : {total-attend:,}  ({(total-attend)/total*100:.1f}%)")

    # -- Outdoor in Rain vs. Outdoor in Sun ------------------------------------
    outdoor = df[df["is_outdoor"] == 1]
    rainy   = outdoor[outdoor["weather_precip_mm"] > 0]
    sunny   = outdoor[outdoor["weather_precip_mm"] == 0]
    rainy_rate = rainy["attended"].mean() if len(rainy) else float("nan")
    sunny_rate = sunny["attended"].mean() if len(sunny) else float("nan")

    print("\n-- Outdoor events: Rain vs. No Rain -------------------------")
    print(f"  Outdoor + Rain   : {len(rainy):>6,} rows | attendance {rainy_rate*100:.1f}%")
    print(f"  Outdoor + No Rain: {len(sunny):>6,} rows | attendance {sunny_rate*100:.1f}%")
    print(f"  Rain penalty delta : {(sunny_rate - rainy_rate)*100:+.1f} pp")

    # -- override_weather = 5 vs. = 1 -----------------------------------------
    ov5 = df[df["override_weather"] == 5]
    ov1 = df[df["override_weather"] == 1]
    ov5_rate = ov5["attended"].mean() if len(ov5) else float("nan")
    ov1_rate = ov1["attended"].mean() if len(ov1) else float("nan")

    print("\n-- override_weather extremes --------------------------------")
    print(f"  override = 5 (ignores weather): {len(ov5):>6,} rows | attendance {ov5_rate*100:.1f}%")
    print(f"  override = 1 (very weather-dep): {len(ov1):>6,} rows | attendance {ov1_rate*100:.1f}%")
    print(f"  Override motivation delta      : {(ov5_rate - ov1_rate)*100:+.1f} pp")

    # -- By event type ---------------------------------------------------------
    print("\n-- Attendance rate by event type ----------------------------")
    by_type = (
        df.groupby("event_type")["attended"]
        .agg(["sum", "count"])
        .assign(rate=lambda x: x["sum"] / x["count"] * 100)
        .rename(columns={"sum": "attended_n", "count": "total"})
        .sort_values("rate", ascending=False)
    )
    print(by_type[["attended_n", "total", "rate"]].to_string())

    # -- By climate zone -------------------------------------------------------
    if "climate_zone" in df.columns:
        print("\n-- Attendance rate by climate zone --------------------------")
        by_zone = (
            df.groupby("climate_zone")
            .agg(
                attend_rate=("attended", "mean"),
                avg_temp=("weather_temp_C", "mean"),
                avg_precip=("weather_precip_mm", "mean"),
                rows=("attended", "count"),
            )
            .assign(attend_pct=lambda x: x["attend_rate"] * 100)
        )
        for zone, row in by_zone.iterrows():
            print(f"  {zone:<12} | attendance {row['attend_pct']:5.1f}% | "
                  f"avg temp {row['avg_temp']:5.1f}C | "
                  f"avg precip {row['avg_precip']:.2f}mm | "
                  f"rows {int(row['rows']):,}")

    print("\n-- Extreme weather attendance breakdown ---------------------")
    xheat_out = df[(df["weather_temp_C"] > 35) & (df["is_outdoor"] == 1)]
    xcold_out = df[(df["weather_temp_C"] < 0)  & (df["is_outdoor"] == 1)]
    xrain_out = df[(df["weather_precip_mm"] >= 2) & (df["is_outdoor"] == 1)]
    for label, subset in [("Outdoor + temp>35C", xheat_out),
                           ("Outdoor + temp<0C",  xcold_out),
                           ("Outdoor + precip>=2mm", xrain_out)]:
        if len(subset):
            print(f"  {label:<28} | {len(subset):>6,} rows | "
                  f"attendance {subset['attended'].mean()*100:.1f}%")

    print("\n-- final_prob distribution ----------------------------------")
    desc = df["final_prob"].describe()
    print(f"  mean={desc['mean']:.3f}  std={desc['std']:.3f}"
          f"  min={desc['min']:.3f}  median={df['final_prob'].median():.3f}"
          f"  max={desc['max']:.3f}")

    print("\n" + sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  WAVE - simulate_labels.py")
    print("=" * 60)

    print("\n[1/6] Loading interaction_with_weather.csv ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

    print("\n[2/6] Computing base probabilities from scenario columns ...")
    base_prob = compute_base_prob(df)
    print(f"  base_prob  mean={base_prob.mean():.3f}  std={base_prob.std():.3f}")

    print("\n[3/6] Computing weather adjustment (outdoor strong, indoor minimal) ...")
    raw    = compute_raw_weather(df)
    adjust = compute_weather_adjust(df, raw)
    print(f"  raw adjust   mean={adjust.mean():.3f}  min={adjust.min():.3f}  max={adjust.max():.3f}")

    print("\n[4/6] Applying motivational override multiplier ...")
    adjust = apply_override(df, adjust)
    print(f"  final adjust mean={adjust.mean():.3f}  min={adjust.min():.3f}")

    print("\n[5/6] Computing affinity boosts, extreme multiplier, final probability ...")
    boost         = compute_affinity_boost(df)
    extreme_mult  = compute_extreme_multiplier(df)
    n_extreme     = (extreme_mult < 0.5).sum()
    print(f"  Rows with extreme multiplier < 0.5: {n_extreme:,}  "
          f"({n_extreme/len(df)*100:.1f}% of dataset)")
    final_prob = compute_final_prob(base_prob, adjust, boost, extreme_mult)
    print(f"  final_prob  mean={final_prob.mean():.3f}  median={final_prob.median():.3f}")

    df["final_prob"] = final_prob
    df["attended"]   = threshold_balanced(final_prob)

    print(f"\n[6/6] Saving to {OUTPUT_CSV} ...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(df):,} rows x {df.shape[1]} columns")

    print_validation(df)


if __name__ == "__main__":
    main()
