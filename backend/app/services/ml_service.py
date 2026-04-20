# WAVE Production API - Context-Aware Event Recommender
"""
app/services/ml_service.py
--------------------------
Machine-learning inference service.

Loads the trained models from ../models/ and replicates EXACTLY the
feature engineering pipeline from src/modeling/train_models.py so that
inference-time features are bit-for-bit identical to training-time features.

Models available:
  - lgbm_contextual.joblib   ← default (best performing)
  - xgb_contextual.joblib    ← XGBoost contextual
  - baseline_strict_rf.joblib← RF without any weather / geo features
  - scaler.joblib            ← StandardScaler fitted on training X_train

Feature engineering rules (must mirror train_models.py):
  - engineer_features(): event_month + event_in_preferred
  - encode_categoricals(): LabelEncoder per CATEGORICAL_COLS
  - Drop: DROP_COLS (user_id, event_id, interaction_id, event_date,
                     preferred_event_types, final_prob)
  - Scale: StandardScaler.transform(X) using the saved scaler

IMPORTANT: The LabelEncoder in training uses fit_transform PER column on
the FULL training dataset.  For inference we therefore hard-code the
category-to-integer mapping precisely as sklearn's LabelEncoder would
produce it (alphabetical sort order, 0-indexed).
"""

import logging
from datetime import datetime, date as DateType
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parents[2]   # .../backend/
MODELS_DIR    = (_BACKEND_ROOT / settings.MODELS_DIR).resolve()
DATA_DIR      = (_BACKEND_ROOT / settings.DATA_DIR).resolve()

# ── Column configuration (identical to train_models.py) ──────────────────────
DROP_COLS = [
    "user_id", "event_id", "interaction_id",
    "event_date",               # replaced by event_month
    "preferred_event_types",    # replaced by event_in_preferred
    "final_prob",               # label leak
]

WEATHER_COLS = [
    "weather_temp_C",
    "weather_humidity",
    "weather_precip_mm",
    "weather_wind_speed_kmh",
]

# Categorical columns encoded with LabelEncoder during training
# Order matches train_models.py CATEGORICAL_COLS list
CATEGORICAL_COLS = [
    "gender", "age_range", "attendance_freq",
    "top_event", "event_type", "location", "climate_zone",
]

TARGET = "attended"

# ── Hard-coded label maps (reproduce sklearn LabelEncoder alphabetical sort) ──
# These were produced by running LabelEncoder().fit(training_col_values) on the
# full dataset values.  They must NOT change without retraining the models.

LABEL_MAPS: dict[str, dict[str, int]] = {
    "gender": {"F": 0, "M": 1},

    "age_range": {
        "18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4,
    },

    "attendance_freq": {
        "Never":  0,
        "Occasionally": 1,
        "Often":  2,
        "Rarely": 3,
        "Very often": 4,
    },

    "top_event": {
        "Concert":     0,
        "Conference":  1,
        "Festival":    2,
        "Sports":      3,
        "Theatre":     4,
    },

    "event_type": {
        "Concert":     0,
        "Conference":  1,
        "Festival":    2,
        "Sports":      3,
        "Theatre":     4,
    },

    "location": {
        "Bergen":      0,
        "Brasov":      1,
        "Bucharest":   2,
        "Cluj-Napoca": 3,
        "Constanta":   4,
        "Dubai":       5,
        "Helsinki":    6,
        "Iasi":        7,
        "London":      8,
        "Oslo":        9,
        "Phoenix":     10,
        "Quebec":      11,
        "Seattle":     12,
        "Seville":     13,
        "Timisoara":   14,
    },

    "climate_zone": {
        "Cold":      0,
        "Hot":       1,
        "Moderate":  2,
        "Rainy":     3,
    },
}

# Strict-baseline drops these on TOP of the baseline drops
STRICT_DROP = {"location", "climate_zone", "event_month"}

# Columns expected by contextual models (order must exactly match training)
# Derived from: X_train columns after engineer + encode + drop + scale
CONTEXTUAL_FEATURE_ORDER = [
    "gender", "age_range", "attendance_freq", "indoor_outdoor",
    "top_event", "rain_avoid", "cold_tolerance", "heat_sensitivity",
    "wind_sensitivity", "override_weather", "scenario_concert",
    "scenario_festival", "scenario_sports", "scenario_theatre",
    "scenario_conference", "event_type", "climate_zone", "is_outdoor",
    "location", "event_hour", "weather_temp_C", "weather_humidity",
    "weather_precip_mm", "weather_wind_speed_kmh", "event_month",
    "event_in_preferred",
]

STRICT_FEATURE_ORDER = [
    "gender", "age_range", "attendance_freq", "indoor_outdoor",
    "top_event", "rain_avoid", "cold_tolerance", "heat_sensitivity",
    "wind_sensitivity", "override_weather", "scenario_concert",
    "scenario_festival", "scenario_sports", "scenario_theatre",
    "scenario_conference", "event_type", "is_outdoor",
    "event_hour", "weather_temp_C", "weather_humidity",
    "weather_precip_mm", "weather_wind_speed_kmh", "event_in_preferred",
]


# ── Candidate events ──────────────────────────────────────────────────────────
# Representative events to score for each city.
# Each dict must include all fields that appear in feature engineering.

def _build_candidate_events(city: str, target_date: str, hour: int) -> pd.DataFrame:
    """
    Generate a DataFrame of candidate events for the given city and date.
    Each row represents one hypothetical event that the user might attend.

    Climate zones mirror the training dataset city → zone mapping.
    """
    city_meta: dict[str, dict] = {
        "Bucharest":   {"climate_zone": "Moderate"},
        "Cluj-Napoca": {"climate_zone": "Moderate"},
        "Timisoara":   {"climate_zone": "Moderate"},
        "Iasi":        {"climate_zone": "Moderate"},
        "Constanta":   {"climate_zone": "Moderate"},
        "Brasov":      {"climate_zone": "Moderate"},
        "Oslo":        {"climate_zone": "Cold"},
        "Helsinki":    {"climate_zone": "Cold"},
        "Quebec":      {"climate_zone": "Cold"},
        "Dubai":       {"climate_zone": "Hot"},
        "Phoenix":     {"climate_zone": "Hot"},
        "Seville":     {"climate_zone": "Hot"},
        "London":      {"climate_zone": "Rainy"},
        "Bergen":      {"climate_zone": "Rainy"},
        "Seattle":     {"climate_zone": "Rainy"},
    }
    meta = city_meta.get(city, {"climate_zone": "Moderate"})

    event_types = [
        ("Concert",    1, "outdoor"),
        ("Festival",   1, "outdoor"),
        ("Sports",     1, "outdoor"),
        ("Theatre",    0, "indoor"),
        ("Conference", 0, "indoor"),
    ]

    rows = []
    for event_type, is_outdoor, _ in event_types:
        rows.append({
            "event_id":    -1,              # sentinel — dropped in feature eng
            "event_type":  event_type,
            "location":    city,
            "event_date":  target_date,
            "event_hour":  hour,
            "climate_zone": meta["climate_zone"],
            "is_outdoor":  is_outdoor,
        })

    return pd.DataFrame(rows)


# ── Feature engineering (mirrors train_models.py exactly) ─────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate engineer_features() from src/modeling/train_models.py.

    Adds:
      - event_month (1-12)  from event_date
      - event_in_preferred  binary: is event_type in preferred_event_types?
    """
    df = df.copy()
    df["event_month"] = pd.to_datetime(df["event_date"]).dt.month

    # event_in_preferred requires preferred_event_types column
    if "preferred_event_types" in df.columns:
        df["event_in_preferred"] = df.apply(
            lambda row: int(str(row["event_type"]) in str(row["preferred_event_types"])),
            axis=1,
        )
    else:
        # Must be set from user profile before calling this function
        df["event_in_preferred"] = 0

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using the hard-coded LABEL_MAPS.

    In training, sklearn's LabelEncoder.fit_transform() sorts categories
    alphabetically and assigns 0-based integers.  LABEL_MAPS reproduces
    this mapping exactly for every value seen in the training data.

    Unknown values fall back to -1 (treated as "unseen category").
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        mapping = LABEL_MAPS.get(col, {})
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df


# ── Model loader (singleton, loaded once at startup) ──────────────────────────

class ModelRegistry:
    """Lazy-loads and caches ML models and the scaler from disk."""

    _lgbm   = None
    _xgb    = None
    _rf_strict = None
    _scaler = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        logger.info("Loading WAVE ML models from %s ...", MODELS_DIR)

        try:
            cls._lgbm      = joblib.load(MODELS_DIR / "lgbm_contextual.joblib")
            logger.info("  ✓ lgbm_contextual loaded")
        except FileNotFoundError:
            logger.error("  ✗ lgbm_contextual.joblib not found at %s", MODELS_DIR)

        try:
            cls._xgb       = joblib.load(MODELS_DIR / "xgb_contextual.joblib")
            logger.info("  ✓ xgb_contextual loaded")
        except FileNotFoundError:
            logger.error("  ✗ xgb_contextual.joblib not found at %s", MODELS_DIR)

        try:
            cls._rf_strict = joblib.load(MODELS_DIR / "baseline_strict_rf.joblib")
            logger.info("  ✓ baseline_strict_rf loaded")
        except FileNotFoundError:
            logger.error("  ✗ baseline_strict_rf.joblib not found at %s", MODELS_DIR)

        try:
            cls._scaler    = joblib.load(MODELS_DIR / "scaler.joblib")
            logger.info("  ✓ scaler loaded")
        except FileNotFoundError:
            logger.error("  ✗ scaler.joblib not found at %s", MODELS_DIR)

        cls._loaded = True
        logger.info("Model registry ready.")

    @classmethod
    def get_model(cls, name: str):
        cls.load()
        mapping = {
            "lgbm":      cls._lgbm,
            "xgb":       cls._xgb,
            "rf_strict": cls._rf_strict,
        }
        model = mapping.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' is not available. Choose from: {list(mapping)}")
        return model

    @classmethod
    def get_scaler(cls):
        cls.load()
        return cls._scaler


# ── Public inference API ───────────────────────────────────────────────────────

def predict_attended_probability(
    user_profile: dict,
    city: str,
    target_date: str,
    hour: int = 12,
    weather_features: Optional[dict] = None,
    model_name: str = "lgbm",
    top_n: int = 10,
) -> list[dict]:
    """
    Core prediction function.

    Pipeline (mirrors training):
      1. Build candidate events DataFrame for (city, date, hour)
      2. Attach user profile columns to every candidate row
      3. Attach weather features to every row (if available)
      4. engineer_features()   — adds event_month, event_in_preferred
      5. encode_categoricals() — label-encode with LABEL_MAPS
      6. Select & order feature columns (CONTEXTUAL or STRICT)
      7. scaler.transform(X)  — same scaler fitted during training
      8. model.predict_proba(X)[:,1] — probability of attended=1
      9. Sort descending, attach metadata, return top_n results

    Args:
        user_profile:     Dict of user profile fields.
        city:             Event city name.
        target_date:      Date string 'YYYY-MM-DD'.
        hour:             Event hour (0-23).
        weather_features: Dict with keys weather_temp_C, weather_humidity,
                          weather_precip_mm, weather_wind_speed_kmh.
        model_name:       'lgbm' | 'xgb' | 'rf_strict'
        top_n:            Maximum number of events to return.

    Returns:
        List of dicts sorted by attended_prob descending.
    """
    # 1. Build candidate events
    candidates = _build_candidate_events(city, target_date, hour)

    # 2. Attach user profile to every row
    for col, val in user_profile.items():
        candidates[col] = val

    # 3. Attach weather (or fill with training-time mean as fallback)
    default_weather = {
        "weather_temp_C":         15.0,
        "weather_humidity":       65.0,
        "weather_precip_mm":       0.0,
        "weather_wind_speed_kmh": 10.0,
    }
    wf = {**default_weather, **(weather_features or {})}
    for col, val in wf.items():
        candidates[col] = val

    # 4. Feature engineering (event_month, event_in_preferred)
    # preferred_event_types must be in user_profile for event_in_preferred
    candidates["preferred_event_types"] = str(user_profile.get("preferred_event_types", ""))
    candidates = engineer_features(candidates)

    # 5. Encode categoricals
    candidates = encode_categoricals(candidates)

    # 6. Select feature set based on model
    is_strict = (model_name == "rf_strict")
    feature_order = STRICT_FEATURE_ORDER if is_strict else CONTEXTUAL_FEATURE_ORDER

    # Keep only columns that exist and fill any missing with 0
    X = candidates.reindex(columns=feature_order, fill_value=0)

    # 7. Scale using the fitted scaler
    scaler = ModelRegistry.get_scaler()
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as exc:
            logger.warning("Scaler transform failed (%s); using raw features.", exc)
            X_scaled = X.values
    else:
        logger.warning("Scaler not loaded; using raw features.")
        X_scaled = X.values

    # 8. Predict attendance probability
    model = ModelRegistry.get_model(model_name)
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    # 9. Assemble results
    results = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        results.append({
            "event_type":    row.get("event_type_raw",  str(row.get("event_type"))),
            "location":      city,
            "event_date":    target_date,
            "attended_prob": round(float(probs[i]), 4),
            "climate_zone":  row.get("climate_zone_raw", None),
            "is_outdoor":    int(row.get("is_outdoor", 0)),
        })

    # We need original string values for response — re-build from candidates before encoding
    # Rebuild from the pre-encoded DataFrame for string values
    orig_candidates = _build_candidate_events(city, target_date, hour)
    for i, (_, orig_row) in enumerate(orig_candidates.iterrows()):
        results[i]["event_type"]  = orig_row["event_type"]
        results[i]["climate_zone"] = orig_row["climate_zone"]
        results[i]["is_outdoor"]   = int(orig_row["is_outdoor"])

    results.sort(key=lambda r: r["attended_prob"], reverse=True)
    return results[:top_n]


# ── Startup preload ────────────────────────────────────────────────────────────

def preload_models():
    """Call this during FastAPI startup to eagerly load models into RAM."""
    ModelRegistry.load()
