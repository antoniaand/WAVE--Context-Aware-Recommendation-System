# WAVE Production API - Context-Aware Event Recommender
"""
app/services/ml_service.py
--------------------------
ML inference service. Accepts a pre-built list of event dicts (from event_service)
and scores each one against the user profile + weather context.

Feature engineering replicates train_models.py exactly so that inference-time
features are bit-for-bit identical to training-time features.
"""

import logging
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR    = (_BACKEND_ROOT / settings.MODELS_DIR).resolve()

# ── Column config (identical to train_models.py) ──────────────────────────────
DROP_COLS = [
    "user_id", "event_id", "interaction_id",
    "event_date", "preferred_event_types", "final_prob",
]
CATEGORICAL_COLS = [
    "gender", "age_range", "attendance_freq",
    "top_event", "event_type", "location", "climate_zone",
]

LABEL_MAPS: dict[str, dict[str, int]] = {
    "gender":          {"F": 0, "M": 1},
    "age_range":       {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4},
    "attendance_freq": {"Never": 0, "Occasionally": 1, "Often": 2, "Rarely": 3, "Very often": 4},
    "top_event":       {"Concert": 0, "Conference": 1, "Festival": 2, "Sports": 3, "Theatre": 4},
    "event_type":      {"Concert": 0, "Conference": 1, "Festival": 2, "Sports": 3, "Theatre": 4},
    "location": {
        "Bergen": 0, "Brasov": 1, "Bucharest": 2, "Cluj-Napoca": 3, "Constanta": 4,
        "Dubai": 5, "Helsinki": 6, "Iasi": 7, "London": 8, "Oslo": 9,
        "Phoenix": 10, "Quebec": 11, "Seattle": 12, "Seville": 13, "Timisoara": 14,
    },
    "climate_zone": {"Cold": 0, "Hot": 1, "Moderate": 2, "Rainy": 3},
}

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

DEFAULT_WEATHER = {
    "weather_temp_C":         15.0,
    "weather_humidity":       65.0,
    "weather_precip_mm":       0.0,
    "weather_wind_speed_kmh": 10.0,
}

DEFAULT_PROFILE = {
    "gender": "F", "age_range": "25-34", "attendance_freq": "Occasionally",
    "top_event": "Concert", "preferred_event_types": "Concert,Festival,Sports,Theatre,Conference",
    # indoor_outdoor: 0=indoor, 1=outdoor
    "indoor_outdoor": 0,
    # rain_avoid, cold_tolerance, heat_sensitivity, wind_sensitivity, override_weather: 1–5 Likert
    "rain_avoid": 3, "cold_tolerance": 3,
    "heat_sensitivity": 3, "wind_sensitivity": 3, "override_weather": 3,
    # scenario_*: 0–3 (Would=3, Probably=2, Probably not=1, Would not=0)
    "scenario_concert": 2, "scenario_festival": 2, "scenario_sports": 2,
    "scenario_theatre": 2, "scenario_conference": 2,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_month"] = pd.to_datetime(df["event_date"]).dt.month
    if "preferred_event_types" in df.columns:
        df["event_in_preferred"] = df.apply(
            lambda row: int(str(row["event_type"]) in str(row["preferred_event_types"])),
            axis=1,
        )
    else:
        df["event_in_preferred"] = 0
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).map(LABEL_MAPS.get(col, {})).fillna(-1).astype(int)
    return df


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    _lgbm = _xgb = _rf_strict = _scaler = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        logger.info("Loading WAVE ML models from %s ...", MODELS_DIR)
        for attr, fname in [("_lgbm", "lgbm_contextual"), ("_xgb", "xgb_contextual"),
                            ("_rf_strict", "baseline_strict_rf"), ("_scaler", "scaler")]:
            try:
                setattr(cls, attr, joblib.load(MODELS_DIR / f"{fname}.joblib"))
                logger.info("  ✓ %s loaded", fname)
            except FileNotFoundError:
                logger.error("  ✗ %s.joblib not found at %s", fname, MODELS_DIR)
        cls._loaded = True

    @classmethod
    def get_model(cls, name: str):
        cls.load()
        m = {"lgbm": cls._lgbm, "xgb": cls._xgb, "rf_strict": cls._rf_strict}.get(name)
        if m is None:
            raise ValueError(f"Model '{name}' unavailable. Choose: lgbm, xgb, rf_strict")
        return m

    @classmethod
    def get_scaler(cls):
        cls.load()
        return cls._scaler


# ── Public inference API ───────────────────────────────────────────────────────

def predict_attended_probability(
    user_profile: dict,
    events: List[dict],
    weather_features: Optional[dict] = None,
    model_name: str = "lgbm",
    top_n: int = 10,
) -> list[dict]:
    """
    Score a list of event dicts against the user profile + weather.

    Args:
        user_profile:     Dict of user profile fields (may be partial/default).
        events:           List of event dicts from event_service (already normalized).
        weather_features: Dict with weather_temp_C, weather_humidity, etc.
        model_name:       'lgbm' | 'xgb' | 'rf_strict'
        top_n:            Max events to return.

    Returns:
        List of result dicts sorted by attended_prob descending, enriched with
        original event metadata (event_name, venue, url, etc.).
    """
    if not events:
        return []

    # Merge with default profile so missing fields don't crash encoding
    profile = {**DEFAULT_PROFILE, **{k: v for k, v in user_profile.items() if v is not None}}

    df = pd.DataFrame(events)

    # Attach user profile columns to every row
    for col, val in profile.items():
        if col not in df.columns:
            df[col] = val

    # Attach weather
    wf = {**DEFAULT_WEATHER, **(weather_features or {})}
    for col, val in wf.items():
        df[col] = val

    # preferred_event_types needed for event_in_preferred
    df["preferred_event_types"] = str(profile.get("preferred_event_types", ""))

    df = engineer_features(df)
    df = encode_categoricals(df)

    feature_order = STRICT_FEATURE_ORDER if model_name == "rf_strict" else CONTEXTUAL_FEATURE_ORDER
    X = df.reindex(columns=feature_order, fill_value=0)

    scaler = ModelRegistry.get_scaler()
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as exc:
            logger.warning("Scaler transform failed (%s); using raw features.", exc)
            X_scaled = X.values
    else:
        X_scaled = X.values

    model = ModelRegistry.get_model(model_name)
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    results = []
    for i, event in enumerate(events):
        results.append({
            "event_type":   event["event_type"],
            "event_name":   event.get("event_name"),
            "location":     event["location"],
            "venue":        event.get("venue"),
            "event_date":   event["event_date"],
            "attended_prob": round(float(probs[i]), 4),
            "climate_zone": event.get("climate_zone"),
            "is_outdoor":   event.get("is_outdoor", 0),
            "source":       event.get("source", "generated"),
            "is_generated": bool(event.get("is_generated", False)),
            "url":          event.get("url"),
            "image_url":    event.get("image_url"),
            "description":  event.get("description"),
        })

    results.sort(key=lambda r: r["attended_prob"], reverse=True)
    return results[:top_n]


def preload_models():
    ModelRegistry.load()
