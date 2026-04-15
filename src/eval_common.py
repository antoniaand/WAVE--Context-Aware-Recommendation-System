"""
eval_common.py
--------------
Shared evaluation helpers: same preprocessing / GroupShuffleSplit as train_models.
Subgroup slice uses RAW weather (pre-StandardScaler) for thresholds.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from train_models import (
    CSV_PATH,
    DROP_COLS,
    MODELS_DIR,
    TARGET,
    WEATHER_COLS,
    engineer_features,
    encode_categoricals,
)

ROOT = Path(__file__).resolve().parents[1]

WEATHER_COLS_SET = set(WEATHER_COLS)
STRICT_BASELINE_EXTRA_DROP = {"location", "climate_zone", "event_month"}

# Model registry: (display label, joblib filename, uses_weather, extra_drop_cols)
MODEL_REGISTRY = [
    ("RF Baseline",           "baseline_rf.joblib",        False, set()),
    ("RF Baseline (strict)",  "baseline_strict_rf.joblib", False, STRICT_BASELINE_EXTRA_DROP),
    ("RF Contextual",         "contextual_rf.joblib",       True, set()),
    ("LGBM Contextual",       "lgbm_contextual.joblib",     True, set()),
    ("XGB Contextual",        "xgb_contextual.joblib",      True, set()),
]


def load_scaled_test_split():
    """
    Reproduce training pipeline through test split; return scaled X_test, y_test,
    and X_test_raw (same rows/columns, values before StandardScaler) for subgroup masks.
    """
    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)
    df = encode_categoricals(df)
    groups = df["user_id"].values
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))

    X_test_raw = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw), columns=X_test_raw.columns
    )

    return X_test, y_test, X_test_raw


def extreme_weather_slice_mask(X_pre_scale: pd.DataFrame) -> pd.Series:
    """
    Outdoor rows with cold (<5°C) or non-trivial precip (>0.5 mm).
    Must be applied to pre-scaler frame so thresholds are in physical units.
    """
    return (
        (X_pre_scale["is_outdoor"] == 1)
        & (
            (X_pre_scale["weather_temp_C"] < 5)
            | (X_pre_scale["weather_precip_mm"] > 0.5)
        )
    )


def get_X_for_model(X_test: pd.DataFrame, uses_weather: bool, extra_drop: set[str] | None = None) -> pd.DataFrame:
    if uses_weather:
        X = X_test
    else:
        X = X_test[[c for c in X_test.columns if c not in WEATHER_COLS_SET]]
    if extra_drop:
        keep = [c for c in X.columns if c not in extra_drop]
        return X[keep]
    return X


def get_pos_probs(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else 1
        return probs_all[:, pos_idx]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)
