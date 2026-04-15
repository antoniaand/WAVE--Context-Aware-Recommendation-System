#!/usr/bin/env python3
"""
WAVE – Train and evaluate four models on the user×event×weather interaction dataset.

Models:
  1. RF Baseline      – RandomForest, NO weather features
  2. RF Contextual    – RandomForest, WITH weather features
  3. LGBM Contextual  – LightGBM,     WITH weather features
  4. XGB Contextual   – XGBoost,      WITH weather features

Split strategy: GroupShuffleSplit on user_id
  - 88 users for training, 22 users for testing
  - No user appears in both train and test (mirrors real-world: new user = cold start)
  - This is academically correct for user-level generalisation

Why not random split?
  With a random row split, rows from the same user appear in both train and test.
  The model can "memorise" user profiles and inflate test metrics. Group split
  prevents this and tests true generalisation to unseen users.

Usage:
    python src/train_models.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROCESSED_DIR / "train_ready_interactions.csv"

# Hard identifiers and leak columns — never used as features
DROP_COLS = [
    "user_id", "event_id", "interaction_id",
    "event_date",           # replaced by event_month below
    "preferred_event_types",# multi-label string; captured by event_in_preferred
    "final_prob",           # continuous version of the label — would leak perfectly
]

# Weather feature names — absent in baseline, present in contextual
WEATHER_COLS = [
    "weather_temp_C",
    "weather_humidity",
    "weather_precip_mm",
    "weather_wind_speed_kmh",
]

# Categorical columns that need label-encoding before scaling
CATEGORICAL_COLS = [
    "gender", "age_range", "attendance_freq",
    "top_event", "event_type", "location", "climate_zone",
]

TARGET = "attended"


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Month of event (1-12) — captures seasonality from the real weather data
    df["event_month"] = pd.to_datetime(df["event_date"]).dt.month

    # Binary: does the user's preferred_event_types include this event_type?
    df["event_in_preferred"] = df.apply(
        lambda row: int(str(row["event_type"]) in str(row["preferred_event_types"])),
        axis=1,
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ─── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_split(path: Path):
    """
    Load interaction CSV, engineer features, encode categoricals,
    then split by user_id group.
    StandardScaler is fitted on X_train only (no leakage to test).
    """
    df = pd.read_csv(path)

    # Feature engineering before dropping anything
    df = engineer_features(df)
    df = encode_categoricals(df)

    # Keep user_id as group key for the split
    groups = df["user_id"].values

    # Drop identifier / leak columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # GroupShuffleSplit: no user appears in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    # Scale after split — fit only on training data
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    n_train_users = len(np.unique(groups[train_idx]))
    n_test_users  = len(np.unique(groups[test_idx]))
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Train users: {n_train_users} | Test users: {n_test_users}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class dist (train) — 0: {(y_train==0).sum():,}  1: {(y_train==1).sum():,}\n")
    return X_train, X_test, y_train, y_test


def get_feature_sets(X_train, X_test):
    """Baseline uses all features except weather; contextual uses all."""
    baseline_cols   = [c for c in X_train.columns if c not in WEATHER_COLS]
    # Strict baseline: also drop geography/season proxies (weather-blind + climate-blind)
    STRICT_DROP = {"location", "climate_zone", "event_month"}
    baseline_strict_cols = [c for c in baseline_cols if c not in STRICT_DROP]
    contextual_cols = list(X_train.columns)
    print(f"Baseline feature count  : {len(baseline_cols)}")
    print(f"Strict baseline features: {len(baseline_strict_cols)}  (drops {sorted(STRICT_DROP)})")
    print(f"Contextual feature count: {len(contextual_cols)}\n")
    return (
        X_train[baseline_cols], X_test[baseline_cols],
        X_train[baseline_strict_cols], X_test[baseline_strict_cols],
        X_train[contextual_cols], X_test[contextual_cols],
        baseline_cols, baseline_strict_cols, contextual_cols,
    )


# ─── Model Definitions ────────────────────────────────────────────────────────

def build_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )


def build_lgbm():
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1,
    )


def build_xgb():
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train(model, X_train, y_train, label: str):
    print(f"Training {label} ({X_train.shape[1]} features)...")
    model.fit(X_train, y_train)
    print(f"  Done.\n")
    return model


def evaluate(model, X_test, y_test, label: str) -> dict:
    y_pred = model.predict(X_test)
    return {
        "Model":     label,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }


def print_comparison(metrics_list: list):
    cols  = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    col_w = [30, 11, 12, 9, 11]
    header = "".join(c.ljust(w) for c, w in zip(cols, col_w))
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print("  NEW DATASET — MODEL COMPARISON")
    print(sep)
    print(header)
    print(sep)
    for m in metrics_list:
        print("".join(str(m[c]).ljust(w) for c, w in zip(cols, col_w)))
    print(sep)

    # vs. old results (only for models that existed historically)
    old = {
        "RF Baseline (no weather)": {"Accuracy": 0.7073, "Precision": 0.5574, "Recall": 0.5033, "F1-Score": 0.4296},
        "RF Contextual":            {"Accuracy": 0.7090, "Precision": 0.5906, "Recall": 0.5041, "F1-Score": 0.4292},
        "LGBM Contextual":          {"Accuracy": 0.7086, "Precision": 0.5827, "Recall": 0.5042, "F1-Score": 0.4301},
        "XGB Contextual":           {"Accuracy": 0.7094, "Precision": 0.5999, "Recall": 0.5046, "F1-Score": 0.4302},
    }
    print(f"\n{sep}")
    print("  DELTA vs. OLD DATASET  (new - old, positive = improvement)")
    print(sep)
    delta_header = "".join(c.ljust(w) for c, w in zip(
        ["Model", "dAccuracy", "dPrecision", "dRecall", "dF1-Score"], col_w))
    print(delta_header)
    print(sep)
    for m in metrics_list:
        name = m["Model"]
        if name not in old:
            continue
        row = [name] + [
            f"{m[k] - old[name][k]:+.4f}"
            for k in ["Accuracy", "Precision", "Recall", "F1-Score"]
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row, col_w)))
    print(sep + "\n")


def save_metrics(metrics_list: list):
    df = pd.DataFrame(metrics_list)
    out = RESULTS_DIR / "metrics_comparison.csv"
    # Pivot to match the old format: rows=metric, cols=model
    pivoted = df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].T
    pivoted.to_csv(out)
    print(f"Metrics saved -> {out}")


def print_feature_importances(model, feature_names: list, label: str, top_n: int = 7):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"Top {top_n} feature importances — {label}:")
    print("-" * 45)
    for feat, score in imp.items():
        print(f"  {feat:<35} {score:.4f}")
    print()


def save_model(model, filename: str):
    out = MODELS_DIR / filename
    joblib.dump(model, out)
    print(f"Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_and_split(CSV_PATH)

    X_tr_base, X_te_base, X_tr_strict, X_te_strict, X_tr_ctx, X_te_ctx, base_cols, strict_cols, ctx_cols = \
        get_feature_sets(X_train, X_test)

    baseline_model = train(build_rf(),   X_tr_base, y_train, "Baseline RF")
    strict_model   = train(build_rf(),   X_tr_strict, y_train, "Strict Baseline RF")
    rf_ctx_model   = train(build_rf(),   X_tr_ctx,  y_train, "Contextual RF")
    lgbm_model     = train(build_lgbm(), X_tr_ctx,  y_train, "Contextual LGBM")
    xgb_model      = train(build_xgb(),  X_tr_ctx,  y_train, "Contextual XGBoost")

    m_base = evaluate(baseline_model, X_te_base, y_test, "RF Baseline (no weather)")
    m_strict = evaluate(strict_model,  X_te_strict, y_test, "RF Baseline (strict)")
    m_rfctx = evaluate(rf_ctx_model,  X_te_ctx,  y_test, "RF Contextual")
    m_lgbm  = evaluate(lgbm_model,    X_te_ctx,  y_test, "LGBM Contextual")
    m_xgb   = evaluate(xgb_model,     X_te_ctx,  y_test, "XGB Contextual")
    print_comparison([m_base, m_strict, m_rfctx, m_lgbm, m_xgb])

    print_feature_importances(rf_ctx_model, ctx_cols, "RF Contextual")
    print_feature_importances(lgbm_model,   ctx_cols, "LGBM Contextual")
    print_feature_importances(xgb_model,    ctx_cols, "XGB Contextual")

    save_metrics([m_base, m_strict, m_rfctx, m_lgbm, m_xgb])

    save_model(baseline_model, "baseline_rf.joblib")
    save_model(strict_model,   "baseline_strict_rf.joblib")
    save_model(rf_ctx_model,   "contextual_rf.joblib")
    save_model(lgbm_model,     "lgbm_contextual.joblib")
    save_model(xgb_model,      "xgb_contextual.joblib")


if __name__ == "__main__":
    main()
