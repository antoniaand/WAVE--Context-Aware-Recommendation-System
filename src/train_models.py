#!/usr/bin/env python3
"""
WAVE – Train and evaluate four models:
  1. RF Baseline      – Random Forest, no weather features
  2. RF Contextual    – Random Forest, with weather features
  3. LGBM Contextual  – LightGBM, with weather features
  4. XGB Contextual   – XGBoost, with weather features

Usage:
    python src/train_models.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier   

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Columns to drop before training (non-predictive identifiers and raw text)
DROP_COLS = ["event_id", "event_datetime", "event_name", "location"]

# Weather columns – present in contextual model only
WEATHER_COLS = ["weather_temp_C", "weather_precip_mm"]

TARGET = "attended"


# ─────────────────────────────────────────────
# 1. Data Loading & Splitting
# ─────────────────────────────────────────────
def load_and_split(path: Path):
    """Load train_ready.csv, drop non-predictive columns, split 80/20 stratified."""
    df = pd.read_csv(path)

    # Drop identifier / raw-text columns that carry no predictive value
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    #fit just for train, bc fit= calculate mean and std for train, and you want to use those ones for test
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Class distribution (train) – 0: {(y_train==0).sum():,}  1: {(y_train==1).sum():,}\n")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
def get_feature_sets(X_train, X_test):
    """Return (baseline, contextual) feature matrices for train and test."""
    # Baseline: all features EXCEPT weather columns
    baseline_cols = [c for c in X_train.columns if c not in WEATHER_COLS]
    # Contextual: all features INCLUDING weather columns
    contextual_cols = list(X_train.columns)

    return (
        X_train[baseline_cols], X_test[baseline_cols],
        X_train[contextual_cols], X_test[contextual_cols],
        baseline_cols, contextual_cols,
    )


# ─────────────────────────────────────────────
# 3. Model Training
# ─────────────────────────────────────────────
def build_rf():
    """Shared RandomForest hyper-parameters tuned to prevent overfitting."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )


def build_lgbm():
    """LightGBM – leaf-wise gradient boosting, efficient on large tabular data."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb():
    """XGBoost – regularized gradient boosting, level-wise tree growth."""
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train(model, X_train, y_train, label: str):
    print(f"Training {label} ({X_train.shape[1]} features)…")
    model.fit(X_train, y_train)
    print(f"  Done.\n")
    return model


# ─────────────────────────────────────────────
# 4. Evaluation & Logging
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, label: str) -> dict:
    """Return a dict of metrics for one model."""
    y_pred = model.predict(X_test)
    return {
        "Model": label,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }


def print_comparison(metrics_list: list):
    """Print a formatted comparison table in the console."""
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    col_w = [26, 10, 11, 8, 10]
    header = "".join(c.ljust(w) for c, w in zip(cols, col_w))
    sep = "-" * len(header)
    print("\n" + sep)
    print("  MODEL COMPARISON")
    print(sep)
    print(header)
    print(sep)
    for m in metrics_list:
        row = "".join(str(m[c]).ljust(w) for c, w in zip(cols, col_w))
        print(row)
    print(sep + "\n")


def print_feature_importances(model, feature_names: list, label: str, top_n: int = 5):
    """Print top-N most important features for a trained model."""
    importance_df = pd.Series(model.feature_importances_, index=feature_names)
    importance_df = importance_df.sort_values(ascending=False).head(top_n)
    print(f"Top {top_n} feature importances – {label}:")
    print("-" * 40)
    for feat, score in importance_df.items():
        print(f"  {feat:<30} {score:.4f}")
    print()


# ─────────────────────────────────────────────
# 5. Export
# ─────────────────────────────────────────────
def save_model(model, filename: str):
    out = MODELS_DIR / filename
    joblib.dump(model, out)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Load & split
    X_train, X_test, y_train, y_test = load_and_split(PROCESSED_DIR / "train_ready.csv")

    # Feature sets
    X_tr_base, X_te_base, X_tr_ctx, X_te_ctx, base_cols, ctx_cols = get_feature_sets(X_train, X_test)

    # Train all four models
    baseline_model  = train(build_rf(),   X_tr_base, y_train, "Baseline RF")
    rf_ctx_model    = train(build_rf(),   X_tr_ctx,  y_train, "Contextual RF")
    lgbm_model      = train(build_lgbm(), X_tr_ctx,  y_train, "Contextual LGBM")
    xgb_model       = train(build_xgb(),  X_tr_ctx,  y_train, "Contextual XGBoost")

    # Evaluate all four models
    baseline_metrics = evaluate(baseline_model, X_te_base, y_test, "RF Baseline (no weather)")
    rf_ctx_metrics   = evaluate(rf_ctx_model,   X_te_ctx,  y_test, "RF Contextual")
    lgbm_metrics     = evaluate(lgbm_model,     X_te_ctx,  y_test, "LGBM Contextual")
    xgb_metrics      = evaluate(xgb_model,      X_te_ctx,  y_test, "XGB Contextual")
    print_comparison([baseline_metrics, rf_ctx_metrics, lgbm_metrics, xgb_metrics])

    # Feature importances for all contextual models (key thesis evidence)
    print_feature_importances(rf_ctx_model, ctx_cols, "RF Contextual",   top_n=5)
    print_feature_importances(lgbm_model,   ctx_cols, "LGBM Contextual", top_n=5)
    print_feature_importances(xgb_model,    ctx_cols, "XGB Contextual",  top_n=5)

    # Save all four models
    save_model(baseline_model, "baseline_rf.joblib")
    save_model(rf_ctx_model,   "contextual_rf.joblib")
    save_model(lgbm_model,     "lgbm_contextual.joblib")
    save_model(xgb_model,      "xgb_contextual.joblib")


if __name__ == "__main__":
    main()
