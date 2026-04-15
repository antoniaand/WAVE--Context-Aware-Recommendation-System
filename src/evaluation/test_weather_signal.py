"""
test_weather_signal.py
----------------------
Isolated experiment to prove that adding real weather data + user weather
sensitivity features produces a measurable improvement over a model that
is completely blind to weather.

Two model configurations — same algorithm (XGBoost), same split, same seed:

  BLIND BASELINE
    Features: event type, location, scenario scores, demographics, event_month,
              is_outdoor, event_in_preferred.
    No weather columns. No weather-preference columns.
    This simulates a recommender that knows what the user likes but has no
    idea about the weather.

  FULL CONTEXTUAL
    Features: everything in BLIND + real weather data (temp, precip, wind)
              + user weather-sensitivity profile (rain_avoid, cold_tolerance,
              heat_sensitivity, wind_sensitivity, override_weather).

The delta between FULL CONTEXTUAL and BLIND BASELINE is the pure causal
contribution of the weather signal.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "train_ready_interactions.csv"

TARGET = "attended"
SEED   = 42

# ── Feature groups ────────────────────────────────────────────────────────────

# What the user chose / prefers (no weather awareness)
USER_PREFERENCE_COLS = [
    "scenario_concert", "scenario_festival", "scenario_sports",
    "scenario_theatre", "scenario_conference",
    "top_event", "indoor_outdoor",
    "attendance_freq",
]

# Basic demographics
DEMOGRAPHIC_COLS = ["gender", "age_range"]

# Event context (no weather)
EVENT_COLS = [
    "event_type", "climate_zone", "is_outdoor", "location",
    "event_month",          # extracted from event_date
    "event_in_preferred",   # engineered: event_type in preferred_event_types
]

# Actual observed weather
WEATHER_DATA_COLS = [
    "weather_temp_C", "weather_precip_mm", "weather_wind_speed_kmh",
]

# User's personal weather tolerance / sensitivity
WEATHER_PREF_COLS = [
    "rain_avoid", "cold_tolerance", "heat_sensitivity",
    "wind_sensitivity", "override_weather",
]

BLIND_COLS    = DEMOGRAPHIC_COLS + USER_PREFERENCE_COLS + EVENT_COLS
FULL_COLS     = BLIND_COLS + WEATHER_DATA_COLS + WEATHER_PREF_COLS

CATEGORICAL   = ["gender", "age_range", "top_event", "event_type",
                 "location", "attendance_freq", "climate_zone"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_month"] = pd.to_datetime(df["event_date"]).dt.month
    df["event_in_preferred"] = df.apply(
        lambda r: int(str(r["event_type"]) in str(r["preferred_event_types"])), axis=1
    )
    le = LabelEncoder()
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_xgb():
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    result = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    if y_proba is not None:
        result["ROC-AUC"] = round(roc_auc_score(y_true, y_proba), 4)
    return result


def scale(X_tr, X_te):
    sc = StandardScaler()
    X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns)
    X_te_s  = pd.DataFrame(sc.transform(X_te),    columns=X_te.columns)
    return X_tr_s, X_te_s


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  WAVE — WEATHER SIGNAL ISOLATION EXPERIMENT")
    print("=" * 65)

    # ── Load & engineer ───────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df = engineer(df)

    groups = df["user_id"].values
    y      = df[TARGET]

    # ── Group split (same as train_models.py — no user leakage) ───────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    n_tr_users = len(np.unique(groups[train_idx]))
    n_te_users = len(np.unique(groups[test_idx]))
    print(f"\n  Split: {len(train_idx):,} train rows ({n_tr_users} users) / "
          f"{len(test_idx):,} test rows ({n_te_users} users)")
    print(f"  Class balance (test): "
          f"0={( y_test==0).sum():,}  1={(y_test==1).sum():,}\n")

    results = {}

    for label, feat_cols in [("BLIND BASELINE", BLIND_COLS),
                              ("FULL CONTEXTUAL", FULL_COLS)]:
        print(f"  {'-'*60}")
        print(f"  Running: {label}  ({len(feat_cols)} features)")
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns skipped: {missing}")
            feat_cols = [c for c in feat_cols if c in df.columns]

        X      = df[feat_cols]
        X_tr   = X.iloc[train_idx].reset_index(drop=True)
        X_te   = X.iloc[test_idx].reset_index(drop=True)
        X_tr, X_te = scale(X_tr, X_te)

        model = build_xgb()
        model.fit(X_tr, y_train)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        results[label] = {"model": model, "metrics": metrics,
                          "feat_cols": feat_cols, "y_pred": y_pred}

        print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1-Score  : {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}\n")

        print(f"  Per-class report ({label}):")
        print(classification_report(y_test, y_pred,
                                    target_names=["Not Attend", "Attend"],
                                    digits=4))

    # ── Delta table ───────────────────────────────────────────────────────────
    sep = "=" * 65
    print(sep)
    print("  RESULTS SUMMARY")
    print(sep)
    cols  = ["Configuration", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    col_w = [22, 11, 12, 9, 11, 10]
    hdr   = "".join(c.ljust(w) for c, w in zip(cols, col_w))
    print(hdr)
    print("-" * len(hdr))
    for name, data in results.items():
        m = data["metrics"]
        row = [name, m["Accuracy"], m["Precision"], m["Recall"], m["F1-Score"], m["ROC-AUC"]]
        print("".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print("-" * len(hdr))
    blind = results["BLIND BASELINE"]["metrics"]
    full  = results["FULL CONTEXTUAL"]["metrics"]
    delta = ["DELTA (+full-blind)"] + [
        f"{full[k] - blind[k]:+.4f}"
        for k in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ]
    print("".join(str(v).ljust(w) for v, w in zip(delta, col_w)))
    print(sep)

    # ── Feature importance — what drives the contextual model ─────────────────
    full_model     = results["FULL CONTEXTUAL"]["model"]
    full_feat_cols = results["FULL CONTEXTUAL"]["feat_cols"]
    imp = pd.Series(full_model.feature_importances_, index=full_feat_cols)
    imp = imp.sort_values(ascending=False)

    print("\n  Top 15 features — FULL CONTEXTUAL model:")
    print("  " + "-" * 50)
    for feat, score in imp.head(15).items():
        bar = "#" * int(score * 200)
        print(f"  {feat:<35} {score:.4f}  {bar}")

    # Highlight weather-related features specifically
    weather_all = WEATHER_DATA_COLS + WEATHER_PREF_COLS
    weather_imp = imp[imp.index.isin(weather_all)].sort_values(ascending=False)
    print(f"\n  Weather-related feature importances ({len(weather_imp)} features):")
    print("  " + "-" * 50)
    total_imp = imp.sum()
    weather_total = weather_imp.sum()
    for feat, score in weather_imp.items():
        print(f"  {feat:<35} {score:.4f}  ({score/total_imp*100:.1f}% of total)")
    print(f"\n  Weather features account for "
          f"{weather_total/total_imp*100:.1f}% of total model importance")
    print(sep + "\n")


if __name__ == "__main__":
    main()
