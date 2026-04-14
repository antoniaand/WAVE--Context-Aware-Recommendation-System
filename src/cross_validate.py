#!/usr/bin/env python3
"""
 – 5-Fold Stratified Cross-Validation pentru detectia overfitting-ului.

Ce face:
  - Incarca train_ready.csv si aplica acelasi split 80/20 ca in train_models.py
  - Ruleaza StratifiedKFold(5) pe X_train (NU pe tot datasetul) pentru fiecare model
  - Raporteaza CV F1 mean +/- std si Test F1 (din metrics_comparison.csv)
  - Calculeaza delta = |CV F1 mean - Test F1|:
      delta < 0.02  -> model stabil, fara overfitting
      delta 0.02-0.05 -> risc scazut
      delta > 0.05  -> posibil overfitting
  - Exporta results/cv_results.csv
  - Genereaza results/cv_comparison.png (boxplot CV per model)

Nota tehnica:
  StandardScaler este aplicat INAUNTRUL fiecarui fold (pe X_fold_train),
  nu o data pe tot X_train. Aceasta previne data leakage in CV.

Usage:
    python src/cross_validate.py
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
TRAIN_PATH   = ROOT / "data" / "processed" / "train_ready_interactions.csv"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS    = ["user_id", "event_id"]
WEATHER_COLS = ["weather_temp_C", "weather_precip_mm"]
TARGET       = "attended"


# ─── Model definitions (same hyperparams as train_models.py) ─────────────────
def make_pipeline(estimator) -> Pipeline:
    """
    Wrap an estimator in a Pipeline with StandardScaler.
    Scaling inside the pipeline ensures no data leakage across CV folds:
    the scaler is fit only on each fold's training portion.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])


MODELS = {
    "RF Baseline": make_pipeline(
        RandomForestClassifier(
            n_estimators=100, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=-1,
        )
    ),
    "RF Contextual": make_pipeline(
        RandomForestClassifier(
            n_estimators=100, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=-1,
        )
    ),
    "LGBM Contextual": make_pipeline(
        LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, max_depth=-1,
            min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
        )
    ),
    "XGB Contextual": make_pipeline(
        XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    ),
}


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_train_split() -> tuple:
    """
    Load interaction CSV and recreate the same GroupShuffleSplit as train_models.py.
    Returns X_train (unscaled), groups_train, y_train.
    Scaling is handled inside the CV pipeline per fold.
    """
    df = pd.read_csv(TRAIN_PATH)
    groups = df["user_id"]
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups=groups))
    return (
        X.iloc[train_idx].reset_index(drop=True),
        groups.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
    )


# ─── Cross-validation ────────────────────────────────────────────────────────
def run_cv(X_train: pd.DataFrame, groups_train: pd.Series,
           y_train: pd.Series, n_splits: int = 5) -> dict[str, list[float]]:
    """
    Run GroupKFold CV on X_train (grouped by user_id) for all 4 models.
    Each fold ensures no user appears in both train and val.
    Returns dict: model_name -> list of per-fold F1 scores (macro).
    """
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores: dict[str, list[float]] = {name: [] for name in MODELS}

    baseline_cols = [c for c in X_train.columns if c not in WEATHER_COLS]

    total_folds = len(MODELS) * n_splits
    done = 0

    for fold_i, (tr_idx, val_idx) in enumerate(
            gkf.split(X_train, y_train, groups=groups_train), 1):
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        for name, pipeline in MODELS.items():
            if name == "RF Baseline":
                X_tr  = X_train[baseline_cols].iloc[tr_idx]
                X_val = X_train[baseline_cols].iloc[val_idx]
            else:
                X_tr  = X_train.iloc[tr_idx]
                X_val = X_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            score  = f1_score(y_val, y_pred, average="macro", zero_division=0)
            cv_scores[name].append(score)

            done += 1
            print(f"  [{done:>2}/{total_folds}] {name} | fold {fold_i} | F1={score:.4f}")

    return cv_scores


# ─── Test F1 loader ──────────────────────────────────────────────────────────
def load_test_f1() -> dict[str, float]:
    """Read the test-set F1 scores saved by evaluate_extended_metrics.py."""
    csv_path = RESULTS_DIR / "metrics_comparison.csv"
    if not csv_path.exists():
        print("  [WARN] metrics_comparison.csv not found. Test F1 set to None.")
        return {}
    df = pd.read_csv(csv_path, index_col=0)
    f1_row = df.loc["F1-Score"] if "F1-Score" in df.index else None
    if f1_row is None:
        return {}
    mapping = {
        "RF Baseline":     f1_row.get("RF Baseline"),
        "RF Contextual":   f1_row.get("RF Contextual"),
        "LGBM Contextual": f1_row.get("LGBM Contextual"),
        "XGB Contextual":  f1_row.get("XGB Contextual"),
    }
    return {k: float(v) for k, v in mapping.items() if v is not None}


# ─── Export & plot ────────────────────────────────────────────────────────────
def build_results_df(cv_scores: dict, test_f1: dict) -> pd.DataFrame:
    rows = []
    for name, scores in cv_scores.items():
        arr       = np.array(scores)
        cv_mean   = arr.mean()
        cv_std    = arr.std()
        t_f1      = test_f1.get(name)
        delta     = abs(cv_mean - t_f1) if t_f1 is not None else None
        if delta is not None:
            status = "No overfitting" if delta < 0.02 else ("Low risk" if delta < 0.05 else "Possible overfitting")
        else:
            status = "N/A"
        rows.append({
            "Model":        name,
            "CV F1 Mean":   round(cv_mean, 4),
            "CV F1 Std":    round(cv_std,  4),
            "Test F1":      round(t_f1, 4) if t_f1 else None,
            "Delta":        round(delta, 4) if delta is not None else None,
            "Status":       status,
            "Fold Scores":  [round(s, 4) for s in scores],
        })
    return pd.DataFrame(rows)


def plot_cv_boxplot(cv_scores: dict, out_path: Path) -> None:
    """Boxplot showing the F1 distribution across 5 folds per model."""
    names  = list(cv_scores.keys())
    scores = [cv_scores[n] for n in names]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(scores, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual fold points
    for i, (name, sc) in enumerate(zip(names, scores), 1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(sc))
        ax.scatter([i + j for j in jitter], sc, color="black", s=30, zorder=5, alpha=0.8)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("F1-Score (macro)", fontsize=11)
    ax.set_title("5-Fold Stratified CV – F1 Distribution per Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0.55, 0.80)
    ax.axhline(y=0.67, color="gray", linestyle="--", linewidth=1, label="Test F1 ref (~0.67)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved CV boxplot: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("WAVE – 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    print("\n[1] Loading data and recreating GroupShuffleSplit (user_id)...")
    X_train, groups_train, y_train = load_train_split()
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Features: {X_train.columns.tolist()}")

    print(f"\n[2] Running 5-fold GroupKFold CV on X_train ({len(X_train):,} rows)...")
    print("  (Grouped by user_id — no user in both train and val per fold)")
    print("  (StandardScaler applied inside each fold – no leakage)\n")
    cv_scores = run_cv(X_train, groups_train, y_train, n_splits=5)

    print("\n[3] Loading test-set F1 from results/metrics_comparison.csv...")
    test_f1 = load_test_f1()

    print("\n[4] Results summary:")
    results_df = build_results_df(cv_scores, test_f1)

    print()
    print(f"  {'Model':<20} {'CV Mean':>8} {'CV Std':>8} {'Test F1':>8} {'Delta':>7} {'Status'}")
    print("  " + "-" * 72)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<20} {row['CV F1 Mean']:>8.4f} {row['CV F1 Std']:>8.4f} "
              f"{str(row['Test F1']):>8} {str(row['Delta']):>7}  {row['Status']}")

    print("\n  Interpretation:")
    print("  delta < 0.02  -> No overfitting (model generalises well)")
    print("  delta 0.02-0.05 -> Low risk")
    print("  delta > 0.05  -> Possible overfitting")

    # Export CSV (without fold scores column for cleanliness)
    export_df = results_df.drop(columns=["Fold Scores"])
    out_csv = RESULTS_DIR / "cv_results.csv"
    export_df.to_csv(out_csv, index=False)
    print(f"\n[5] Saved: {out_csv}")

    # Plot
    out_png = RESULTS_DIR / "cv_comparison.png"
    plot_cv_boxplot(cv_scores, out_png)

    print("\n" + "=" * 60)
    print("Cross-validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
