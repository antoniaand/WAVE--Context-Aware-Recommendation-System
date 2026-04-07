#!/usr/bin/env python3
"""
Extended metrics and comparative charts for all four WAVE models.

Outputs (saved to results/):
 - metrics_comparison.csv   (Accuracy, Precision, Recall, F1 for all 4 models)
 - metrics_barchart.png     (grouped bar chart)
 - pr_curve_comparison.png  (Precision-Recall curves for all 4 models)

Recreates the same stratified 80/20 split (random_state=42) used during training.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]   # project root (one level above src/)
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS    = {"event_id", "event_datetime", "event_name", "location"}
WEATHER_COLS = {"weather_temp_C", "weather_precip_mm"}
TARGET = "attended"

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    ("RF Baseline",      "baseline_rf.joblib",     False),
    ("RF Contextual",    "contextual_rf.joblib",    True),
    ("LGBM Contextual",  "lgbm_contextual.joblib",  True),
    ("XGB Contextual",   "xgb_contextual.joblib",   True),
]


def load_data_and_split():
    """Load processed CSV and return stratified X_test, y_test (80/20 split)."""
    df = pd.read_csv(PROCESSED_DIR / "train_ready.csv")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler=joblib.load(MODELS_DIR / "scaler.joblib")
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_test, y_test


def get_X_for_model(X_test: pd.DataFrame, uses_weather: bool) -> pd.DataFrame:
    if uses_weather:
        return X_test
    return X_test[[c for c in X_test.columns if c not in WEATHER_COLS]]


def get_pos_probs(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else 1
        return probs_all[:, pos_idx]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def plot_metrics_barchart(df_metrics: pd.DataFrame, out_path: Path):
    """Grouped bar chart — one bar group per metric, one bar per model."""
    metrics      = df_metrics.index.tolist()
    model_labels = df_metrics.columns.tolist()
    n_models     = len(model_labels)
    x            = np.arange(len(metrics))
    width        = 0.18
    colors       = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (col, color) in enumerate(zip(model_labels, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_metrics[col].values, width, label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison — All Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(pr_data: list, out_path: Path):
    """Precision-Recall curves for all models on one plot."""
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for (label, y_true, probs), color in zip(pr_data, colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — All Models")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    X_test, y_test = load_data_and_split()

    all_metrics = {}   # label -> metrics dict
    pr_data     = []   # (label, y_true, probs)

    for label, filename, uses_weather in MODEL_REGISTRY:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"WARNING: {filename} not found, skipping.")
            continue

        model  = joblib.load(model_path)
        X      = get_X_for_model(X_test, uses_weather)
        y_pred = model.predict(X)
        probs  = get_pos_probs(model, X)

        all_metrics[label] = compute_metrics(y_test, y_pred)
        pr_data.append((label, y_test, probs))

    # Metrics comparison DataFrame (metrics as rows, models as columns)
    df_metrics = pd.DataFrame(all_metrics).loc[["Accuracy", "Precision", "Recall", "F1-Score"]]

    # Export CSV
    metrics_csv = RESULTS_DIR / "metrics_comparison.csv"
    df_metrics.to_csv(metrics_csv)
    print("Saved metrics comparison CSV:", metrics_csv)

    # Bar chart
    barchart_path = RESULTS_DIR / "metrics_barchart.png"
    plot_metrics_barchart(df_metrics, barchart_path)
    print("Saved metrics bar chart:", barchart_path)

    # Precision-Recall curves
    pr_path = RESULTS_DIR / "pr_curve_comparison.png"
    plot_pr_curves(pr_data, pr_path)
    print("Saved precision-recall curve:", pr_path)


if __name__ == "__main__":
    main()
