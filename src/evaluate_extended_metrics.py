#!/usr/bin/env python3
"""
Evaluate extended metrics and produce comparative charts/tables.

Outputs (saved to results/):
 - metrics_comparison.csv
 - metrics_barchart.png
 - pr_curve_comparison.png

Recreates the same stratified 80/20 split (random_state=42) used during training.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[0]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = {"event_id", "event_datetime", "event_name", "location"}
TARGET = "attended"


def load_data_and_split():
    """Load processed CSV and return stratified X_test, y_test (80/20 split)."""
    df = pd.read_csv(PROCESSED_DIR / "train_ready.csv")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return X_test, y_test


def align_features_for_model(model, X):
    """Ensure feature ordering matches model expectations (best-effort)."""
    if hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)
        return X[feat_names]
    if hasattr(model, "n_features_in_"):
        n = int(model.n_features_in_)
        return X.iloc[:, :n]
    return X


def compute_classification_metrics(y_true, y_pred):
    """Return dict with Accuracy, Precision, Recall, F1 (macro average)."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def plot_metrics_barchart(df_metrics: pd.DataFrame, out_path: Path):
    """Grouped bar chart comparing metrics between models."""
    metrics = df_metrics.index.tolist()
    baseline_vals = df_metrics["Baseline"].values
    contextual_vals = df_metrics["Contextual"].values

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, contextual_vals, width, label="Contextual", color="#dd8452")

    # Annotations
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison — Baseline vs Contextual")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(y_true, proba_dict, out_path: Path):
    """Plot Precision-Recall curves for each model and include AP in legend."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, probs in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Baseline vs Contextual")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # Load test split
    X_test, y_test = load_data_and_split()

    # Load models
    baseline = joblib.load(MODELS_DIR / "baseline_rf.joblib")
    contextual = joblib.load(MODELS_DIR / "contextual_rf.joblib")

    # Align features
    X_test_base = align_features_for_model(baseline, X_test)
    X_test_ctx = align_features_for_model(contextual, X_test)

    # Predictions and probabilities
    y_pred_base = baseline.predict(X_test_base)
    y_pred_ctx = contextual.predict(X_test_ctx)

    # Probabilities for positive class (1) – fallback to predictions if not available
    def get_pos_probs(model, X):
        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(X)
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                pos_idx = classes.index(1) if 1 in classes else 1
            else:
                pos_idx = 1
            return probs_all[:, pos_idx]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return model.predict(X)

    proba_base = get_pos_probs(baseline, X_test_base)
    proba_ctx = get_pos_probs(contextual, X_test_ctx)

    # Compute metrics
    metrics_base = compute_classification_metrics(y_test, y_pred_base)
    metrics_ctx = compute_classification_metrics(y_test, y_pred_ctx)

    # Build comparison DataFrame
    df_metrics = pd.DataFrame({"Baseline": metrics_base, "Contextual": metrics_ctx})
    # Reorder index for consistent presentation
    df_metrics = df_metrics.loc[["Accuracy", "Precision", "Recall", "F1-Score"]]

    # Export CSV
    metrics_csv = RESULTS_DIR / "metrics_comparison.csv"
    df_metrics.to_csv(metrics_csv)

    # Plot grouped bar chart
    barchart_path = RESULTS_DIR / "metrics_barchart.png"
    plot_metrics_barchart(df_metrics, barchart_path)

    # Plot Precision-Recall curves
    pr_path = RESULTS_DIR / "pr_curve_comparison.png"
    plot_pr_curves(y_test, {"Baseline": proba_base, "Contextual": proba_ctx}, pr_path)

    # Print success message with saved file locations
    print("Saved metrics comparison CSV:", metrics_csv)
    print("Saved metrics bar chart:", barchart_path)
    print("Saved precision-recall curve:", pr_path)


if __name__ == "__main__":
    main()

