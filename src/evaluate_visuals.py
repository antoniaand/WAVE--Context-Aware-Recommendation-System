#!/usr/bin/env python3
"""
Evaluate and produce visuals for Baseline vs Contextual models.

Outputs saved to project_root/results/:
 - confusion_matrix_comparison.png
 - roc_comparison.png
 - feature_importances.csv

This script recreates the exact 80/20 stratified split (random_state=42)
so the test set matches training-time evaluation.
"""
from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns that were considered non-predictive / raw text in the pipeline
DROP_COLS = {"event_id", "event_datetime", "event_name", "location"}
TARGET = "attended"


def load_data():
    """Load processed CSV and recreate the same train/test split used for training."""
    df = pd.read_csv(PROCESSED_DIR / "train_ready.csv")
    # Drop non-predictive raw columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Separate X/y and perform stratified split (same random_state=42)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def align_features_for_model(model, X):
    """
    Ensure feature ordering matches what the model expects.
    If the model exposes feature_names_in_, use it; otherwise fall back to
    taking the first n_features_in_ columns from X (best-effort).
    """
    if hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)
        return X[feat_names]
    if hasattr(model, "n_features_in_"):
        n = int(model.n_features_in_)
        return X.iloc[:, :n]
    # Last resort: return X as-is
    return X


def plot_confusion_matrices(y_true, preds_dict, out_path: Path):
    """
    preds_dict: {"label": y_pred, ...}
    Saves a side-by-side heatmap comparison.
    """
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        # Use matplotlib to draw heatmap with annotations (no seaborn dependency)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix — {label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        # annotate cells with counts
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(y_true, proba_dict, out_path: Path):
    """
    proba_dict: {"label": y_proba_for_positive_class, ...}
    Plots ROC for each model on the same axes and includes AUC in legend.
    """
    plt.figure(figsize=(8, 6))
    for label, probs in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Baseline vs Contextual")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_feature_importances(model, feature_names, out_csv: Path):
    """Save contextual model feature importances to CSV (sorted desc)."""
    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        print("Model has no feature_importances_. Skipping export.")
        return
    df = pd.DataFrame({"feature": feature_names, "importance": fi})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved feature importances: {out_csv}")


def main():
    # Recreate split
    X_train, X_test, y_train, y_test = load_data()

    # Load models
    baseline_path = MODELS_DIR / "baseline_rf.joblib"
    contextual_path = MODELS_DIR / "contextual_rf.joblib"
    baseline = joblib.load(baseline_path)
    contextual = joblib.load(contextual_path)

    # Align features to what each model expects
    X_test_base = align_features_for_model(baseline, X_test)
    X_test_ctx = align_features_for_model(contextual, X_test)

    # Predictions
    y_pred_base = baseline.predict(X_test_base)
    y_pred_ctx = contextual.predict(X_test_ctx)

    # Confusion matrices (side-by-side)
    cm_out = RESULTS_DIR / "confusion_matrix_comparison.png"
    plot_confusion_matrices(y_test, {"Baseline": y_pred_base, "Contextual": y_pred_ctx}, cm_out)
    print(f"Saved confusion matrices: {cm_out}")

    # ROC curves: need probability for positive class (1)
    proba_dict = {}
    for label, model_obj, X in [("Baseline", baseline, X_test_base), ("Contextual", contextual, X_test_ctx)]:
        probs = None
        if hasattr(model_obj, "predict_proba"):
            probs_all = model_obj.predict_proba(X)
            # column ordering: classes_ -> find index of class '1'
            if hasattr(model_obj, "classes_"):
                classes = list(model_obj.classes_)
                pos_idx = classes.index(1) if 1 in classes else 1
            else:
                pos_idx = 1
            probs = probs_all[:, pos_idx]
        elif hasattr(model_obj, "decision_function"):
            probs = model_obj.decision_function(X)
            # decision_function can be used directly for roc_curve
        else:
            # fallback: use predictions (produces poor ROC but keeps pipeline working)
            probs = model_obj.predict(X)
        proba_dict[label] = probs

    roc_out = RESULTS_DIR / "roc_comparison.png"
    plot_roc_curves(y_test, proba_dict, roc_out)
    print(f"Saved ROC comparison: {roc_out}")

    # Export contextual feature importances
    # Determine feature names used by contextual model
    if hasattr(contextual, "feature_names_in_"):
        feat_names = list(contextual.feature_names_in_)
    elif hasattr(contextual, "n_features_in_"):
        feat_names = list(X_test.columns[: int(contextual.n_features_in_)])
    else:
        feat_names = list(X_test.columns)

    export_feature_importances(contextual, feat_names, RESULTS_DIR / "feature_importances.csv")


if __name__ == "__main__":
    main()

