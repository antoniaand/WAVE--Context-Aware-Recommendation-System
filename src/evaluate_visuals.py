#!/usr/bin/env python3
"""
Evaluate and produce visuals for all four WAVE models.

Outputs saved to project_root/results/:
 - confusion_matrix_comparison.png  (2x2 grid, one panel per model)
 - roc_comparison.png               (all 4 ROC curves on one plot)
 - feature_importances.csv          (importances for all 3 contextual models)

Recreates the exact 80/20 stratified split (random_state=42) so the test
set is identical to the one used during training.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS   = {"event_id", "event_datetime", "event_name", "location"}
WEATHER_COLS = {"weather_temp_C", "weather_precip_mm"}
TARGET = "attended"

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (display label, joblib filename, uses_weather)
MODEL_REGISTRY = [
    ("RF Baseline",      "baseline_rf.joblib",     False),
    ("RF Contextual",    "contextual_rf.joblib",    True),
    ("LGBM Contextual",  "lgbm_contextual.joblib",  True),
    ("XGB Contextual",   "xgb_contextual.joblib",   True),
]


def load_data():
    """Recreate the same stratified 80/20 split used during training."""
    df = pd.read_csv(PROCESSED_DIR / "train_ready.csv")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_test, y_test


def get_X_for_model(X_test: pd.DataFrame, uses_weather: bool) -> pd.DataFrame:
    """Return the correct feature subset depending on whether the model uses weather."""
    if uses_weather:
        return X_test
    return X_test[[c for c in X_test.columns if c not in WEATHER_COLS]]


def get_pos_probs(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities for class 1."""
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else 1
        return probs_all[:, pos_idx]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def plot_confusion_matrices(models_data: list, y_true: pd.Series, out_path: Path):
    """
    2×2 grid of confusion matrices, one per model.
    models_data: list of (label, y_pred)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (label, y_pred) in zip(axes, models_data):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not attended (0)", "Attended (1)"], fontsize=8)
        ax.set_yticklabels(["Not attended (0)", "Attended (1)"], fontsize=8)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrix Comparison — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(roc_data: list, out_path: Path):
    """
    All 4 ROC curves on one plot.
    roc_data: list of (label, y_true, y_probs)
    """
    plt.figure(figsize=(8, 6))
    for label, y_true, probs in roc_data:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_feature_importances(models_data: list, feature_names_map: dict, out_csv: Path):
    """
    Export feature importances for all contextual models into one CSV.
    models_data: list of (label, model)
    feature_names_map: {label: [feature_names]}
    """
    rows = []
    for label, model in models_data:
        fi = getattr(model, "feature_importances_", None)
        if fi is None:
            continue
        names = feature_names_map.get(label, [])
        for feat, imp in zip(names, fi):
            rows.append({"model": label, "feature": feat, "importance": imp})

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["model", "importance"], ascending=[True, False])
        df.to_csv(out_csv, index=False)
        print(f"Saved feature importances: {out_csv}")


def main():
    X_test, y_test = load_data()

    cm_preds   = []  # (label, y_pred)
    roc_data   = []  # (label, y_true, probs)
    fi_models  = []  # (label, model) — contextual only
    fi_names   = {}  # label -> feature name list

    for label, filename, uses_weather in MODEL_REGISTRY:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"WARNING: {filename} not found, skipping.")
            continue

        model = joblib.load(model_path)
        X = get_X_for_model(X_test, uses_weather)

        y_pred = model.predict(X)
        probs  = get_pos_probs(model, X)

        cm_preds.append((label, y_pred))
        roc_data.append((label, y_test, probs))

        if uses_weather:
            fi_models.append((label, model))
            fi_names[label] = list(X.columns)

    # Confusion matrices — 2x2 grid
    cm_out = RESULTS_DIR / "confusion_matrix_comparison.png"
    plot_confusion_matrices(cm_preds, y_test, cm_out)
    print(f"Saved confusion matrices: {cm_out}")

    # ROC curves — all 4 on one plot
    roc_out = RESULTS_DIR / "roc_comparison.png"
    plot_roc_curves(roc_data, roc_out)
    print(f"Saved ROC comparison: {roc_out}")

    # Feature importances — all contextual models
    export_feature_importances(fi_models, fi_names, RESULTS_DIR / "feature_importances.csv")


if __name__ == "__main__":
    main()
