#!/usr/bin/env python3
"""
Evaluate and produce visuals for all four WAVE models.

Outputs (project_root/results/):
 - confusion_matrix_comparison.png     — 2x2 confusion matrices (global test)
 - roc_comparison.png                  — ROC curves (global test)
 - feature_importances.csv             — SKLearn split importances (contextual)
 - f1_extreme_weather_slice.png        — F1 bar chart, outdoor cold/wet slice
 - xgb_subgroup_permutation_importance.png — XGB on slice (permutation importance)

Uses the same preprocessing and GroupShuffleSplit as train_models.py (via eval_common).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from src.evaluation.eval_common import (
    MODEL_REGISTRY,
    WEATHER_COLS_SET,
    extreme_weather_slice_mask,
    get_pos_probs,
    get_X_for_model,
    load_scaled_test_split,
)

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_FEATURE_LABELS = {
    "weather_temp_C": "Temp (C)",
    "weather_humidity": "Humidity (%)",
    "weather_precip_mm": "Precip (mm)",
    "weather_wind_speed_kmh": "Wind (km/h)",
}


def plot_confusion_matrices(models_data: list, y_true: pd.Series, out_path: Path, title: str):
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
                ax.text(
                    j,
                    i,
                    format(int(cm[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(roc_data: list, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    for label, y_true, probs in roc_data:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_extreme_weather(f1_by_model: dict, n_slice: int, out_path: Path):
    labels = list(f1_by_model.keys())
    values = [f1_by_model[k] for k in labels]
    base_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    colors = (base_colors * ((len(labels) // len(base_colors)) + 1))[: len(labels)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, min(1.15, max(values) * 1.15 + 0.05))
    ax.set_ylabel("Macro F1-Score")
    ax.set_title(
        f"Extreme-weather slice (n={n_slice})\n"
        r"outdoor & (T$<$5°C or precip$>$0.5mm)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_xgb_subgroup_permutation(
    model,
    X_sub: pd.DataFrame,
    y_sub: np.ndarray,
    out_path: Path,
    title: str,
    n_repeats: int = 15,
):
    """
    Permutation importance on the subgroup — reflects which features drive
    XGB predictions on hard outdoor weather rows (including weather channels).
    """
    r = permutation_importance(
        model,
        X_sub,
        y_sub,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="f1_macro",
    )
    names = list(X_sub.columns)
    order = np.argsort(r.importances_mean)[::-1]
    top_k = min(18, len(order))
    order = order[:top_k]

    means = r.importances_mean[order]
    stds = r.importances_std[order]
    feat_names = [names[i] for i in order]

    def _lab(n):
        return WEATHER_FEATURE_LABELS.get(n, n)

    display = [_lab(n) for n in feat_names]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(display))
    colors = ["#c44e52" if n in WEATHER_COLS_SET else "#4c72b0" for n in feat_names]
    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in macro-F1 (permutation)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def export_feature_importances(models_data: list, feature_names_map: dict, out_csv: Path):
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
    X_test, y_test, X_test_raw = load_scaled_test_split()
    slice_mask = extreme_weather_slice_mask(X_test_raw)
    n_sub = int(slice_mask.sum())
    y_np = y_test.to_numpy()
    m = np.asarray(slice_mask, dtype=bool)

    cm_preds = []
    roc_data = []
    fi_models = []
    fi_names = {}
    f1_sub = {}

    for label, filename, uses_weather, extra_drop in MODEL_REGISTRY:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"WARNING: {filename} not found, skipping.")
            continue

        model = joblib.load(model_path)
        X = get_X_for_model(X_test, uses_weather, extra_drop)

        y_pred = model.predict(X)
        probs = get_pos_probs(model, X)

        cm_preds.append((label, y_pred))
        roc_data.append((label, y_test, probs))

        if n_sub > 0:
            f1_sub[label] = f1_score(
                y_np[m], y_pred[m], average="macro", zero_division=0
            )

        if uses_weather:
            fi_models.append((label, model))
            fi_names[label] = list(X.columns)

    cm_out = RESULTS_DIR / "confusion_matrix_comparison.png"
    plot_confusion_matrices(
        cm_preds,
        y_test,
        cm_out,
        "Confusion Matrix — All Models (global test)",
    )
    print(f"Saved: {cm_out}")

    roc_out = RESULTS_DIR / "roc_comparison.png"
    plot_roc_curves(roc_data, roc_out, "ROC Curve — All Models (global test)")
    print(f"Saved: {roc_out}")

    export_feature_importances(
        fi_models, fi_names, RESULTS_DIR / "feature_importances.csv"
    )

    # ----- Subgroup: F1 bar + XGB permutation -----
    if n_sub == 0:
        print("Extreme-weather slice empty; skip subgroup figures.")
        return

    f1_out = RESULTS_DIR / "f1_extreme_weather_slice.png"
    plot_f1_extreme_weather(f1_sub, n_sub, f1_out)
    print(f"Saved: {f1_out}")
    print("\n--- F1 (macro) on extreme-weather slice ---")
    for k, v in f1_sub.items():
        print(f"  {k:<18} {v:.4f}")

    xgb_path = MODELS_DIR / "xgb_contextual.joblib"
    if xgb_path.exists():
        xgb = joblib.load(xgb_path)
        X_xgb = get_X_for_model(X_test, True)
        X_sub = X_xgb.loc[m].reset_index(drop=True)
        y_sub = y_np[m]
        perm_out = RESULTS_DIR / "xgb_subgroup_permutation_importance.png"
        plot_xgb_subgroup_permutation(
            xgb,
            X_sub,
            y_sub,
            perm_out,
            title=f"XGB Contextual — permutation importance\n"
            f"extreme-weather slice (n={n_sub})",
        )
        print(f"Saved: {perm_out}")
    else:
        print("xgb_contextual.joblib missing; skip permutation plot.")


if __name__ == "__main__":
    main()
