#!/usr/bin/env python3
"""
Extended metrics and comparative charts for all four WAVE models.

Outputs (saved to results/):
 - metrics_comparison.csv           — global test metrics
 - metrics_barchart.png             — grouped bar chart (global)
 - pr_curve_comparison.png          — PR curves (global)
 - metrics_subgroup_extreme_weather.csv — subgroup: outdoor + (cold or wet)
 - metrics_barchart_extreme_weather.png — grouped bar for subgroup

Subgroup definition (physical units, pre-scaler):
  is_outdoor == 1  AND  (weather_temp_C < 5  OR  weather_precip_mm > 0.5)

Relative Error Reduction (RF Baseline vs XGB Contextual) is printed for
global test set and for the subgroup (accuracy-based and F1-based).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from eval_common import (
    MODEL_REGISTRY,
    extreme_weather_slice_mask,
    get_pos_probs,
    get_X_for_model,
    load_scaled_test_split,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "F1-Score": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
    }


def plot_metrics_barchart(df_metrics: pd.DataFrame, out_path: Path, title: str):
    metrics = df_metrics.index.tolist()
    model_labels = df_metrics.columns.tolist()
    n_models = len(model_labels)
    x = np.arange(len(metrics))
    width = 0.18
    base_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    colors = (base_colors * ((n_models // len(base_colors)) + 1))[:n_models]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (col, color) in enumerate(zip(model_labels, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_metrics[col].values, width, label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(pr_data: list, out_path: Path):
    base_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    colors = (base_colors * ((len(pr_data) // len(base_colors)) + 1))[: len(pr_data)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for (label, y_true, probs), color in zip(pr_data, colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — All Models (global test)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def relative_error_reduction(score_base: float, score_ctx: float, name: str = "score"):
    """
    RER using error = 1 - score (higher score = better).
    RER = (err_base - err_ctx) / err_base * 100  if err_base > 0
    """
    err_b = 1.0 - float(score_base)
    err_c = 1.0 - float(score_ctx)
    if err_b <= 1e-12:
        return float("nan"), err_b, err_c
    return (err_b - err_c) / err_b * 100.0, err_b, err_c


def print_rer_block(
    label: str,
    m_base: dict,
    m_xgb: dict,
):
    print(f"\n--- Relative Error Reduction: RF Baseline vs XGB Contextual ({label}) ---")
    for key in ("Accuracy", "F1-Score"):
        sb, sx = m_base[key], m_xgb[key]
        rer, eb, ec = relative_error_reduction(sb, sx, key)
        print(f"  {key}: baseline={sb:.4f}  xgb={sx:.4f}  "
              f"err: {1-sb:.4f} -> {1-sx:.4f}  RER={rer:.1f}%")


def run_models_collect_metrics(X_test, slice_mask, y_test) -> dict:
    """Predict on full X_test; metrics computed on y_test[slice_mask] for subgroup."""
    out = {}
    y_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
    for label, filename, uses_weather, extra_drop in MODEL_REGISTRY:
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        model = joblib.load(path)
        X = get_X_for_model(X_test, uses_weather, extra_drop)
        y_pred = model.predict(X)
        if slice_mask is None:
            out[label] = compute_metrics(y_test, y_pred)
        else:
            m = np.asarray(slice_mask, dtype=bool)
            out[label] = compute_metrics(y_np[m], y_pred[m])
    return out


def main():
    X_test, y_test, X_test_raw = load_scaled_test_split()
    slice_mask = extreme_weather_slice_mask(X_test_raw)
    n_sub = int(slice_mask.sum())
    n_all = len(y_test)
    print(f"Global test rows: {n_all:,}")
    print(
        f"Extreme-weather slice (outdoor & (T<5C or precip>0.5mm)): {n_sub:,} "
        f"({100*n_sub/n_all:.1f}% of test)\n"
    )

    # ----- Global metrics -----
    all_metrics = run_models_collect_metrics(X_test, None, y_test)
    pr_data = []
    for label, filename, uses_weather, extra_drop in MODEL_REGISTRY:
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"WARNING: {filename} missing, skip.")
            continue
        model = joblib.load(path)
        X = get_X_for_model(X_test, uses_weather, extra_drop)
        probs = get_pos_probs(model, X)
        pr_data.append((label, y_test, probs))

    df_metrics = pd.DataFrame(all_metrics).loc[
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    ]
    metrics_csv = RESULTS_DIR / "metrics_comparison.csv"
    df_metrics.to_csv(metrics_csv)
    print("Saved:", metrics_csv)

    plot_metrics_barchart(
        df_metrics,
        RESULTS_DIR / "metrics_barchart.png",
        "Metrics Comparison — All Models (global test)",
    )
    print("Saved:", RESULTS_DIR / "metrics_barchart.png")

    plot_pr_curves(pr_data, RESULTS_DIR / "pr_curve_comparison.png")
    print("Saved:", RESULTS_DIR / "pr_curve_comparison.png")

    # ----- Subgroup metrics -----
    if n_sub == 0:
        print("WARNING: empty extreme-weather slice; skip subgroup outputs.")
    else:
        sub_metrics = run_models_collect_metrics(X_test, slice_mask, y_test)
        df_sub = pd.DataFrame(sub_metrics).loc[
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        ]
        sub_csv = RESULTS_DIR / "metrics_subgroup_extreme_weather.csv"
        df_sub.to_csv(sub_csv)
        print("\n--- Extreme-weather slice: metrics ---")
        print(df_sub.to_string())
        print("\nSaved:", sub_csv)

        plot_metrics_barchart(
            df_sub,
            RESULTS_DIR / "metrics_barchart_extreme_weather.png",
            f"Metrics — Extreme-weather slice (n={n_sub})",
        )
        print("Saved:", RESULTS_DIR / "metrics_barchart_extreme_weather.png")

    # ----- Relative error reduction -----
    if "RF Baseline" in all_metrics and "XGB Contextual" in all_metrics:
        print_rer_block("global test", all_metrics["RF Baseline"], all_metrics["XGB Contextual"])
        if n_sub > 0:
            print_rer_block(
                "extreme-weather slice",
                sub_metrics["RF Baseline"],
                sub_metrics["XGB Contextual"],
            )


if __name__ == "__main__":
    main()
