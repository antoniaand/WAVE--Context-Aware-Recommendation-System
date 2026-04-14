#!/usr/bin/env python3
"""
Local hypothesis tests for small baseline-vs-contextual F1 gap.
Does NOT modify build_interaction_dataset.py or train_models.py.

Run: python src/hypothesis_probe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "processed" / "train_ready_interactions.csv"

WEATHER = ["weather_temp_C", "weather_precip_mm"]
USER_WEATHER_TRAITS = [
    "user_rain_avoid",
    "user_cold_tolerance",
    "user_heat_sensitivity",
    "user_wind_sensitivity",
    "user_override_weather",
]
DROP = ["user_id", "event_id"]
TARGET = "attended"
RNG = 42


def split_group(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RNG)
    tr, te = next(gss.split(X, y, groups=groups))
    return (
        X.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        y.iloc[te].reset_index(drop=True),
    )


def scale_fit_train(X_tr, X_te):
    sc = StandardScaler()
    X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns)
    X_te = pd.DataFrame(sc.transform(X_te), columns=X_te.columns)
    return X_tr, X_te


def macro_f1(m, X_te, y_te):
    pred = m.predict(X_te)
    return f1_score(y_te, pred, average="macro", zero_division=0)


def agreement(a, b):
    return float((np.array(a) == np.array(b)).mean())


def main():
    print("=" * 72)
    print("WAVE hypothesis_probe (local, no pipeline edits)")
    print("=" * 72)

    df = pd.read_csv(CSV)
    groups = df["user_id"]
    X = df.drop(columns=[c for c in DROP if c in df.columns] + [TARGET])
    y = df[TARGET]

    print("\n--- Data sanity ---")
    print(f"Rows: {len(df):,} | Users: {groups.nunique()} | Events: {df['event_id'].nunique()}")
    print(f"attended=1: {(y == 1).mean() * 100:.2f}%")
    print(f"is_outdoor=1: {df['is_outdoor'].mean() * 100:.2f}% of all rows")
    print(f"Of is_outdoor=1, attended=1: {df.loc[df['is_outdoor'] == 1, 'attended'].mean() * 100:.2f}%")
    print(f"Of is_outdoor=0, attended=1: {df.loc[df['is_outdoor'] == 0, 'attended'].mean() * 100:.2f}%")

    X_tr, X_te, y_tr, y_te = split_group(X, y, groups)
    # Keep raw binary mask BEFORE StandardScaler (scaling breaks is_outdoor == 1)
    outdoor_te_raw = X_te["is_outdoor"].values
    X_tr, X_te = scale_fit_train(X_tr, X_te)

    all_cols = list(X_tr.columns)
    base_cols = [c for c in all_cols if c not in WEATHER]
    lean_cols = [c for c in base_cols if c not in USER_WEATHER_TRAITS]

    # --- Experiment 1: redundancy (user weather traits in baseline) ---
    print("\n--- Exp1: Feature sets (GroupShuffleSplit, same RF hyperparams as train_models) ---")
    rf_full_base = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rf_lean_base = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rf_ctx = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )

    rf_full_base.fit(X_tr[base_cols], y_tr)
    rf_lean_base.fit(X_tr[lean_cols], y_tr)
    rf_ctx.fit(X_tr, y_tr)

    f1_fb = macro_f1(rf_full_base, X_te[base_cols], y_te)
    f1_lb = macro_f1(rf_lean_base, X_te[lean_cols], y_te)
    f1_c = macro_f1(rf_ctx, X_te, y_te)

    p_fb = rf_full_base.predict(X_te[base_cols])
    p_lb = rf_lean_base.predict(X_te[lean_cols])
    p_c = rf_ctx.predict(X_te)

    print(f"RF baseline (event+user INCLUDING weather traits, NO weather cols): macro F1 = {f1_fb:.4f}")
    print(f"RF baseline (LEAN: drop user rain/cold/heat/wind/override):     macro F1 = {f1_lb:.4f}")
    print(f"RF contextual (all features incl. weather):                    macro F1 = {f1_c:.4f}")
    print(
        f"Delta contextual - full baseline (pp): {(f1_c - f1_fb) * 100:+.2f} | "
        f"agreement full_base vs ctx: {agreement(p_fb, p_c):.4f}"
    )
    print(
        f"Delta contextual - LEAN baseline (pp): {(f1_c - f1_lb) * 100:+.2f} | "
        f"agreement lean_base vs ctx: {agreement(p_lb, p_c):.4f}"
    )

    # --- Experiment 2: outdoor-only test subset ---
    print("\n--- Exp2: Test rows where is_outdoor == 1 only ---")
    outdoor_mask = outdoor_te_raw == 1
    n_out = int(outdoor_mask.sum())
    print(f"Outdoor test rows: {n_out} / {len(X_te)} ({100 * n_out / len(X_te):.1f}%)")
    if n_out > 200:
        f1_fb_o = f1_score(y_te[outdoor_mask], p_fb[outdoor_mask], average="macro", zero_division=0)
        f1_c_o = f1_score(y_te[outdoor_mask], p_c[outdoor_mask], average="macro", zero_division=0)
        print(f"macro F1 full baseline (outdoor test): {f1_fb_o:.4f}")
        print(f"macro F1 contextual    (outdoor test): {f1_c_o:.4f}")
        print(f"Delta (pp) outdoor-only: {(f1_c_o - f1_fb_o) * 100:+.2f}")

    # --- Experiment 3: naive majority baseline ---
    maj = int(y_tr.mode().iloc[0])
    pred_maj = np.full(len(y_te), maj)
    f1_maj = f1_score(y_te, pred_maj, average="macro", zero_division=0)
    print("\n--- Exp3: Always predict majority class from train ---")
    print(f"macro F1 = {f1_maj:.4f} (ceiling for 'no skill' on macro with imbalance)")

    # --- Experiment 4: LogReg with explicit interactions (weather * traits) ---
    print("\n--- Exp4: LogisticRegression on engineered interactions (same group split) ---")
    Xe_tr = X_tr.copy()
    Xe_te = X_te.copy()
    for ucol in ["user_rain_avoid", "user_cold_tolerance", "user_heat_sensitivity"]:
        Xe_tr[f"{ucol}_x_temp"] = Xe_tr[ucol] * Xe_tr["weather_temp_C"]
        Xe_te[f"{ucol}_x_temp"] = Xe_te[ucol] * Xe_te["weather_temp_C"]
        Xe_tr[f"{ucol}_x_precip"] = Xe_tr[ucol] * Xe_tr["weather_precip_mm"]
        Xe_te[f"{ucol}_x_precip"] = Xe_te[ucol] * Xe_te["weather_precip_mm"]

    lr_lean = LogisticRegression(max_iter=2000, random_state=RNG)
    lr_full = LogisticRegression(max_iter=2000, random_state=RNG)
    lr_lean.fit(Xe_tr[lean_cols], y_tr)
    # contextual uses all original + interaction cols
    int_cols = [c for c in Xe_tr.columns if "_x_" in c]
    ctx_lr_cols = lean_cols + WEATHER + int_cols
    lr_full.fit(Xe_tr[ctx_lr_cols], y_tr)

    p_lr_l = lr_lean.predict(Xe_te[lean_cols])
    p_lr_f = lr_full.predict(Xe_te[ctx_lr_cols])
    f1_lr_l = f1_score(y_te, p_lr_l, average="macro", zero_division=0)
    f1_lr_f = f1_score(y_te, p_lr_f, average="macro", zero_division=0)
    try:
        proba_f = lr_full.predict_proba(Xe_te[ctx_lr_cols])[:, 1]
        auc_f = roc_auc_score(y_te, proba_f)
    except Exception:
        auc_f = float("nan")
    print(f"LR lean (no user traits, no weather):     macro F1 = {f1_lr_l:.4f}")
    print(f"LR contextual lean+weather+interactions: macro F1 = {f1_lr_f:.4f} | AUC = {auc_f:.4f}")
    print(f"Delta LR contextual - LR lean (pp): {(f1_lr_f - f1_lr_l) * 100:+.2f}")

    # --- Experiment 5: row-level random split (leaky) reminder ---
    print("\n--- Exp5: Stratified random split (LEAKY: same user in train+test) ---")
    Xr_tr, Xte_r, yr_tr, yr_te = train_test_split(X, y, test_size=0.2, random_state=RNG, stratify=y)
    scr = StandardScaler()
    Xr_trs = pd.DataFrame(scr.fit_transform(Xr_tr), columns=Xr_tr.columns)
    Xte_rs = pd.DataFrame(scr.transform(Xte_r), columns=Xte_r.columns)
    rfb = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rfc = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rfb.fit(Xr_trs[base_cols], yr_tr)
    rfc.fit(Xr_trs, yr_tr)
    f1b = f1_score(yr_te, rfb.predict(Xte_rs[base_cols]), average="macro", zero_division=0)
    f1c2 = f1_score(yr_te, rfc.predict(Xte_rs), average="macro", zero_division=0)
    print(f"RF macro F1 baseline vs contextual (random stratified): {f1b:.4f} vs {f1c2:.4f} | delta pp: {(f1c2 - f1b) * 100:+.2f}")

    # --- Summary diagnosis ---
    print("\n" + "=" * 72)
    print("DIAGNOSIS (read this block for thesis / next approval step)")
    print("=" * 72)
    print(
        "Problem type: TASK DESIGN + FEATURE REDUNDANCY, not a broken sklearn pipeline.\n"
        "  - Baseline already sees USER weather-tolerance columns that ENTER the label\n"
        "    formula multiplicatively with weather. RF can approximate weather-driven\n"
        "    decisions from (traits × implicit row structure) without reading temp/precip.\n"
        "  - Exp1 'LEAN baseline' (drop traits) is the fairer ablation; compare its delta\n"
        "    to contextual to isolate 'value of weather columns'.\n"
        "  - Outdoor share is small; weather penalty is weak for indoor rows (outdoor_mult=0.4),\n"
        "    so population-level macro F1 dilutes any weather signal.\n"
        "  - Boosters ~= RF because Bayes decision boundaries are already well-captured / noise-limited.\n"
    )
    print("Likely resolutions (require pipeline / thesis decisions — do not auto-apply here):")
    print("  1) Redefine baseline: exclude user weather-trait columns from baseline only.")
    print("  2) Strengthen label weather sensitivity for outdoor (formula) +/or balance classes.")
    print("  3) Report subset metrics (is_outdoor==1) alongside global macro F1.")
    print("  4) If you need LGBM > RF: tune/boost capacity OR use data with sharper interaction signal.")
    print("=" * 72)


if __name__ == "__main__":
    main()
