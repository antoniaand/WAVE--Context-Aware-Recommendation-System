#!/usr/bin/env python3
"""
Debug: why baseline vs contextual F1 gap is small on train_ready_interactions.csv.

Writes NDJSON to project_root/debug-e3238b.log (session e3238b).
Run: python src/debug_interaction_f1.py
"""
# region agent log helper
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "debug-e3238b.log"
SESSION_ID = "e3238b"


def agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix"):
    entry = {
        "sessionId": SESSION_ID,
        "timestamp": int(time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
# endregion

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

CSV_PATH = ROOT / "data" / "processed" / "train_ready_interactions.csv"
DROP = ["user_id", "event_id"]
WEATHER = ["weather_temp_C", "weather_precip_mm"]
TARGET = "attended"
RNG = 42


def main():
    # region agent log
    agent_log("INIT", "debug_interaction_f1.py:main", "starting interaction F1 debug", {"csv": str(CSV_PATH)}, "pre-fix")
    # endregion

    df = pd.read_csv(CSV_PATH)
    groups = df["user_id"]
    df_feat = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df_feat[TARGET]
    X_raw = df_feat.drop(columns=[TARGET])
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())

    # region agent log — hypothesis C (class balance)
    agent_log(
        "C",
        "debug_interaction_f1.py:class_dist",
        "target distribution",
        {"n_rows": len(df), "n0": n0, "n1": n1, "pct_0": round(100 * n0 / len(df), 2), "pct_1": round(100 * n1 / len(df), 2)},
        "pre-fix",
    )
    # endregion

    base_cols = [c for c in X_raw.columns if c not in WEATHER]
    ctx_cols = list(X_raw.columns)

    # --- Group split (same as train_models) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RNG)
    tr_g, te_g = next(gss.split(X_raw, y, groups=groups))
    X_tr_g = X_raw.iloc[tr_g].reset_index(drop=True)
    X_te_g = X_raw.iloc[te_g].reset_index(drop=True)
    y_tr_g = y.iloc[tr_g].reset_index(drop=True)
    y_te_g = y.iloc[te_g].reset_index(drop=True)
    groups_tr = groups.iloc[tr_g].reset_index(drop=True)

    sc_g = StandardScaler()
    X_tr_g_s = pd.DataFrame(sc_g.fit_transform(X_tr_g), columns=X_tr_g.columns)
    X_te_g_s = pd.DataFrame(sc_g.transform(X_te_g), columns=X_te_g.columns)

    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rf_ctx = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    lgb = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=20, random_state=RNG, n_jobs=-1, verbose=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=RNG, n_jobs=-1,
    )

    rf_base.fit(X_tr_g_s[base_cols], y_tr_g)
    rf_ctx.fit(X_tr_g_s, y_tr_g)
    lgb.fit(X_tr_g_s, y_tr_g)
    xgb.fit(X_tr_g_s, y_tr_g)

    pb = rf_base.predict(X_te_g_s[base_cols])
    pc = rf_ctx.predict(X_te_g_s)
    pl = lgb.predict(X_te_g_s)
    px = xgb.predict(X_te_g_s)

    agree_bc = float((pb == pc).mean())
    agree_bl = float((pb == pl).mean())
    agree_bx = float((pb == px).mean())
    agree_cl = float((pc == pl).mean())
    agree_cx = float((pc == px).mean())

    f1b_m = f1_score(y_te_g, pb, average="macro", zero_division=0)
    f1c_m = f1_score(y_te_g, pc, average="macro", zero_division=0)
    f1l_m = f1_score(y_te_g, pl, average="macro", zero_division=0)
    f1x_m = f1_score(y_te_g, px, average="macro", zero_division=0)
    f1b_0 = f1_score(y_te_g, pb, labels=[0], average="macro", zero_division=0)
    f1c_0 = f1_score(y_te_g, pc, labels=[0], average="macro", zero_division=0)

    # region agent log — hypothesis A, E
    agent_log(
        "A",
        "debug_interaction_f1.py:group_split_preds",
        "prediction agreement and macro F1 (GroupShuffleSplit)",
        {
            "agree_baseline_vs_contextual_rf": round(agree_bc, 4),
            "agree_baseline_vs_lgbm": round(agree_bl, 4),
            "agree_baseline_vs_xgb": round(agree_bx, 4),
            "agree_contextual_rf_vs_lgbm": round(agree_cl, 4),
            "agree_contextual_rf_vs_xgb": round(agree_cx, 4),
            "f1_macro_baseline_rf": round(f1b_m, 4),
            "f1_macro_contextual_rf": round(f1c_m, 4),
            "f1_macro_lgbm": round(f1l_m, 4),
            "f1_macro_xgb": round(f1x_m, 4),
            "f1_class0_only_baseline": round(f1b_0, 4),
            "f1_class0_only_contextual": round(f1c_0, 4),
            "delta_f1_macro_ctx_minus_base_pp": round((f1c_m - f1b_m) * 100, 2),
            "delta_f1_macro_lgbm_minus_rf_pp": round((f1l_m - f1c_m) * 100, 2),
            "delta_f1_macro_xgb_minus_rf_pp": round((f1x_m - f1c_m) * 100, 2),
        },
        "pre-fix",
    )
    # endregion

    # --- Shuffle weather on test (contextual uses full X) — hypothesis B signal test ---
    X_te_shuf = X_te_g_s.copy()
    for wcol in WEATHER:
        X_te_shuf[wcol] = np.random.default_rng(RNG).permutation(X_te_shuf[wcol].values)
    pc_shuf = rf_ctx.predict(X_te_shuf)
    f1c_shuf = f1_score(y_te_g, pc_shuf, average="macro", zero_division=0)
    # region agent log
    agent_log(
        "B",
        "debug_interaction_f1.py:shuffle_weather_test",
        "contextual RF F1 after permuting weather on test only",
        {"f1_macro_contextual_rf_normal": round(f1c_m, 4), "f1_macro_after_shuffle_weather": round(f1c_shuf, 4), "drop_pp": round((f1c_m - f1c_shuf) * 100, 2)},
        "pre-fix",
    )
    # endregion

    # --- Mutual information: full X vs attended on TRAIN sample (speed) ---
    sample_n = min(20000, len(X_tr_g_s))
    idx = np.random.default_rng(RNG).choice(len(X_tr_g_s), size=sample_n, replace=False)
    Xs = X_tr_g_s.iloc[idx]
    ys = y_tr_g.iloc[idx]
    mi_all = mutual_info_classif(Xs, ys, discrete_features=False, random_state=RNG)
    mi_map = {c: float(mi_all[i]) for i, c in enumerate(Xs.columns)}
    mi_weather_sum = sum(mi_map.get(w, 0) for w in WEATHER)
    mi_baseline_sum = sum(mi_map.get(c, 0) for c in base_cols)
    # region agent log
    agent_log(
        "B",
        "debug_interaction_f1.py:mutual_info_train_sample",
        "mutual information with target (train subsample)",
        {"n_sample": sample_n, "mi_weather_cols": {w: round(mi_map.get(w, 0), 5) for w in WEATHER}, "sum_mi_weather": round(mi_weather_sum, 5), "sum_mi_baseline_features": round(mi_baseline_sum, 5)},
        "pre-fix",
    )
    # endregion

    # --- Random stratified split (same users can appear in train and test) — hypothesis D ---
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_raw, y, test_size=0.20, random_state=RNG, stratify=y
    )
    sc_r = StandardScaler()
    X_tr_rs = pd.DataFrame(sc_r.fit_transform(X_tr_r), columns=X_tr_r.columns)
    X_te_rs = pd.DataFrame(sc_r.transform(X_te_r), columns=X_te_r.columns)
    rf_br = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rf_cr = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, random_state=RNG, n_jobs=-1
    )
    rf_br.fit(X_tr_rs[base_cols], y_tr_r)
    rf_cr.fit(X_tr_rs, y_tr_r)
    prb = rf_br.predict(X_te_rs[base_cols])
    prc = rf_cr.predict(X_te_rs)
    f1br = f1_score(y_te_r, prb, average="macro", zero_division=0)
    f1cr = f1_score(y_te_r, prc, average="macro", zero_division=0)
    agree_r = float((prb == prc).mean())
    # region agent log
    agent_log(
        "D",
        "debug_interaction_f1.py:random_stratified_split",
        "leaky split: stratified random rows (users may overlap train/test)",
        {"f1_macro_baseline": round(f1br, 4), "f1_macro_contextual": round(f1cr, 4), "delta_pp": round((f1cr - f1br) * 100, 2), "pred_agreement_b_vs_c": round(agree_r, 4)},
        "pre-fix",
    )
    # endregion

    print("Debug run complete. See debug-e3238b.log for NDJSON lines.")


if __name__ == "__main__":
    main()
