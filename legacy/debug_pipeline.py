#!/usr/bin/env python3
"""
WAVE – Debug script: investigheaza de ce toate 4 modele au F1 identic.
Testeaza 5 ipoteze si scrie rezultatele in debug-e3238b.log (NDJSON).
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

ROOT        = Path(__file__).resolve().parents[1]
TRAIN_PATH  = ROOT / "data" / "processed" / "train_ready.csv"
MODELS_DIR  = ROOT / "models"
LOG_PATH    = ROOT / "debug-e3238b.log"

DROP_COLS    = ["event_id", "event_datetime", "event_name", "location"]
WEATHER_COLS = ["weather_temp_C", "weather_precip_mm"]
TARGET       = "attended"

SESSION_ID = "e3238b"

def log(hypothesis_id: str, message: str, data: dict, run_id: str = "run1"):
    entry = {
        "sessionId":   SESSION_ID,
        "id":          f"log_{int(time.time()*1000)}_{hypothesis_id}",
        "timestamp":   int(time.time() * 1000),
        "location":    "debug_pipeline.py",
        "hypothesisId": hypothesis_id,
        "runId":       run_id,
        "message":     message,
        "data":        data,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{hypothesis_id}] {message}: {json.dumps(data, indent=2)}")


def main():
    LOG_PATH.unlink(missing_ok=True)
    print("=" * 65)
    print("WAVE DEBUG – Pipeline Investigation")
    print("=" * 65)

    # ──────────────────────────────────────────────────────────────
    # IPOTEZA A: Duplicate features cu labels mixte in acelasi grup
    # ──────────────────────────────────────────────────────────────
    print("\n[A] Structura dataset – duplicate features cu labels mixte")
    df = pd.read_csv(TRAIN_PATH)

    total_rows     = len(df)
    unique_events  = df["event_id"].nunique()
    rows_per_event = df.groupby("event_id").size()

    # Pentru fiecare event_id, exista ambele labels (0 si 1)?
    mixed_label_events = df.groupby("event_id")["attended"].nunique()
    events_with_mixed  = (mixed_label_events > 1).sum()
    events_only_1      = (mixed_label_events == 1).sum()

    # Sample event_id=1: rate 0 vs 1
    e1 = df[df["event_id"] == 1]
    e1_label_dist = e1["attended"].value_counts().to_dict()

    # Feature columns ce sunt efectiv prezise
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    
    # Cate randuri au feature vector identic?
    dupes = df[feature_cols].duplicated().sum()
    unique_feature_vecs = df[feature_cols].drop_duplicates().shape[0]

    log("A", "Dataset structure – duplicate features with mixed labels", {
        "total_rows":           total_rows,
        "unique_event_ids":     int(unique_events),
        "rows_per_event_mean":  round(float(rows_per_event.mean()), 1),
        "rows_per_event_min":   int(rows_per_event.min()),
        "rows_per_event_max":   int(rows_per_event.max()),
        "events_with_BOTH_labels_0_and_1": int(events_with_mixed),
        "events_with_ONLY_label_1":        int(events_only_1),
        "event_id_1_label_distribution":   {str(k): int(v) for k,v in e1_label_dist.items()},
        "duplicate_feature_vectors":       int(dupes),
        "unique_feature_vectors":          int(unique_feature_vecs),
        "HYPOTHESIS_A_verdict": "CONFIRMED if events_with_BOTH_labels > 0 and duplicate_feature_vectors >> 0",
    })

    # ──────────────────────────────────────────────────────────────
    # IPOTEZA B: Deconectare labels (sintetic) vs weather (real)
    # ──────────────────────────────────────────────────────────────
    print("\n[B] Corelatie weather real vs labels (ar trebui aproape 0 daca deconectate)")
    from scipy.stats import pointbiserialr
    corr_temp,  p_temp  = pointbiserialr(df["weather_temp_C"],    df[TARGET])
    corr_precip, p_precip = pointbiserialr(df["weather_precip_mm"], df[TARGET])

    # Attendance rate: vreme buna vs rea (praguri zilnice reale)
    good = (df["weather_temp_C"].between(12, 26)) & (df["weather_precip_mm"] < 5)
    bad  = (df["weather_temp_C"] < 2) | (df["weather_temp_C"] > 30) | (df["weather_precip_mm"] > 20)
    att_good = float(df.loc[good, TARGET].mean()) if good.sum() > 0 else None
    att_bad  = float(df.loc[bad,  TARGET].mean()) if bad.sum()  > 0 else None

    log("B", "Weather-label disconnect analysis", {
        "pearson_r_temp_vs_attended":       round(corr_temp, 4),
        "pearson_p_temp":                   round(p_temp, 6),
        "pearson_r_precip_vs_attended":     round(corr_precip, 4),
        "pearson_p_precip":                 round(p_precip, 6),
        "attendance_rate_good_weather":     round(att_good, 4) if att_good else "N/A",
        "attendance_rate_bad_weather":      round(att_bad,  4) if att_bad  else "N/A",
        "n_good_weather_rows":              int(good.sum()),
        "n_bad_weather_rows":               int(bad.sum()),
        "HYPOTHESIS_B_verdict": "CONFIRMED if |r| < 0.05 and p > 0.05 OR if att_good ≈ att_bad",
    })

    # ──────────────────────────────────────────────────────────────
    # IPOTEZA C: Data leakage prin grup — acelasi event_id in train si test
    # ──────────────────────────────────────────────────────────────
    print("\n[C] Data leakage – event_id overlap intre train si test")
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train_event_ids = set(X_train["event_id"].unique())
    test_event_ids  = set(X_test["event_id"].unique())
    overlap = train_event_ids & test_event_ids
    only_in_train = train_event_ids - test_event_ids
    only_in_test  = test_event_ids  - train_event_ids

    log("C", "Event_id overlap between train and test splits", {
        "train_unique_event_ids": len(train_event_ids),
        "test_unique_event_ids":  len(test_event_ids),
        "overlap_count":          len(overlap),
        "only_in_train":          len(only_in_train),
        "only_in_test":           len(only_in_test),
        "overlap_pct_of_test":    round(len(overlap) / len(test_event_ids) * 100, 1),
        "HYPOTHESIS_C_verdict": "CONFIRMED if overlap_count > 0 (most events split across train/test)",
    })

    # ──────────────────────────────────────────────────────────────
    # IPOTEZA D: Mutual Information – weather vs alte features
    # ──────────────────────────────────────────────────────────────
    print("\n[D] Mutual Information: weather vs alte features fata de attended")
    feat_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    X_feat = df[feat_cols].fillna(0)
    mi_scores = mutual_info_classif(X_feat, df[TARGET], random_state=42)
    mi_dict = {feat_cols[i]: round(float(mi_scores[i]), 6) for i in range(len(feat_cols))}
    mi_sorted = dict(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))

    log("D", "Mutual Information scores per feature", {
        "mi_scores_ranked": mi_sorted,
        "weather_temp_C_rank": list(mi_sorted.keys()).index("weather_temp_C") + 1,
        "weather_precip_mm_rank": list(mi_sorted.keys()).index("weather_precip_mm") + 1,
        "total_features": len(feat_cols),
        "HYPOTHESIS_D_verdict": "CONFIRMED if weather_temp_C MI ≈ 0 and ranked near last",
    })

    # ──────────────────────────────────────────────────────────────
    # IPOTEZA E: Predicțiile sunt byte-identice pentru toate 4 modele
    # ──────────────────────────────────────────────────────────────
    print("\n[E] Overlap predicții – sunt modelele identice byte-by-byte?")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_drop  = X_test.drop(columns=[c for c in DROP_COLS if c in X_test.columns])
    X_test_sc = pd.DataFrame(scaler.transform(X_drop), columns=X_drop.columns)
    base_cols = [c for c in X_test_sc.columns if c not in WEATHER_COLS]

    baseline = joblib.load(MODELS_DIR / "baseline_rf.joblib")
    ctx_rf   = joblib.load(MODELS_DIR / "contextual_rf.joblib")
    lgbm_m   = joblib.load(MODELS_DIR / "lgbm_contextual.joblib")
    xgb_m    = joblib.load(MODELS_DIR / "xgb_contextual.joblib")

    y_base = baseline.predict(X_test_sc[base_cols])
    y_ctx  = ctx_rf.predict(X_test_sc)
    y_lgbm = lgbm_m.predict(X_test_sc)
    y_xgb  = xgb_m.predict(X_test_sc)

    f1_base = f1_score(y_test, y_base, average="macro")
    f1_ctx  = f1_score(y_test, y_ctx,  average="macro")
    f1_lgbm = f1_score(y_test, y_lgbm, average="macro")
    f1_xgb  = f1_score(y_test, y_xgb,  average="macro")

    log("E", "Model predictions overlap analysis", {
        "f1_baseline":   round(f1_base, 4),
        "f1_rf_ctx":     round(f1_ctx,  4),
        "f1_lgbm":       round(f1_lgbm, 4),
        "f1_xgb":        round(f1_xgb,  4),
        "identical_base_vs_rfctx_pct":  round(float((y_base == y_ctx).mean())  * 100, 2),
        "identical_base_vs_lgbm_pct":   round(float((y_base == y_lgbm).mean()) * 100, 2),
        "identical_base_vs_xgb_pct":    round(float((y_base == y_xgb).mean())  * 100, 2),
        "identical_rfctx_vs_lgbm_pct":  round(float((y_ctx  == y_lgbm).mean()) * 100, 2),
        "HYPOTHESIS_E_verdict": "CONFIRMED if identical_pct close to 100%",
    })

    # ──────────────────────────────────────────────────────────────
    # BONUS: Experiment – retrain FARA event_month si season
    # ──────────────────────────────────────────────────────────────
    print("\n[BONUS] Experiment: retrain fara event_month si season → creste weather?")
    X_no_time = df.drop(columns=DROP_COLS + [TARGET, "event_month", "season"])
    base_no_time = [c for c in X_no_time.columns if c not in WEATHER_COLS]

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_no_time, df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    sc2 = StandardScaler()
    X_tr2_sc = pd.DataFrame(sc2.fit_transform(X_tr2), columns=X_tr2.columns)
    X_te2_sc  = pd.DataFrame(sc2.transform(X_te2),   columns=X_te2.columns)

    rf_b2 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    rf_c2 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    rf_b2.fit(X_tr2_sc[base_no_time], y_tr2)
    rf_c2.fit(X_tr2_sc, y_tr2)

    f1_b2 = f1_score(y_te2, rf_b2.predict(X_te2_sc[base_no_time]), average="macro")
    f1_c2 = f1_score(y_te2, rf_c2.predict(X_te2_sc), average="macro")

    log("BONUS", "Experiment: F1 without event_month and season", {
        "f1_baseline_no_time":   round(f1_b2, 4),
        "f1_contextual_no_time": round(f1_c2, 4),
        "delta_ctx_vs_base":     round(f1_c2 - f1_b2, 4),
        "original_f1_baseline":  round(f1_base, 4),
        "original_f1_ctx":       round(f1_ctx,  4),
        "INSIGHT": "If weather still doesnt help after removing time features, confirms Hypothesis A is root cause",
    })

    print("\n" + "=" * 65)
    print(f"Debug complete. Log written to: {LOG_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
