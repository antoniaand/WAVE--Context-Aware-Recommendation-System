#!/usr/bin/env python3
"""
tune_models_optuna.py
---------------------
Advanced Hyperparameter tuning using Optuna (Bayesian Optimization).
Respects GroupKFold to avoid data leakage.

=========================================
REZULTATE OPTUNA:
XGBoost Cel mai bun scor: 0.9491
XGBoost Cei mai buni parametri: {'n_estimators': 500, 'learning_rate': 0.09099071502444517, 'max_depth': 4, 'subsample': 0.823645666957371, 'colsample_bytree': 0.869667799958888}

LightGBM Cel mai bun scor: 0.9477
LightGBM Cei mai buni parametri: {'n_estimators': 600, 'learning_rate': 0.09270772807912192, 'num_leaves': 33, 'min_child_samples': 29}
=========================================
"""

from pathlib import Path
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.modeling.train_models import (
    CSV_PATH,
    DROP_COLS,
    TARGET,
    engineer_features,
    encode_categoricals,
)

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_xy_groups():
    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)
    df = encode_categoricals(df)

    groups = df["user_id"].values
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y, groups

def main():
    X, y, groups = load_xy_groups()
    cv = GroupKFold(n_splits=3)
    scoring = "f1_macro"

    # --- 1. Definim functia obiectiv pentru XGBoost ---
    def objective_xgb(trial):
        # Optuna "sugereaza" valori in mod inteligent
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # Setam n_jobs=1 aici pt model, si lasam cross_val_score sa paralelizeze
        model = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, **params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        
        # Calculam scorul pe cele 3 fold-uri (pastrand utilizatorii separati)
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
        return scores.mean()

    # --- 2. Definim functia obiectiv pentru LightGBM ---
    def objective_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
        }
        
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, **params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
        return scores.mean()

    # --- 3. Executam cautarea Optuna ---
    print("\n--- Incepem Optuna tuning pentru XGBoost ---")
    study_xgb = optuna.create_study(direction="maximize", study_name="XGBoost_Optuna")
    # n_trials = 20 inseamna ca va face 20 de incercari "inteligente" (poti mari numarul daca ai timp)
    study_xgb.optimize(objective_xgb, n_trials=20) 
    
    print("\n--- Incepem Optuna tuning pentru LightGBM ---")
    study_lgbm = optuna.create_study(direction="maximize", study_name="LightGBM_Optuna")
    study_lgbm.optimize(objective_lgbm, n_trials=20)

    # --- 4. Salvam rezultatele ---
    print("\n=========================================")
    print("REZULTATE OPTUNA:")
    print(f"XGBoost Cel mai bun scor: {study_xgb.best_value:.4f}")
    print(f"XGBoost Cei mai buni parametri: {study_xgb.best_params}")
    
    print(f"\nLightGBM Cel mai bun scor: {study_lgbm.best_value:.4f}")
    print(f"LightGBM Cei mai buni parametri: {study_lgbm.best_params}")
    print("=========================================")

if __name__ == "__main__":
    main()