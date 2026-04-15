#!/usr/bin/env python3
"""

FOR LATER IMPLEMENTATION -- takes too long, and my processor can barely hold on, so i switched to Optuna, 
a more intelligent search for hyperparameters.(for now, in the future i may try to run it again)
tune_models.py
--------------
Hyperparameter tuning for the WAVE models using GROUPED CV (by user_id).

IMPORTANT:
- We must apply the SAME preprocessing as train_models.py (encoding + feature engineering),
  otherwise sklearn fails with: "could not convert string to float: 'Female'".
- Scaling must be done INSIDE the CV folds to avoid leakage -> we use sklearn Pipeline.

Outputs:
  results/hyperparam_tuning.csv  (best params + best macro-F1 per model)
"""

from pathlib import Path

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    # Make progress logs visible even when output is buffered
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    X, y, groups = load_xy_groups()
    cv = GroupKFold(n_splits=3)

    # Use macro-F1 (consistent with the rest of the project)
    scoring = "f1_macro"

    models_and_grids = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [12, 16, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
        },
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            "param_grid": {
                "model__n_estimators": [300, 600],
                "model__learning_rate": [0.03, 0.05],
                "model__num_leaves": [31, 63, 127],
                "model__min_child_samples": [10, 20],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "param_grid": {
                "model__n_estimators": [300, 600],
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [4, 6, 8],
                "model__subsample": [0.7, 0.85],
                "model__colsample_bytree": [0.7, 0.85],
            },
        },
    }

    rows = []

    for model_name, cfg in models_and_grids.items():
        print(f"\n--- Incepem tuning pentru {model_name} ---")

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", cfg["estimator"]),
            ]
        )

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=2,
        )

        gs.fit(X, y, groups=groups)

        print(f"Cel mai bun scor {scoring} ({model_name}): {gs.best_score_:.4f}")
        print(f"Cei mai buni parametri ({model_name}): {gs.best_params_}")

        row = {
            "model": model_name,
            "best_score_f1_macro": round(float(gs.best_score_), 6),
            **gs.best_params_,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "hyperparam_tuning.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()