import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

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
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_test_split():
    """
    Load the test partition only — same GroupShuffleSplit(test_size=0.20, random_state=42)
    as train_models.py. SHAP must be computed on held-out data so explanations reflect
    genuine generalisation, not memorised training patterns.
    """
    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)
    df = encode_categoricals(df)

    groups = df["user_id"].values
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_test_scaled = X_test.copy()

    return X_test, X_test_scaled, y_test

def main():
    X, X_scaled, y = load_test_split()
    
    model_path = MODELS_DIR / "lgbm_contextual.joblib"
    lgbm_model = joblib.load(model_path)
    
    explainer = shap.TreeExplainer(lgbm_model)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")
        shap_values = explainer.shap_values(X_scaled)
    
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_plot_lgbm.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("weather_temp_C", shap_values_to_plot, X, interaction_index="weather_precip_mm", show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_dependence_temp_precip_lgbm.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_bar_importance_lgbm.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()