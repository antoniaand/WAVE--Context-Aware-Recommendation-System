import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

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

def load_preprocessed_data():
    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)
    df = encode_categoricals(df)
    
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    else:
        X_scaled = X.copy()
        
    return X, X_scaled, y

def main():
    X, X_scaled, y = load_preprocessed_data()
    
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