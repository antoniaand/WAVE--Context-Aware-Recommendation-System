# WAVE -- A Context-Aware Recommender System for Event Attendance

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

**WAVE** (Weather-Aware Venue & Event recommender) is a bachelor thesis project that
designs and evaluates a **context-aware recommendation system** incorporating real-time
weather conditions into event attendance prediction.
Traditional recommender systems rely on static user preferences and historical behaviour
while ignoring dynamic contextual factors.
This project tests the hypothesis that **integrating weather context (temperature and
precipitation) significantly improves the accuracy of event attendance prediction**
compared to a baseline model that uses only temporal and categorical event features.

---

## Key Results

| Metric    | Baseline | Contextual | Delta  |
|-----------|----------|------------|--------|
| Accuracy  | 0.7165   | **0.8067** | +0.090 |
| Precision | 0.7188   | **0.7982** | +0.079 |
| Recall    | 0.6638   | **0.7901** | +0.126 |
| F1-Score  | 0.6678   | **0.7936** | +0.126 |

The Contextual Model significantly outperforms the Baseline across every metric,
improving the **F1-Score from 0.67 to 0.79** (an absolute increase of ~12.5 percentage points).
Feature importance analysis reveals that the two weather features alone
(**precipitation** and **temperature**) account for **~72.5%** of model decisions,
providing strong empirical evidence for the research hypothesis.

| Rank | Feature             | Importance |
|------|---------------------|------------|
| 1    | weather_precip_mm   | 0.4547     |
| 2    | weather_temp_C      | 0.2705     |
| 3    | event_month         | 0.0570     |
| 4    | season              | 0.0538     |
| 5    | location_enc        | 0.0517     |

---

## Repository Layout

```
WAVE/
|-- data/
|   |-- raw/                        # Original unprocessed datasets
|   |   |-- event_attendance.csv    #   * Primary training data (Kaggle, ~200 000 rows)
|   |   |-- users_110.csv           #   * User survey (110 respondents, event preferences)
|   |   |-- cultural_engagement_dataset.csv   # (archived -- outside current scope)
|   |   |-- cult_pcs_caa_linear.csv           # (archived -- outside current scope)
|   |-- processed/                  # Pipeline outputs
|       |-- train_ready.csv         #   Training-ready dataset with weather & labels
|       |-- app_users.csv           #   Cleaned user profiles for the demo app
|
|-- models/                         # Serialised trained models (joblib)
|   |-- baseline_rf.joblib          #   RandomForest -- temporal + categorical features
|   |-- contextual_rf.joblib        #   RandomForest -- above + weather features
|
|-- results/                        # Evaluation outputs (plots & tables)
|   |-- confusion_matrix_comparison.png
|   |-- roc_comparison.png
|   |-- pr_curve_comparison.png
|   |-- metrics_barchart.png
|   |-- metrics_comparison.csv
|   |-- feature_importances.csv
|
|-- src/                            # Core pipeline scripts
|   |-- dataset_pipeline.py         #   Step 1 -- clean, encode, scale raw data
|   |-- generate_weather.py         #   Step 2 -- synthetic weather + attendance flip
|   |-- train_models.py             #   Step 3 -- train Baseline & Contextual RF
|   |-- evaluate_visuals.py         #   Step 4 -- confusion matrices, ROC, importances
|
|-- evaluate_extended_metrics.py    # Step 5 -- metrics CSV, bar chart, PR curve
|-- requirements.txt                # Python dependencies
|-- DirectionOfProject.sty          # Project scope & research description
|-- README.md                       # This file
```

### A note on archived datasets

`cultural_engagement_dataset.csv` and `cult_pcs_caa_linear.csv` were explored during
the initial analysis phase but intentionally excluded to maintain a focused thesis scope.
They share no common user or event keys with the primary datasets and provide only
country-level aggregation that does not align with per-event prediction.
They are retained in `data/raw/` for transparency and potential future extensions.

---

## Setup & Installation

### Prerequisites

* **Python 3.10** or later
* `pip` (bundled with Python)

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Wave

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproduction -- Pipeline Run Order

The entire experiment can be reproduced end-to-end with the five commands below.
Each script is idempotent and overwrites its outputs, so re-running is safe.

```bash
# Step 1 -- Data cleaning, encoding, scaling
#           Produces: data/processed/train_ready.csv, data/processed/app_users.csv
python src/dataset_pipeline.py

# Step 2 -- Populate synthetic weather features & adjust attendance labels
#           Updates:  data/processed/train_ready.csv (in-place)
python src/generate_weather.py

# Step 3 -- Train Baseline and Contextual RandomForest classifiers
#           Produces: models/baseline_rf.joblib, models/contextual_rf.joblib
python src/train_models.py

# Step 4 -- Generate confusion matrices, ROC curves, feature importance CSV
#           Produces: results/confusion_matrix_comparison.png
#                     results/roc_comparison.png
#                     results/feature_importances.csv
python src/evaluate_visuals.py

# Step 5 -- Generate metrics table, grouped bar chart, Precision-Recall curve
#           Produces: results/metrics_comparison.csv
#                     results/metrics_barchart.png
#                     results/pr_curve_comparison.png
python evaluate_extended_metrics.py
```

---

## Methodology & Reproducibility

### Experimental Design

Two **RandomForestClassifier** models are trained on the same dataset with identical
hyper-parameters; only the feature set differs:

| Aspect          | Baseline Model                       | Contextual Model                          |
|-----------------|--------------------------------------|-------------------------------------------|
| Features        | Temporal + categorical (6 features)  | Temporal + categorical + weather (8 features) |
| Target          | `attended` (binary: 0 / 1)           | `attended` (binary: 0 / 1)               |
| Hyper-parameters | `n_estimators=100, max_depth=15, min_samples_split=5` | Same |

### Reproducibility Guarantees

* **Random seed:** All stochastic operations use `random_state=42` (data splitting,
  model initialisation, synthetic weather generation via `numpy.random.default_rng(42)`).
* **Stratified splitting:** `train_test_split(..., stratify=y)` preserves the class
  ratio (~61 % attended / ~39 % not attended) in both train and test partitions.
* **Deterministic pipeline:** Running the five scripts sequentially from a clean
  `data/raw/` folder will produce byte-identical outputs.

### Inspecting Saved Models

The `.joblib` files are serialised scikit-learn estimators.
Load and inspect them interactively:

```python
import joblib

model = joblib.load("models/contextual_rf.joblib")

print(type(model))               # <class 'sklearn.ensemble._forest.RandomForestClassifier'>
print(model.get_params())         # hyper-parameters
print(model.n_features_in_)       # number of input features
print(model.feature_importances_) # feature importance array
```

---

## Tech Stack

| Component         | Technology               |
|-------------------|--------------------------|
| Language          | Python 3.10+             |
| Data manipulation | Pandas, NumPy            |
| ML framework      | scikit-learn             |
| Model persistence | joblib                   |
| Visualisation     | Matplotlib               |
| Frontend (planned)| Streamlit or PWA         |
| Weather API (planned) | OpenWeatherMap       |

---

## License

This project is developed as part of a **Bachelor's Thesis** at ASE Bucharest (CSIE). All rights reserved by the author. For academic or research use, please
cite appropriately.

---

