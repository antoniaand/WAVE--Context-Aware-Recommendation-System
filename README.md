# WAVE -- A Context-Aware Recommender System for Event Attendance

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen?logo=data:image/svg+xml;base64,)
![XGBoost](https://img.shields.io/badge/XGBoost-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

**WAVE** (Weather-Aware Venue & Event recommender) is a bachelor thesis project that
designs and evaluates a **context-aware recommendation system** incorporating
**real historical weather conditions** into event attendance prediction.

Traditional recommender systems rely on static user preferences while ignoring dynamic
contextual factors such as weather. This project tests the hypothesis that
**integrating real weather data and user weather-sensitivity profiles significantly
improves attendance prediction**, especially for outdoor events under extreme conditions.

> The previous pipeline used synthetically generated weather.
> The current pipeline fetches **real historical data from the Open-Meteo archive API**
> for 18 cities across 4 climate zones. See `legacy/README_old_pipeline.md` for the
> old results.

---

## Key Results (current pipeline -- 49,500 interactions, 18 cities)

### Global test set (9,900 rows, 22 held-out users)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RF Baseline *(no weather)* | 0.9128 | 0.9134 | 0.9125 | 0.9127 |
| RF Contextual | 0.9446 | 0.9449 | 0.9445 | 0.9446 |
| LGBM Contextual | 0.9559 | 0.9558 | 0.9558 | **0.9558** |
| XGB Contextual | 0.9563 | 0.9563 | 0.9562 | **0.9562** |

RF Baseline → XGB Contextual: **+3.19 pp F1** on the global test set.

### Extreme-weather subgroup (n = 1,606 | outdoor & T < 5°C or precip > 0.5 mm)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RF Baseline *(no weather)* | 0.9271 | 0.9008 | 0.8956 | 0.8981 |
| RF Contextual | 0.9595 | 0.9634 | 0.9232 | 0.9413 |
| LGBM Contextual | 0.9726 | 0.9695 | 0.9537 | 0.9613 |
| **XGB Contextual** | **0.9788** | **0.9804** | **0.9605** | **0.9700** |

**Relative Error Reduction (RF Baseline → XGB Contextual)**

| Scope | Accuracy RER | F1 RER |
|---|---|---|
| Global test | 49.9 % | 49.8 % |
| Extreme-weather slice | **70.9 %** | **70.6 %** |

The contextual models eliminate **~71 % of errors** that the weather-blind baseline
makes on hard outdoor scenarios (cold / wet conditions).

### Outdoor attendance by weather condition (simulation validation)

| Condition | Attendance rate |
|---|---|
| Outdoor + extreme heat (T > 35°C) | 7.7 % |
| Outdoor + extreme cold (T < 0°C) | 16.3 % |
| Outdoor + no rain | 41.4 % |
| Indoor (any weather) | ~75 % |

---

## Dataset

| Property | Value |
|---|---|
| Users | 110 real survey respondents (`data/processed/app_users.csv`) |
| Events | 450 synthetic events across 4 climate zones |
| Interactions | 110 × 450 = **49,500 rows** |
| Weather source | Open-Meteo Historical Archive API (real hourly data) |
| Label | `attended` (0/1) -- 50/50 balanced via median threshold |
| Split | GroupShuffleSplit on `user_id` -- 88 train / 22 test users |

### Climate zones

| Zone | Cities | Season window | Events |
|---|---|---|---|
| Moderate | Bucharest, Cluj-Napoca, Timisoara, Iasi, Constanta, Brasov | Jun 2024 -- Jun 2025 | 270 |
| Cold | Oslo, Helsinki, Quebec | Dec 2024 -- Feb 2025 | 60 |
| Heat | Dubai, Phoenix, Seville | Jun -- Aug 2024 | 60 |
| Rain | London, Bergen, Seattle | Oct -- Nov 2024 | 60 |

Extreme zones = **40 %** of all events.

---

## Repository Layout

```
WAVE/
|-- data/
|   |-- raw/
|   |   |-- weather_archive_cache.csv   # Hourly weather per city (Open-Meteo)
|   |-- processed/
|       |-- app_users.csv               # 110 user profiles (survey)
|       |-- interaction_foundation.csv  # Step 1 output: 49,500 user x event rows
|       |-- interaction_with_weather.csv# Step 2 output: + real weather columns
|       |-- train_ready_interactions.csv# Step 3 output: + attended label
|
|-- models/
|   |-- baseline_rf.joblib             # RF -- no weather features
|   |-- contextual_rf.joblib           # RF -- full features
|   |-- lgbm_contextual.joblib         # LightGBM -- full features
|   |-- xgb_contextual.joblib          # XGBoost -- full features
|   |-- scaler.joblib                  # StandardScaler (fit on train only)
|
|-- results/
|   |-- confusion_matrix_comparison.png
|   |-- roc_comparison.png
|   |-- pr_curve_comparison.png
|   |-- metrics_barchart.png
|   |-- metrics_barchart_extreme_weather.png  # NEW -- subgroup
|   |-- metrics_comparison.csv
|   |-- metrics_subgroup_extreme_weather.csv  # NEW -- subgroup table
|   |-- feature_importances.csv
|   |-- f1_extreme_weather_slice.png          # NEW -- F1 bar (subgroup)
|   |-- xgb_subgroup_permutation_importance.png  # NEW -- permutation importance
|
|-- src/
|   |-- generate_foundation.py    # Step 1 -- 450 events x 110 users Cartesian grid
|   |-- fetch_weather_api.py      # Step 2 -- real weather from Open-Meteo (18 cities)
|   |-- simulate_labels.py        # Step 3 -- behavioural label simulation
|   |-- train_models.py           # Step 4 -- train 4 models (GroupShuffleSplit)
|   |-- eval_common.py            # Shared preprocessing for evaluation scripts
|   |-- evaluate_visuals.py       # Step 5 -- confusion, ROC, F1 subgroup, permutation
|   |-- evaluate_extended_metrics.py  # Step 6 -- metrics CSV, barcharts, RER
|   |-- test_weather_signal.py    # Isolated blind vs contextual experiment
|   |-- cross_validate.py         # K-fold cross-validation
|
|-- legacy/                       # Archived old pipeline (pre-rebuild)
|   |-- README_old_pipeline.md    # Old README with old results
|   |-- dataset_pipeline.py
|   |-- generate_weather.py       # Old synthetic weather generation
|   |-- build_interaction_dataset.py
|   |-- fetch_weather.py
|   |-- ...
|
|-- docs/
|   |-- DATASET_METHODOLOGY.md
|
|-- requirements.txt
|-- README.md
```

---

## Setup & Installation

### Prerequisites

* Python 3.10 or later
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

Run each script in order from the project root. Every step is idempotent.

```bash
# Step 1 -- Generate 450 events x 110 users interaction grid (no weather yet)
#           Produces: data/processed/interaction_foundation.csv  (49,500 rows)
python src/generate_foundation.py

# Step 2 -- Fetch real hourly weather from Open-Meteo for all 18 cities
#           Produces: data/raw/weather_archive_cache.csv
#                     data/processed/interaction_with_weather.csv
python src/fetch_weather_api.py

# Step 3 -- Simulate attended labels using weather x user sensitivity profile
#           Produces: data/processed/train_ready_interactions.csv
python src/simulate_labels.py

# Step 4 -- Train 4 models (RF Baseline, RF Contextual, LGBM, XGB)
#           Produces: models/*.joblib, results/metrics_comparison.csv
python src/train_models.py

# Step 5 -- Confusion matrices, ROC, F1 subgroup bar, XGB permutation importance
#           Produces: results/confusion_matrix_comparison.png
#                     results/roc_comparison.png
#                     results/f1_extreme_weather_slice.png
#                     results/xgb_subgroup_permutation_importance.png
python src/evaluate_visuals.py

# Step 6 -- Global + subgroup metrics tables, barcharts, Relative Error Reduction
#           Produces: results/metrics_barchart.png
#                     results/metrics_barchart_extreme_weather.png
#                     results/metrics_subgroup_extreme_weather.csv
#                     results/pr_curve_comparison.png
python src/evaluate_extended_metrics.py
```

---

## Methodology

### Experimental Design

Four models are trained on the same 49,500-row dataset with the same hyperparameters.
Only the feature set differs between Baseline and Contextual:

| Aspect | RF Baseline | RF / LGBM / XGB Contextual |
|---|---|---|
| User preferences | Yes | Yes |
| Event type, location, month | Yes | Yes |
| Real weather (temp, precip, wind) | **No** | **Yes** |
| User weather sensitivity profile | **No** | **Yes** |
| Algorithm | RandomForest | RF / LightGBM / XGBoost |

### Reproducibility Guarantees

* **Random seed:** `random_state=42` / `numpy.random.default_rng(42)` throughout.
* **Group split:** `GroupShuffleSplit(test_size=0.20)` on `user_id` — no user appears
  in both train and test, preventing profile memorisation.
* **No data leakage:** `StandardScaler` fitted on train fold only; `final_prob`
  (the continuous probability used to generate labels) is excluded from features.
* **Deterministic:** Re-running steps 1--6 from a clean state produces identical outputs.

### Weather Data

Weather is fetched from the **Open-Meteo Historical Archive API** (free, no key needed).
One API call per city covers the full event date window for that city; 15 API calls total
for 15 unique cities in the dataset.

Fetched variables: `temperature_2m`, `precipitation`, `windspeed_10m` (hourly).

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| ML framework | scikit-learn, LightGBM, XGBoost |
| Weather data | Open-Meteo Historical Archive API |
| Model persistence | joblib |
| Visualisation | Matplotlib |
| Feature analysis | sklearn.inspection.permutation_importance |

---

## Version History

| Tag / Branch | Description |
|---|---|
| `archive/old-dataset-and-models` | Last state before modular rebuild: synthetic weather, old `train_ready.csv`, 2-model RF experiment |
| `legacy/old-dataset-pipeline` | Same snapshot as a branch for easy checkout |
| `main` (current) | Modular pipeline: real weather, 18 cities, 4 models, subgroup analysis |

---

## License

This project is developed as part of a **Bachelor's Thesis** at ASE Bucharest (CSIE).
All rights reserved by the author. For academic or research use, please cite appropriately.
