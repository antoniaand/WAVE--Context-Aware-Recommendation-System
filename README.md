# WAVE -- A Context-Aware Recommender System for Event Attendance

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen)
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

The project comprises three layers:

- **ML pipeline** — modular 3-step data generation + 5 trained models + SHAP explainability
- **Backend** — FastAPI REST API serving real-time contextual recommendations
- **Frontend** — React + Vite PWA with auth, onboarding, and recommendation view

> The previous pipeline used synthetically generated weather.
> The current pipeline fetches **real historical data from the Open-Meteo archive API**
> for 18 cities across 4 climate zones. See `legacy/README_old_pipeline.md` for the
> old results.

---

## Key Results (current pipeline -- 49,500 interactions, 18 cities)

### Global test set (9,900 rows, 22 held-out users)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RF Baseline *(no weather)* | 0.8852 | 0.8865 | 0.8849 | 0.8850 |
| RF Baseline *(strict)* | 0.8677 | 0.8689 | 0.8675 | 0.8675 |
| RF Contextual | 0.9049 | 0.9053 | 0.9048 | 0.9049 |
| LGBM Contextual | 0.9325 | 0.9328 | 0.9324 | 0.9325 |
| **XGB Contextual** | **0.9334** | **0.9338** | **0.9333** | **0.9334** |

Baseline → Contextual gap (RF): **+1.99 pp F1**.
Strict baseline (drops `location`, `climate_zone`, `event_month`) → Contextual gap (RF): **+3.74 pp F1**.

### Extreme-weather subgroup (n = 1,606 | outdoor & T < 5°C or precip > 0.5 mm)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RF Baseline *(no weather)* | 0.9352 | 0.9282 | 0.8377 | 0.8744 |
| RF Baseline *(strict)* | 0.9103 | 0.8369 | 0.8665 | 0.8505 |
| RF Contextual | 0.9334 | 0.9528 | 0.8139 | 0.8640 |
| **LGBM Contextual** | **0.9626** | 0.9519 | **0.9151** | **0.9322** |
| XGB Contextual | 0.9589 | **0.9550** | 0.8987 | 0.9238 |

**Relative Error Reduction (RF Baseline → XGB Contextual)**

| Scope | Accuracy RER | F1 RER |
|---|---|---|
| Global test | 42.0 % | 42.1 % |
| Extreme-weather slice | **36.6 %** | **39.3 %** |

The contextual models eliminate a large fraction of errors that the weather-blind baseline
makes on hard outdoor scenarios (cold / wet conditions), and the gap grows further when
geography/season proxies are removed (strict baseline).

### Outdoor attendance by weather condition (simulation validation)

| Condition | Attendance rate |
|---|---|
| Outdoor + extreme heat (T > 35°C) | 9.9 % |
| Outdoor + extreme cold (T < 0°C) | 13.4 % |
| Outdoor + no rain | 29.0 % |
| Indoor (any weather) | ~75 % |

### Scenario calibration check (survey hypotheticals)

We validate that the simulator aligns with the survey's scenario responses by running
the 110 users through 4 scenario mappings (one per scenario question) and computing
binary agreement (scenario score ≥ 2 → attend). This produces **84.3% agreement** overall.
See `src/scenario_validation.py` and `results/scenario_validation.csv`.

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
|-- backend/                          # FastAPI production API
|   |-- app/
|   |   |-- core/
|   |   |   |-- config.py             # Pydantic v2 settings (Supabase, Open-Meteo)
|   |   |   |-- database.py           # Supabase client + user profile helpers
|   |   |   |-- security.py           # JWT creation & verification
|   |   |-- models/
|   |   |   |-- recommendation.py     # Pydantic schemas (request/response)
|   |   |-- routers/
|   |   |   |-- auth.py               # Register, login, profile CRUD
|   |   |   |-- recommend.py          # POST /recommend — contextual scoring
|   |   |-- services/
|   |   |   |-- ml_service.py         # ML inference: feature engineering + model registry
|   |   |   |-- weather_service.py    # Open-Meteo forecast API integration
|   |   |   |-- event_service.py      # Fetch events from Supabase (synthetic fallback)
|   |-- scripts/
|   |   |-- common.py                 # Shared city/climate/event-type constants
|   |-- main.py                       # FastAPI app entry point
|   |-- requirements.txt              # Backend Python dependencies
|   |-- .env                          # Secrets (not committed) -- see .env.example
|
|-- frontend/                         # React 18 + Vite PWA
|   |-- src/
|   |   |-- pages/                    # Landing, Login, Register, Onboarding, Home, Profile
|   |   |-- components/               # EventCard, SurveyForm, WeatherWidget, ThemeToggle
|   |   |-- services/                 # api.ts (Axios + JWT), authService, recommendService
|   |   |-- contexts/                 # AuthContext (user, role, token)
|   |   |-- hooks/                    # useAuth, useTheme
|   |   |-- types/                    # TypeScript interfaces mirrored from backend schemas
|   |-- package.json
|   |-- vite.config.ts
|
|-- data/
|   |-- raw/
|   |   |-- weather_archive_cache.csv # Hourly weather per city (Open-Meteo)
|   |-- processed/
|       |-- app_users.csv             # 110 user profiles (survey)
|       |-- interaction_foundation.csv# Step 1 output: 49,500 user x event rows
|       |-- interaction_with_weather.csv # Step 2 output: + real weather columns
|       |-- train_ready_interactions.csv # Step 3 output: + attended label
|
|-- models/
|   |-- baseline_rf.joblib            # RF -- no weather features
|   |-- baseline_strict_rf.joblib     # RF -- strict baseline (no weather/location/month)
|   |-- contextual_rf.joblib          # RF -- full features
|   |-- lgbm_contextual.joblib        # LightGBM -- full features
|   |-- xgb_contextual.joblib         # XGBoost -- full features
|   |-- scaler.joblib                 # StandardScaler (fit on train only)
|
|-- results/
|   |-- metrics/
|   |   |-- metrics_comparison.csv
|   |   |-- metrics_subgroup_extreme_weather.csv
|   |   |-- cv_results.csv
|   |   |-- scenario_validation.csv
|   |   |-- feature_importances.csv
|   |-- visual_performance/
|   |   |-- metrics_barchart.png
|   |   |-- cv_comparison.png
|   |   |-- roc_comparison.png
|   |   |-- pr_curve_comparison.png
|   |   |-- confusion_matrix_comparison.png
|   |-- visuals_subgroups/
|   |   |-- metrics_barchart_extreme_weather.png
|   |   |-- f1_extreme_weather_slice.png
|   |-- explainability/
|   |   |-- shap_summary_plot_lgbm.png
|   |   |-- shap_bar_importance_lgbm.png
|   |   |-- shap_dependence_temp_precip_lgbm.png
|   |   |-- xgb_subgroup_permutation_importance.png
|
|-- src/
|   |-- data/
|   |   |-- generate_foundation.py    # Step 1 -- 450 events x 110 users Cartesian grid
|   |   |-- fetch_weather_api.py      # Step 2 -- real weather from Open-Meteo
|   |   |-- simulate_labels.py        # Step 3 -- behavioural label simulation
|   |-- modeling/
|   |   |-- train_models.py           # Step 4 -- train 5 models
|   |   |-- tune_models_OPTUNA.py     # Hyperparameter optimization (Optuna)
|   |   |-- tune_models_GRIDSearch.py
|   |   |-- cross_validate_overfitting.py # Group-based cross-validation
|   |-- evaluation/
|   |   |-- eval_common.py            # Shared preprocessing for evaluation
|   |   |-- evaluate_visuals.py       # Step 5 -- confusion, ROC, F1 subgroup
|   |   |-- evaluate_extended_metrics.py  # Step 6 -- metrics CSV, barcharts, RER
|   |   |-- scenario_validation.py    # Calibrate simulator on survey scenarios
|   |   |-- test_weather_signal.py    # Isolated blind vs contextual experiment
|   |-- xAI_SHAP/
|   |   |-- explain_models.py         # SHAP value extraction and explanation plots
|
|-- legacy/                           # Archived old pipeline (pre-rebuild)
|-- docs/
|   |-- DATASET_METHODOLOGY.md
|   |-- ML_AUDIT_REPORT.md
|-- requirements.txt                  # ML pipeline Python dependencies
|-- README.md
```

---

## Setup & Installation

### Prerequisites

* Python 3.10 or later
* Node.js 18 or later (for the frontend)
* A [Supabase](https://supabase.com) project (free tier is enough) — provides auth + events database

### 1. Clone and install ML dependencies

```bash
git clone <repository-url>
cd Wave

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure the backend environment

Create `backend/.env` (never committed):

```ini
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SECRET_KEY=any-long-random-string-for-jwt
```

Install backend dependencies (uses the same venv or a separate one):

```bash
pip install -r backend/requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the Web App

Open **two terminals** from the project root.

**Terminal 1 — Backend (FastAPI)**

```bash
# Activate the Python venv first (see Setup step 1)
cd backend
uvicorn main:app --reload --port 8000
```

API is now at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`.

**Terminal 2 — Frontend (React + Vite)**

```bash
cd frontend
npm run dev
```

App is now at `http://localhost:5173`.

### Quick smoke test

1. Open `http://localhost:5173` in the browser.
2. Register a new account → complete the onboarding survey.
3. Open the Home page — you should see event cards with attendance probability scores and current weather context.
4. Change the city or date — scores update based on real weather forecast.

---

## Reproduction -- ML Pipeline Run Order

Run each script in order from the project root. Every step is idempotent.

```bash
# Step 1 -- Generate 450 events x 110 users interaction grid (no weather yet)
#           Produces: data/processed/interaction_foundation.csv  (49,500 rows)
python src/data/generate_foundation.py

# Step 2 -- Fetch real hourly weather from Open-Meteo for all 18 cities
#           Produces: data/raw/weather_archive_cache.csv
#                     data/processed/interaction_with_weather.csv
python src/data/fetch_weather_api.py

# Step 3 -- Simulate attended labels using weather x user sensitivity profile
#           Produces: data/processed/train_ready_interactions.csv
python src/data/simulate_labels.py

# Step 4 -- Train 5 models (RF Baseline, RF Strict, RF Contextual, LGBM, XGB)
#           Produces: models/*.joblib, results/metrics/metrics_comparison.csv
python src/modeling/train_models.py

# Step 5 -- Confusion matrices, ROC, F1 subgroup bar, XGB permutation importance
#           Produces charts in results/visual_performance/ and results/visuals_subgroups/
python src/evaluation/evaluate_visuals.py

# Step 6 -- Global + subgroup metrics tables, barcharts, Relative Error Reduction
#           Produces metrics in results/metrics/ and charts in results/visual_performance/
python src/evaluation/evaluate_extended_metrics.py

# Optional -- Scenario calibration check (survey hypotheticals)
python src/evaluation/scenario_validation.py
```

---

## Methodology

### Experimental Design

Five models are trained on the same 49,500-row dataset with the same hyperparameters.
Only the feature set differs between Baseline and Contextual:

| Aspect | RF Baseline | RF Baseline (strict) | RF / LGBM / XGB Contextual |
|---|---|---|---|
| User preferences | Yes | Yes | Yes |
| Event type, location, month | Yes | No | Yes |
| Real weather (temp, precip, wind) | No | No | Yes |
| User weather sensitivity profile | No | No | Yes |
| Algorithm | RandomForest | RandomForest | RF / LightGBM / XGBoost |

The strict baseline drops `location`, `climate_zone`, and `event_month` to remove
all geography/season proxies, making it a stricter weather-contribution control.

### Backend — Recommendation Logic

At inference time (`backend/app/routers/recommend.py`):

1. **Events** are fetched from Supabase for the requested city + date range.
   If fewer than 5 real events exist, synthetic placeholders are generated.
2. **Weather** is fetched from Open-Meteo forecast API for the selected city and hour.
   Falls back to mild clear-sky defaults if the forecast is unavailable.
3. **ML scoring** — each event is scored by the selected model (default: LGBM).
   The user profile + weather context are attached to every event row before inference.
4. Results are returned sorted by `attended_prob` descending.

The horizon determines which weather strategy is used:

| Horizon | Weather source |
|---|---|
| Today | Real current-hour forecast |
| This week (≤ 7 days) | Open-Meteo hourly forecast |
| This month (> 7 days) | No weather — model falls back to mild defaults |

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

Fetched variables (hourly): `temperature_2m`, `relative_humidity_2m`, `precipitation`, `windspeed_10m`.

---

## Tech Stack

| Component | Technology |
|---|---|
| ML language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| ML framework | scikit-learn, LightGBM, XGBoost |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP (TreeSHAP) |
| Weather data | Open-Meteo Historical Archive & Forecast API |
| Model persistence | joblib |
| Visualisation | Matplotlib |
| Backend framework | FastAPI + Uvicorn |
| Backend auth | Supabase + PyJWT (HS256) |
| Backend async HTTP | httpx |
| Database / auth provider | Supabase (PostgreSQL) |
| Frontend framework | React 18 + TypeScript |
| Frontend build | Vite 5 + PWA plugin |
| Frontend styling | TailwindCSS 4 |
| Frontend routing | React Router v6 |
| Frontend animations | Framer Motion |
| Frontend HTTP | Axios (JWT interceptor) |

---

## Recent Changes

### ml_service.py — inference correctness fixes (May 2026)

**`climate_zone` LABEL_MAPS** — the map now accepts both the capitalized forms used
by `event_service` and scrapers at inference time (`"Cold"`, `"Hot"`, `"Moderate"`,
`"Rainy"`) *and* the lowercase forms stored in training data by `generate_foundation.py`
(`"cold"`, `"heat"`, `"moderate"`, `"rain"`). Previously, any lowercase value arriving
from a Supabase row would silently encode to `-1`, corrupting predictions.

**`STRICT_FEATURE_ORDER`** — corrected from 23 features to 19. The `baseline_strict_rf`
model was trained without weather columns (`weather_temp_C`, `weather_humidity`,
`weather_precip_mm`, `weather_wind_speed_kmh`) and without `location`, `climate_zone`,
and `event_month`. The old list incorrectly included all four weather columns, which
would cause a feature matrix dimension mismatch every time `model_name="rf_strict"`
was used.

---

## Version History

| Tag / Branch | Description |
|---|---|
| `archive/old-dataset-and-models` | Last state before modular rebuild: synthetic weather, old `train_ready.csv`, 2-model RF experiment |
| `legacy/old-dataset-pipeline` | Same snapshot as a branch for easy checkout |
| `main` (current) | Modular ML pipeline + FastAPI backend + React frontend; real weather; 5 models; subgroup analysis; SHAP explainability |

---

## License

This project is developed as part of a **Bachelor's Thesis** at ASE Bucharest (CSIE).
All rights reserved by the author. For academic or research use, please cite appropriately.
