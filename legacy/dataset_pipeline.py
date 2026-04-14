#!/usr/bin/env python3
"""
WAVE Dataset Pipeline – Context-Aware Recommender System.

Two separate datasets, two processing paths (no merge):
- Kaggle dataset → train_ready.csv (model training)
- Users survey → app_users.csv (Streamlit app users)
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: survey event types → canonical categories (aligned with training)
EVENT_TYPE_MAP = {
    "Concerte": "Concert",
    "Concerts": "Concert",
    "Festivaluri": "Festival",
    "Festivals": "Festival",
    "Teatru": "Theatre",
    "Theatre": "Theatre",
    "Conferințe": "Conference",
    "Conferences": "Conference",
    "Expoziții": "Exhibition",
    "Exhibitions": "Exhibition",
    "Evenimente sportive": "Sports",
    "Sports events": "Sports",
}


def process_training_data(filepath: Path) -> pd.DataFrame:
    """
    Process Kaggle event attendance data for model training.
    Output: train_ready.csv
    """
    df = pd.read_csv(filepath, dtype=str)

    # --- Data Cleaning ---
    df = df.dropna(subset=["event_id", "event_name", "date_time"])
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df = df.dropna(subset=["event_id"])
    df["event_datetime"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["event_datetime"])
    df["location"] = df["location"].fillna("Unknown")

    # --- Context features (temporal; weather placeholders for later API) ---
    df["event_hour"] = df["event_datetime"].dt.hour
    df["event_weekday"] = df["event_datetime"].dt.weekday
    df["event_month"] = df["event_datetime"].dt.month
    df["season"] = df["event_month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["weather_temp_C"] = pd.NA  # placeholder for weather API
    df["weather_precip_mm"] = pd.NA

    # --- Target: attended = 1 (implicit positive) ---
    df["attended"] = 1

    # --- Label Encoding ---
    le_event = LabelEncoder()
    le_location = LabelEncoder()
    df["event_name_enc"] = le_event.fit_transform(df["event_name"].astype(str))
    df["location_enc"] = le_location.fit_transform(df["location"].astype(str))

    # --- Numeric columns for scaling ---
    numeric_cols = ["event_hour", "event_weekday", "event_month", "season", "event_name_enc", "location_enc"]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


    # --- Output columns ---
    out_cols = [
        "event_id", "event_name", "location", "event_datetime",
        "event_hour", "event_weekday", "event_month", "season",
        "event_name_enc", "location_enc", "weather_temp_C", "weather_precip_mm",
        "attended"
    ]
    out = df[[c for c in out_cols if c in df.columns]].copy()

    out_path = PROCESSED_DIR / "train_ready.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved train_ready.csv: {len(out)} rows")
    return out


def process_real_users(filepath) -> pd.DataFrame:
    """
    Process survey users for Streamlit app & SQLite Database.
    Output: app_users.csv containing clean weather tolerances and mapped event types.
    """
    df = pd.read_csv(filepath, dtype=str, engine="python")

    # 1. Pastram doar randurile valide
    df = df.dropna(how="all")
    df = df.dropna(subset=[df.columns[0]])  

    # 2. Redenumim TOATE coloanele exact cum aveai in scriptul vechi
    df.columns = [
        "timestamp", "consent", "gender", "age_range", "attendance_freq",
        "event_types", "indoor_outdoor", "top_event", "rain_avoid",
        "cold_tolerance", "heat_sensitivity", "wind_sensitivity",
        "override_weather", "scenario_concert", "scenario_festival",
        "scenario_sports", "scenario_theatre", "scenario_conference"
    ]

    # 3. Adaugam ID-ul de utilizator
    df["user_id"] = range(1, len(df) + 1)

    # 4. Magia din prima functie: Mapam preferintele de evenimente
    def map_preferences(val):
        if pd.isna(val):
            return ""
        s = str(val)
        mapped = []
        # Asigura-te ca EVENT_TYPE_MAP este definit mai sus in scriptul tau!
        for survey_key, canonical in EVENT_TYPE_MAP.items():
            if survey_key in s:
                mapped.append(canonical)
        return "|".join(sorted(set(mapped))) if mapped else ""

    df["preferred_event_types"] = df["event_types"].apply(map_preferences)

    # 5. Magia din a doua functie: Mapam scenariile in cifre (0-3)
    mapping = {
        "Aș participa / I would attend": 3,
        "Probabil aș participa / Probably": 2,
        "Probabil nu / Probably not": 1,
        "Nu aș participa / Would not attend": 0
    }
    
    scenario_columns = [
        "scenario_concert", "scenario_festival", "scenario_sports",
        "scenario_theatre", "scenario_conference"
    ]
    for col in scenario_columns:
        df[col] = df[col].map(mapping)

    # 6. Curatam si salvam
    df = df.drop(columns=["timestamp", "consent"])
    df = df.fillna("") # Evitam erorile de JSON

    # Salvam fisierul final (pastram numele app_users.csv ca sa nu stricam restul codului tau)
    out_path = PROCESSED_DIR / "app_users.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved app_users.csv: {len(df)} rows with full weather profiles!")
    
    return df


def main():
    attendance_path = RAW_DIR / "event_attendance.csv"
    users_path = RAW_DIR / "users_110.csv"

    if attendance_path.exists():
        process_training_data(attendance_path)
    else:
        print(f"Missing: {attendance_path}")

    if users_path.exists():
        process_real_users(users_path)
    else:
        print(f"Missing: {users_path}")

if __name__ == "__main__":
    main()
