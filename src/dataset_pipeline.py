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

    # --- Standardization ---
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

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


def process_real_users(filepath: Path) -> pd.DataFrame:
    """
    Process survey users for Streamlit app.
    Output: app_users.csv (no merge with training data).
    """
    df = pd.read_csv(filepath, dtype=str, engine="python")

    # --- Data Cleaning ---
    df = df.dropna(how="all")
    df = df.dropna(subset=[df.columns[0]])  # keep rows with timestamp/consent

    # --- Map column names (fuzzy) ---
    cols_lower = {c.lower(): c for c in df.columns}
    gender_col = next((cols_lower[k] for k in cols_lower if "gender" in k), None)
    age_col = next((cols_lower[k] for k in cols_lower if "age" in k and "range" in k), None)
    freq_col = next((cols_lower[k] for k in cols_lower if "often" in k or "des participi" in k), None)
    pref_col = next((cols_lower[k] for k in cols_lower if "types of events" in k or "tipuri" in k), None)

    # --- Standardize preferred event types to match training categories ---
    def map_preferences(val):
        if pd.isna(val):
            return ""
        s = str(val)
        mapped = []
        for survey_key, canonical in EVENT_TYPE_MAP.items():
            if survey_key in s:
                mapped.append(canonical)
        return "|".join(sorted(set(mapped))) if mapped else ""

    df["user_id"] = range(1, len(df) + 1)
    df["gender"] = df[gender_col] if gender_col else ""
    df["age_range"] = df[age_col] if age_col else ""
    df["attendance_freq"] = df[freq_col] if freq_col else ""
    df["preferred_event_types"] = df[pref_col].apply(map_preferences) if pref_col else ""

    # --- Drop nulls in key fields ---
    df = df.dropna(subset=["gender", "age_range"], how="all")

    out_cols = ["user_id", "gender", "age_range", "attendance_freq", "preferred_event_types"]
    out = df[[c for c in out_cols if c in df.columns]].copy()

    out_path = PROCESSED_DIR / "app_users.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved app_users.csv: {len(out)} rows")
    return out


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
