#!/usr/bin/env python3
"""
Initialize the SQLite database for the WAVE Streamlit application.

Reads data/processed/app_users.csv (fully preprocessed, weather tolerances
and scenario scores already numeric) and populates db/wave_users.db.

- First 3 rows  → VIP accounts with fixed emails/passwords
- Remaining rows → normal accounts with generated dummy credentials

Plain-text credentials are written to admin_passwords.csv for admin reference.

Run:
    python init_db.py
"""

import sqlite3
import json
import os
import random
import string
import csv

import pandas as pd
from werkzeug.security import generate_password_hash

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ROOT, "db")
DB_PATH = os.path.join(DB_DIR, "wave_users.db")
APP_USERS_CSV = os.path.join(ROOT, "data", "processed", "app_users.csv")
ADMIN_PASSWORDS_CSV = os.path.join(ROOT, "admin_passwords.csv")

os.makedirs(DB_DIR, exist_ok=True)

# ── VIP account definitions (mapped to CSV row indices 0, 1, 2) ───────────────
VIP_ACCOUNTS = [
    {"row_index": 0, "email": "nia@wave.ro",    "password": "nia22004"},
    {"row_index": 1, "email": "steli@wave.ro",  "password": "wave123"},
    {"row_index": 2, "email": "andrei@wave.ro", "password": "wave123"},
]
VIP_INDICES = {v["row_index"] for v in VIP_ACCOUNTS}


def random_password(length: int = 8) -> str:
    """Generate a random alphanumeric password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def main():
    random.seed(42)

    # Load preprocessed users; replace NaN with empty string for clean JSON
    df = pd.read_csv(APP_USERS_CSV, dtype=str)
    df = df.fillna("")

    # Connect and recreate schema from scratch
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS users")
    conn.execute("""
        CREATE TABLE users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            preferences   TEXT
        )
    """)
    conn.commit()

    credentials = []  # collect (email, plain_password) for admin_passwords.csv
    inserted = 0

    for idx, row in df.iterrows():
        # Serialize the entire row as JSON for the preferences column
        prefs_json = json.dumps(row.to_dict(), ensure_ascii=False)

        # Determine email and password based on VIP status
        vip = next((v for v in VIP_ACCOUNTS if v["row_index"] == idx), None)
        if vip:
            email = vip["email"]
            plain_password = vip["password"]
        else:
            user_id = row.get("user_id", str(idx + 1))
            email = f"user_{user_id}@wave.com"
            plain_password = random_password(8)

        hashed = generate_password_hash(plain_password)

        conn.execute(
            "INSERT INTO users (email, password_hash, preferences) VALUES (?, ?, ?)",
            (email, hashed, prefs_json),
        )
        credentials.append({"email": email, "plain_password": plain_password})
        inserted += 1

    conn.commit()
    conn.close()

    # Write plain-text credentials (admin reference only, excluded from git)
    with open(ADMIN_PASSWORDS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["email", "plain_password"])
        writer.writeheader()
        writer.writerows(credentials)

    print(f"Database initialised: {DB_PATH}")
    print(f"Total users inserted : {inserted}")
    print(f"  VIP accounts       : {len(VIP_ACCOUNTS)}")
    print(f"  Normal accounts    : {inserted - len(VIP_ACCOUNTS)}")
    print(f"Credentials saved to : {ADMIN_PASSWORDS_CSV}")


if __name__ == "__main__":
    main()
