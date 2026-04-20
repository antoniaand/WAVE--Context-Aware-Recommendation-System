import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Încărcăm parolele din fișierul .env
load_dotenv()

url = os.environ.get("SUPABASE_URL")
# Folosim cheia ANON sau SERVICE_ROLE pe care o ai în .env
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    print("❌ Eroare: Nu am găsit cheile Supabase în .env!")
    exit()

supabase: Client = create_client(url, key)

# 2. Lista cu evenimentele generate automat
mock_events = [
    {"event_name": "Concert Rock Arena", "event_type": "Concert", "location": "București", "is_outdoor": True, "event_date": "2026-05-15", "event_hour": 20, "climate_zone": "moderate"},
    {"event_name": "Piesă de teatru TNB", "event_type": "Theatre", "location": "București", "is_outdoor": False, "event_date": "2026-05-16", "event_hour": 19, "climate_zone": "moderate"},
    {"event_name": "Derby Fotbal", "event_type": "Sports", "location": "București", "is_outdoor": True, "event_date": "2026-05-20", "event_hour": 21, "climate_zone": "moderate"},
    {"event_name": "Festival de Jazz", "event_type": "Festival", "location": "București", "is_outdoor": True, "event_date": "2026-06-01", "event_hour": 18, "climate_zone": "moderate"},
    {"event_name": "Conferință Tech WAVE", "event_type": "Conference", "location": "București", "is_outdoor": False, "event_date": "2026-05-25", "event_hour": 10, "climate_zone": "moderate"},
    {"event_name": "Simfonia a 9-a (Ateneu)", "event_type": "Concert", "location": "București", "is_outdoor": False, "event_date": "2026-05-18", "event_hour": 19, "climate_zone": "moderate"},
    {"event_name": "Maratonul București", "event_type": "Sports", "location": "București", "is_outdoor": True, "event_date": "2026-06-10", "event_hour": 8, "climate_zone": "moderate"},
    {"event_name": "Street Food Festival", "event_type": "Festival", "location": "București", "is_outdoor": True, "event_date": "2026-05-28", "event_hour": 14, "climate_zone": "moderate"},
    {"event_name": "Teatru de Comedie", "event_type": "Theatre", "location": "București", "is_outdoor": False, "event_date": "2026-05-22", "event_hour": 20, "climate_zone": "moderate"},
    {"event_name": "AI & Future Summit", "event_type": "Conference", "location": "București", "is_outdoor": False, "event_date": "2026-06-05", "event_hour": 9, "climate_zone": "moderate"}
]

# 3. Trimitem datele către Supabase (Bulk Insert)
print(f"Încerc să adaug {len(mock_events)} evenimente în baza de date...")

try:
    response = supabase.table("events").insert(mock_events).execute()
    print("✅ Succes! Toate evenimentele au fost adăugate în Supabase.")
except Exception as e:
    print(f"❌ A apărut o eroare: {e}")