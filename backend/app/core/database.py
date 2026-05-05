# WAVE Production API - Context-Aware Event Recommender
"""
app/core/database.py
--------------------
Supabase client factory and helpers.

The WAVE app uses Supabase for:
  - Authentication (handled by Supabase Auth natively)
  - Storing extended user profiles in the `users` table

SQL DDL for the users table (run once in Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS public.users (
        id                  UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
        email               TEXT        UNIQUE NOT NULL,
        -- Demographic profile
        gender              TEXT,
        age_range           TEXT,
        attendance_freq     TEXT,
        top_event           TEXT,
        preferred_event_types TEXT,
        -- Weather tolerance scores (0-10 scale)
        rain_avoid          INTEGER     DEFAULT 5,
        cold_tolerance      INTEGER     DEFAULT 5,
        heat_sensitivity    INTEGER     DEFAULT 5,
        wind_sensitivity    INTEGER     DEFAULT 5,
        override_weather    INTEGER     DEFAULT 0,
        -- Indoor/outdoor preference (0=indoor, 1=outdoor)
        indoor_outdoor      INTEGER     DEFAULT 0,
        -- Scenario interest scores (0-10 scale)
        scenario_concert    INTEGER     DEFAULT 5,
        scenario_festival   INTEGER     DEFAULT 5,
        scenario_sports     INTEGER     DEFAULT 5,
        scenario_theatre    INTEGER     DEFAULT 5,
        scenario_conference INTEGER     DEFAULT 5,
        -- Timestamps
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        updated_at          TIMESTAMPTZ DEFAULT NOW()
    );

    -- Row-level security: each user can only read/write their own row
    ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
    CREATE POLICY "Users can view own profile"
        ON public.users FOR SELECT
        USING (auth.uid() = id);
    CREATE POLICY "Users can update own profile"
        ON public.users FOR UPDATE
        USING (auth.uid() = id);
"""

from functools import lru_cache
from typing import Optional

from supabase import create_client, Client

from app.core.config import settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Return a cached Supabase client using the ANON key.
    Suitable for operations where the user's JWT is forwarded in the request.
    Thread-safe because Supabase-py clients are stateless per-call.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in the .env file."
        )
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client:
    """
    Return a cached Supabase client using the SERVICE_ROLE key.
    Use ONLY for server-side operations that bypass Row Level Security
    (e.g., creating user profile rows after sign-up).
    Never expose this client or its key to the frontend.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the .env file."
        )
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


async def get_user_profile(user_id: str) -> Optional[dict]:
    """
    Fetch a user's extended profile from the `users` table by auth UUID.
    Returns None if the profile row does not exist yet.
    """
    client = get_supabase_admin_client()
    response = client.table("users").select("*").eq("id", user_id).single().execute()
    return response.data if response.data else None


async def upsert_user_profile(
    user_id: str,
    email: str,
    profile_data: dict | None = None,
    role: str = "user",
) -> dict:
    """
    Insert or update a user's extended profile in the `users` table.
    Profile fields are optional — callers may set only email + role on first insert.
    """
    client = get_supabase_admin_client()
    payload: dict = {"id": user_id, "email": email, "role": role}
    if profile_data:
        payload.update(profile_data)
    response = (
        client.table("users")
        .upsert(payload, on_conflict="id")
        .execute()
    )
    return response.data[0] if response.data else {}
