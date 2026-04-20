# WAVE Production API - Context-Aware Event Recommender
"""
app/models/recommendation.py
-----------------------------
Pydantic v2 schemas for request / response bodies.

Keeping schemas in one file avoids circular imports and makes it trivial to
audit what data the API accepts and returns.
"""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ── User Profile ──────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """
    Full user preference profile used as ML feature input.

    Column mapping mirrors the training dataset schema exactly
    (see data/processed/train_ready_interactions.csv).
    """

    # Demographic
    gender:              str   = Field(..., examples=["F"])
    age_range:           str   = Field(..., examples=["18-24"])
    attendance_freq:     str   = Field(..., examples=["Often"])
    top_event:           str   = Field(..., examples=["Concert"])

    # Preference flags
    preferred_event_types: str  = Field(..., examples=["Concert,Festival"])
    indoor_outdoor:        int  = Field(..., ge=0, le=1, description="0=indoor preference, 1=outdoor")

    # Weather tolerance (0–10 scale; higher = more tolerant / sensitive)
    rain_avoid:       int = Field(..., ge=0, le=10, description="How much the user avoids rain (0=never, 10=always)")
    cold_tolerance:   int = Field(..., ge=0, le=10, description="Cold weather tolerance (0=very low, 10=very high)")
    heat_sensitivity: int = Field(..., ge=0, le=10, description="Heat sensitivity (0=not sensitive, 10=very sensitive)")
    wind_sensitivity: int = Field(..., ge=0, le=10, description="Wind sensitivity (0=not sensitive, 10=very sensitive)")
    override_weather: int = Field(..., ge=0, le=1,  description="1 if user ignores weather for event attendance")

    # Scenario interest scores (0–10)
    scenario_concert:    int = Field(..., ge=0, le=10)
    scenario_festival:   int = Field(..., ge=0, le=10)
    scenario_sports:     int = Field(..., ge=0, le=10)
    scenario_theatre:    int = Field(..., ge=0, le=10)
    scenario_conference: int = Field(..., ge=0, le=10)


# ── Registration ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Payload for POST /auth/register."""
    email:    str = Field(..., examples=["user@example.com"])
    password: str = Field(..., min_length=6)
    profile:  UserProfile


class LoginRequest(BaseModel):
    """Payload for POST /auth/login."""
    email:    str
    password: str


class TokenResponse(BaseModel):
    """JWT token response returned after successful auth."""
    access_token:  str
    token_type:    str = "bearer"
    expires_in:    int           # seconds
    user_id:       str


# ── Recommendation request ────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Payload for POST /recommend.

    Either provide a `user_id` (fetches profile from Supabase) or pass an
    inline `user_profile` directly (useful for anonymous / demo calls).
    """
    user_id:      Optional[str]         = Field(None, description="Supabase auth user UUID")
    user_profile: Optional[UserProfile] = Field(None, description="Inline profile (if user_id not given)")

    # Context
    city:  Optional[str]  = Field(None, examples=["Bucharest"])
    date:  Optional[str]  = Field(None, examples=["2025-07-20"], description="YYYY-MM-DD; defaults to today")
    hour:  Optional[int]  = Field(12, ge=0, le=23, description="Hour of event (0-23)")
    top_n: int            = Field(10, ge=1, le=50, description="Number of recommendations to return")

    # Model selection
    model: str = Field(
        "lgbm",
        description="Which model to use: 'lgbm' (default), 'xgb', or 'rf_strict'",
    )

    @model_validator(mode="after")
    def must_have_user_source(self) -> "RecommendRequest":
        if self.user_id is None and self.user_profile is None:
            raise ValueError("Provide either 'user_id' or 'user_profile'.")
        return self


# ── Event result ──────────────────────────────────────────────────────────────

class WeatherContext(BaseModel):
    """Snapshot of forecast weather used during scoring."""
    city:            str
    date:            str
    temp_C:          Optional[float]
    humidity_pct:    Optional[float]
    precip_mm:       Optional[float]
    wind_speed_kmh:  Optional[float]


class EventRecommendation(BaseModel):
    """Single ranked event recommendation."""
    event_type:       str
    location:         str
    event_date:       str
    attended_prob:    float = Field(..., description="Predicted attendance probability (0–1)")
    climate_zone:     Optional[str] = None
    is_outdoor:       Optional[int] = None
    weather:          Optional[WeatherContext] = None


class RecommendResponse(BaseModel):
    """Full response payload from POST /recommend."""
    user_id:        Optional[str]
    city:           Optional[str]
    date:           Optional[str]
    model_used:     str
    weather:        Optional[WeatherContext]
    recommendations: List[EventRecommendation]
    total_scored:   int


# ── User profile response ─────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    """Public-safe user profile returned by GET /auth/me."""
    user_id:       str
    email:         str
    profile:       Optional[dict] = None
