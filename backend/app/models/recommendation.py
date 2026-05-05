# WAVE Production API - Context-Aware Event Recommender
"""
app/models/recommendation.py
-----------------------------
Pydantic v2 schemas for request / response bodies.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ── User Profile ──────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Full user preference profile used as ML feature input."""

    gender:              str   = Field(..., examples=["F"])
    age_range:           str   = Field(..., examples=["18-24"])
    attendance_freq:     str   = Field(..., examples=["Often"])
    top_event:           str   = Field(..., examples=["Concert"])
    preferred_event_types: str = Field(..., examples=["Concert,Festival"])
    indoor_outdoor:      int   = Field(..., ge=0, le=1)
    rain_avoid:          int   = Field(..., ge=0, le=10)
    cold_tolerance:      int   = Field(..., ge=0, le=10)
    heat_sensitivity:    int   = Field(..., ge=0, le=10)
    wind_sensitivity:    int   = Field(..., ge=0, le=10)
    override_weather:    int   = Field(..., ge=0, le=1)
    scenario_concert:    int   = Field(..., ge=0, le=10)
    scenario_festival:   int   = Field(..., ge=0, le=10)
    scenario_sports:     int   = Field(..., ge=0, le=10)
    scenario_theatre:    int   = Field(..., ge=0, le=10)
    scenario_conference: int   = Field(..., ge=0, le=10)


# ── Registration / Auth ───────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Payload for POST /auth/register. Profile is optional — set via PUT /auth/profile."""
    email:    str = Field(..., examples=["user@example.com"])
    password: str = Field(..., min_length=6)
    profile:  Optional[UserProfile] = None


class LoginRequest(BaseModel):
    """Payload for POST /auth/login."""
    email:    str
    password: str


class TokenResponse(BaseModel):
    """JWT token response returned after successful auth."""
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int
    user_id:      str


# ── Recommendation request ────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """Payload for POST /recommend."""
    user_id:      Optional[str]         = Field(None)
    user_profile: Optional[UserProfile] = Field(None)

    city:    Optional[str] = Field(None, examples=["Bucharest"])
    date:    Optional[str] = Field(None, examples=["2026-05-10"])
    hour:    Optional[int] = Field(12, ge=0, le=23)
    top_n:   int           = Field(10, ge=1, le=50)
    model:   str           = Field("lgbm")
    horizon: Optional[str] = Field(None, description="'today' | 'week' | 'month'")

    @model_validator(mode="after")
    def must_have_user_source(self) -> "RecommendRequest":
        if self.user_id is None and self.user_profile is None:
            raise ValueError("Provide either 'user_id' or 'user_profile'.")
        return self


# ── Event result ──────────────────────────────────────────────────────────────

class WeatherContext(BaseModel):
    city:           str
    date:           str
    temp_C:         Optional[float] = None
    humidity_pct:   Optional[float] = None
    precip_mm:      Optional[float] = None
    wind_speed_kmh: Optional[float] = None


class EventRecommendation(BaseModel):
    """Single ranked event recommendation."""
    event_type:    str
    event_name:    Optional[str]  = None
    location:      str
    venue:         Optional[str]  = None
    event_date:    str
    attended_prob: float = Field(..., description="Predicted attendance probability (0–1)")
    climate_zone:  Optional[str]  = None
    is_outdoor:    Optional[int]  = None
    source:        Optional[str]  = None
    is_generated:  bool           = False
    url:           Optional[str]  = None
    image_url:     Optional[str]  = None
    description:   Optional[str]  = None
    weather:       Optional[WeatherContext] = None


class RecommendResponse(BaseModel):
    """Full response payload from POST /recommend."""
    user_id:         Optional[str]
    city:            Optional[str]
    date:            Optional[str]
    model_used:      str
    weather:         Optional[WeatherContext]
    recommendations: List[EventRecommendation]
    total_scored:    int


# ── User profile response ─────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    """Public-safe user profile returned by GET /auth/me."""
    user_id: str
    email:   str
    role:    str = "user"
    profile: Optional[dict] = None
