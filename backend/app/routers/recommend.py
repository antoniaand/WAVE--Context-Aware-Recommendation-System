# WAVE Production API - Context-Aware Event Recommender
"""
app/routers/recommend.py
------------------------
Recommendation endpoint.

POST /recommend
  - Accepts user_id (Supabase UUID) OR inline user_profile
  - Optional: city, date (YYYY-MM-DD), hour, model, top_n
  - Fetches Open-Meteo weather forecast for the given city/date/hour
  - Runs ML inference via ml_service.predict_attended_probability()
  - Returns JSON list of events sorted by attended_prob descending

Context-aware formula (implemented in ml_service — summarising here for thesis):
  The feature vector fed to the model includes:
    • Static user traits   : gender, age_range, attendance_freq, top_event,
                             indoor_outdoor, scenario_* scores
    • Weather tolerance    : rain_avoid, cold_tolerance, heat_sensitivity,
                             wind_sensitivity, override_weather
    • Event context        : event_type, location, climate_zone, is_outdoor,
                             event_month, event_in_preferred, event_hour
    • Real-time weather    : weather_temp_C, weather_humidity,
                             weather_precip_mm, weather_wind_speed_kmh
  The scaler normalises all numeric features; LabelEncoder maps categories
  to integers using the same alphabetical ordering used at training time.
"""

import logging
from datetime import date as DateType
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.core.database import get_user_profile
from app.models.recommendation import (
    EventRecommendation,
    RecommendRequest,
    RecommendResponse,
    WeatherContext,
)
from app.services.ml_service import predict_attended_probability, preload_models  # noqa: F401
from app.services.weather_service import get_weather_forecast

logger = logging.getLogger(__name__)
router = APIRouter()




# ── POST /recommend ───────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=RecommendResponse,
    summary="Get context-aware event recommendations",
    description=(
        "Returns a ranked list of events with predicted attendance probabilities. "
        "Combines the user's preference profile with Open-Meteo real-time weather "
        "and trained LightGBM / XGBoost models."
    ),
)
async def recommend(
    body: RecommendRequest,
):
    """
    Context-aware event recommendation endpoint.

    Authentication is OPTIONAL — callers may pass an inline user_profile
    for anonymous / demo access, or a user_id to pull the profile from Supabase
    (protected, requires Bearer token).
    """

    # ── 1. Resolve user profile ────────────────────────────────────────────────
    if body.user_id:
        # Pull from Supabase (server-side, bypasses RLS)
        try:
            profile_data = await get_user_profile(body.user_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not fetch user profile from Supabase: {exc}",
            )
        if profile_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No profile found for user_id='{body.user_id}'.",
            )
        user_profile = profile_data

    elif body.user_profile:
        user_profile = body.user_profile.model_dump()

    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either user_id or user_profile must be provided.",
        )

    # ── 2. Resolve city & date ─────────────────────────────────────────────────
    city        = body.city or "Bucharest"
    target_date = body.date or str(DateType.today())
    hour        = body.hour if body.hour is not None else 12
    model_name  = body.model or "lgbm"
    top_n       = body.top_n or 10

    # ── 3. Fetch weather forecast ──────────────────────────────────────────────
    weather_obj  = None
    weather_feat = None

    try:
        weather_obj = await get_weather_forecast(city, target_date, hour)
        if weather_obj:
            weather_feat = weather_obj.to_feature_dict()
            logger.info(
                "Weather for %s on %s@%dh: %.1f°C, %.1fmm rain",
                city, target_date, hour,
                weather_obj.temp_C or 0, weather_obj.precip_mm or 0,
            )
        else:
            logger.warning("No weather data for %s on %s — using defaults.", city, target_date)
    except Exception as exc:
        logger.error("Weather fetch error: %s — using defaults.", exc)

    # ── 4. ML inference ────────────────────────────────────────────────────────
    try:
        raw_results = predict_attended_probability(
            user_profile   = user_profile,
            city           = city,
            target_date    = target_date,
            hour           = hour,
            weather_features = weather_feat,
            model_name     = model_name,
            top_n          = top_n,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML inference error: {exc}",
        )

    # ── 5. Build response ──────────────────────────────────────────────────────
    weather_ctx = None
    if weather_obj:
        weather_ctx = WeatherContext(
            city           = weather_obj.city,
            date           = weather_obj.date,
            temp_C         = weather_obj.temp_C,
            humidity_pct   = weather_obj.humidity_pct,
            precip_mm      = weather_obj.precip_mm,
            wind_speed_kmh = weather_obj.wind_speed_kmh,
        )

    recommendations = [
        EventRecommendation(
            event_type    = r["event_type"],
            location      = r["location"],
            event_date    = r["event_date"],
            attended_prob = r["attended_prob"],
            climate_zone  = r.get("climate_zone"),
            is_outdoor    = r.get("is_outdoor"),
            weather       = weather_ctx,
        )
        for r in raw_results
    ]

    return RecommendResponse(
        user_id         = body.user_id,
        city            = city,
        date            = target_date,
        model_used      = model_name,
        weather         = weather_ctx,
        recommendations = recommendations,
        total_scored    = len(raw_results),
    )


# ── GET /recommend/cities ─────────────────────────────────────────────────────

@router.get(
    "/cities",
    summary="List supported cities",
    response_model=list[str],
)
async def supported_cities():
    """Return the list of cities for which weather forecasts are available."""
    from app.services.weather_service import CITY_COORDS
    return sorted(CITY_COORDS.keys())


# ── GET /recommend/models ─────────────────────────────────────────────────────

@router.get(
    "/models",
    summary="List available ML models",
)
async def available_models():
    """Describe the ML models available for recommendation."""
    return [
        {
            "id":          "lgbm",
            "name":        "LightGBM Contextual",
            "description": "Best overall F1. Trained on all contextual features including weather.",
            "default":     True,
        },
        {
            "id":          "xgb",
            "name":        "XGBoost Contextual",
            "description": "Highest precision. Contextual features + weather.",
            "default":     False,
        },
        {
            "id":          "rf_strict",
            "name":        "Random Forest Strict Baseline",
            "description": "Blind to weather, location and seasonality. Preference-only.",
            "default":     False,
        },
    ]
