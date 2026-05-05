# WAVE Production API - Context-Aware Event Recommender
"""
app/routers/recommend.py
------------------------
Recommendation endpoint. Supports three time horizons:
  today  — real weather + contextual model (lgbm/xgb)
  week   — 7-day Open-Meteo forecast + contextual model
  month  — no weather (> 7 days ahead), rf_strict baseline
"""

import logging
from datetime import date as DateType, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.core.database import get_user_profile
from app.models.recommendation import (
    EventRecommendation,
    RecommendRequest,
    RecommendResponse,
    WeatherContext,
)
from app.services.event_service import get_events_for_date, get_events_for_range
from app.services.ml_service import predict_attended_probability, preload_models  # noqa: F401
from app.services.weather_service import get_weather_forecast

logger = logging.getLogger(__name__)
router = APIRouter()

# Profile used when a user hasn't filled in preferences yet
_ANON_PROFILE = {
    "gender": "F", "age_range": "25-34", "attendance_freq": "Occasionally",
    "top_event": "Concert", "preferred_event_types": "Concert,Festival,Sports,Theatre,Conference",
    "indoor_outdoor": 0, "rain_avoid": 5, "cold_tolerance": 5,
    "heat_sensitivity": 5, "wind_sensitivity": 5, "override_weather": 0,
    "scenario_concert": 5, "scenario_festival": 5, "scenario_sports": 5,
    "scenario_theatre": 5, "scenario_conference": 5,
}


def _horizon_from_date(target_date: str) -> str:
    today = DateType.today()
    try:
        target = DateType.fromisoformat(target_date)
    except ValueError:
        return "today"
    days_ahead = (target - today).days
    if days_ahead > 7:
        return "month"
    if days_ahead > 0:
        return "week"
    return "today"


# ── POST /recommend ───────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=RecommendResponse,
    summary="Get context-aware event recommendations",
)
async def recommend(body: RecommendRequest):
    # ── 1. Resolve user profile ────────────────────────────────────────────────
    if body.user_id:
        try:
            profile_data = await get_user_profile(body.user_id)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        # Fall back to anonymous defaults if profile not filled yet
        user_profile = profile_data if profile_data and profile_data.get("gender") else _ANON_PROFILE
    elif body.user_profile:
        user_profile = body.user_profile.model_dump()
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Provide user_id or user_profile.")

    # ── 2. Resolve context ─────────────────────────────────────────────────────
    city        = body.city or "Bucharest"
    target_date = body.date or str(DateType.today())
    hour        = body.hour if body.hour is not None else 12
    top_n       = body.top_n or 10
    horizon     = body.horizon or _horizon_from_date(target_date)

    # ── 3. Select model based on horizon ──────────────────────────────────────
    if horizon == "month":
        model_name = "rf_strict"
    else:
        model_name = body.model or "lgbm"

    # ── 4. Fetch events ────────────────────────────────────────────────────────
    if horizon == "week":
        end_date = str(DateType.fromisoformat(target_date) + timedelta(days=7))
        events = await get_events_for_range(city, target_date, end_date, hour)
    elif horizon == "month":
        end_date = str(DateType.fromisoformat(target_date) + timedelta(days=30))
        events = await get_events_for_range(city, target_date, end_date, hour)
    else:
        events = await get_events_for_date(city, target_date, hour)

    # ── 5. Fetch weather (today/week only) ────────────────────────────────────
    weather_obj  = None
    weather_feat = None

    if horizon != "month":
        try:
            weather_obj = await get_weather_forecast(city, target_date, hour)
            if weather_obj:
                weather_feat = weather_obj.to_feature_dict()
        except Exception as exc:
            logger.error("Weather fetch error: %s — using defaults.", exc)

    # ── 6. ML inference ────────────────────────────────────────────────────────
    try:
        raw_results = predict_attended_probability(
            user_profile    = user_profile,
            events          = events,
            weather_features = weather_feat,
            model_name      = model_name,
            top_n           = top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # ── 7. Build response ──────────────────────────────────────────────────────
    weather_ctx = None
    if weather_obj:
        weather_ctx = WeatherContext(
            city=weather_obj.city, date=weather_obj.date,
            temp_C=weather_obj.temp_C, humidity_pct=weather_obj.humidity_pct,
            precip_mm=weather_obj.precip_mm, wind_speed_kmh=weather_obj.wind_speed_kmh,
        )

    recommendations = [
        EventRecommendation(
            event_type   = r["event_type"],
            event_name   = r.get("event_name"),
            location     = r["location"],
            venue        = r.get("venue"),
            event_date   = r["event_date"],
            attended_prob = r["attended_prob"],
            climate_zone = r.get("climate_zone"),
            is_outdoor   = r.get("is_outdoor"),
            source       = r.get("source"),
            is_generated = r.get("is_generated", False),
            url          = r.get("url"),
            image_url    = r.get("image_url"),
            description  = r.get("description"),
            weather      = weather_ctx,
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

@router.get("/cities", response_model=list[str], summary="List supported cities")
async def supported_cities():
    from app.services.weather_service import CITY_COORDS
    return sorted(CITY_COORDS.keys())


# ── GET /recommend/models ─────────────────────────────────────────────────────

@router.get("/models", summary="List available ML models")
async def available_models():
    return [
        {"id": "lgbm",      "name": "LightGBM Contextual",          "description": "Best overall F1. Uses real-time weather + user profile.", "default": True},
        {"id": "xgb",       "name": "XGBoost Contextual",           "description": "Highest precision. Contextual features + weather.",       "default": False},
        {"id": "rf_strict", "name": "Random Forest Strict Baseline", "description": "Preference-only. No weather, location or seasonality.",   "default": False},
    ]
