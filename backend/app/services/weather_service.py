# WAVE Production API - Context-Aware Event Recommender
"""
app/services/weather_service.py
--------------------------------
Fetches weather FORECAST from the Open-Meteo free forecast API.

Replicates and extends the logic from src/data/fetch_weather_api.py which
used the *historical archive* endpoint.  For live recommendations we need
the *forecast* endpoint (up to 16 days ahead).

Open-Meteo forecast endpoint:
    GET https://api.open-meteo.com/v1/forecast
    Params: latitude, longitude, hourly, timezone, start_date, end_date

No API key required — Open-Meteo is free and open.
"""

import asyncio
import logging
from datetime import date as DateType
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── City coordinates — identical to fetch_weather_api.py ─────────────────────
# Coordinates (WGS-84): latitude, longitude
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Romanian cities
    "Bucharest":   (44.43,  26.10),
    "Cluj-Napoca": (46.77,  23.62),
    "Timisoara":   (45.75,  21.23),
    "Iasi":        (47.16,  27.58),
    "Constanta":   (44.17,  28.63),
    "Brasov":      (45.65,  25.60),
    # Cold hotspots
    "Oslo":        (59.91,  10.75),
    "Helsinki":    (60.17,  24.94),
    "Quebec":      (46.81, -71.21),
    # Heat hotspots
    "Dubai":       (25.20,  55.27),
    "Phoenix":     (33.45, -112.07),
    "Seville":     (37.39,  -5.99),
    # Rain hotspots
    "London":      (51.51,  -0.13),
    "Bergen":      (60.39,   5.33),
    "Seattle":     (47.61, -122.33),
}


# ── Weather data container ────────────────────────────────────────────────────

class WeatherData:
    """Simple weather snapshot for a single (city, date, hour)."""

    def __init__(
        self,
        city: str,
        date: str,
        hour: int,
        temp_C: Optional[float],
        humidity_pct: Optional[float],
        precip_mm: Optional[float],
        wind_speed_kmh: Optional[float],
    ):
        self.city           = city
        self.date           = date
        self.hour           = hour
        self.temp_C         = temp_C
        self.humidity_pct   = humidity_pct
        self.precip_mm      = precip_mm
        self.wind_speed_kmh = wind_speed_kmh

    def to_dict(self) -> dict:
        return {
            "city":           self.city,
            "date":           self.date,
            "temp_C":         self.temp_C,
            "humidity_pct":   self.humidity_pct,
            "precip_mm":      self.precip_mm,
            "wind_speed_kmh": self.wind_speed_kmh,
        }

    def to_feature_dict(self) -> dict:
        """
        Return keys that match the feature names used in training.
        (weather_temp_C, weather_humidity, weather_precip_mm, weather_wind_speed_kmh)
        """
        return {
            "weather_temp_C":         self.temp_C,
            "weather_humidity":       self.humidity_pct,
            "weather_precip_mm":      self.precip_mm,
            "weather_wind_speed_kmh": self.wind_speed_kmh,
        }


# ── Main fetch function ───────────────────────────────────────────────────────

async def get_weather_forecast(
    city: str,
    target_date: str,
    hour: int = 12,
) -> Optional[WeatherData]:
    """
    Fetch hourly weather forecast for a given city, date, and hour.

    Args:
        city:        City name (must be in CITY_COORDS).
        target_date: Date string in 'YYYY-MM-DD' format.
        hour:        Event hour (0-23).  Default is noon.

    Returns:
        WeatherData object if successful, None on failure.

    The function is async and uses httpx for non-blocking HTTP.
    Open-Meteo provides up to 16-day forecasts at no cost.
    """
    if city not in CITY_COORDS:
        logger.warning("City '%s' not in CITY_COORDS — weather unavailable.", city)
        return None

    lat, lon = CITY_COORDS[city]

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m",
        "start_date": target_date,
        "end_date":   target_date,
        "timezone":   "Europe/Bucharest",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.WEATHER_TIMEOUT_S) as client:
            resp = await client.get(settings.OPEN_METEO_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Open-Meteo HTTP error for %s: %s", city, exc)
        return None
    except httpx.RequestError as exc:
        logger.error("Open-Meteo request failed for %s: %s", city, exc)
        return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        logger.warning("Empty forecast payload for %s on %s.", city, target_date)
        return None

    # Find the record matching the requested hour
    target_time = f"{target_date}T{hour:02d}:00"
    try:
        idx = times.index(target_time)
    except ValueError:
        # Fall back to midnight if the exact hour is not in the payload
        logger.warning(
            "Hour %d not found in forecast for %s on %s; using index 0.",
            hour, city, target_date
        )
        idx = 0

    temps    = hourly.get("temperature_2m",      [None] * len(times))
    humidity = hourly.get("relative_humidity_2m",[None] * len(times))
    precip   = hourly.get("precipitation",       [None] * len(times))
    wind     = hourly.get("windspeed_10m",       [None] * len(times))

    return WeatherData(
        city           = city,
        date           = target_date,
        hour           = hour,
        temp_C         = temps[idx],
        humidity_pct   = humidity[idx],
        precip_mm      = precip[idx],
        wind_speed_kmh = wind[idx],
    )


def get_weather_forecast_sync(
    city: str,
    target_date: str,
    hour: int = 12,
) -> Optional[WeatherData]:
    """
    Synchronous wrapper around get_weather_forecast for use outside async context.
    Uses asyncio.run() — do NOT call from inside an already-running event loop.
    """
    return asyncio.run(get_weather_forecast(city, target_date, hour))
