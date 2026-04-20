# WAVE Production API - Context-Aware Event Recommender
"""
app/core/config.py
------------------
Centralised application settings loaded from environment variables / .env file.
Uses Pydantic v2 BaseSettings for automatic validation and type coercion.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable values for the WAVE backend.

    Values are read from environment variables or from the .env file
    located in the backend directory.  Variable names are case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "WAVE Production API"
    ENV: str = "development"          # "development" | "production"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production-please"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",   # Next.js / React dev
        "http://localhost:8501",   # Streamlit dev
        "http://localhost:5173",   # Vite dev
    ]

    # ── Open-Meteo ────────────────────────────────────────────────────────────
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_TIMEOUT_S: int = 15

    # ── ML Models ─────────────────────────────────────────────────────────────
    # Paths are resolved at runtime relative to the backend directory
    MODELS_DIR: str = "../models"
    DATA_DIR: str = "../data/processed"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level alias for convenience: `from app.core.config import settings`
settings: Settings = get_settings()
