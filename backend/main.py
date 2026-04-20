#!/usr/bin/env python3
"""
WAVE Production API - Context-Aware Event Recommender
======================================================
FastAPI application entry point.

Starts the WAVE backend which exposes:
  - /auth   → registration, login, token refresh
  - /recommend → context-aware event recommendations

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

from app.core.config import settings
from app.routers import auth, recommend
from app.services.ml_service import preload_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load ML models on startup, release on shutdown."""
    preload_models()
    yield
    # Cleanup (if needed) goes here

# ── Application factory ───────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="WAVE – Context-Aware Event Recommender API",
    description=(
        "Production API for the WAVE recommendation system. "
        "Combines user preference profiles, real-time weather forecasts, "
        "and trained ML models (LightGBM / XGBoost) to rank events by "
        "predicted attendance probability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,      prefix="/auth",      tags=["Authentication"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {
        "service": "WAVE Production API",
        "status": "ok",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health probe for orchestration layers."""
    return {"status": "healthy", "models_loaded": True}
