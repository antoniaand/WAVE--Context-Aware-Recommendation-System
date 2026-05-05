# WAVE Production API - Context-Aware Event Recommender
"""
app/routers/auth.py
-------------------
Authentication routes using Supabase Auth.

  POST /auth/register  — create Supabase user + role-assigned profile row
  POST /auth/login     — sign in, return JWT with role claim
  GET  /auth/me        — current user + profile (protected)
  PUT  /auth/profile   — update preference profile (protected)
  POST /auth/logout    — invalidate Supabase session (protected)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.core.database import get_supabase_client, get_supabase_admin_client, get_user_profile, upsert_user_profile
from app.core.security import create_access_token, get_current_user
from app.models.recommendation import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserProfile,
    UserProfileResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _role_for_email(email: str) -> str:
    domain = email.split("@")[-1].lower()
    if domain == settings.ADMIN_EMAIL_DOMAIN.lower():
        return "admin"
    if domain == settings.MANAGER_EMAIL_DOMAIN.lower():
        return "event_manager"
    return "user"


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(body: RegisterRequest):
    admin = get_supabase_admin_client()

    try:
        auth_response = admin.auth.admin.create_user({
            "email": body.email,
            "password": body.password,
            "email_confirm": True,
        })
    except Exception as exc:
        logger.error("Supabase create_user failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration failed: {exc}")

    user = auth_response.user
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Supabase returned no user object.")

    user_id = str(user.id)
    role = _role_for_email(body.email)
    profile_data = body.profile.model_dump() if body.profile else None

    try:
        await upsert_user_profile(user_id, body.email, profile_data, role)
    except Exception as exc:
        logger.error("Profile upsert failed for %s: %s", user_id, exc)

    access_token = create_access_token(data={"sub": user_id, "email": body.email, "role": role})
    return TokenResponse(access_token=access_token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, user_id=user_id)


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse, summary="Login with email and password")
async def login(body: LoginRequest):
    client = get_supabase_client()

    try:
        auth_response = client.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception as exc:
        logger.warning("Login failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_response.user
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed.", headers={"WWW-Authenticate": "Bearer"})

    user_id = str(user.id)

    # Fetch role from users table (set at registration)
    try:
        profile = await get_user_profile(user_id)
        role = profile.get("role", "user") if profile else "user"
    except Exception:
        role = "user"

    access_token = create_access_token(data={"sub": user_id, "email": body.email, "role": role})
    return TokenResponse(access_token=access_token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, user_id=user_id)


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserProfileResponse, summary="Get current user profile")
async def me(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["sub"]
    email   = current_user.get("email", "")
    role    = current_user.get("role", "user")

    try:
        profile = await get_user_profile(user_id)
    except Exception as exc:
        logger.error("Profile fetch error for %s: %s", user_id, exc)
        profile = None

    return UserProfileResponse(user_id=user_id, email=email, role=role, profile=profile)


# ── PUT /auth/profile ─────────────────────────────────────────────────────────

@router.put("/profile", status_code=status.HTTP_200_OK, summary="Update user preference profile")
async def update_profile(
    profile: UserProfile,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["sub"]
    email   = current_user.get("email", "")
    role    = current_user.get("role", "user")

    try:
        await upsert_user_profile(user_id, email, profile.model_dump(), role)
    except Exception as exc:
        logger.error("Profile update failed for %s: %s", user_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save profile.")

    return {"message": "Profile updated successfully."}


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Logout current user")
async def logout(current_user: Annotated[dict, Depends(get_current_user)]):
    client = get_supabase_client()
    try:
        client.auth.sign_out()
    except Exception as exc:
        logger.warning("Supabase sign_out error (non-fatal): %s", exc)
    return
