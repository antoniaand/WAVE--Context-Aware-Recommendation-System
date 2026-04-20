# WAVE Production API - Context-Aware Event Recommender
"""
app/routers/auth.py
-------------------
Authentication routes using Supabase Auth.

Endpoints:
  POST /auth/register  — create a new Supabase user + profile row
  POST /auth/login     — sign in with email/password, return JWT
  GET  /auth/me        — fetch current user profile (protected)
  POST /auth/logout    — sign out (invalidate Supabase session)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.core.database import get_supabase_client, get_user_profile, upsert_user_profile
from app.core.security import create_access_token, get_current_user
from app.models.recommendation import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description=(
        "Creates a Supabase Auth account and inserts an extended profile row "
        "in the `public.users` table with all weather-tolerance and scenario "
        "preference columns."
    ),
)
async def register(body: RegisterRequest):
    """
    Register a new WAVE user.

    Steps:
      1. Call Supabase Auth sign_up() → returns user UUID
      2. Upsert a profile row in public.users with the full preference profile
      3. Issue a short-lived JWT (using app SECRET_KEY) for immediate use
    """
    client = get_supabase_client()

    try:
        auth_response = client.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
    except Exception as exc:
        logger.error("Supabase sign_up failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {exc}",
        )

    user = auth_response.user
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supabase returned no user object. Check email confirmation settings.",
        )

    user_id = str(user.id)

    # Persist extended profile in public.users
    profile_data = body.profile.model_dump()
    try:
        await upsert_user_profile(user_id, body.email, profile_data)
    except Exception as exc:
        logger.error("Profile upsert failed for user %s: %s", user_id, exc)
        # Non-fatal: user exists in auth but profile row missing
        # They can update it later via PUT /auth/profile

    access_token = create_access_token(
        data={"sub": user_id, "email": body.email}
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=user_id,
    )


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(body: LoginRequest):
    """
    Authenticate with Supabase and return a JWT.

    Supabase validates credentials server-side; we then issue a local JWT
    with the user's UUID as the `sub` claim.
    """
    client = get_supabase_client()

    try:
        auth_response = client.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as exc:
        logger.warning("Login failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_response.user
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user.id)
    access_token = create_access_token(
        data={"sub": user_id, "email": body.email}
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=user_id,
    )


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
)
async def me(current_user: Annotated[dict, Depends(get_current_user)]):
    """
    Return the authenticated user's extended profile from Supabase.

    Requires: Authorization: Bearer <token>
    """
    user_id = current_user["sub"]
    email   = current_user.get("email", "")

    try:
        profile = await get_user_profile(user_id)
    except Exception as exc:
        logger.error("Profile fetch error for %s: %s", user_id, exc)
        profile = None

    return UserProfileResponse(user_id=user_id, email=email, profile=profile)


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout current user",
)
async def logout(current_user: Annotated[dict, Depends(get_current_user)]):
    """
    Sign out the current user from Supabase (invalidates the server-side session).
    The client should also discard their local JWT.
    """
    client = get_supabase_client()
    try:
        client.auth.sign_out()
    except Exception as exc:
        logger.warning("Supabase sign_out error (non-fatal): %s", exc)
    return
