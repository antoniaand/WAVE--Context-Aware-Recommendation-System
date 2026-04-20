# WAVE Production API - Context-Aware Event Recommender
"""
app/core/security.py
--------------------
JWT token creation, validation, and FastAPI dependency for the current user.

Although Supabase manages auth natively (including JWT issuance), this module
provides:
  - A local JWT decoder to validate Supabase-issued tokens in protected routes
  - An `get_current_user` FastAPI dependency that extracts the user from the
    Authorization: Bearer <token> header
  - Utility helpers for password hashing (used only if you implement a custom
    auth flow alongside Supabase)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

# OAuth2 / Bearer scheme — reads the Authorization header automatically
bearer_scheme = HTTPBearer(auto_error=True)


# ── Token creation (used for locally-issued tokens if needed) ─────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Encode a JWT access token signed with the app's SECRET_KEY.
    Primary use: issuing short-lived tokens after Supabase auth succeeds and
    you want an app-specific token with custom claims.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ── Token decoding ────────────────────────────────────────────────────────────

def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTPException 401 on any validation failure.

    Supports:
      - Locally-issued tokens (signed with SECRET_KEY)
      - Supabase-issued tokens (if SUPABASE_JWT_SECRET is set, it would go here)
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency: current authenticated user ────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency injected into protected routes.

    Extracts the Bearer token from the Authorization header, decodes it,
    and returns the payload dict (contains `sub` = user UUID, plus any
    custom claims added at token creation time).

    Usage in a route:
        @router.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    payload = decode_token(credentials.credentials)
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing 'sub' claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# ── Optional: verify Supabase JWT without a round-trip ───────────────────────

def verify_supabase_token(token: str, supabase_jwt_secret: str) -> dict:
    """
    Verify a token signed by Supabase using the project's JWT secret.
    The SUPABASE_JWT_SECRET can be found in:
        Supabase Dashboard → Project Settings → API → JWT Settings

    This lets you validate tokens locally (no network call to Supabase).
    """
    try:
        payload = jwt.decode(
            token,
            supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Supabase token invalid: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
