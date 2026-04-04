"""
feedback/auth.py — JWT Authentication Utilities
================================================

Provides:
  - Password hashing / verification (bcrypt via passlib)
  - JWT token creation and decoding (python-jose)
  - FastAPI dependency `get_current_user` for protected routes

Environment variables
---------------------
SECRET_KEY      — HMAC signing key (change in production!)
                  Generate with: python -c "import secrets; print(secrets.token_hex(32))"
ACCESS_TOKEN_TTL_MINUTES — Token lifetime in minutes (default: 60)
"""

import os
import hmac
import hashlib
import base64
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .database import get_db, UserDB

# ── Config ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM  = "HS256"
TOKEN_TTL  = int(os.getenv("ACCESS_TOKEN_TTL_MINUTES", "60"))

bearer_scheme = HTTPBearer()


# ── Password helpers (pure stdlib — no passlib needed) ─────────────────────────

def _hash_password(plain: str) -> str:
    """
    Hash a plaintext password with PBKDF2-HMAC-SHA256 + random salt.
    Returns a self-contained string: "pbkdf2:sha256:<iterations>:<salt_hex>:<hash_hex>"

    NOTE: For production consider installing passlib[bcrypt] and replacing
    this with bcrypt (more resistant to GPU cracking). This implementation
    uses 260 000 iterations of PBKDF2 which is NIST-recommended as of 2024.
    """
    salt = os.urandom(16)
    iterations = 260_000
    key = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, iterations)
    return f"pbkdf2:sha256:{iterations}:{salt.hex()}:{key.hex()}"


def _verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verification of a plaintext password against a stored hash."""
    try:
        _, algo, iterations_str, salt_hex, stored_key_hex = hashed.split(":")
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(algo, plain.encode(), salt, iterations)
        return hmac.compare_digest(key.hex(), stored_key_hex)
    except Exception:
        return False


# ── JWT helpers (pure stdlib — no python-jose needed) ──────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT token containing `data` as the payload.
    Expiry defaults to ACCESS_TOKEN_TTL_MINUTES from env.
    """
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=TOKEN_TTL))
    payload.update({"exp": int(expire.timestamp()), "iat": int(datetime.now(timezone.utc).timestamp())})

    header  = _b64url_encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    body    = _b64url_encode(json.dumps(payload).encode())
    signing_input = f"{header}.{body}".encode()
    sig     = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def _decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Raises ValueError on invalid signature or expiry.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token")

        header_b, body_b, sig_b = parts
        signing_input = f"{header_b}.{body_b}".encode()

        # Verify signature
        expected_sig = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, _b64url_decode(sig_b)):
            raise ValueError("Invalid signature")

        payload = json.loads(_b64url_decode(body_b))

        # Check expiry
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise ValueError("Token expired")

        return payload

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise ValueError(str(e))


# ── FastAPI dependencies ────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> UserDB:
    """
    FastAPI dependency that validates the Bearer token and returns the
    authenticated UserDB instance. Raises 401 on any failure.

    Usage:
        @app.get("/protected")
        async def protected(user: UserDB = Depends(get_current_user)):
            return {"hello": user.username}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = _decode_token(credentials.credentials)
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except ValueError:
        raise credentials_exception

    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def get_optional_user(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[UserDB]:
    """
    Like get_current_user but returns None for unauthenticated requests.
    Used on endpoints that allow anonymous access but record the submitter
    if a valid token is provided.
    """
    if credentials is None:
        return None
    try:
        payload  = _decode_token(credentials.credentials)
        username = payload.get("sub")
        if not username:
            return None
        return db.query(UserDB).filter(
            UserDB.username == username,
            UserDB.is_active == True
        ).first()
    except (ValueError, Exception):
        return None


# ── User CRUD helpers (used by auth router) ────────────────────────────────────

def create_user(db: Session, username: str, email: str, password: str) -> UserDB:
    """Insert a new user with a hashed password. Raises ValueError on duplicate."""
    if db.query(UserDB).filter(
        (UserDB.username == username) | (UserDB.email == email)
    ).first():
        raise ValueError("Username or email already registered")

    user = UserDB(
        username  = username,
        email     = email,
        hashed_pw = _hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[UserDB]:
    """Return UserDB if credentials are valid, else None."""
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user and _verify_password(password, user.hashed_pw):
        return user
    return None
