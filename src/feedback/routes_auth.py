"""
feedback/routes_auth.py — Authentication Routes
================================================

Routes
------
POST /auth/register   — create a new user account
POST /auth/login      — exchange credentials for a JWT access token
GET  /auth/me         — return the currently authenticated user's profile
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .database import get_db, UserDB
from .schemas  import UserRegister, UserLogin, TokenResponse, UserOut
from .auth     import create_user, authenticate_user, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    """
    Create a new user account.

    - Username must be unique (3–64 chars).
    - Email must be unique.
    - Password minimum length is 8 characters (stored as PBKDF2 hash).

    **Example request**
    ```json
    {
      "username": "alice",
      "email": "alice@example.com",
      "password": "securepass123"
    }
    ```
    """
    try:
        user = create_user(db, payload.username, payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT access token",
)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate with username + password and receive a Bearer token.

    Pass the token as `Authorization: Bearer <token>` on protected endpoints.

    **Example request**
    ```json
    { "username": "alice", "password": "securepass123" }
    ```

    **Example response**
    ```json
    { "access_token": "eyJ...", "token_type": "bearer" }
    ```
    """
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Get the currently authenticated user",
)
def me(current_user: UserDB = Depends(get_current_user)):
    """
    Returns the profile of the user identified by the Bearer token.
    Requires a valid JWT in the `Authorization` header.
    """
    return current_user
