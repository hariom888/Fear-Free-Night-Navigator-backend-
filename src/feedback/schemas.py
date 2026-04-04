"""
feedback/schemas.py — Pydantic models for Road Safety Feedback API
===================================================================

Separation of concerns:
  - *Request* models validate incoming JSON / form data.
  - *Response* models define what is serialised back to the caller.
  - *DB* models (in database.py) are SQLAlchemy ORM classes.

Having all three distinct keeps validation logic, ORM logic,
and API contract decoupled from each other.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Auth ────────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    """Payload for POST /auth/register"""
    username: str = Field(..., min_length=3, max_length=64,
                          description="Unique username (3–64 chars)")
    email:    str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8,
                          description="Minimum 8 characters")


class UserLogin(BaseModel):
    """Payload for POST /auth/login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT access token returned after successful login."""
    access_token: str
    token_type:   str = "bearer"


class UserOut(BaseModel):
    """Safe user representation — never exposes hashed_pw."""
    id:         int
    username:   str
    email:      str
    is_active:  bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Feedback ────────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    """
    Payload for POST /feedback/submit (multipart form, image uploaded separately).

    Location is provided as EITHER lat/lon OR address (or both).
    At least one must be supplied.
    """
    latitude:      Optional[float] = Field(None, ge=-90,  le=90,
                                           description="Decimal latitude")
    longitude:     Optional[float] = Field(None, ge=-180, le=180,
                                           description="Decimal longitude")
    address:       Optional[str]   = Field(None, max_length=512,
                                           description="Human-readable address")
    description:   str             = Field(..., min_length=10, max_length=2000,
                                           description="Describe the road safety issue")
    safety_rating: int             = Field(..., ge=1, le=10,
                                           description="1 = Very unsafe, 10 = Very safe")

    @model_validator(mode="after")
    def require_location(self) -> "FeedbackCreate":
        has_coords  = self.latitude is not None and self.longitude is not None
        has_address = bool(self.address and self.address.strip())
        if not has_coords and not has_address:
            raise ValueError(
                "Provide either (latitude + longitude) or address (or both)."
            )
        return self


class FeedbackOut(BaseModel):
    """Full feedback record returned to API callers."""
    id:            int
    latitude:      Optional[float]
    longitude:     Optional[float]
    address:       Optional[str]
    description:   str
    safety_rating: int
    image_url:     Optional[str]   = Field(None, description="URL to retrieve uploaded image")
    submitted_by:  Optional[str]
    is_resolved:   bool
    created_at:    datetime
    updated_at:    datetime

    model_config = {"from_attributes": True}


class FeedbackUpdate(BaseModel):
    """
    Partial update for PATCH /feedback/{id}.
    All fields are optional — only supplied fields are modified.
    """
    description:   Optional[str] = Field(None, min_length=10, max_length=2000)
    safety_rating: Optional[int] = Field(None, ge=1, le=10)
    address:       Optional[str] = Field(None, max_length=512)
    is_resolved:   Optional[bool] = None


class FeedbackListResponse(BaseModel):
    """Paginated list wrapper."""
    total:   int
    page:    int
    size:    int
    results: List[FeedbackOut]


# ── Filter / Query params ───────────────────────────────────────────────────────

class AreaFilter(BaseModel):
    """
    Bounding-box area filter used by GET /feedback/area.
    All four corners are required for a spatial query.
    """
    min_lat:    float = Field(..., ge=-90,  le=90)
    max_lat:    float = Field(..., ge=-90,  le=90)
    min_lon:    float = Field(..., ge=-180, le=180)
    max_lon:    float = Field(..., ge=-180, le=180)
    min_rating: int   = Field(1,  ge=1, le=10,
                              description="Minimum safety rating to include")
    max_rating: int   = Field(10, ge=1, le=10,
                              description="Maximum safety rating to include")

    @model_validator(mode="after")
    def validate_bbox(self) -> "AreaFilter":
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if self.min_rating > self.max_rating:
            raise ValueError("min_rating must be ≤ max_rating")
        return self
