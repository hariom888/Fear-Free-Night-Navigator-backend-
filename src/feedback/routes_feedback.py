"""
feedback/routes_feedback.py — Road Safety Feedback Routes
==========================================================

Routes
------
POST   /feedback/submit           — Submit new feedback (with optional image)
GET    /feedback/{id}             — Get a single feedback record
GET    /feedback/list             — Paginated list with rating filter
GET    /feedback/area             — Feedback within a bounding box
GET    /feedback/filter           — Filter by safety rating range
GET    /feedback/area/stats       — Aggregate statistics for an area
PATCH  /feedback/{id}             — Partial update (owner or any auth user)
DELETE /feedback/{id}             — Delete feedback + its image (auth required)
GET    /feedback/image/{filename} — Retrieve an uploaded image file
"""

import json
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException,
    Query, UploadFile, status
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database      import get_db, UserDB
from .schemas       import (
    FeedbackCreate, FeedbackOut, FeedbackUpdate,
    FeedbackListResponse, AreaFilter,
)
from .auth          import get_current_user, get_optional_user
from .image_handler import save_image, delete_image, build_image_url, get_image_path
from . import crud

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ── Helper ──────────────────────────────────────────────────────────────────────

def _enrich(record, base_url: str = "") -> FeedbackOut:
    """
    Convert a FeedbackDB ORM record to FeedbackOut, injecting the image URL.
    """
    out = FeedbackOut.model_validate(record)
    out.image_url = build_image_url(record.image_path)
    return out


# ── Submit ──────────────────────────────────────────────────────────────────────

@router.post(
    "/submit",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="Submit road safety feedback",
)
async def submit_feedback(
    # ── Form fields (multipart so image can be included in same request) ──
    latitude:      Optional[float] = Form(None,  description="Decimal latitude"),
    longitude:     Optional[float] = Form(None,  description="Decimal longitude"),
    address:       Optional[str]   = Form(None,  description="Human-readable address"),
    description:   str             = Form(...,   description="Describe the safety issue (min 10 chars)"),
    safety_rating: int             = Form(...,   description="1 (Unsafe) to 10 (Safe)"),
    # ── Optional image ────────────────────────────────────────────────────
    image: Optional[UploadFile] = File(None, description="JPEG / PNG / WebP, max 10 MB"),
    # ── Dependencies ──────────────────────────────────────────────────────
    db:           Session            = Depends(get_db),
    current_user: Optional[UserDB]   = Depends(get_optional_user),
):
    """
    Submit a new road safety feedback entry.

    - Authentication is **optional**. If a valid Bearer token is provided the
      submission is linked to the user's account; otherwise it is recorded as
      anonymous.
    - Location must be supplied as **lat + lon**, **address**, or **both**.
    - `safety_rating` is an integer slider from **1** (Very Unsafe) to **10** (Very Safe).
    - Attach an image via `image` form field (JPEG / PNG / WebP, ≤ 10 MB).

    **Example cURL**
    ```bash
    curl -X POST http://localhost:8000/feedback/submit \\
      -F "latitude=12.9716" -F "longitude=77.5946" \\
      -F "description=Large pothole at the intersection near KR Circle" \\
      -F "safety_rating=3" \\
      -F "image=@/path/to/photo.jpg"
    ```
    """
    # Validate via Pydantic schema
    try:
        payload = FeedbackCreate(
            latitude=latitude,
            longitude=longitude,
            address=address,
            description=description,
            safety_rating=safety_rating,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc))

    # Save image if provided
    image_path: Optional[str] = None
    if image and image.filename:
        image_path = await save_image(image)

    submitted_by = current_user.username if current_user else None
    record = crud.create_feedback(db, payload, image_path, submitted_by)
    return _enrich(record)


# ── Get single ──────────────────────────────────────────────────────────────────

@router.get(
    "/{feedback_id}",
    response_model=FeedbackOut,
    summary="Get a feedback record by ID",
)
def get_feedback(feedback_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single feedback record by its numeric ID.

    **Example response**
    ```json
    {
      "id": 1,
      "latitude": 12.9716,
      "longitude": 77.5946,
      "address": null,
      "description": "Large pothole near KR Circle",
      "safety_rating": 3,
      "image_url": "http://localhost:8000/feedback/image/abc123.jpg",
      "submitted_by": "alice",
      "is_resolved": false,
      "created_at": "2025-06-01T10:30:00+00:00",
      "updated_at": "2025-06-01T10:30:00+00:00"
    }
    ```
    """
    record = crud.get_feedback_by_id(db, feedback_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Feedback id={feedback_id} not found")
    return _enrich(record)


# ── List (paginated) ────────────────────────────────────────────────────────────

@router.get(
    "/list",
    response_model=FeedbackListResponse,
    summary="List all feedback with optional rating filter",
)
def list_all_feedback(
    page:       int = Query(1,  ge=1,  description="Page number (1-based)"),
    size:       int = Query(20, ge=1, le=100, description="Records per page"),
    min_rating: int = Query(1,  ge=1, le=10,  description="Minimum safety rating"),
    max_rating: int = Query(10, ge=1, le=10,  description="Maximum safety rating"),
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of feedback entries sorted by newest first.

    Use `min_rating` / `max_rating` to surface dangerous roads:
    - `min_rating=1&max_rating=3` — very unsafe roads
    - `min_rating=8&max_rating=10` — safe roads
    """
    total, records = crud.list_feedback(db, page, size, min_rating, max_rating)
    return FeedbackListResponse(
        total=total, page=page, size=size,
        results=[_enrich(r) for r in records],
    )


# ── Area filter ─────────────────────────────────────────────────────────────────

@router.get(
    "/area",
    response_model=FeedbackListResponse,
    summary="Fetch feedback for a specific geographic area",
)
def feedback_by_area(
    min_lat:    float = Query(..., ge=-90,  le=90),
    max_lat:    float = Query(..., ge=-90,  le=90),
    min_lon:    float = Query(..., ge=-180, le=180),
    max_lon:    float = Query(..., ge=-180, le=180),
    min_rating: int   = Query(1,  ge=1, le=10),
    max_rating: int   = Query(10, ge=1, le=10),
    page:       int   = Query(1,  ge=1),
    size:       int   = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    Return all feedback whose lat/lon falls within the supplied bounding box.

    Results are sorted by **safety_rating ascending** (most unsafe first) so
    that a frontend heatmap can render high-risk areas prominently.

    **Example — Bengaluru city centre**
    ```
    GET /feedback/area?min_lat=12.95&max_lat=13.00&min_lon=77.57&max_lon=77.62
    ```
    """
    # Basic sanity check
    if min_lat >= max_lat or min_lon >= max_lon:
        raise HTTPException(400, "Invalid bounding box: min must be less than max")
    if min_rating > max_rating:
        raise HTTPException(400, "min_rating must be ≤ max_rating")

    total, records = crud.list_feedback_by_area(
        db, min_lat, max_lat, min_lon, max_lon,
        min_rating, max_rating, page, size,
    )
    return FeedbackListResponse(
        total=total, page=page, size=size,
        results=[_enrich(r) for r in records],
    )


# ── Safety rating filter ────────────────────────────────────────────────────────

@router.get(
    "/filter",
    response_model=FeedbackListResponse,
    summary="Filter roads by safety rating range",
)
def filter_by_rating(
    min_rating: int = Query(1,  ge=1, le=10, description="Lower bound (inclusive)"),
    max_rating: int = Query(10, ge=1, le=10, description="Upper bound (inclusive)"),
    page:       int = Query(1,  ge=1),
    size:       int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Filter feedback by safety rating slider range.

    | Range     | Meaning               |
    |-----------|-----------------------|
    | 1 – 3     | Very unsafe           |
    | 4 – 6     | Moderate              |
    | 7 – 10    | Safe                  |

    **Example — fetch the most dangerous reports**
    ```
    GET /feedback/filter?min_rating=1&max_rating=3
    ```
    """
    total, records = crud.filter_by_safety_rating(db, min_rating, max_rating, page, size)
    return FeedbackListResponse(
        total=total, page=page, size=size,
        results=[_enrich(r) for r in records],
    )


# ── Area statistics ─────────────────────────────────────────────────────────────

@router.get(
    "/area/stats",
    summary="Aggregate statistics for a geographic area",
)
def area_statistics(
    min_lat: float = Query(..., ge=-90,  le=90),
    max_lat: float = Query(..., ge=-90,  le=90),
    min_lon: float = Query(..., ge=-180, le=180),
    max_lon: float = Query(..., ge=-180, le=180),
    db: Session = Depends(get_db),
):
    """
    Returns aggregate stats for feedback within a bounding box:
    - total_reports
    - average_rating
    - min_rating / max_rating

    **Example response**
    ```json
    {
      "total_reports": 42,
      "average_rating": 4.1,
      "min_rating": 1,
      "max_rating": 9
    }
    ```
    """
    return crud.get_area_statistics(db, min_lat, max_lat, min_lon, max_lon)


# ── Update ──────────────────────────────────────────────────────────────────────

@router.patch(
    "/{feedback_id}",
    response_model=FeedbackOut,
    summary="Partially update a feedback record (auth required)",
)
def update_feedback(
    feedback_id: int,
    data:        FeedbackUpdate,
    db:          Session = Depends(get_db),
    _user:       UserDB  = Depends(get_current_user),   # auth enforced
):
    """
    Partially update description, safety_rating, address, or resolved status.
    Only fields included in the request body are modified.
    Requires a valid Bearer token.
    """
    record = crud.update_feedback(db, feedback_id, data)
    if not record:
        raise HTTPException(404, f"Feedback id={feedback_id} not found")
    return _enrich(record)


# ── Delete ──────────────────────────────────────────────────────────────────────

@router.delete(
    "/{feedback_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a feedback record and its image (auth required)",
)
def delete_feedback_entry(
    feedback_id: int,
    db:    Session = Depends(get_db),
    _user: UserDB  = Depends(get_current_user),
):
    """
    Permanently delete a feedback record and remove its associated image file.
    Requires a valid Bearer token.
    """
    image_path = crud.delete_feedback(db, feedback_id)
    if image_path is None:
        raise HTTPException(404, f"Feedback id={feedback_id} not found")
    delete_image(image_path)   # remove file from disk (no-op if path is empty)


# ── Image retrieval ─────────────────────────────────────────────────────────────

@router.get(
    "/image/{filename}",
    summary="Retrieve an uploaded image",
    response_class=FileResponse,
    tags=["feedback", "images"],
)
def get_image(filename: str):
    """
    Serve a previously uploaded image by its stored filename.
    The URL is returned in the `image_url` field of every FeedbackOut response.

    **Example**
    ```
    GET /feedback/image/a1b2c3d4.jpg
    ```
    """
    path = get_image_path(filename)
    return FileResponse(path)
