"""
feedback/crud.py — Database CRUD Operations for Feedback
=========================================================

All database interactions are isolated here so that routers remain thin
and business logic can be tested independently of FastAPI.

Each function accepts a SQLAlchemy Session injected by FastAPI's dependency
system (see database.get_db).
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from .database import FeedbackDB
from .schemas import FeedbackCreate, FeedbackUpdate


# ── Create ──────────────────────────────────────────────────────────────────────

def create_feedback(
    db:           Session,
    data:         FeedbackCreate,
    image_path:   Optional[str] = None,
    submitted_by: Optional[str] = None,
) -> FeedbackDB:
    """
    Persist a new feedback record.

    Parameters
    ----------
    db           : Active SQLAlchemy session.
    data         : Validated FeedbackCreate payload.
    image_path   : Relative filename of the uploaded image (or None).
    submitted_by : Username of the authenticated caller (or None / "anonymous").
    """
    record = FeedbackDB(
        latitude      = data.latitude,
        longitude     = data.longitude,
        address       = data.address,
        description   = data.description,
        safety_rating = data.safety_rating,
        image_path    = image_path,
        submitted_by  = submitted_by or "anonymous",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Read ────────────────────────────────────────────────────────────────────────

def get_feedback_by_id(db: Session, feedback_id: int) -> Optional[FeedbackDB]:
    """Fetch a single feedback record by primary key. Returns None if not found."""
    return db.query(FeedbackDB).filter(FeedbackDB.id == feedback_id).first()


def list_feedback(
    db:         Session,
    page:       int = 1,
    size:       int = 20,
    min_rating: int = 1,
    max_rating: int = 10,
) -> Tuple[int, List[FeedbackDB]]:
    """
    Paginated list of all feedback, optionally filtered by safety_rating range.

    Returns
    -------
    (total_count, records_for_this_page)
    """
    query = db.query(FeedbackDB).filter(
        and_(
            FeedbackDB.safety_rating >= min_rating,
            FeedbackDB.safety_rating <= max_rating,
        )
    )
    total   = query.count()
    records = (
        query
        .order_by(FeedbackDB.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return total, records


def list_feedback_by_area(
    db:         Session,
    min_lat:    float,
    max_lat:    float,
    min_lon:    float,
    max_lon:    float,
    min_rating: int = 1,
    max_rating: int = 10,
    page:       int = 1,
    size:       int = 50,
) -> Tuple[int, List[FeedbackDB]]:
    """
    Return feedback within a geographic bounding box and rating range.

    Only records that have lat/lon are considered (address-only records
    are excluded because they have no spatial coordinates).
    """
    query = db.query(FeedbackDB).filter(
        and_(
            FeedbackDB.latitude  >= min_lat,
            FeedbackDB.latitude  <= max_lat,
            FeedbackDB.longitude >= min_lon,
            FeedbackDB.longitude <= max_lon,
            FeedbackDB.safety_rating >= min_rating,
            FeedbackDB.safety_rating <= max_rating,
        )
    )
    total   = query.count()
    records = (
        query
        .order_by(FeedbackDB.safety_rating.asc())   # most unsafe first
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return total, records


def filter_by_safety_rating(
    db:         Session,
    min_rating: int = 1,
    max_rating: int = 10,
    page:       int = 1,
    size:       int = 20,
) -> Tuple[int, List[FeedbackDB]]:
    """
    Return feedback filtered to a specific safety rating range.
    Useful for dashboards that want to surface the most dangerous roads.
    """
    return list_feedback(
        db, page=page, size=size,
        min_rating=min_rating, max_rating=max_rating,
    )


def get_area_statistics(
    db:      Session,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> dict:
    """
    Aggregate statistics for a bounding box:
      - total_reports
      - average_rating
      - min_rating
      - max_rating
    """
    result = db.query(
        func.count(FeedbackDB.id).label("total"),
        func.avg(FeedbackDB.safety_rating).label("avg_rating"),
        func.min(FeedbackDB.safety_rating).label("min_rating"),
        func.max(FeedbackDB.safety_rating).label("max_rating"),
    ).filter(
        and_(
            FeedbackDB.latitude  >= min_lat,
            FeedbackDB.latitude  <= max_lat,
            FeedbackDB.longitude >= min_lon,
            FeedbackDB.longitude <= max_lon,
        )
    ).one()

    return {
        "total_reports": result.total or 0,
        "average_rating": round(float(result.avg_rating), 2) if result.avg_rating else None,
        "min_rating": result.min_rating,
        "max_rating": result.max_rating,
    }


# ── Update ──────────────────────────────────────────────────────────────────────

def update_feedback(
    db:          Session,
    feedback_id: int,
    data:        FeedbackUpdate,
) -> Optional[FeedbackDB]:
    """
    Partial update — only fields explicitly set in `data` are applied.
    Returns the updated record, or None if not found.
    """
    record = get_feedback_by_id(db, feedback_id)
    if not record:
        return None

    # Only update fields that were explicitly provided (not None)
    update_fields = data.model_dump(exclude_none=True)
    for field, value in update_fields.items():
        setattr(record, field, value)

    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


# ── Delete ──────────────────────────────────────────────────────────────────────

def delete_feedback(db: Session, feedback_id: int) -> Optional[str]:
    """
    Delete a feedback record.

    Returns
    -------
    str  — image_path of the deleted record (so the caller can remove the file)
    None — if the record was not found
    """
    record = get_feedback_by_id(db, feedback_id)
    if not record:
        return None
    image_path = record.image_path
    db.delete(record)
    db.commit()
    return image_path or ""
