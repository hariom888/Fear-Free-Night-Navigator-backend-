"""
feedback/database.py — Database layer for Road Safety Feedback API
==================================================================

Uses SQLite via SQLAlchemy (zero-config, file-based).
In production swap DATABASE_URL for PostgreSQL / MySQL without changing
any other code — SQLAlchemy handles the dialect differences.

Environment variables
---------------------
DATABASE_URL  — SQLAlchemy connection string
                default: sqlite:///./feedback.db
                example: postgresql://user:pass@localhost:5432/fearfree
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

# ── Connection ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./feedback.db")

# connect_args only required for SQLite (allows multi-threaded access)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ──────────────────────────────────────────────────────────────────

class UserDB(Base):
    """
    Stores registered users for JWT authentication.
    Passwords are stored as bcrypt hashes — never plaintext.
    """
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64), unique=True, index=True, nullable=False)
    email      = Column(String(255), unique=True, index=True, nullable=False)
    hashed_pw  = Column(String(255), nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class FeedbackDB(Base):
    """
    Core feedback record linking a user-reported road safety issue
    to a geographic location, image, and a 1–10 safety rating.
    """
    __tablename__ = "feedback"

    id           = Column(Integer, primary_key=True, index=True)

    # ── Location ──────────────────────────────────────────────────────────────
    latitude     = Column(Float,       nullable=True,  index=True)
    longitude    = Column(Float,       nullable=True,  index=True)
    address      = Column(String(512), nullable=True)

    # ── Content ───────────────────────────────────────────────────────────────
    description  = Column(Text,        nullable=False)
    safety_rating= Column(Integer,     nullable=False)   # 1 (unsafe) – 10 (safe)
    image_path   = Column(String(512), nullable=True)    # relative path under uploads/

    # ── Meta ──────────────────────────────────────────────────────────────────
    submitted_by = Column(String(64),  nullable=True)    # username or "anonymous"
    is_resolved  = Column(Boolean,     default=False)
    created_at   = Column(DateTime(timezone=True),
                          default=lambda: datetime.now(timezone.utc))
    updated_at   = Column(DateTime(timezone=True),
                          default=lambda: datetime.now(timezone.utc),
                          onupdate=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Create all tables if they don't exist yet. Called once at startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency — yields a SQLAlchemy session and ensures it is
    closed after the request completes, even on error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
