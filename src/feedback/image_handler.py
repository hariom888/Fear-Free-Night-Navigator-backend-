"""
feedback/image_handler.py — Image Upload & Retrieval
=====================================================

Handles:
  - Saving uploaded images to local disk (configurable upload directory)
  - Generating safe, collision-resistant filenames
  - MIME-type validation (JPEG / PNG / WebP only)
  - File size validation
  - Building public image URLs for responses

Swap Strategy
-------------
To use AWS S3 instead of local storage, replace `save_image` and
`delete_image` with boto3 calls. The rest of the codebase is unaffected
because all image paths flow through this module.

Environment variables
---------------------
UPLOAD_DIR          — Directory for stored images (default: ./uploads)
MAX_IMAGE_SIZE_MB   — Maximum allowed upload size in MB (default: 10)
BASE_URL            — API base URL used to build image retrieval URLs
                      default: http://localhost:8000
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status

# ── Config ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR        = Path(os.getenv("UPLOAD_DIR", "./uploads"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES   = MAX_IMAGE_SIZE_MB * 1024 * 1024
BASE_URL          = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# Allowed MIME types → file extensions
ALLOWED_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png":  ".png",
    "image/webp": ".webp",
}


def ensure_upload_dir() -> None:
    """Create the upload directory tree if it doesn't exist yet."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_image(file: UploadFile) -> str:
    """
    Validate and persist an uploaded image file.

    Parameters
    ----------
    file : UploadFile
        The multipart file object from the FastAPI request.

    Returns
    -------
    str
        The relative path (from UPLOAD_DIR) stored in the database,
        e.g. "a1b2c3d4.jpg".

    Raises
    ------
    HTTPException 400 — unsupported content type or file too large.
    HTTPException 500 — disk write failure.
    """
    ensure_upload_dir()

    # ── Validate MIME type ───────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Allowed: {', '.join(ALLOWED_TYPES)}"
            ),
        )

    extension = ALLOWED_TYPES[content_type]
    filename  = f"{uuid.uuid4().hex}{extension}"
    dest_path = UPLOAD_DIR / filename

    # ── Read & size check ────────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB} MB.",
        )

    # ── Write to disk ────────────────────────────────────────────────────────
    try:
        dest_path.write_bytes(contents)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {exc}",
        )

    return filename


def delete_image(filename: str) -> None:
    """
    Remove an image file from local storage (best-effort, silently ignores
    missing files so that DB records can be cleaned up independently).
    """
    if not filename:
        return
    try:
        path = UPLOAD_DIR / filename
        if path.exists():
            path.unlink()
    except OSError:
        pass   # log in production; never block the caller


def build_image_url(filename: Optional[str]) -> Optional[str]:
    """
    Convert a stored filename to the full public URL the caller can use
    to retrieve the image via GET /feedback/image/{filename}.

    Returns None if filename is None (no image attached).
    """
    if not filename:
        return None
    return f"{BASE_URL}/feedback/image/{filename}"


def get_image_path(filename: str) -> Path:
    """
    Return the absolute filesystem path for a given stored filename.
    Used by the /feedback/image/{filename} retrieval endpoint.

    Raises HTTPException 404 if the file doesn't exist.
    """
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{filename}' not found.",
        )
    return path
