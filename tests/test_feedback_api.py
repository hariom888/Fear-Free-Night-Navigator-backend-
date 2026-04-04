"""
tests/test_feedback_api.py — Unit & Integration Tests for Road Safety Feedback API
====================================================================================

Run with:
    pytest tests/test_feedback_api.py -v

Uses an in-memory SQLite database so no external services are needed.
"""

import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Set up an in-memory test DB before importing the app ──────────────────────
import os
os.environ["DATABASE_URL"]  = "sqlite://"          # pure in-memory
os.environ["SECRET_KEY"]    = "test-secret-key"
os.environ["BASE_URL"]      = "http://testserver"
os.environ["AUTOPING_ENABLED"] = "false"           # don't fire background task

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feedback.database import Base, get_db
from feedback.auth import _hash_password

# Build a fresh in-memory engine for every test session
engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Patch the app's DB dependency BEFORE importing the app routes
# (The app module imports get_db at module level.)
from feedback import database as _db_module
_db_module.SessionLocal = TestingSessionLocal

# Now build a minimal FastAPI app for testing (avoids loading large .npz files)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from feedback.routes_auth     import router as auth_router
from feedback.routes_feedback import router as feedback_router

test_app = FastAPI()
test_app.add_middleware(CORSMiddleware, allow_origins=["*"],
                        allow_methods=["*"], allow_headers=["*"])
test_app.include_router(auth_router)
test_app.include_router(feedback_router)
test_app.dependency_overrides[get_db] = override_get_db

client = TestClient(test_app)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def register_and_login(username="testuser", password="testpass123"):
    """Register a user and return a valid Bearer token."""
    client.post("/auth/register", json={
        "username": username,
        "email": f"{username}@example.com",
        "password": password,
    })
    resp = client.post("/auth/login", json={"username": username, "password": password})
    return resp.json()["access_token"]


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


def submit_feedback(token=None, **overrides):
    """Submit a minimal feedback entry via the API."""
    data = {
        "latitude":      "12.9716",
        "longitude":     "77.5946",
        "description":   "Deep pothole at the junction near MG Road metro",
        "safety_rating": "3",
    }
    data.update({k: str(v) for k, v in overrides.items()})
    headers = auth_header(token) if token else {}
    return client.post("/feedback/submit", data=data, headers=headers)


# ─────────────────────────────────────────────────────────────────────────────
# Auth tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_success(self):
        resp = client.post("/auth/register", json={
            "username": "alice",
            "email":    "alice@example.com",
            "password": "securepass",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["username"] == "alice"
        assert "hashed_pw" not in body          # never leak password hash

    def test_register_duplicate_username(self):
        client.post("/auth/register", json={
            "username": "bob", "email": "bob@example.com", "password": "pass1234"
        })
        resp = client.post("/auth/register", json={
            "username": "bob", "email": "bob2@example.com", "password": "pass1234"
        })
        assert resp.status_code == 409

    def test_login_success(self):
        client.post("/auth/register", json={
            "username": "carol", "email": "carol@example.com", "password": "pass1234"
        })
        resp = client.post("/auth/login", json={"username": "carol", "password": "pass1234"})
        assert resp.status_code == 200
        assert "access_token" in resp.json()
        assert resp.json()["token_type"] == "bearer"

    def test_login_wrong_password(self):
        client.post("/auth/register", json={
            "username": "dave", "email": "dave@example.com", "password": "correct"
        })
        resp = client.post("/auth/login", json={"username": "dave", "password": "wrong"})
        assert resp.status_code == 401

    def test_me_authenticated(self):
        token = register_and_login("eve", "pass1234x")
        resp  = client.get("/auth/me", headers=auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "eve"

    def test_me_unauthenticated(self):
        resp = client.get("/auth/me")
        assert resp.status_code == 403   # no credentials at all


# ─────────────────────────────────────────────────────────────────────────────
# Feedback submission tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackSubmit:
    def test_submit_with_coords_anonymous(self):
        resp = submit_feedback()
        assert resp.status_code == 201
        body = resp.json()
        assert body["safety_rating"] == 3
        assert body["submitted_by"] == "anonymous"
        assert body["latitude"]  == pytest.approx(12.9716)
        assert body["longitude"] == pytest.approx(77.5946)

    def test_submit_with_address_only(self):
        resp = client.post("/feedback/submit", data={
            "address":       "MG Road, Bengaluru",
            "description":   "Broken streetlight makes this stretch very dark at night",
            "safety_rating": "2",
        })
        assert resp.status_code == 201
        assert resp.json()["address"] == "MG Road, Bengaluru"

    def test_submit_missing_location_fails(self):
        resp = client.post("/feedback/submit", data={
            "description":   "Something unsafe here",
            "safety_rating": "5",
        })
        assert resp.status_code == 422

    def test_submit_invalid_rating_fails(self):
        resp = client.post("/feedback/submit", data={
            "latitude": "12.9716", "longitude": "77.5946",
            "description":   "Some issue here on the road",
            "safety_rating": "11",   # out of range
        })
        assert resp.status_code == 422

    def test_submit_short_description_fails(self):
        resp = client.post("/feedback/submit", data={
            "latitude": "12.9716", "longitude": "77.5946",
            "description":   "bad",   # too short
            "safety_rating": "5",
        })
        assert resp.status_code == 422

    def test_submit_authenticated_links_user(self):
        token = register_and_login("frank", "frank1234")
        resp  = submit_feedback(token=token)
        assert resp.status_code == 201
        assert resp.json()["submitted_by"] == "frank"

    def test_submit_with_image(self):
        fake_image = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100)   # minimal JPEG header
        resp = client.post("/feedback/submit",
            data={
                "latitude": "12.9716", "longitude": "77.5946",
                "description": "Pothole at the crossroads near the bus stop",
                "safety_rating": "4",
            },
            files={"image": ("photo.jpg", fake_image, "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["image_url"] is not None
        assert "feedback/image/" in body["image_url"]


# ─────────────────────────────────────────────────────────────────────────────
# Feedback retrieval tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackRetrieval:
    def setup_method(self):
        """Submit a known record before each test."""
        resp = submit_feedback(safety_rating=3)
        self.fid = resp.json()["id"]

    def test_get_by_id(self):
        resp = client.get(f"/feedback/{self.fid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == self.fid

    def test_get_nonexistent(self):
        resp = client.get("/feedback/999999")
        assert resp.status_code == 404

    def test_list_returns_results(self):
        resp = client.get("/feedback/list")
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert body["total"] >= 1

    def test_list_rating_filter(self):
        resp = client.get("/feedback/list?min_rating=1&max_rating=3")
        assert resp.status_code == 200
        for item in resp.json()["results"]:
            assert 1 <= item["safety_rating"] <= 3

    def test_area_query(self):
        resp = client.get(
            "/feedback/area"
            "?min_lat=12.9&max_lat=13.0&min_lon=77.5&max_lon=77.6"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["results"], list)

    def test_area_invalid_bbox(self):
        resp = client.get(
            "/feedback/area"
            "?min_lat=13.0&max_lat=12.9&min_lon=77.5&max_lon=77.6"
        )
        assert resp.status_code == 400

    def test_filter_by_safety_rating(self):
        resp = client.get("/feedback/filter?min_rating=1&max_rating=5")
        assert resp.status_code == 200
        for item in resp.json()["results"]:
            assert item["safety_rating"] <= 5

    def test_area_stats(self):
        resp = client.get(
            "/feedback/area/stats"
            "?min_lat=12.9&max_lat=13.0&min_lon=77.5&max_lon=77.6"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "total_reports" in body
        assert "average_rating" in body


# ─────────────────────────────────────────────────────────────────────────────
# Update & Delete tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackMutation:
    def setup_method(self):
        self.token = register_and_login(
            f"mutuser_{id(self)}", "mutablepass"
        )
        resp     = submit_feedback(token=self.token)
        self.fid = resp.json()["id"]

    def test_update_description(self):
        resp = client.patch(
            f"/feedback/{self.fid}",
            json={"description": "Updated: large crack spanning the full lane width"},
            headers=auth_header(self.token),
        )
        assert resp.status_code == 200
        assert "Updated:" in resp.json()["description"]

    def test_update_safety_rating(self):
        resp = client.patch(
            f"/feedback/{self.fid}",
            json={"safety_rating": 7},
            headers=auth_header(self.token),
        )
        assert resp.status_code == 200
        assert resp.json()["safety_rating"] == 7

    def test_update_mark_resolved(self):
        resp = client.patch(
            f"/feedback/{self.fid}",
            json={"is_resolved": True},
            headers=auth_header(self.token),
        )
        assert resp.status_code == 200
        assert resp.json()["is_resolved"] is True

    def test_update_unauthenticated_fails(self):
        resp = client.patch(
            f"/feedback/{self.fid}",
            json={"safety_rating": 8},
        )
        assert resp.status_code == 403

    def test_delete_success(self):
        resp = client.delete(
            f"/feedback/{self.fid}",
            headers=auth_header(self.token),
        )
        assert resp.status_code == 204
        # Confirm it's gone
        assert client.get(f"/feedback/{self.fid}").status_code == 404

    def test_delete_unauthenticated_fails(self):
        resp = client.delete(f"/feedback/{self.fid}")
        assert resp.status_code == 403
