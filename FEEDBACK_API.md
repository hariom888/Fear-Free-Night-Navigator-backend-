# Road Safety Feedback API

A production-ready Python (FastAPI) module integrated into the
**Fear-Free Night Navigator** backend.  
Users can submit geo-tagged road safety reports with images and a 1–10 safety
rating slider, then query, filter, and manage those reports through a clean
RESTful interface protected by JWT authentication.

---

## Folder Structure

```
src/
├── main.py                     ← FastAPI app (routers registered here)
├── feedback/                   ← Road Safety Feedback sub-package
│   ├── __init__.py
│   ├── database.py             ← SQLAlchemy ORM models + DB session
│   ├── schemas.py              ← Pydantic request / response models
│   ├── auth.py                 ← JWT creation, password hashing, dependencies
│   ├── crud.py                 ← All DB queries (create / read / update / delete)
│   ├── image_handler.py        ← Image upload validation, storage, retrieval
│   ├── routes_auth.py          ← /auth/* endpoints
│   └── routes_feedback.py      ← /feedback/* endpoints
tests/
└── test_feedback_api.py        ← pytest suite (in-memory SQLite, no external deps)
```

---

## Tech Stack

| Layer        | Library                          |
|-------------|----------------------------------|
| Framework    | FastAPI 0.115                    |
| Validation   | Pydantic v2                      |
| ORM / DB     | SQLAlchemy 2 + SQLite (default)  |
| Auth         | JWT (HMAC-SHA256, stdlib only)   |
| Passwords    | PBKDF2-SHA256, 260 000 iters     |
| File uploads | FastAPI `UploadFile` (multipart) |
| Image store  | Local disk (swap to S3 easily)   |

> **Production swap**: set `DATABASE_URL=postgresql://...` and the ORM
> handles the dialect automatically. For images, replace `save_image` /
> `delete_image` in `image_handler.py` with `boto3` S3 calls.

---

## Environment Variables

| Variable                  | Default                          | Description                          |
|--------------------------|----------------------------------|--------------------------------------|
| `DATABASE_URL`            | `sqlite:///./feedback.db`        | SQLAlchemy connection string         |
| `SECRET_KEY`              | *(insecure default)*             | JWT signing key — **change this!**   |
| `ACCESS_TOKEN_TTL_MINUTES`| `60`                             | Token lifetime in minutes            |
| `UPLOAD_DIR`              | `./uploads`                      | Directory for stored images          |
| `MAX_IMAGE_SIZE_MB`       | `10`                             | Max upload size in MB                |
| `BASE_URL`                | `http://localhost:8000`          | Used to build image retrieval URLs   |

Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## API Routes

### Authentication

| Method | Path              | Auth     | Description                        |
|--------|-------------------|----------|------------------------------------|
| POST   | `/auth/register`  | —        | Create a new user account          |
| POST   | `/auth/login`     | —        | Get a JWT access token             |
| GET    | `/auth/me`        | Required | Get current user profile           |

### Feedback

| Method | Path                    | Auth     | Description                              |
|--------|-------------------------|----------|------------------------------------------|
| POST   | `/feedback/submit`      | Optional | Submit road safety feedback + image      |
| GET    | `/feedback/list`        | —        | Paginated list with rating filter        |
| GET    | `/feedback/{id}`        | —        | Get a single record                      |
| GET    | `/feedback/area`        | —        | Feedback within a lat/lon bounding box   |
| GET    | `/feedback/area/stats`  | —        | Aggregate stats for an area              |
| GET    | `/feedback/filter`      | —        | Filter by safety rating range            |
| PATCH  | `/feedback/{id}`        | Required | Partial update                           |
| DELETE | `/feedback/{id}`        | Required | Delete record + image                    |
| GET    | `/feedback/image/{fn}`  | —        | Retrieve uploaded image                  |

---

## Example Requests & Responses

### Register

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"securepass123"}'
```
```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@example.com",
  "is_active": true,
  "created_at": "2025-06-01T10:00:00+00:00"
}
```

### Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"securepass123"}'
```
```json
{ "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...", "token_type": "bearer" }
```

### Submit Feedback (with image)

```bash
curl -X POST http://localhost:8000/feedback/submit \
  -H "Authorization: Bearer <token>" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946" \
  -F "description=Large pothole at KR Circle — covers half the lane" \
  -F "safety_rating=2" \
  -F "image=@/path/to/photo.jpg"
```
```json
{
  "id": 1,
  "latitude": 12.9716,
  "longitude": 77.5946,
  "address": null,
  "description": "Large pothole at KR Circle — covers half the lane",
  "safety_rating": 2,
  "image_url": "http://localhost:8000/feedback/image/a1b2c3d4.jpg",
  "submitted_by": "alice",
  "is_resolved": false,
  "created_at": "2025-06-01T10:05:00+00:00",
  "updated_at": "2025-06-01T10:05:00+00:00"
}
```

### Fetch Feedback for an Area

```bash
curl "http://localhost:8000/feedback/area?min_lat=12.95&max_lat=13.00&min_lon=77.57&max_lon=77.62"
```
```json
{
  "total": 3,
  "page": 1,
  "size": 50,
  "results": [
    {
      "id": 1,
      "latitude": 12.9716,
      "longitude": 77.5946,
      "safety_rating": 2,
      "description": "Large pothole at KR Circle...",
      "image_url": "http://localhost:8000/feedback/image/a1b2c3d4.jpg",
      "submitted_by": "alice",
      "is_resolved": false,
      "created_at": "2025-06-01T10:05:00+00:00",
      "updated_at": "2025-06-01T10:05:00+00:00"
    }
  ]
}
```

### Filter by Safety Rating

```bash
# Fetch the most dangerous roads (rating 1–3)
curl "http://localhost:8000/feedback/filter?min_rating=1&max_rating=3"
```

### Area Statistics

```bash
curl "http://localhost:8000/feedback/area/stats?min_lat=12.95&max_lat=13.00&min_lon=77.57&max_lon=77.62"
```
```json
{
  "total_reports": 8,
  "average_rating": 3.6,
  "min_rating": 1,
  "max_rating": 7
}
```

### Update a Record

```bash
curl -X PATCH http://localhost:8000/feedback/1 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"is_resolved": true, "safety_rating": 8}'
```

### Delete a Record

```bash
curl -X DELETE http://localhost:8000/feedback/1 \
  -H "Authorization: Bearer <token>"
# HTTP 204 No Content
```

---

## Running Tests

```bash
# From the project root
pip install pytest httpx
pytest tests/test_feedback_api.py -v
```

Tests use an **in-memory SQLite database** — no external services needed.

---

## Safety Rating Scale

| Rating | Meaning         |
|--------|-----------------|
| 1 – 3  | Very unsafe     |
| 4 – 6  | Moderate        |
| 7 – 9  | Safe            |
| 10     | Excellent / safe|

---

## Interactive Docs

Once the server is running, visit:

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc**       → http://localhost:8000/redoc
