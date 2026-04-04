# 🚀 Fear-Free Night Navigator — Full Render Deployment Guide

> **Read this fully before touching GitHub or Render.** The biggest mistake is pushing without training first.

---

## 🗺️ Overview of Steps

```
1. Train model locally  →  2. Push to GitHub  →  3. Deploy on Render  →  4. Set env vars  →  5. Test live API
```

---

## STEP 1 — Train the Model Locally (MUST do first)

Render can't train — it has no GPU and limited RAM. Train on your machine, commit the outputs.

```bash
# Install dependencies
pip install -r requirements.txt

# Train (fast — 80k sample, ~3-5 min)
python3 src/train_fast.py

# OR train on full dataset (~10-15 min, better accuracy)
python3 src/train_fast.py --full

# Verify the outputs were created
ls -lh outputs/
# You MUST see:
#   safety_model.pkl   (the trained ML model)
#   css_cache.csv      (pre-computed safety scores — may be ~280 MB)
```

**⚠️ If `outputs/` is empty, DO NOT proceed. The API will crash on startup without these files.**

---

## STEP 2 — Prepare the `.gitignore`

Create a `.gitignore` file in your project root (rename `gitignore.txt` → `.gitignore`):

```bash
mv gitignore.txt .gitignore
```

The `.gitignore` already excludes plots and demos, but **keeps** `safety_model.pkl` and `css_cache.csv` — those **must** be committed.

---

## STEP 3 — Handle Large Files (css_cache.csv is ~280 MB)

GitHub's hard limit is 100 MB per file. `css_cache.csv` is ~280 MB, so you **must** use Git LFS.

```bash
# Install Git LFS (once per machine)
# macOS:
brew install git-lfs

# Ubuntu/Debian:
sudo apt install git-lfs

# Activate LFS
git lfs install

# Track large files
git lfs track "outputs/css_cache.csv"
git lfs track "outputs/safety_model.pkl"

# This creates .gitattributes — commit it too
git add .gitattributes
```

---

## STEP 4 — Push to GitHub

```bash
# Initialize repo
git init
git add .
git commit -m "feat: initial fear-free navigator with trained model and feedback API"

# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/fear-free-navigator.git
git push -u origin main
```

**Check on GitHub:** Go to your repo → `outputs/` folder → `css_cache.csv` should show "Stored with Git LFS" badge.

---

## STEP 5 — Fix the `run.sh` for Render

The current `run.sh` tries to use `pyenv`, which isn't available on Render's Python runtime. Create a Render-safe start script:

Create a file called `start_render.sh` in your project root:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[startup] Installing dependencies..."
pip install -r requirements.txt

echo "[startup] Starting FastAPI server..."
cd src
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 65 \
  --log-level info
```

Then update `render.yaml` to use it:

```yaml
# render.yaml
services:
  - type: web
    name: fear-free-navigator
    runtime: python
    region: singapore
    plan: free

    buildCommand: pip install -r requirements.txt

    startCommand: bash start_render.sh

    healthCheckPath: /health

    envVars:
      - key: PYTHON_VERSION
        value: "3.12.0"
      - key: AUTOPING_ENABLED
        value: "true"
      - key: SECRET_KEY
        generateValue: true        # Render auto-generates a secure random value
```

Commit this:

```bash
git add start_render.sh render.yaml
git commit -m "fix: render-compatible start script (no pyenv)"
git push
```

---

## STEP 6 — Create the Web Service on Render

1. Go to **https://dashboard.render.com**
2. Click **New** → **Web Service**
3. Click **Connect a repository** → Authorize GitHub → Select your repo
4. Fill in the settings:

| Field | Value |
|-------|-------|
| **Name** | `fear-free-navigator` |
| **Region** | Singapore |
| **Branch** | `main` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `bash start_render.sh` |
| **Instance Type** | Free |

5. Click **Advanced** to add environment variables (see Step 7 below)
6. Click **Create Web Service**

---

## STEP 7 — Set Environment Variables on Render

Go to your service → **Environment** tab → Add these:

| Key | Value | Notes |
|-----|-------|-------|
| `PYTHON_VERSION` | `3.12.0` | Pins Python version |
| `AUTOPING_ENABLED` | `true` | Prevents cold starts on free tier |
| `SECRET_KEY` | *(click "Generate")* | JWT signing key — must be secret |
| `ACCESS_TOKEN_TTL_MINUTES` | `60` | JWT expiry (optional, default is 60) |
| `MAX_IMAGE_SIZE_MB` | `10` | Max feedback image size (optional) |
| `DATABASE_URL` | *(leave blank for SQLite)* | Or set PostgreSQL URL for production |

> **⚠️ `SECRET_KEY` is critical.** Never use the default. Click "Generate" in Render's dashboard or run locally: `python -c "import secrets; print(secrets.token_hex(32))"`

> **`RENDER_EXTERNAL_URL`** — Do NOT set this manually. Render injects it automatically as your service's public URL (e.g., `https://fear-free-navigator.onrender.com`). AutoPing reads it to keep the dyno warm.

---

## STEP 8 — Verify the Deployment

Watch the **Deploy Logs** in the Render dashboard. You should see:

```
[startup] Starting FastAPI server...
INFO: Application startup complete.
```

Once green, test your live API (replace with your actual URL):

```bash
BASE="https://fear-free-navigator.onrender.com"

# 1. Health check
curl $BASE/health

# 2. Ping (AutoPing target)
curl $BASE/ping

# 3. Route request
curl -X POST $BASE/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin":      {"lat": 12.9758, "lon": 77.6011},
    "destination": {"lat": 12.9139, "lon": 77.6419},
    "departure_epoch": 1700000000,
    "profile": {
      "persona": "solo_woman",
      "safety_threshold": 0.65,
      "speed_weight": 0.3
    }
  }'

# 4. Register a user (Feedback API)
curl -X POST $BASE/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"securepass123"}'

# 5. Login
curl -X POST $BASE/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"securepass123"}'
```

**Interactive Swagger docs:** `https://fear-free-navigator.onrender.com/docs`
**ReDoc:** `https://fear-free-navigator.onrender.com/redoc`

---

## STEP 9 — Full API Reference (for your React Native dev)

Give this base URL and endpoint list to whoever builds the mobile app:

### Base URL
```
https://fear-free-navigator.onrender.com
```

### Core Navigation Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Liveness check |
| `GET` | `/ping` | — | Lightweight echo |
| `POST` | `/route` | — | Get 3 Pareto-optimal route tiers |
| `GET` | `/safety/segment?edge_id=X&time_band=Y` | — | CSS score for one road segment |
| `POST` | `/heatmap` | — | Safety overlay for a lat/lon bounding box |

### Auth Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | — | Create account |
| `POST` | `/auth/login` | — | Get JWT token |
| `GET` | `/auth/me` | Bearer token | Current user profile |

### Feedback Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/feedback/submit` | Optional | Submit geo-tagged report + image |
| `GET` | `/feedback/list` | — | Paginated list |
| `GET` | `/feedback/{id}` | — | Single record |
| `GET` | `/feedback/area` | — | Reports in a bounding box |
| `GET` | `/feedback/area/stats` | — | Aggregate stats for an area |
| `GET` | `/feedback/filter` | — | Filter by safety rating |
| `PATCH` | `/feedback/{id}` | Bearer token | Update record |
| `DELETE` | `/feedback/{id}` | Bearer token | Delete record |
| `GET` | `/feedback/image/{filename}` | — | Retrieve uploaded image |

### Route Request Body Example

```json
{
  "origin":      { "lat": 12.9758, "lon": 77.6011 },
  "destination": { "lat": 12.9139, "lon": 77.6419 },
  "departure_epoch": 1700000000,
  "profile": {
    "persona": "solo_woman",
    "safety_threshold": 0.65,
    "speed_weight": 0.3
  }
}
```

**Personas:** `solo_woman` | `elderly` | `delivery` | `general`

### Route Response — 3 Tiers

```json
{
  "tiers": [
    {
      "tier": "safe_express",
      "n_segments": 42,
      "path_safety": 0.81,
      "total_cost": 1240.5,
      "extra_vs_fastest": 0.0,
      "explanation": "...",
      "path_node_indices": [101, 204, 338, ...]
    },
    { "tier": "balanced", ... },
    { "tier": "safe_scenic", ... }
  ],
  "time_band": 22,
  "persona": "solo_woman",
  "mun_alerts": []
}
```

### Feedback Submit (multipart/form-data)

```
POST /feedback/submit
Authorization: Bearer <token>   (optional)

Form fields:
  latitude       float
  longitude      float
  description    string
  safety_rating  int (1–10)
  image          file (jpg/png, optional, max 10 MB)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: css_cache.csv` | You didn't commit `outputs/`. Train locally, commit, push. |
| `css_cache.csv` push rejected (too large) | Enable Git LFS (Step 3) |
| `pip install` fails on scipy/numpy | Set `PYTHON_VERSION=3.12.0` in Render env vars |
| Cold starts still happening | Check `AUTOPING_ENABLED=true` is set in Render |
| JWT errors on feedback endpoints | Set a real `SECRET_KEY` env var (not the default) |
| Routes slow (>5 sec) | Free tier has 512 MB RAM. Upgrade to Starter ($7/mo) or Standard ($25/mo) |
| First deploy AutoPing failures | `RENDER_EXTERNAL_URL` is set after first deploy — normal. Redeploy once. |

---

## Upgrading from Free Tier

| Plan | Cost | RAM | Sleep | AutoPing |
|------|------|-----|-------|----------|
| Free | $0 | 512 MB | After 15 min idle | Set `AUTOPING_ENABLED=true` |
| Starter | $7/mo | 512 MB | Never | Set `AUTOPING_ENABLED=false` |
| Standard | $25/mo | 2 GB | Never | Recommended if CSS cache > 500 MB |
