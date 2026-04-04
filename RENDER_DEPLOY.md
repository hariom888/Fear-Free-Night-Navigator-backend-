# Fear-Free Night Navigator — Render Deployment Guide

## Project structure (what you push to GitHub)

```
fear-free-navigator/
├── data/
│   ├── edges_list.csv
│   ├── adjacency_matrix.npz
│   └── nodes_features.csv
├── outputs/                    ← commit this AFTER running train locally
│   ├── safety_model.pkl
│   └── css_cache.csv
├── src/
│   ├── features.py
│   ├── model.py
│   ├── router.py
│   ├── schemas.py
│   └── main.py                 ← includes AutoPing
├── tests/
│   └── test_router.py
├── requirements.txt
├── run.sh
└── render.yaml                 ← Render reads this automatically
```

---

## Step 1 — Train locally first

Run training on your machine before touching Render.
The model and CSS cache are large binary files — train once, commit, deploy.

```bash
# Install deps
pip install -r requirements.txt

# Train (80k sample, ~3-5 min)
python3 src/train_fast.py

# Or train on full 393k dataset (~10-15 min)
python3 src/train_fast.py --full

# Verify outputs exist
ls -lh outputs/
# You should see: safety_model.pkl, css_cache.csv
```

---

## Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial fear-free navigator with trained model"
git remote add origin https://github.com/YOUR_USERNAME/fear-free-navigator.git
git push -u origin main
```

**Important:** Make sure `outputs/safety_model.pkl` and `outputs/css_cache.csv`
are included. Add this to `.gitignore` to keep the rest clean:

```
# .gitignore
.venv/
__pycache__/
*.pyc
outputs/roc_curve.png
outputs/feature_importance.png
outputs/ablation_plot.png
outputs/edges_with_coords.csv
outputs/demo.html
```

If `css_cache.csv` is over 100 MB, use Git LFS:
```bash
git lfs install
git lfs track "outputs/css_cache.csv"
git lfs track "outputs/safety_model.pkl"
git add .gitattributes
git commit -m "chore: track large files with git lfs"
```

---

## Step 3 — Create Web Service on Render

1. Go to **https://dashboard.render.com** → **New** → **Web Service**
2. Connect your GitHub repo
3. Fill in these settings:

| Field | Value |
|-------|-------|
| **Name** | `fear-free-navigator` |
| **Region** | Singapore (closest to Bengaluru data) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `bash run.sh --skip-train` |
| **Instance Type** | Free (or Starter $7/mo for no sleep) |

4. Click **Advanced** → **Add Environment Variable**:

| Key | Value | Notes |
|-----|-------|-------|
| `AUTOPING_ENABLED` | `true` | Keeps free tier alive |
| `PORT` | `8000` | Render injects this automatically |
| `PYTHON_VERSION` | `3.12.0` | Pin your Python version |

5. Click **Create Web Service**

Render will:
- Clone your repo
- Run `pip install -r requirements.txt`
- Run `bash run.sh --skip-train` (loads model + starts uvicorn)

---

## Step 4 — Verify deployment

Once the deploy log shows `[startup] Ready.`, test it:

```bash
# Replace with your actual Render URL
BASE_URL="https://fear-free-navigator.onrender.com"

# Health check
curl $BASE_URL/health

# Ping (AutoPing target)
curl $BASE_URL/ping

# Route request
curl -X POST $BASE_URL/route \
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
```

Your interactive docs are at: `https://your-app.onrender.com/docs`

---

## How AutoPing works

```
Render Free Tier
  └─ spins down after 15 min of no requests
  └─ cold start takes ~30-60 sec (very bad UX)

AutoPing solution (inside main.py):
  1. At startup, asyncio.create_task(autoping_loop()) runs in background
  2. Every 13 minutes it sends GET /ping to itself
  3. /ping returns instantly without touching any state
  4. Render sees activity → dyno stays warm → no cold start

Environment variable control:
  AUTOPING_ENABLED=true   → autopings every 13 min  (default, use on free tier)
  AUTOPING_ENABLED=false  → disabled  (use on paid Starter plan — no sleep)
```

The `RENDER_EXTERNAL_URL` env variable is automatically set by Render to your
service's public URL (e.g. `https://fear-free-navigator.onrender.com`).
AutoPing reads it — no hardcoding needed.

---

## Troubleshooting

**Deploy fails at startup — "FileNotFoundError: css_cache.csv"**
→ You forgot to commit `outputs/`. Run training locally first, then commit.

**"pip install" fails on scipy or numpy**
→ Add `PYTHON_VERSION=3.12.0` to environment variables in Render dashboard.

**Routes take > 5 sec**
→ The free tier has 512 MB RAM. If the CSS cache is very large (>300 MB),
  upgrade to Starter ($7/mo) or reduce cache size by scoring fewer time bands.

**AutoPing logs show failures**
→ Render hasn't set `RENDER_EXTERNAL_URL` yet (first deploy only). It will
  resolve on the next deploy once the URL is assigned.

**Cold starts still happening**
→ Check that `AUTOPING_ENABLED=true` is set in your Render env vars.
  On free tier, if your service URL changes, update the env var accordingly.

---

## Upgrade path

| Tier | Cost | RAM | Sleep | Recommendation |
|------|------|-----|-------|----------------|
| Free | $0 | 512 MB | Yes (15 min) | Use AutoPing — works fine |
| Starter | $7/mo | 512 MB | Never | Disable AutoPing, cleaner logs |
| Standard | $25/mo | 2 GB | Never | Use if CSS cache > 500 MB |

---

## Re-deploying after retraining

```bash
# Retrain with new data
python3 src/train_fast.py --full

# Commit updated outputs
git add outputs/safety_model.pkl outputs/css_cache.csv
git commit -m "model: retrain with updated edge data"
git push

# Render auto-deploys on push (if auto-deploy is on)
# Or click Manual Deploy in the Render dashboard
```
