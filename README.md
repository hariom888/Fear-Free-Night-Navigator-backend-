# Fear-Free Night Navigator 

> A safety-aware routing backend for Bengaluru that ranks routes by **psychological safety**, not just speed.

---

## Table of Contents

- [Overview](#overview)
- [Live Links](#live-links)
- [System Architecture](#system-architecture)
- [Algorithms & Techniques](#algorithms--techniques)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Ablation Study](#ablation-study)
- [Route Quality Demo](#route-quality-demo)
- [How to Run Locally](#how-to-run-locally)
- [API Reference](#api-reference)
- [Repository Structure](#repository-structure)
- [Interview Notes](#interview-notes)

---

## Overview

Fear-Free Night Navigator takes an **origin**, **destination**, and **departure time** and returns **3 Pareto-optimal routes** simultaneously, each trading off travel time against a learned **Composite Safety Score (CSS)** per road segment.

Built on real OpenStreetMap data for Bengaluru:
- ~155,000 nodes
- ~392,000 directed edges
- 4.7M pre-computed CSS scores (393k edges × 12 time bands)

The system is designed with **solo women travellers** as the primary persona but supports configurable safety profiles.

---

## Live Links

| Resource | URL |
|----------|-----|
| **Frontend (Demo Map)** | [Open Demo Map](outputs/demo.html) |
| **Backend API** | `http://localhost:8000` |
| **API Docs (Swagger)** | `http://localhost:8000/docs` |
| **Health Check** | `http://localhost:8000/health` |

---

## System Architecture

```
OSM Road Graph (154,929 nodes · 392,199 edges)
        │
        ▼
Feature Engineering  ←── POI density, lighting, hospital/police proximity,
        │                  road type, dead-end topology, time-band encoding
        ▼
GBM Safety Classifier  ──→  CSS score [0–1] per segment × 12 time bands
        │
        ▼
CSS Cache (4.7M rows)  ←── pre-computed offline, loaded at startup
        │
        ▼
A* / Dijkstra Router  ──→  3 Pareto tiers: safe_express · balanced · safe_scenic
        │
        ▼
FastAPI REST API  ──→  /route  /heatmap  /safety/segment  /health  /feedback
```

**Key design decision:** ML inference is **not** in the hot request path. All CSS scores are pre-computed and cached. API latency stays under 200ms.

---

## Algorithms & Techniques

### 1. Graph Construction — OSM Road Network
The city road network is modelled as a **directed weighted graph** using OpenStreetMap data parsed into:
- A sparse adjacency matrix (`adjacency_matrix.npz`) with 392,199 edges
- A node feature table (`nodes_features.csv`) with coordinates and safety statistics

### 2. Safety Scoring — Gradient Boosting Classifier (GBM)
A **Gradient Boosting Machine (GBM)** is trained on 22 engineered features to predict a **Composite Safety Score (CSS)** ∈ [0, 1] for each road segment. The model:
- Is trained on a 50k-edge stratified sample with 3-fold cross-validation
- Outperforms RandomForest and Logistic Regression baselines (see Evaluation)
- Scores all 393k edges across 12 time bands **offline**, producing 4.7M cached predictions

This keeps ML out of the live request path entirely.

### 3. Routing — Bi-Objective Dijkstra / A\*
Routes are computed using a **bi-objective cost function** that balances travel time and safety:

```
cost = α · travel_time + β · (1 − CSS)
```

Three Pareto tiers are returned per request by varying (α, β):

| Tier | α | β | Priority |
|------|---|---|----------|
| Safe Express | 0.4 | 0.6 | Balanced, leans safe |
| Balanced | 0.5 | 0.5 | True trade-off |
| Safe Scenic | 0.2 | 0.8 | Maximum safety |

### 4. Time-Band Encoding
Time is discretised into **12 bands** (e.g., late night, early morning, rush hour) to capture how safety scores change with the time of day. Interaction terms (`night_x_road`, `night_x_safe_poi`, etc.) are computed to let the model learn non-linear temporal effects.

### 5. POI Buffer Analysis
Safe, neutral, and risky Points of Interest (cafes, hospitals, bars, etc.) are counted within **100m and 300m buffers** around each road segment using spatial indexing on OSM amenity data.

### 6. Graph Topology Feature — Dead-End Detection
Degree-1 nodes (dead ends) are flagged as `dead_end_flag`, a binary feature that meaningfully impacts safety prediction (ablation Δ AUC = −0.0255).

---

## Feature Engineering

22 features are used across 5 categories:

| Feature | Type | Source |
|---------|------|--------|
| `road_type_encoded` | int 1–4 | OSM highway tag |
| `length_m` | float | OSM geometry |
| `safe_poi_count_100m`, `safe_poi_count_300m` | int | OSM POI buffer |
| `neutral_poi_count_100m`, `neutral_poi_count_300m` | int | OSM POI buffer |
| `risky_poi_count_100m`, `risky_poi_count_300m` | int | OSM POI buffer |
| `time_band` | int 0–11 | Departure timestamp |
| `is_night`, `is_weekend` | binary | Derived from timestamp |
| `lighting_score` | float 0–1 | Road type heuristic |
| `perceived_risk_score` | float | Domain proxy |
| `nearest_hospital_m`, `nearest_police_m` | float | OSM amenity |
| `dead_end_flag` | binary | Graph topology (degree-1 node) |
| `night_x_road` | float | Interaction term |
| `night_x_safe_poi` | float | Interaction term |
| `night_x_risky_poi` | float | Interaction term |
| `night_x_lighting` | float | Interaction term |

---

## Model Evaluation

Evaluated on a **50k edge sample** with **3-fold stratified cross-validation**:

| Model | CV AUC | Train AUC | Precision | Recall | F1 | Brier |
|-------|--------|-----------|-----------|--------|-----|-------|
| **GBM (ours)** | **0.8705** | **0.8713** | **0.7832** | **0.7299** | **0.7556** | **0.1440** |
| RandomForest | 0.8663 | 0.8667 | 0.7948 | 0.7016 | 0.7453 | 0.1475 |
| LogisticRegression | 0.8599 | 0.8600 | 0.8199 | 0.6413 | 0.7197 | 0.1533 |

### Target Thresholds — All Passed 

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| CV AUC | > 0.80 | 0.8705 |  Pass |
| Train AUC | > 0.82 | 0.8713 |  Pass |
| Precision | > 0.75 | 0.7832 |  Pass |
| Recall | > 0.70 | 0.7299 |  Pass |
| F1 | > 0.74 | 0.7556 |  Pass |
| Brier Score | < 0.20 | 0.1440 |  Pass |

---

## Ablation Study

Each feature group was removed one at a time to measure its independent contribution to model AUC. A larger Δ AUC means the group carries more predictive signal.

| Removed Group | AUC Without | Δ AUC | Features Removed | Interpretation |
|---------------|-------------|-------|-----------------|----------------|
| **safe_POI** | 0.8304 | **−0.0401** | `safe_poi_count_100m`, `safe_poi_count_300m`, `night_x_safe_poi` | Strongest independent signal — crowd safety proxies matter most |
| **dead_end_flag** | 0.8451 | −0.0255 | `dead_end_flag` | Graph topology is a powerful single-feature signal |
| **neutral_POI** | 0.8560 | −0.0146 | `neutral_poi_count_100m`, `neutral_poi_count_300m` | Neutral POIs contribute meaningfully to context |
| **road_struct** | 0.8665 | −0.0040 | `road_type_encoded`, `length_m` | Road type contributes modest structural signal |
| **risky_POI** | 0.8681 | −0.0025 | `risky_poi_count_100m`, `risky_poi_count_300m`, `night_x_risky_poi` | Risky POIs add on top of structural features |
| **temporal** | 0.8682 | −0.0023 | `is_night`, `time_band`, `is_weekend`, `night_x_road`, `night_x_safe_poi`, `night_x_risky_poi` | Night/day variation confirmed — important for time-aware routing |

**Key takeaway:** Safe POI density and dead-end topology are the two highest-impact feature groups. Removing either causes the largest accuracy drop. Temporal features confirm that time-of-day matters for safety estimation, even if the marginal AUC contribution is modest.

---

## Route Quality Demo

**Route:** MG Road → Koramangala | **Time:** Midnight | **Profile:** `solo_woman`

| Tier | Path Safety Score | Extra Time Cost | MUN Alerts |
|------|-------------------|-----------------|------------|
| Safe Express | 0.994 | 0.0s | 0 |
| Balanced | 0.730 | +4.84s | 1 |
| Safe Scenic | 0.994 | −9.68s | 0 |

The balanced route crosses 1 unsafe segment that both safety-dominant tiers route around entirely.

---

## How to Run Locally

### Prerequisites
- Python 3.9+
- `pip` and `uvicorn`

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/yourname/fear-free-navigator
cd fear-free-navigator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files in data/
#    - compressed_data_csv.gz     (393k edges × 25 features)
#    - adjacency_matrix.npz       (sparse directed graph)
#    - nodes_features.csv         (node coordinates + safety stats)

# 4. Train the GBM model and build the CSS cache (4.7M rows)
python3 src/train_fast.py

# 5. Start the FastAPI backend
cd src && uvicorn main:app --reload --port 8000

# 6. Run the test suite (26 tests)
python3 tests/test_router.py

# 7. Generate and open the interactive demo map
python3 demo_map.py
open outputs/demo.html
```

### Sample API Call

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": {"lat": 12.9758, "lon": 77.6011},
    "destination": {"lat": 12.9139, "lon": 77.6419},
    "departure_epoch": 1700000000,
    "profile": {
      "persona": "solo_woman",
      "safety_threshold": 0.65,
      "speed_weight": 0.3
    }
  }'
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/route` | POST | Returns 3 Pareto-optimal routes for a given O-D pair and time |
| `/heatmap` | GET | Returns CSS scores across the city for map overlay |
| `/safety/segment` | GET | Returns the CSS score for a specific road segment |
| `/health` | GET | Health check — confirms cache loaded and API is live |
| `/feedback` | POST | Accepts user feedback to log perceived safety ratings |

Full interactive docs available at `http://localhost:8000/docs` once the server is running.

---

## Repository Structure

```
fear-free-navigator/
├── data/
│   ├── compressed_data_csv.gz     # 393k edges × 25 features (Bengaluru OSM)
│   ├── adjacency_matrix.npz       # Sparse directed graph (154k nodes)
│   └── nodes_features.csv         # Node coordinates + safety statistics
├── src/
│   ├── features.py                # Feature engineering + interaction terms
│   ├── model.py                   # GBM training, evaluation, ablation, plots
│   ├── router.py                  # Dijkstra / A* safety routing engine
│   ├── schemas.py                 # Pydantic request/response models
│   ├── main.py                    # FastAPI application (5 endpoints)
│   └── train_fast.py              # End-to-end pipeline runner
├── outputs/
│   ├── safety_model.pkl           # Trained GBM classifier
│   ├── css_cache_full.csv         # 4.7M rows: all edges × 12 time bands
│   ├── roc_curve.png              # Model vs baseline ROC curves
│   ├── feature_importance.png     # Top 20 feature importances
│   ├── ablation_plot.png          # AUC delta per feature group
│   ├── ablation_results.csv       # Ablation table (CSV)
│   ├── evaluation_report.md       # Full metrics report
│   └── demo.html                  # Interactive Folium map
├── tests/
│   └── test_router.py             # 26 tests, all passing
├── demo_map.py                    # Generates demo.html
└── README.md
```

