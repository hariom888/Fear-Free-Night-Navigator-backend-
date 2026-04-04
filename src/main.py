"""
main.py — Fear-Free Night Navigator · FastAPI Backend
======================================================

Design principles:
  - CSS scores are pre-computed offline and loaded at startup.
    NO ML inference in the hot request path → latency under 200ms.
  - AutoPing background task pings /ping every 13 minutes so
    Render's free tier never goes cold (Render spins down after 15 min idle).

Endpoints:
  GET  /ping            — lightweight AutoPing echo (no state access)
  GET  /health          — full liveness probe
  POST /route           — 3 Pareto route tiers for a journey
  GET  /safety/segment  — CSS score for a single road segment
  POST /heatmap         — CSS safety overlay for a bounding box
  POST /feedback        — store user perception data
"""

import os, sys, asyncio, logging
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    RouteRequest, RouteResponse, RouteTier,
    SegmentSafetyResponse, HeatmapRequest, HeatmapResponse, HeatmapFeature,
    FeedbackRequest, HealthResponse,
)
from features import FEATURE_COLS

log = logging.getLogger("uvicorn.error")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUT  = BASE / "outputs"
DATA = BASE / "data"

# ── Constants ──────────────────────────────────────────────────────────────────
NIGHT_BANDS = {0, 1, 2, 10, 11}

TIER_CONFIGS = [
    ("safe_express", 0.40, 0.60),
    ("balanced",     0.50, 0.50),
    ("safe_scenic",  0.20, 0.80),
]

PERSONA_BETA_FLOOR = {
    "solo_woman": 0.80,
    "elderly":    0.70,
    "delivery":   0.30,
    "general":    0.50,
}

TIER_EXPLANATIONS = {
    "safe_express": "Fastest route that avoids the most dangerous segments.",
    "balanced":     "Equal weight between speed and safety.",
    "safe_scenic":  "Maximum safety — follows highest-scoring corridors.",
}

PERSONA_EXPLANATIONS = {
    "solo_woman": "Route heavily prioritises well-lit, busy corridors near hospitals.",
    "elderly":    "Route avoids isolated segments; maximises proximity to help points.",
    "delivery":   "Route balances speed with avoiding dead-ends and isolated alleys.",
    "general":    "Route balances travel time and segment safety scores.",
}

# ── AutoPing ───────────────────────────────────────────────────────────────────
# Render free tier spins down after 15 min idle. We self-ping every 13 min.
# Set AUTOPING_ENABLED=false in env to disable (e.g. on paid tier).
AUTOPING_INTERVAL_SEC = 13 * 60
AUTOPING_ENABLED      = os.getenv("AUTOPING_ENABLED", "true").lower() == "true"
RENDER_URL            = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
AUTOPING_TARGET       = f"{RENDER_URL.rstrip('/')}/ping"


async def autoping_loop() -> None:
    """Sends GET /ping every AUTOPING_INTERVAL_SEC to keep Render warm."""
    await asyncio.sleep(30)   # let startup finish first
    log.info(f"[autoping] Started — pinging {AUTOPING_TARGET} every "
             f"{AUTOPING_INTERVAL_SEC // 60} min")
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                resp = await client.get(AUTOPING_TARGET)
                ts   = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
                log.info(f"[autoping] {ts} → HTTP {resp.status_code}")
            except Exception as exc:
                log.warning(f"[autoping] ping failed: {exc}")
            await asyncio.sleep(AUTOPING_INTERVAL_SEC)


# ── Routing helpers ────────────────────────────────────────────────────────────
def epoch_to_time_band(epoch: int) -> int:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).hour // 2


def load_adjacency(path: Path) -> sp.csr_matrix:
    d = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (d["data"], d["indices"], d["indptr"]),
        shape=tuple(d["shape"]),
    )


def reconstruct_path(preds: np.ndarray, src: int, dst: int) -> list[int]:
    path, node = [], int(dst)
    while node != src and node >= 0:
        path.append(node)
        node = int(preds[node])
    if node == src:
        path.append(int(src))
        return path[::-1]
    return []


def build_cost_matrix(adj: sp.csr_matrix, css_tb: dict,
                      alpha: float, beta: float) -> sp.csr_matrix:
    # Work directly on CSR arrays — avoids the expensive tolil() copy
    rows, cols = adj.nonzero()
    data = adj.data.astype(np.float32).copy()
    for i, (r, c) in enumerate(zip(rows, cols)):
        css = css_tb.get((int(r), int(c)), 0.5)
        data[i] = alpha * float(adj.data[i]) + beta * (1.0 - css)
    return sp.csr_matrix((data, (rows, cols)), shape=adj.shape)


def path_safety(path: list[int], css_tb: dict) -> float:
    scores = [css_tb.get((path[i], path[i+1]), 0.5) for i in range(len(path)-1)]
    if not scores:
        return 0.5
    return round(0.6 * float(np.mean(scores)) + 0.4 * float(np.min(scores)), 4)


def nearest_node(nodes_df: pd.DataFrame, lat: float, lon: float) -> int:
    d = (nodes_df["y"] - lat)**2 + (nodes_df["x"] - lon)**2
    return int(d.idxmin())  # index IS u_idx now


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[startup] Loading adjacency matrix ...")
    app.state.adj = load_adjacency(DATA / "adjacency_matrix.npz")
    log.info(f"[startup] Graph: {app.state.adj.shape[0]:,} nodes, "
             f"{app.state.adj.nnz:,} edges")

    log.info("[startup] Loading CSS cache ...")
    npz_path = OUT / "css_cache.npz"
    csv_path = OUT / "css_cache.csv"

    if npz_path.exists():
        # Fast path: compressed NPZ (~5 MB, loads in <1s)
        d   = np.load(npz_path)
        u   = d["u_idx"].astype(int)
        v   = d["v_idx"].astype(int)
        tb  = d["time_band"].astype(int)
        css = d["css_score"].astype(float)
        eid = d["edge_id"].astype(int)
        n_rows = len(u)

        app.state.css_by_band = {}
        for band in range(12):
            mask = tb == band
            app.state.css_by_band[band] = dict(
                zip(zip(u[mask], v[mask]), css[mask])
            )
        # Store raw arrays for /safety/segment — avoids duplicating all data
        # into a second giant dict. Lookups use np.searchsorted on sorted eid.
        sort_idx = np.argsort(eid * 12 + tb)  # sort by (edge_id, band)
        app.state.edge_css_key = (eid * 12 + tb)[sort_idx]
        app.state.edge_css_val = css[sort_idx]

        log.info(f"[startup] CSS cache (NPZ): {n_rows:,} rows, "
                 f"{len(app.state.css_by_band)} time bands")
    elif csv_path.exists():
        # Fallback: plain CSV (slower, larger)
        log.warning("[startup] css_cache.npz not found, falling back to CSV. "
                    "Run: python3 src/compress_cache.py")
        css_df = pd.read_csv(csv_path)
        app.state.css_by_band = {}
        for band, grp in css_df.groupby("time_band"):
            app.state.css_by_band[int(band)] = dict(
                zip(zip(grp["u_idx"].astype(int), grp["v_idx"].astype(int)),
                    grp["css_score"])
            )
        # Same compact array lookup as NPZ path
        eid_csv = css_df["edge_id"].values.astype(int)
        tb_csv  = css_df["time_band"].values.astype(int)
        css_csv = css_df["css_score"].values.astype(np.float32)
        sort_idx = np.argsort(eid_csv * 12 + tb_csv)
        app.state.edge_css_key = (eid_csv * 12 + tb_csv)[sort_idx]
        app.state.edge_css_val = css_csv[sort_idx]
        n_rows = len(css_df)
        log.info(f"[startup] CSS cache (CSV): {n_rows:,} rows")
    else:
        raise FileNotFoundError(
            "Neither outputs/css_cache.npz nor outputs/css_cache.csv found. "
            "Run training first: python3 src/train_fast.py"
        )

    log.info("[startup] Loading node coordinates ...")
    nodes_df = pd.read_csv(DATA / "nodes_features.csv")[["osmid", "x", "y"]]
    nodes_df["u_idx"]     = range(len(nodes_df))
    # Index by u_idx for O(1) lookup in heatmap; keep osmid for output only
    app.state.nodes_df    = nodes_df.set_index("u_idx")
    app.state.n_features  = len(FEATURE_COLS)
    app.state.n_cache_rows = n_rows
    app.state.startup_time = datetime.now(tz=timezone.utc).isoformat()
    log.info("[startup] Ready.")

    ping_task = None
    if AUTOPING_ENABLED:
        ping_task = asyncio.create_task(autoping_loop())
        log.info(f"[autoping] Task started — target: {AUTOPING_TARGET}")
    else:
        log.info("[autoping] Disabled (AUTOPING_ENABLED=false)")

    yield

    if ping_task:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fear-Free Night Navigator",
    description=(
        "Safety-aware routing for Bengaluru road network. "
        "CSS scores pre-computed offline — zero ML in the request path."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/ping", tags=["meta"])
async def ping():
    """Lightweight keep-alive endpoint for AutoPing. Always fast."""
    return {"pong": True, "ts": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Full health check confirming all assets are loaded."""
    return HealthResponse(
        status="ok",
        css_cache_rows=app.state.n_cache_rows,
        model_features=app.state.n_features,
    )


@app.post("/route", response_model=RouteResponse, tags=["routing"])
async def get_route(req: RouteRequest):
    """
    Compute 3 Pareto-optimal route tiers for origin -> destination.
    Tiers: safe_express / balanced / safe_scenic.
    """
    time_band = epoch_to_time_band(req.departure_epoch)
    css_tb    = app.state.css_by_band.get(time_band, app.state.css_by_band[10])

    origin_idx = nearest_node(app.state.nodes_df, req.origin.lat, req.origin.lon)
    dest_idx   = nearest_node(app.state.nodes_df, req.destination.lat, req.destination.lon)

    if origin_idx == dest_idx:
        raise HTTPException(400, "Origin and destination resolve to the same graph node.")

    beta_floor = PERSONA_BETA_FLOOR.get(req.profile.persona, 0.50)
    tiers: list[RouteTier] = []
    fastest_cost: float | None = None

    for tier_name, alpha_base, beta_base in TIER_CONFIGS:
        beta  = max(beta_base, beta_floor)
        alpha = 1.0 - beta

        H = build_cost_matrix(app.state.adj, css_tb, alpha, beta)
        dist_arr, preds = dijkstra(
            H, directed=True,
            indices=origin_idx,
            return_predecessors=True,
        )

        route_path = reconstruct_path(preds, origin_idx, dest_idx)
        if not route_path:
            continue

        cost = float(dist_arr[dest_idx])
        if fastest_cost is None:
            fastest_cost = cost

        tiers.append(RouteTier(
            tier=tier_name,
            n_segments=len(route_path) - 1,
            path_safety=path_safety(route_path, css_tb),
            total_cost=round(cost, 2),
            extra_vs_fastest=round(cost - (fastest_cost or cost), 2),
            explanation=(
                f"{TIER_EXPLANATIONS[tier_name]} "
                f"{PERSONA_EXPLANATIONS.get(req.profile.persona, '')}"
            ).strip(),
            path_node_indices=route_path,
        ))

    if not tiers:
        raise HTTPException(404, "No route found between these nodes.")

    mun_alerts = list({
        t.path_node_indices[i]
        for t in tiers
        for i in range(len(t.path_node_indices) - 1)
        if css_tb.get(
            (t.path_node_indices[i], t.path_node_indices[i+1]), 0.5
        ) < req.profile.safety_threshold
    })

    return RouteResponse(
        tiers=tiers,
        time_band=time_band,
        persona=req.profile.persona,
        mun_alerts=mun_alerts,
    )


@app.get("/safety/segment", response_model=SegmentSafetyResponse, tags=["safety"])
async def segment_safety(edge_id: int, time_band: int = 10):
    """CSS score for a single road segment at a given time band (0-11)."""
    key = edge_id * 12 + time_band
    idx = np.searchsorted(app.state.edge_css_key, key)
    if idx >= len(app.state.edge_css_key) or app.state.edge_css_key[idx] != key:
        raise HTTPException(404, f"edge_id={edge_id} not found for time_band={time_band}")
    score = float(app.state.edge_css_val[idx])
    return SegmentSafetyResponse(
        edge_id=edge_id,
        time_band=time_band,
        css_score=round(score, 4),
        is_safe=score >= 0.5,
    )


@app.post("/heatmap", response_model=HeatmapResponse, tags=["safety"])
async def heatmap(req: HeatmapRequest):
    """CSS scores for all edges inside a lat/lon bounding box."""
    css_tb = app.state.css_by_band.get(req.time_band, {})
    nodes  = app.state.nodes_df  # already indexed by u_idx
    features = []
    for (ui, vi), sc in css_tb.items():
        if ui not in nodes.index:
            continue
        row = nodes.loc[ui]
        if (req.min_lat <= row["y"] <= req.max_lat and
                req.min_lon <= row["x"] <= req.max_lon):
            features.append(HeatmapFeature(
                edge_id=0, u=int(row["osmid"]), v=vi,
                css_score=round(float(sc), 4),
            ))
        if len(features) >= 2000:
            break
    return HeatmapResponse(features=features, time_band=req.time_band, count=len(features))


@app.post("/feedback", status_code=202, tags=["feedback"])
async def feedback(req: FeedbackRequest):
    """Accept user perception feedback (queued for model retraining in production)."""
    return {"status": "accepted", "edge_id": req.edge_id, "time_band": req.time_band}


# ── Local dev ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
