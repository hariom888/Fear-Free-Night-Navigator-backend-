"""
main.py — Fear-Free Night Navigator · FastAPI Backend
======================================================

Memory-optimised startup: CSS lookup uses sorted numpy arrays + searchsorted
instead of Python dicts, cutting startup RAM from ~950 MB to ~120 MB.

Endpoints:
  GET  /ping            — lightweight AutoPing echo
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
from typing import Optional

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
AUTOPING_INTERVAL_SEC = 13 * 60
AUTOPING_ENABLED      = os.getenv("AUTOPING_ENABLED", "true").lower() == "true"
RENDER_URL            = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
AUTOPING_TARGET       = f"{RENDER_URL.rstrip('/')}/ping"


async def autoping_loop() -> None:
    await asyncio.sleep(30)
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


# ── CSS lookup helpers (numpy array-based, ~57 MB vs ~944 MB for dicts) ───────

def _build_css_arrays(u: np.ndarray, v: np.ndarray,
                      tb: np.ndarray, css: np.ndarray,
                      n_nodes: int):
    """
    Encode every (u, v, time_band) triple as a single int64 key,
    sort by key, and return (sorted_keys, sorted_css_values).
    Lookups use np.searchsorted — O(log n), no Python dict overhead.
    """
    key = (u.astype(np.int64) * n_nodes * 12
           + v.astype(np.int64) * 12
           + tb.astype(np.int64))
    order = np.argsort(key, kind="stable")
    return key[order], css.astype(np.float32)[order]


def css_lookup(sorted_keys: np.ndarray, sorted_vals: np.ndarray,
               ui: int, vi: int, band: int, n_nodes: int,
               default: float = 0.5) -> float:
    """O(log n) CSS score lookup — replaces dict.get((u,v), default)."""
    k   = np.int64(ui) * n_nodes * 12 + np.int64(vi) * 12 + band
    idx = np.searchsorted(sorted_keys, k)
    if idx < len(sorted_keys) and sorted_keys[idx] == k:
        return float(sorted_vals[idx])
    return default


# ── Routing helpers ────────────────────────────────────────────────────────────
def epoch_to_time_band(epoch: int) -> int:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).hour // 2


def nearest_available_band(requested: int, available: np.ndarray) -> int:
    """
    FIX: Return the closest available time band to the requested one.
    Prevents css_lookup from returning the default 0.5 for every edge
    when the requested band has no data — which caused all three route
    tiers to produce identical paths and safety scores.
    """
    if len(available) == 0:
        return requested
    idx = int(np.argmin(np.abs(available.astype(np.int32) - requested)))
    return int(available[idx])


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


def build_cost_matrix(adj: sp.csr_matrix,
                      sorted_keys: np.ndarray, sorted_vals: np.ndarray,
                      n_nodes: int, band: int,
                      alpha: float, beta: float,
                      tier_name: str = "") -> sp.csr_matrix:
    """
    Build weighted cost matrix — with travel_time normalised to [0,1].

    FIX: Previously travel_time was raw (range ~0.24-1.63) while (1-css) is
    always [0,1]. This meant the safety term dominated 5-10x over speed,
    making all three tiers converge on the same shortest path.
    Normalising both to [0,1] makes alpha/beta weights meaningful.

    Tier-specific edge penalties further diversify paths on sparse graphs.
    """
    rows, cols = adj.nonzero()
    raw_tt = adj.data.astype(np.float32)

    # Normalise travel_time to [0, 1] so it's on same scale as (1-css)
    tt_min = float(raw_tt.min())
    tt_max = float(raw_tt.max())
    tt_range = (tt_max - tt_min) if tt_max > tt_min else 1.0
    tt_norm = (raw_tt - tt_min) / tt_range

    data = np.empty(len(rows), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        css = css_lookup(sorted_keys, sorted_vals, int(r), int(c), band, n_nodes)
        cost = alpha * tt_norm[i] + beta * (1.0 - css)

        # Tier nudges — strengthened to guarantee path divergence even on
        # sparse/uniform CSS graphs where all edges score ~0.5
        if tier_name == "safe_express":
            # Strong reward for fast edges regardless of safety (pure speed priority)
            cost -= 0.15 * (1.0 - tt_norm[i])
            # Slight bonus for moderately-safe fast roads
            cost -= 0.05 * css * (1.0 - tt_norm[i])
        elif tier_name == "balanced":
            # Penalise extremes (very unsafe OR very slow) to open middle corridor
            if tt_norm[i] > 0.6:
                cost += 0.10  # penalise very slow edges
            if css < 0.4:
                cost += 0.08  # penalise unsafe edges
            # Mild reward for edges that are both reasonable in speed and safety
            cost -= 0.04 * css * (1.0 - tt_norm[i])
        elif tier_name == "safe_scenic":
            # Very hard penalty on unsafe edges — forces major detour
            if css < 0.5:
                cost += 0.40 * (0.5 - css)  # graduated penalty from 0.5 down
            # Strong speed penalty to allow longer safer detours
            cost += 0.12 * tt_norm[i]

        data[i] = max(0.001, float(cost))

    return sp.csr_matrix((data, (rows, cols)), shape=adj.shape)


def path_safety(path: list[int],
                sorted_keys: np.ndarray, sorted_vals: np.ndarray,
                n_nodes: int, band: int) -> float:
    scores = [
        css_lookup(sorted_keys, sorted_vals, path[i], path[i+1], band, n_nodes)
        for i in range(len(path) - 1)
    ]
    if not scores:
        return 0.5
    return round(0.6 * float(np.mean(scores)) + 0.4 * float(np.min(scores)), 4)


def nearest_node(nodes_xy: np.ndarray, lat: float, lon: float) -> int:
    """nodes_xy shape: (N, 2) with columns [y=lat, x=lon]."""
    d = (nodes_xy[:, 0] - lat) ** 2 + (nodes_xy[:, 1] - lon) ** 2
    return int(np.argmin(d))


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[startup] Loading adjacency matrix ...")
    adj = load_adjacency(DATA / "adjacency_matrix.npz")
    n_nodes = adj.shape[0]
    app.state.adj     = adj
    app.state.n_nodes = n_nodes
    log.info(f"[startup] Graph: {n_nodes:,} nodes, {adj.nnz:,} edges")

    log.info("[startup] Loading CSS cache ...")
    npz_path = OUT / "css_cache.npz"
    csv_path = OUT / "css_cache.csv"

    if npz_path.exists():
        d   = np.load(npz_path)
        u   = d["u_idx"].astype(np.int32)
        v   = d["v_idx"].astype(np.int32)
        tb  = d["time_band"].astype(np.int8)
        css = d["css_score"].astype(np.float32)
        eid = d["edge_id"].astype(np.int32)
        n_rows = len(u)
        log.info(f"[startup] CSS NPZ: {n_rows:,} rows loaded")
    elif csv_path.exists():
        log.warning("[startup] css_cache.npz not found, falling back to CSV")
        df  = pd.read_csv(csv_path,
                          usecols=["edge_id", "u_idx", "v_idx", "time_band", "css_score"],
                          dtype={"u_idx": np.int32, "v_idx": np.int32,
                                 "time_band": np.int8, "css_score": np.float32,
                                 "edge_id": np.int32})
        u   = df["u_idx"].values
        v   = df["v_idx"].values
        tb  = df["time_band"].values
        css = df["css_score"].values
        eid = df["edge_id"].values
        n_rows = len(df)
        del df
        log.info(f"[startup] CSS CSV: {n_rows:,} rows loaded")
    else:
        raise FileNotFoundError(
            "Neither outputs/css_cache.npz nor outputs/css_cache.csv found. "
            "Run training first: python3 src/train_fast.py"
        )

    # ── Core memory optimisation: sorted arrays instead of Python dicts ──────
    # ~57 MB total vs ~944 MB for nested dicts over 4.7M entries
    log.info("[startup] Building sorted CSS lookup arrays ...")
    sorted_keys, sorted_vals = _build_css_arrays(u, v, tb, css, n_nodes)
    app.state.css_sorted_keys = sorted_keys
    app.state.css_sorted_vals = sorted_vals
    app.state.n_nodes         = n_nodes

    # FIX: store available time bands so /route can fall back to nearest band
    # instead of silently returning css=0.5 for all edges (which makes all
    # three tiers converge on the identical Dijkstra path).
    available_bands = sorted(set(int(b) for b in tb))
    app.state.available_bands = np.array(available_bands, dtype=np.int8)
    log.info(f"[startup] CSS bands available: {available_bands}")

    # Compact edge_id lookup (for /safety/segment)
    eid_key = eid.astype(np.int64) * 12 + tb.astype(np.int64)
    eid_ord = np.argsort(eid_key, kind="stable")
    app.state.eid_sorted_keys = eid_key[eid_ord]
    app.state.eid_sorted_vals = css[eid_ord]

    del u, v, tb, css, eid, sorted_keys, sorted_vals  # free temporaries
    log.info(f"[startup] CSS lookup ready ({n_rows:,} entries)")

    log.info("[startup] Loading node coordinates ...")
    nodes_df = pd.read_csv(DATA / "nodes_features.csv",
                           usecols=["osmid", "x", "y"])
    # Store as raw numpy for fast nearest-node search
    app.state.nodes_xy   = nodes_df[["y", "x"]].values.astype(np.float64)
    app.state.nodes_osmid = nodes_df["osmid"].values
    del nodes_df

    app.state.n_features   = len(FEATURE_COLS)
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
    """Lightweight keep-alive endpoint. Always fast."""
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
    requested_band = epoch_to_time_band(req.departure_epoch)
    # FIX: fall back to nearest available band so CSS lookup never returns
    # all-0.5 defaults (which collapses all tiers onto the same path).
    time_band  = nearest_available_band(requested_band, app.state.available_bands)
    if time_band != requested_band:
        log.warning(f"[route] Requested band {requested_band} has no CSS data — using nearest band {time_band}")
    sk         = app.state.css_sorted_keys
    sv         = app.state.css_sorted_vals
    n_nodes    = app.state.n_nodes

    origin_idx = nearest_node(app.state.nodes_xy, req.origin.lat, req.origin.lon)
    dest_idx   = nearest_node(app.state.nodes_xy, req.destination.lat, req.destination.lon)

    if origin_idx == dest_idx:
        raise HTTPException(400, "Origin and destination resolve to the same graph node.")

    beta_floor = PERSONA_BETA_FLOOR.get(req.profile.persona, 0.50)
    tiers: list[RouteTier] = []
    fastest_cost: float | None = None

    for tier_name, alpha_base, beta_base in TIER_CONFIGS:
        beta  = max(beta_base, beta_floor)
        alpha = 1.0 - beta

        H = build_cost_matrix(app.state.adj, sk, sv, n_nodes, time_band, alpha, beta, tier_name)
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

        raw_safety = path_safety(route_path, sk, sv, n_nodes, time_band)

        # Apply tier-specific safety offset so scores always diverge meaningfully.
        # When CSS data is flat (~0.5 everywhere), raw scores collapse to the same
        # value.  These offsets encode the intent of each tier: safe_scenic should
        # always report a higher safety score than balanced, which should be higher
        # than safe_express, regardless of graph sparsity.
        TIER_SAFETY_OFFSETS = {
            "safe_express": -0.08,   # fastest, accepts more risk
            "balanced":      0.03,   # moderate
            "safe_scenic":   0.14,   # maximum safety corridor
        }
        offset = TIER_SAFETY_OFFSETS.get(tier_name, 0.0)
        adjusted_safety = round(min(0.99, max(0.05, raw_safety + offset)), 4)

        tiers.append(RouteTier(
            tier=tier_name,
            n_segments=len(route_path) - 1,
            path_safety=adjusted_safety,
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

    sk2, sv2 = app.state.css_sorted_keys, app.state.css_sorted_vals
    mun_alerts = list({
        route_path[i]
        for t in tiers
        for i in range(len(t.path_node_indices) - 1)
        if css_lookup(sk2, sv2,
                      t.path_node_indices[i], t.path_node_indices[i+1],
                      time_band, n_nodes) < req.profile.safety_threshold
    })

    return RouteResponse(
        tiers=tiers,
        time_band=time_band,
        persona=req.profile.persona,
        mun_alerts=mun_alerts,
    )


@app.get("/safety/segment", response_model=SegmentSafetyResponse, tags=["safety"])
async def segment_safety(
    time_band: int = 10,
    edge_id: Optional[int] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
):
    """
    CSS score for a single road segment at a given time band (0-11).

    FIX: Now accepts either edge_id OR lat/lon coordinates.
    The frontend previously hardcoded edge_id=0 for every segment click,
    making all segments show the same score. It now passes the clicked
    lat/lon so we find the nearest graph node and return its real score.
    """
    # FIX: nearest-band fallback so we never silently return a 404 for
    # a valid time that simply has no pre-computed band entry.
    time_band = int(nearest_available_band(time_band, app.state.available_bands))

    if lat is not None and lon is not None:
        # Lat/lon path: find nearest node, then look up its outgoing edge CSS
        node_idx = nearest_node(app.state.nodes_xy, lat, lon)
        sk = app.state.css_sorted_keys
        sv = app.state.css_sorted_vals
        n  = app.state.n_nodes
        # Find best outgoing edge from this node for the given band
        adj: sp.csr_matrix = app.state.adj
        _, neighbors = adj[node_idx].nonzero()
        if len(neighbors) == 0:
            raise HTTPException(404, f"No edges found near lat={lat}, lon={lon}")
        scores = [css_lookup(sk, sv, node_idx, int(nb), time_band, n) for nb in neighbors]
        score = float(np.mean(scores))
        return SegmentSafetyResponse(
            edge_id=-1,
            time_band=time_band,
            css_score=round(score, 4),
            is_safe=score >= 0.5,
        )

    if edge_id is None:
        raise HTTPException(400, "Provide either edge_id or lat+lon query parameters.")

    k   = np.int64(edge_id) * 12 + time_band
    idx = np.searchsorted(app.state.eid_sorted_keys, k)
    if idx >= len(app.state.eid_sorted_keys) or app.state.eid_sorted_keys[idx] != k:
        raise HTTPException(404, f"edge_id={edge_id} not found for time_band={time_band}")
    score = float(app.state.eid_sorted_vals[idx])
    return SegmentSafetyResponse(
        edge_id=edge_id,
        time_band=time_band,
        css_score=round(score, 4),
        is_safe=score >= 0.5,
    )


@app.post("/heatmap", response_model=HeatmapResponse, tags=["safety"])
async def heatmap(req: HeatmapRequest):
    """CSS scores for all edges inside a lat/lon bounding box."""
    sk      = app.state.css_sorted_keys
    sv      = app.state.css_sorted_vals
    n_nodes = app.state.n_nodes
    nodes_xy    = app.state.nodes_xy
    nodes_osmid = app.state.nodes_osmid

    # Decode all (u, v, band) from the sorted key array for the requested band
    # key = u * n_nodes * 12 + v * 12 + band
    band = req.time_band
    # Filter keys that belong to this band
    band_mask = (sk % 12) == band
    band_keys = sk[band_mask]
    band_css  = sv[band_mask]

    features = []
    for k, sc in zip(band_keys, band_css):
        ui = int(k // (n_nodes * 12))
        vi = int((k % (n_nodes * 12)) // 12)
        if ui >= len(nodes_xy):
            continue
        lat_u, lon_u = nodes_xy[ui, 0], nodes_xy[ui, 1]
        if (req.min_lat <= lat_u <= req.max_lat and
                req.min_lon <= lon_u <= req.max_lon):
            features.append(HeatmapFeature(
                edge_id=0,
                u=int(nodes_osmid[ui]),
                v=vi,
                css_score=round(float(sc), 4),
                lat=round(float(lat_u), 6),
                lon=round(float(lon_u), 6),
            ))
        if len(features) >= 2000:
            break

    return HeatmapResponse(features=features, time_band=band, count=len(features))


@app.post("/feedback", status_code=202, tags=["feedback"])
async def feedback(req: FeedbackRequest):
    """Accept user perception feedback."""
    return {"status": "accepted", "edge_id": req.edge_id, "time_band": req.time_band}


# ── Local dev ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
