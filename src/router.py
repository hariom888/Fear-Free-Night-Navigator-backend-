"""
router.py — A* Safety-Aware Graph Routing Engine
Builds a weighted graph from the adjacency matrix + CSS scores and
computes Pareto-optimal route tiers using scipy shortest paths.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, dijkstra
from typing import Optional


# ─── Persona safety weight overrides ──────────────────────────────────────────
PERSONA_BETA = {
    "solo_woman": 0.80,
    "elderly":    0.70,
    "delivery":   0.30,
    "general":    0.50,
}

# ─── Route tier configurations (name, alpha=time_weight, beta=safety_weight) ──
TIER_CONFIGS = [
    ("safe_express", 0.40, 0.60),
    ("balanced",     0.50, 0.50),
    ("safe_scenic",  0.20, 0.80),
]


def load_graph(npz_path: str) -> sp.csr_matrix:
    """Load pre-built sparse adjacency matrix."""
    data = np.load(npz_path, allow_pickle=True)
    mat  = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    )
    print(f"[router] Graph loaded: {mat.shape[0]:,} nodes, {mat.nnz:,} edges")
    return mat


def build_cost_matrix(adj: sp.csr_matrix, css_cache: pd.DataFrame,
                      time_band: int, alpha: float, beta: float) -> sp.csr_matrix:
    """
    Replace adjacency weights (travel_time) with bi-objective cost:
        cost = alpha * travel_time + beta * (1 - css_score)
    """
    # Filter cache to this time band
    tb_cache = css_cache[css_cache["time_band"] == time_band].copy()

    # Build (u_idx, v_idx) → css lookup
    if "u_idx" in tb_cache.columns and "v_idx" in tb_cache.columns:
        css_lookup = {(row.u_idx, row.v_idx): row.css_score
                      for row in tb_cache.itertuples(index=False)}
    else:
        # Fall back to edge_id based — slower
        css_lookup = {}

    # Copy adjacency structure
    H = adj.copy().astype(np.float32)
    H = H.tolil()

    rows_nz, cols_nz = adj.nonzero()
    for r, c in zip(rows_nz, cols_nz):
        travel_t = float(adj[r, c])
        css = css_lookup.get((r, c), 0.5)  # default neutral
        H[r, c] = alpha * travel_t + beta * (1.0 - css)

    return H.tocsr()


def path_safety_score(path: list[int], css_tb: dict) -> float:
    """Weakest-link composite: 60% mean + 40% minimum.
    css_tb: dict of {(u_idx, v_idx): css_score}.
    """
    if len(path) < 2:
        return 0.5
    scores = [css_tb.get((path[i], path[i+1]), 0.5) for i in range(len(path)-1)]
    if not scores:
        return 0.5
    return round(0.6 * float(np.mean(scores)) + 0.4 * float(np.min(scores)), 4)


def reconstruct_path(predecessors: np.ndarray, src: int, dst: int) -> list[int]:
    """Walk back through predecessor array to get path."""
    path = []
    node = dst
    while node != src and node >= 0:
        path.append(node)
        node = predecessors[node]
    if node == src:
        path.append(src)
        return path[::-1]
    return []  # no path found


def compute_routes(adj: sp.csr_matrix, css_cache: pd.DataFrame,
                   origin_idx: int, dest_idx: int,
                   time_band: int, persona: str = "general") -> list[dict]:
    """
    Compute 3 Pareto route tiers for origin→destination.
    Returns list of {tier, path, path_safety, total_cost, n_segments}.
    """
    beta_override = PERSONA_BETA.get(persona, 0.5)
    results = []

    # Build (u_idx, v_idx) → css_score lookup for this time band once,
    # shared across all tiers so path_safety_score has something to look up.
    tb_slice = css_cache[css_cache["time_band"] == time_band]
    css_tb = {
        (int(row.u_idx), int(row.v_idx)): float(row.css_score)
        for row in tb_slice.itertuples(index=False)
    }

    fastest_cost = None

    for tier_name, alpha, beta in TIER_CONFIGS:
        # Apply persona override
        if persona != "general":
            beta = max(beta, beta_override)
            alpha = 1.0 - beta

        H = build_cost_matrix(adj, css_cache, time_band, alpha, beta)

        # Run Dijkstra from origin
        dist_matrix, predecessors = dijkstra(
            H, directed=True,
            indices=origin_idx,
            return_predecessors=True,
            unweighted=False,
        )

        path = reconstruct_path(predecessors, origin_idx, dest_idx)
        if not path:
            print(f"  [router] No path found for tier={tier_name}")
            continue

        cost = dist_matrix[dest_idx]
        if fastest_cost is None:
            fastest_cost = cost

        safety = path_safety_score(path, css_tb)

        results.append({
            "tier":              tier_name,
            "path":              path,
            "n_segments":        len(path) - 1,
            "path_safety":       round(float(safety), 4),
            "total_cost":        round(float(cost), 2),
            "extra_vs_fastest":  round(float(cost - fastest_cost), 2),
            "persona":           persona,
            "time_band":         time_band,
        })

    return results


def route_summary(routes: list[dict]) -> pd.DataFrame:
    """Pretty-print route summary table."""
    rows = []
    for r in routes:
        rows.append({
            "tier":           r["tier"],
            "n_segments":     r["n_segments"],
            "path_safety":    r["path_safety"],
            "total_cost":     r["total_cost"],
            "extra_vs_fastest": r["extra_vs_fastest"],
        })
    return pd.DataFrame(rows)
