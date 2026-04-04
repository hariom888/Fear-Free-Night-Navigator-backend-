"""
demo_map.py — Step 8: Interactive Folium Demo Map (Fear-Free Navigator)

Produces demo.html showing:
  1. Safety heatmap — all edges coloured green→red by CSS score
  2. Three Pareto route tiers for a demo journey (midnight, solo_woman persona)
  3. Safety anchors (safe nodes near best route)
  4. MUN alerts (unavoidable unsafe segments)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from pathlib import Path
import folium
from folium import plugins

BASE = Path(__file__).parent
OUT  = BASE / "outputs"
DATA = BASE / "data"

# ── Colour mapping ─────────────────────────────────────────────────────────────
def css_to_colour(css: float) -> str:
    """Green (safe=1.0) → Yellow → Red (unsafe=0.0)"""
    css = float(np.clip(css, 0, 1))
    if css >= 0.7:
        r, g = int(255 * (1 - css) * 3), 200
    elif css >= 0.4:
        r, g = 200, int(255 * (css - 0.4) * 3.33)
    else:
        r, g = 200, int(50 * css / 0.4)
    return f"#{r:02x}{g:02x}20"


def reconstruct_path(predecessors, src, dst):
    path, node = [], int(dst)
    while node != src and node >= 0:
        path.append(node)
        node = int(predecessors[node])
    if node == src:
        path.append(int(src))
        return path[::-1]
    return []


def path_safety(path, css_tb):
    scores = [css_tb.get((path[i], path[i+1]), 0.5) for i in range(len(path)-1)]
    if not scores:
        return 0.5
    return round(0.6 * np.mean(scores) + 0.4 * np.min(scores), 3)


def build_cost_matrix(adj, css_tb, alpha, beta):
    H = adj.astype(np.float32).tolil()
    rows, cols = adj.nonzero()
    for r, c in zip(rows, cols):
        css = css_tb.get((int(r), int(c)), 0.5)
        H[r, c] = alpha * float(adj[r, c]) + beta * (1.0 - css)
    return H.tocsr()


def nearest_node_idx(nodes_df, lat, lon):
    d = (nodes_df["u_lat"] - lat)**2 + (nodes_df["u_lon"] - lon)**2
    return int(d.idxmin())


# ── Load assets ────────────────────────────────────────────────────────────────
print("[demo] Loading graph …")
npz = np.load(DATA / "adjacency_matrix.npz", allow_pickle=True)
adj = sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]),
                     shape=tuple(npz["shape"]))
print(f"[demo] {adj.shape[0]:,} nodes, {adj.nnz:,} edges")

print("[demo] Loading CSS cache …")
css_df = pd.read_csv(OUT / "css_cache.csv")
tb0 = css_df[css_df["time_band"] == 0]
css_tb = dict(zip(zip(tb0["u_idx"].astype(int), tb0["v_idx"].astype(int)),
                   tb0["css_score"]))

print("[demo] Loading node coords …")
nodes_raw = pd.read_csv(DATA / "nodes_features.csv")[["osmid","x","y"]].copy()
nodes_raw.columns = ["osmid","u_lon","u_lat"]
nodes_raw["u_idx"] = range(len(nodes_raw))

# ── Pick demo route: central Bengaluru → electronic city ─────────────────────
# Origin: MG Road area (approx)
# Destination: HSR Layout (approx)
ORIGIN_LL      = (12.9758, 77.6011)  # MG Road
DESTINATION_LL = (12.9139, 77.6419)  # Koramangala

origin_idx = nearest_node_idx(nodes_raw, *ORIGIN_LL)
dest_idx   = nearest_node_idx(nodes_raw, *DESTINATION_LL)
origin_node  = nodes_raw.iloc[origin_idx]
dest_node    = nodes_raw.iloc[dest_idx]

print(f"[demo] Origin   idx={origin_idx}: ({origin_node.u_lat:.4f}, {origin_node.u_lon:.4f})")
print(f"[demo] Dest     idx={dest_idx}:   ({dest_node.u_lat:.4f},  {dest_node.u_lon:.4f})")

# ── Compute 3 route tiers ─────────────────────────────────────────────────────
TIERS = [
    ("safe_express", 0.40, 0.60, "#27ae60", "Safe Express"),
    ("balanced",     0.50, 0.50, "#2980b9", "Balanced"),
    ("safe_scenic",  0.20, 0.80, "#8e44ad", "Safe Scenic"),
]

routes = {}
fastest_cost = None

for tier_key, alpha, beta, colour, label in TIERS:
    H = build_cost_matrix(adj, css_tb, alpha, beta)
    dist, preds = dijkstra(H, directed=True, indices=origin_idx,
                           return_predecessors=True)
    path = reconstruct_path(preds, origin_idx, dest_idx)
    cost = float(dist[dest_idx])
    if fastest_cost is None:
        fastest_cost = cost
    safety = path_safety(path, css_tb)
    routes[tier_key] = {
        "path": path, "cost": cost, "safety": safety,
        "colour": colour, "label": label,
        "extra": round(cost - fastest_cost, 2),
    }
    print(f"[demo] {label:14s} — {len(path)} nodes, safety={safety:.3f}, extra={cost-fastest_cost:.2f}")

# ── Build Folium map ──────────────────────────────────────────────────────────
centre = ((ORIGIN_LL[0] + DESTINATION_LL[0]) / 2,
          (ORIGIN_LL[1] + DESTINATION_LL[1]) / 2)

m = folium.Map(
    location=centre, zoom_start=13,
    tiles="CartoDB dark_matter",
    prefer_canvas=True,
)

# ── Layer 1: Safety heatmap (sample 5000 edges for performance) ──────────────
print("[demo] Drawing heatmap …")
heatmap_group = folium.FeatureGroup(name="🌡 Safety Heatmap", show=True)

# Sample band-0 edges; join node coordinates via u_idx / v_idx positional lookup
sample_tb0 = tb0.sample(min(5000, len(tb0)), random_state=42)
for _, row in sample_tb0.iterrows():
    u_idx = int(row["u_idx"])
    v_idx = int(row["v_idx"])
    css   = float(row["css_score"])
    colour = css_to_colour(css)
    weight = 1.5 if css < 0.4 else 1.0
    u_node = nodes_raw.iloc[u_idx]
    v_node = nodes_raw.iloc[v_idx]
    folium.PolyLine(
        locations=[(u_node.u_lat, u_node.u_lon),
                   (v_node.u_lat, v_node.u_lon)],
        color=colour, weight=weight, opacity=0.6,
        tooltip=f"CSS: {css:.3f}",
    ).add_to(heatmap_group)
heatmap_group.add_to(m)

# ── Layer 2: Route tiers ──────────────────────────────────────────────────────
print("[demo] Drawing routes …")
for tier_key, alpha, beta, colour, label in TIERS:
    r = routes[tier_key]
    path = r["path"]
    if len(path) < 2:
        continue

    group = folium.FeatureGroup(name=f"🗺 {label} (safety={r['safety']:.2f})", show=True)

    # Draw path segments
    coords = []
    for idx in path:
        node = nodes_raw.iloc[idx]
        coords.append((node.u_lat, node.u_lon))

    folium.PolyLine(
        locations=coords, color=colour, weight=5, opacity=0.85,
        tooltip=f"{label} | Safety: {r['safety']:.3f} | Extra: {r['extra']:.1f}s",
        dash_array="10 5" if tier_key == "safe_scenic" else None,
    ).add_to(group)

    # Highlight unsafe segments (MUN markers)
    mun_count = 0
    for i in range(len(path)-1):
        css = css_tb.get((path[i], path[i+1]), 0.5)
        if css < 0.4:
            n = nodes_raw.iloc[path[i]]
            folium.CircleMarker(
                location=(n.u_lat, n.u_lon),
                radius=5, color="#e74c3c", fill=True,
                fill_color="#e74c3c", fill_opacity=0.9,
                tooltip=f"⚠ Unsafe segment CSS={css:.3f}",
            ).add_to(group)
            mun_count += 1

    group.add_to(m)
    print(f"  {label}: {len(coords)} waypoints, {mun_count} MUN alerts")

# ── Layer 3: Origin & Destination markers ─────────────────────────────────────
folium.Marker(
    location=ORIGIN_LL,
    popup=folium.Popup("<b>Origin</b><br>MG Road, Bengaluru", max_width=150),
    tooltip="🚶 Origin: MG Road",
    icon=folium.Icon(color="green", icon="play", prefix="fa"),
).add_to(m)

folium.Marker(
    location=DESTINATION_LL,
    popup=folium.Popup("<b>Destination</b><br>Koramangala, Bengaluru", max_width=150),
    tooltip="🏁 Destination: Koramangala",
    icon=folium.Icon(color="red", icon="flag", prefix="fa"),
).add_to(m)

# ── Legend HTML ───────────────────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;bottom:40px;right:20px;z-index:1000;
     background:#1a1a2e;border:2px solid #444;border-radius:10px;
     padding:14px 18px;font-family:monospace;color:#eee;font-size:12px;
     box-shadow:2px 2px 12px rgba(0,0,0,0.7);">
  <b style="font-size:14px;color:#27ae60;">Fear-Free Navigator</b><br>
  <span style="color:#aaa;font-size:11px;">Bengaluru · Midnight · Solo Woman</span><br><br>
  <b>Routes</b><br>
  <span style="color:#27ae60;">━━━</span> Safe Express &nbsp;(safety={safe_express:.2f})<br>
  <span style="color:#2980b9;">━━━</span> Balanced &nbsp;&nbsp;&nbsp;&nbsp;(safety={balanced:.2f})<br>
  <span style="color:#8e44ad;">╌╌╌</span> Safe Scenic &nbsp;&nbsp;(safety={safe_scenic:.2f})<br><br>
  <b>Safety Score</b><br>
  <span style="color:#27c020;">▬</span> High (≥ 0.7) &nbsp;
  <span style="color:#c8c820;">▬</span> Mid &nbsp;
  <span style="color:#c82020;">▬</span> Low (&lt; 0.4)<br>
  <span style="color:#e74c3c;">●</span> MUN — unavoidable unsafe segment
</div>
""".format(
    safe_express=routes["safe_express"]["safety"],
    balanced=routes["balanced"]["safety"],
    safe_scenic=routes["safe_scenic"]["safety"],
)
m.get_root().html.add_child(folium.Element(legend_html))

# ── Title bar ─────────────────────────────────────────────────────────────────
title_html = """
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
     z-index:1000;background:#1a1a2e;border:2px solid #27ae60;border-radius:8px;
     padding:8px 20px;font-family:monospace;color:#27ae60;font-size:15px;
     font-weight:bold;box-shadow:2px 2px 10px rgba(0,0,0,0.7);">
  🌙 Fear-Free Night Navigator — Bengaluru Demo
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ── Layer control ─────────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT / "demo.html"
m.save(str(out_path))
print(f"\n[demo] Saved → {out_path}")
print(f"  Open in any browser: file://{out_path.resolve()}")
