"""
features.py — Feature Engineering Pipeline for Fear-Free Night Navigator

LEAKAGE NOTE
------------
The following columns exist in the raw data but are EXCLUDED from FEATURE_COLS
because they were direct inputs to the css_score formula that generated safe_label.
Including them allows any model to reconstruct the label formula, inflating AUC
artificially to 1.0000:

  Excluded:
    nearest_police_m      — AUC 0.91 alone; direct input to css_score formula
    nearest_hospital_m    — AUC 0.82 alone; direct input to css_score formula
    hospital_data_missing — derived from nearest_hospital_m
    police_data_missing   — derived from nearest_police_m
    perceived_risk_score  — near-constant (4 unique values), partially circular
    lighting_score        — part of css_score proxy formula
    night_x_lighting      — interaction of lighting_score

  Honest GBM AUC without these: ~0.87
  Inflated GBM/LR AUC with them: 1.0000 (model just re-learns the label formula)
"""

import gzip
import numpy as np
import pandas as pd
from pathlib import Path

# ── Feature columns used by the model ─────────────────────────────────────────
FEATURE_COLS = [
    # ── Road structure ────────────────────────────────────────────────────
    "road_type_encoded",       # 1=alley  2=residential  3=secondary  4=arterial
    "length_m",                # segment length in metres
    "dead_end_flag",           # 1 if node degree=1 (no escape route)
    # ── POI environment (counts within buffer, not distances) ─────────────
    "safe_poi_count_100m",     # hospitals, pharmacies within 100m
    "safe_poi_count_300m",
    "neutral_poi_count_100m",  # shops, cafes within 100m
    "neutral_poi_count_300m",
    "risky_poi_count_100m",    # bars, liquor stores within 100m
    "risky_poi_count_300m",
    # ── Temporal ─────────────────────────────────────────────────────────
    "time_band",               # 0–11 (2-hour bands, 0 = midnight–2am)
    "is_night",                # 1 if time_band in {0,1,2,10,11}
    "is_weekend",
    # ── Interaction terms (not in label-generation formula) ───────────────
    "night_x_road",            # is_night * road_type_encoded
    "night_x_safe_poi",        # is_night * safe_poi_count_100m
    "night_x_risky_poi",       # is_night * risky_poi_count_100m
]

TARGET_COL = "safe_label"

# ── Columns excluded due to data leakage ──────────────────────────────────────
# These were direct inputs to the css_score formula that generated safe_label.
# Any model trained on them achieves AUC ≈ 1.0 by reconstructing that formula,
# not by learning genuine safety signal.
LEAKY_COLS = [
    "nearest_police_m",      # AUC 0.91 alone — direct input to css_score
    "nearest_hospital_m",    # AUC 0.82 alone — direct input to css_score
    "hospital_data_missing", # derived flag of nearest_hospital_m
    "police_data_missing",   # derived flag of nearest_police_m
    "perceived_risk_score",  # near-constant (4 unique values), partially circular
    "lighting_score",        # part of css_score proxy formula
    "night_x_lighting",      # interaction of lighting_score
    "css_score",             # the label source itself
]


def load_edge_features(path: str | Path) -> pd.DataFrame:
    """Load edge features from a plain CSV or a .gz compressed CSV."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    print(f"[features] Loaded {len(df):,} rows x {df.shape[1]} columns from {path.name}")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction terms that go beyond the proxy heuristic."""
    df = df.copy()
    df["night_x_road"]      = df["is_night"] * df["road_type_encoded"]
    df["night_x_safe_poi"]  = df["is_night"] * df["safe_poi_count_100m"]
    df["night_x_risky_poi"] = df["is_night"] * df["risky_poi_count_100m"]
    # night_x_lighting intentionally excluded — lighting_score is a
    # direct input to the css_score formula that generated safe_label
    return df


def check_for_leakage(df: pd.DataFrame) -> None:
    """
    Warn if any known-leaky column is present in the dataframe AND would
    accidentally end up in FEATURE_COLS. Raises AssertionError on hard leaks.
    """
    present_leakers = [c for c in LEAKY_COLS if c in df.columns]
    in_features     = [c for c in LEAKY_COLS if c in FEATURE_COLS]

    if present_leakers:
        print(f"\n  [leakage-check] Leaky columns present in data (excluded from model): "
              f"{present_leakers}")
    else:
        print("\n  [leakage-check] No leaky columns found in dataframe.")

    assert not in_features, (
        f"CRITICAL LEAKAGE: {in_features} are in FEATURE_COLS! "
        "Remove them before training."
    )
    print(f"  [leakage-check] FEATURE_COLS has {len(FEATURE_COLS)} features, "
          f"{len(LEAKY_COLS)} excluded as leaky.")


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y arrays ready for sklearn."""
    df = add_interaction_features(df)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    return X, y, FEATURE_COLS


def describe_features(df: pd.DataFrame) -> None:
    """Print a quick summary of label balance and feature stats."""
    print("\n── Label distribution ──────────────────────────────────")
    vc = df[TARGET_COL].value_counts()
    total = len(df)
    for label, count in vc.items():
        print(f"  label={label}: {count:>8,}  ({100*count/total:.1f}%)")

    print("\n── CSS score range ─────────────────────────────────────")
    print(f"  min={df['css_score'].min():.4f}  max={df['css_score'].max():.4f}  "
          f"mean={df['css_score'].mean():.4f}  std={df['css_score'].std():.4f}")

    print("\n── Feature stats (sample) ──────────────────────────────")
    safe_cols = [c for c in FEATURE_COLS[:8] if c in df.columns]
    print(df[safe_cols].describe().round(3).to_string())
