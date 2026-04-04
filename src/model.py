"""
model.py — ML Training, Evaluation & CSS Cache Generation
Trains a GradientBoostingClassifier on edge safety features,
evaluates against baselines, runs an ablation study, and
pre-computes the CSS score cache for all edges × time bands.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from typing import Optional

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Feature groups for ablation ────────────────────────────────────────────────
# Ablation groups — only features that remain in FEATURE_COLS after leakage removal.
# Removed from ablation: nearest_police_m, nearest_hospital_m, perceived_risk_score,
# lighting_score, night_x_lighting — these caused AUC inflation (see features.py).
ABLATION_GROUPS = {
    "risky_POI":     ["risky_poi_count_100m", "risky_poi_count_300m", "night_x_risky_poi"],
    "safe_POI":      ["safe_poi_count_100m", "safe_poi_count_300m", "night_x_safe_poi"],
    "neutral_POI":   ["neutral_poi_count_100m", "neutral_poi_count_300m"],
    "temporal":      ["is_night", "time_band", "is_weekend",
                      "night_x_road", "night_x_safe_poi", "night_x_risky_poi"],
    "road_struct":   ["road_type_encoded", "length_m"],
    "dead_end_flag": ["dead_end_flag"],
}


def build_gbm():
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )


def build_baselines(feature_names: list[str]):
    """Return {name: estimator} dict for baseline comparison."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
    }


def cross_validate_model(model, X, y, n_splits=5, label="Model"):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  [{label}] CV AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")
    return aucs


def full_evaluation(model, X, y, feature_names, label="GBM") -> dict:
    """Train on full data, compute all metrics, return results dict."""
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "label":     label,
        "auc":       roc_auc_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall":    recall_score(y, pred, zero_division=0),
        "f1":        f1_score(y, pred, zero_division=0),
        "brier":     brier_score_loss(y, proba),
        "model":     model,
        "proba":     proba,
    }
    return metrics


def run_ablation(X, y, feature_names: list[str], base_auc: float) -> pd.DataFrame:
    """Remove one feature group at a time, report AUC delta."""
    rows = []
    for group_name, drop_cols in ABLATION_GROUPS.items():
        # Find indices of columns to drop
        drop_idx = [i for i, f in enumerate(feature_names) if f in drop_cols]
        if not drop_idx:
            continue
        keep_idx = [i for i in range(len(feature_names)) if i not in drop_idx]
        X_abl = X[:, keep_idx]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = cross_val_score(build_gbm(), X_abl, y, cv=cv,
                               scoring="roc_auc", n_jobs=-1)
        auc_abl = aucs.mean()
        rows.append({
            "removed_group":  group_name,
            "auc_without":    round(auc_abl, 4),
            "auc_delta":      round(auc_abl - base_auc, 4),
            "features_removed": ", ".join(drop_cols),
        })
        print(f"  ablation [{group_name:20s}]: AUC={auc_abl:.4f}  Δ={auc_abl-base_auc:+.4f}")

    return pd.DataFrame(rows).sort_values("auc_delta")


def plot_roc_curves(results: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["proba"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{r['label']}  (AUC={r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Fear-Free Navigator Safety Classifier", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curve → {out_path}")


def plot_feature_importance(model, feature_names: list[str], out_path: str, top_n=20) -> None:
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names, vals, color=plt.cm.RdYlGn(vals / vals.max()))
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Top Feature Importances — GBM Safety Classifier", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance → {out_path}")


def plot_ablation(ablation_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if d < 0 else "#95a5a6" for d in ablation_df["auc_delta"]]
    ax.barh(ablation_df["removed_group"], ablation_df["auc_delta"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("AUC Delta (negative = feature group matters)", fontsize=11)
    ax.set_title("Ablation Study — Impact of Removing Each Feature Group", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    for i, (_, row) in enumerate(ablation_df.iterrows()):
        ax.text(row["auc_delta"] - 0.0005, i,
                f"{row['auc_delta']:+.4f}", va="center", ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ablation plot → {out_path}")


def save_css_cache(model, df_full: pd.DataFrame, feature_names: list[str], out_path: str) -> None:
    """
    Pre-compute CSS scores for all edges across all time bands (0–11).
    Saves a cache CSV keyed by (edge_id, time_band).
    Since the dataset only has time_band=10, we synthesize the other bands
    by copying features and varying the time-related columns.
    """
    TIME_BANDS = list(range(12))
    NIGHT_BANDS = {0, 1, 2, 10, 11}

    base_cols = [c for c in feature_names if c not in
                 {"time_band", "is_night", "night_x_road", "night_x_safe_poi",
                  "night_x_risky_poi"}]

    records = []
    # Use unique edges from the dataset
    unique_edges = df_full[["edge_id", "u", "v"]].drop_duplicates("edge_id").copy()

    for tb in TIME_BANDS:
        is_night = int(tb in NIGHT_BANDS)
        sub = unique_edges.copy()
        sub["time_band"] = tb
        sub["is_night"]  = is_night

        # Merge back static features
        merged = sub.merge(
            df_full[["edge_id"] + [c for c in feature_names
                                    if c not in {"time_band", "is_night", "night_x_road",
                                                 "night_x_safe_poi", "night_x_risky_poi",
                                                 "is_weekend"}]].drop_duplicates("edge_id"),
            on="edge_id", how="left"
        )
        # is_weekend: use mode from original
        merged["is_weekend"] = 0

        # Rebuild interaction features (lighting_score excluded — leaky)
        merged["night_x_road"]      = merged["is_night"] * merged["road_type_encoded"]
        merged["night_x_safe_poi"]  = merged["is_night"] * merged["safe_poi_count_100m"]
        merged["night_x_risky_poi"] = merged["is_night"] * merged["risky_poi_count_100m"]

        # Fill any missing cols with 0
        for col in feature_names:
            if col not in merged.columns:
                merged[col] = 0

        X_tb = merged[feature_names].fillna(0).values.astype(np.float32)
        css  = model.predict_proba(X_tb)[:, 1]

        merged["css_score"] = css
        merged["time_band"] = tb
        records.append(merged[["edge_id", "u", "v", "time_band", "css_score"]])

    cache_df = pd.concat(records, ignore_index=True)
    cache_df.to_csv(out_path, index=False)
    print(f"  Saved CSS cache ({len(cache_df):,} rows) → {out_path}")
    return cache_df
