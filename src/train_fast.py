"""
train_fast.py  —  Fear-Free Night Navigator · End-to-End Training Pipeline
===========================================================================

REQUIRED FILES  (all go in the  data/  folder next to src/)
------------------------------------------------------------
  data/
  ├── edges_list.csv           ← main edge feature table  (393k edges x 25 cols)
  ├── adjacency_matrix.npz     ← sparse directed road graph (154k nodes)
  └── nodes_features.csv       ← node coordinates + safety stats

USAGE
-----
  # Train on a fast sample (default 80k rows):
  python3 src/train_fast.py

  # Train on the FULL 393k edge dataset:
  python3 src/train_fast.py --full

  # Skip ablation (saves ~2 min):
  python3 src/train_fast.py --skip-ablation

OUTPUTS  (written to  outputs/)
--------------------------------
  safety_model.pkl         trained GBM classifier
  css_cache.csv            CSS scores: all 393k edges x 12 time bands
  roc_curve.png            ROC curves vs baselines
  feature_importance.png   top-20 feature importances
  ablation_plot.png        AUC delta per feature group
  ablation_results.csv     ablation table (CSV)
  evaluation_report.md     full metrics report
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from features import (
    load_edge_features, add_interaction_features,
    get_feature_matrix, describe_features, check_for_leakage,
    FEATURE_COLS, LEAKY_COLS, TARGET_COL,
)
from model import (
    build_gbm, build_baselines,
    cross_validate_model, full_evaluation,
    plot_roc_curves, plot_feature_importance,
    plot_ablation, ABLATION_GROUPS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ── Required data files ────────────────────────────────────────────────────────
#
#   Put these three files into your  data/  directory:
#
#     data/compressed_data_csv.gz    <- from your uploaded files
#     data/adjacency_matrix.npz      <- from your uploaded files
#     data/nodes_features.csv        <- from your uploaded files
#
EDGE_FEATURES_CSV = DATA / "edges_list.csv"   # main training data
ADJACENCY_NPZ    = DATA / "adjacency_matrix.npz"      # road graph
NODES_CSV        = DATA / "nodes_features.csv"         # node coordinates

NIGHT_BANDS = {0, 1, 2, 10, 11}


# ── Step 0: File checker ───────────────────────────────────────────────────────

def check_data_files():
    required = {
        EDGE_FEATURES_CSV: "Main edge feature table  (393k edges x 25 cols)",
        ADJACENCY_NPZ:    "Sparse directed road graph (154k nodes)",
        NODES_CSV:        "Node coordinates + safety stats",
    }
    print("\n-- Checking data/  folder -----------------------------------")
    missing = []
    for path, desc in required.items():
        exists = path.exists()
        size   = f"({path.stat().st_size/1e6:.1f} MB)" if exists else ""
        mark   = "OK" if exists else "MISSING"
        print(f"  [{mark}]  {path.name:<35} {size}  {desc}")
        if not exists:
            missing.append(path)

    if missing:
        print(f"\n  ERROR: {len(missing)} file(s) not found.")
        print(f"  Copy them into:  {DATA}/")
        print("  Then re-run this script.\n")
        sys.exit(1)
    print("  All files present.\n")


# ── Step 1: Load training data ─────────────────────────────────────────────────

def load_training_data(use_full: bool, sample_size: int) -> pd.DataFrame:
    """
    Load edge features from edges_list.csv.
    use_full=False  → sample sample_size rows for faster iteration
    use_full=True   → load all 393k rows for production training
    """
    print(f"  Source: {EDGE_FEATURES_CSV}")
    df = pd.read_csv(EDGE_FEATURES_CSV)
    total = len(df)
    print(f"  Rows in file: {total:,}")

    if not use_full:
        df = df.sample(min(sample_size, total), random_state=42)
        print(f"  Using sample: {len(df):,} rows  "
              f"(use --full to train on all {total:,})")
    else:
        print(f"  Using full dataset: {total:,} rows")
    return df


# ── Step 5a: Build full CSS cache ──────────────────────────────────────────────

def build_css_cache(model, feat_names: list, out_path: str) -> None:
    """
    Score EVERY edge in edges_list.csv across all 12 time bands.
    Writes  css_cache.csv  used by the API and demo map.
    """
    print(f"\n  Source: {EDGE_FEATURES_CSV}")
    df_all = pd.read_csv(EDGE_FEATURES_CSV)

    df_all = add_interaction_features(df_all)

    static_cols = [c for c in feat_names if c not in {
        "time_band", "is_night", "is_weekend",
        "night_x_road", "night_x_safe_poi", "night_x_risky_poi",
    }]

    unique_edges = df_all[
        ["edge_id", "u", "v", "u_idx", "v_idx"] + static_cols
    ].drop_duplicates("edge_id")
    print(f"  Unique edges: {len(unique_edges):,}  x  12 time bands  = "
          f"{len(unique_edges)*12:,} predictions")

    records = []
    for tb in range(12):
        is_night = int(tb in NIGHT_BANDS)
        sub = unique_edges.copy()
        sub["time_band"]         = tb
        sub["is_night"]          = is_night
        sub["is_weekend"]        = 0
        sub["night_x_road"]      = sub["is_night"] * sub["road_type_encoded"]
        sub["night_x_safe_poi"]  = sub["is_night"] * sub["safe_poi_count_100m"]
        sub["night_x_risky_poi"] = sub["is_night"] * sub["risky_poi_count_100m"]
        # night_x_lighting removed — lighting_score excluded as leaky feature
        for col in feat_names:
            if col not in sub.columns:
                sub[col] = 0

        X   = sub[feat_names].fillna(0).values.astype("float32")
        css = model.predict_proba(X)[:, 1]

        out = sub[["edge_id", "u", "v", "u_idx", "v_idx"]].copy()
        out["time_band"] = tb
        out["css_score"] = css
        records.append(out)
        print(f"    band={tb:>2}  mean_css={css.mean():.4f}")

    cache = pd.concat(records, ignore_index=True)
    cache.to_csv(out_path, index=False)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Saved {len(cache):,} rows ({size_mb:.0f} MB) -> {out_path}")


# ── Report writer ──────────────────────────────────────────────────────────────

def write_report(all_results, ablation_df, gbm_r, out_dir):
    lines = [
        "# Fear-Free Night Navigator - Evaluation Report\n",
        "## Model Comparison\n",
        "| Model | CV AUC | Train AUC | Precision | Recall | F1 | Brier |",
        "|-------|--------|-----------|-----------|--------|-----|-------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['label']} | {r['cv_auc']:.4f} | {r['auc']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['brier']:.4f} |"
        )

    if not ablation_df.empty:
        lines += [
            "\n## Ablation Study\n",
            "| Removed Group | AUC Without | AUC Delta | Features |",
            "|---------------|-------------|-----------|---------|",
        ]
        for _, row in ablation_df.iterrows():
            lines.append(
                f"| {row['removed_group']} | {row['auc_without']:.4f} | "
                f"{row['auc_delta']:+.4f} | {row['features_removed']} |"
            )

    lines += [
        "\n## Target Checks\n",
        "| Metric | Target | Achieved | Pass? |",
        "|--------|--------|----------|-------|",
        f"| CV AUC    | >0.80 | {gbm_r['cv_auc']:.4f}    | {'Yes' if gbm_r['cv_auc']>0.80 else 'No'} |",
        f"| Train AUC | >0.82 | {gbm_r['auc']:.4f}       | {'Yes' if gbm_r['auc']>0.82 else 'No'} |",
        f"| Precision | >0.75 | {gbm_r['precision']:.4f}  | {'Yes' if gbm_r['precision']>0.75 else 'No'} |",
        f"| Recall    | >0.70 | {gbm_r['recall']:.4f}     | {'Yes' if gbm_r['recall']>0.70 else 'No'} |",
        f"| F1        | >0.74 | {gbm_r['f1']:.4f}         | {'Yes' if gbm_r['f1']>0.74 else 'No'} |",
        f"| Brier     | <0.20 | {gbm_r['brier']:.4f}      | {'Yes' if gbm_r['brier']<0.20 else 'No'} |",
    ]

    p = out_dir / "evaluation_report.md"
    p.write_text("\n".join(lines))
    print(f"  Report saved -> {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Train on all 393k rows instead of 80k sample")
    parser.add_argument("--sample-size", type=int, default=80_000,
                        help="Sample size when not using --full (default: 80000)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation study to save time")
    args = parser.parse_args()

    # Step 0: check files
    check_data_files()

    # Step 1: load
    print("=" * 60)
    print("STEP 1  Load edge features")
    print("=" * 60)
    df = load_training_data(use_full=args.full, sample_size=args.sample_size)
    describe_features(df)
    check_for_leakage(df)

    # Explicit sanity check: css_score must NOT be in FEATURE_COLS
    assert "css_score" not in FEATURE_COLS, "LEAK: css_score in FEATURE_COLS!"
    assert "perceived_risk_score" not in FEATURE_COLS, "LEAK: perceived_risk_score in FEATURE_COLS!"
    print(f"  [leakage-check] FEATURE_COLS has {len(FEATURE_COLS)} features, {len(LEAKY_COLS)} excluded.")

    # Step 2: features
    print("\n" + "=" * 60)
    print("STEP 2  Build feature matrix (22 features + 4 interaction terms)")
    print("=" * 60)
    df_feat    = add_interaction_features(df)
    X, y, feat_names = get_feature_matrix(df_feat)
    print(f"  X: {X.shape}  |  Features: {len(feat_names)}")

    # Step 3: label check
    print("\n" + "=" * 60)
    print("STEP 3  Label distribution check")
    print("=" * 60)
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  label={lbl}: {cnt:>8,}  ({100*cnt/len(y):.1f}%)")
    ratio = counts.max() / counts.min()
    print(f"  Imbalance: {ratio:.2f}x  {'OK' if ratio < 3 else 'WARNING: imbalanced'}")

    # Step 4: train + CV
    print("\n" + "=" * 60)
    print("STEP 4  Train & evaluate (3-fold stratified CV)")
    print("=" * 60)
    gbm_aucs = cross_validate_model(build_gbm(), X, y, n_splits=3, label="GBM")

    baselines = build_baselines(feat_names)
    bl_aucs   = {}
    for name, est in baselines.items():
        bl_aucs[name] = cross_validate_model(est, X, y, n_splits=3, label=name).mean()

    all_results = []
    gbm_res = full_evaluation(build_gbm(), X, y, feat_names, label="GBM (our model)")
    gbm_res.update({"y_true": y, "cv_auc": gbm_aucs.mean()})
    all_results.append(gbm_res)

    for name, est in build_baselines(feat_names).items():
        res = full_evaluation(est, X, y, feat_names, label=name)
        res.update({"y_true": y, "cv_auc": bl_aucs[name]})
        all_results.append(res)

    print(f"\n{'Model':<25} {'CV AUC':>8} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Brier':>8}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['label']:<25} {r['cv_auc']:>8.4f} {r['auc']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['brier']:>8.4f}")

    gbm_r = all_results[0]
    print("\n  Target checks:")
    print("\n  NOTE on high AUC:")
    print("  The label was derived as css_score > 0.5, and css_score was built from")
    print("  lighting, road_type, police/hospital proximity, and POI density.")
    print("  These same features are in our training set, so the model learns")
    print("  the labelling formula back — hence AUC near 1.0.")
    print("  css_score and perceived_risk_score are explicitly excluded (leakers).")
    print("  AUC reflects formula-learning, not blind generalisation.")
    print()
    for desc, ok in [
        ("CV AUC > 0.80",    gbm_r["cv_auc"]   > 0.80),
        ("Train AUC > 0.82", gbm_r["auc"]       > 0.82),
        ("Precision > 0.75", gbm_r["precision"] > 0.75),
        ("Recall > 0.70",    gbm_r["recall"]    > 0.70),
        ("F1 > 0.74",        gbm_r["f1"]        > 0.74),
        ("Brier < 0.20",     gbm_r["brier"]     < 0.20),
    ]:
        print(f"    {'PASS' if ok else 'FAIL'}  {desc}")

    plot_roc_curves(all_results, str(OUT / "roc_curve.png"))
    plot_feature_importance(gbm_res["model"], feat_names, str(OUT / "feature_importance.png"))

    # Step 4e: ablation
    ablation_df = pd.DataFrame()
    if not args.skip_ablation:
        print("\n" + "=" * 60)
        print("STEP 4e  Ablation study")
        print("=" * 60)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        cv   = StratifiedKFold(3, shuffle=True, random_state=42)
        rows = []
        for grp, drop in ABLATION_GROUPS.items():
            keep = [i for i, f in enumerate(feat_names) if f not in drop]
            m    = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=42)
            auc  = cross_val_score(m, X[:, keep], y, cv=cv,
                                   scoring="roc_auc", n_jobs=1).mean()
            delta = round(auc - gbm_r["cv_auc"], 4)
            rows.append({"removed_group": grp, "auc_without": round(auc, 4),
                         "auc_delta": delta, "features_removed": ", ".join(drop)})
            print(f"  {grp:20s}: AUC={auc:.4f}  delta={delta:+.4f}")

        ablation_df = pd.DataFrame(rows).sort_values("auc_delta")
        ablation_df.to_csv(OUT / "ablation_results.csv", index=False)
        plot_ablation(ablation_df, str(OUT / "ablation_plot.png"))
    else:
        print("\n  (ablation skipped)")

    write_report(all_results, ablation_df, gbm_r, OUT)

    # Step 5: save model + build cache
    print("\n" + "=" * 60)
    print("STEP 5  Save model & build CSS cache (all 393k edges x 12 bands)")
    print("=" * 60)
    joblib.dump(gbm_res["model"], OUT / "safety_model.pkl")
    print(f"  Model saved -> outputs/safety_model.pkl")
    build_css_cache(gbm_res["model"], feat_names, str(OUT / "css_cache.csv"))

    # Compress CSV → NPZ (5x smaller, commit this instead of the CSV)
    print("\n  Compressing CSS cache to NPZ ...")
    from compress_cache import csv_to_npz
    csv_to_npz(OUT / "css_cache.csv", OUT / "css_cache.npz")
    (OUT / "css_cache.csv").unlink()   # delete the CSV, keep only NPZ
    print("  outputs/css_cache.csv deleted — commit outputs/css_cache.npz instead")

    # Done
    print("\n" + "=" * 60)
    print("DONE — outputs/")
    print("=" * 60)
    for f in sorted(OUT.glob("*")):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name:<35} {mb:.1f} MB")

    print("\n  Run next:")
    print("    python3 tests/test_router.py        # 26 tests")
    print("    python3 demo_map.py                 # interactive map")
    print("    cd src && uvicorn main:app --reload  # API on :8000")


if __name__ == "__main__":
    main()
