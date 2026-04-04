"""
compress_cache.py — Convert css_cache.csv to css_cache.npz

Reduces the CSS cache from ~37-302 MB CSV down to ~5-7 MB compressed NPZ.
Run this after training, before committing to git or deploying to Render.

Usage:
    python3 src/compress_cache.py
    python3 src/compress_cache.py --input outputs/css_cache_full.csv
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT  = BASE / "outputs"


def csv_to_npz(csv_path: Path, npz_path: Path) -> None:
    print(f"  Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    np.savez_compressed(
        npz_path,
        u_idx     = df["u_idx"].values.astype(np.int32),
        v_idx     = df["v_idx"].values.astype(np.int32),
        time_band = df["time_band"].values.astype(np.int8),
        css_score = df["css_score"].values.astype(np.float32),
        edge_id   = df["edge_id"].values.astype(np.int32),
    )

    csv_mb = csv_path.stat().st_size / 1e6
    npz_mb = npz_path.stat().st_size / 1e6
    print(f"  {csv_path.name}  {csv_mb:.1f} MB  →  {npz_path.name}  {npz_mb:.2f} MB  "
          f"({csv_mb/npz_mb:.1f}x smaller)")


def load_css_npz(npz_path: Path) -> dict[int, dict]:
    """
    Load css_cache.npz and return a nested dict:
        { time_band: { (u_idx, v_idx): css_score } }
    This is what main.py uses at startup.
    """
    d    = np.load(npz_path)
    u    = d["u_idx"].astype(int)
    v    = d["v_idx"].astype(int)
    tb   = d["time_band"].astype(int)
    css  = d["css_score"].astype(float)

    css_by_band: dict[int, dict] = {}
    for band in range(12):
        mask = tb == band
        css_by_band[band] = dict(zip(zip(u[mask], v[mask]), css[mask]))

    return css_by_band


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(OUT / "css_cache.csv"),
                        help="Input CSV path (default: outputs/css_cache.csv)")
    parser.add_argument("--output", default=str(OUT / "css_cache.npz"),
                        help="Output NPZ path (default: outputs/css_cache.npz)")
    parser.add_argument("--keep-csv", action="store_true",
                        help="Keep the CSV after conversion (default: delete it)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    npz_path = Path(args.output)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run training first.")
        sys.exit(1)

    csv_to_npz(csv_path, npz_path)

    if not args.keep_csv:
        csv_path.unlink()
        print(f"  Deleted {csv_path.name} (use --keep-csv to retain it)")

    print(f"\n  Done. Commit outputs/css_cache.npz to git.")
    print(f"  Git will track {npz_path.stat().st_size/1e6:.2f} MB instead of the CSV.")
