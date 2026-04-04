#!/usr/bin/env bash
set -euo pipefail

echo "[startup] Fear-Free Night Navigator starting..."
echo "[startup] Python: $(python3 --version)"
echo "[startup] Checking required output files..."

MISSING=0
for f in outputs/css_cache.npz outputs/safety_model.pkl data/adjacency_matrix.npz data/nodes_features.csv; do
  if [ -f "$f" ]; then
    SIZE=$(du -sh "$f" | cut -f1)
    echo "[startup] OK  $f ($SIZE)"
  else
    echo "[startup] MISSING: $f"
    MISSING=$((MISSING+1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo "[startup] ERROR: $MISSING required file(s) missing. Train locally and commit outputs/."
  exit 1
fi

echo "[startup] All files present. Starting API..."
cd src
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 65 \
  --log-level info
