#!/usr/bin/env bash
# start_render.sh — Render-safe startup script for Fear-Free Night Navigator
# Does NOT use pyenv (Render provides Python directly via PYTHON_VERSION env var)
set -euo pipefail

echo "[startup] Fear-Free Night Navigator starting..."
echo "[startup] Python: $(python --version)"
echo "[startup] Checking required output files..."

# Verify trained model files exist
MISSING=0
for f in outputs/safety_model.pkl outputs/css_cache.csv; do
  if [ -f "$f" ]; then
    SIZE=$(du -sh "$f" | cut -f1)
    echo "[startup] ✓ $f ($SIZE)"
  else
    echo "[startup] ✗ MISSING: $f"
    MISSING=$((MISSING+1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo "[startup] ERROR: $MISSING required file(s) missing."
  echo "[startup] Train locally first: python3 src/train_fast.py"
  echo "[startup] Then commit outputs/ and push to GitHub."
  exit 1
fi

echo "[startup] All required files present. Starting API server..."

cd src
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 65 \
  --log-level info
