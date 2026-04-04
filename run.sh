#!/usr/bin/env bash
# =============================================================================
# run.sh — Fear-Free Night Navigator · Full Automation Script
#
# USAGE:
#   chmod +x run.sh
#   ./run.sh                   # train (80k sample) + start API
#   ./run.sh --full            # train (full 393k dataset) + start API
#   ./run.sh --skip-train      # skip training, start API immediately  ← Render
#   ./run.sh --skip-ablation   # faster training, skip ablation
#   ./run.sh --port 9000       # different port
#
# RENDER DEPLOY:
#   Build Command:  pip install -r requirements.txt
#   Start Command:  bash run.sh --skip-train
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

FULL_TRAIN=false
SKIP_TRAIN=false
SKIP_ABLATION=false
PORT="${PORT:-8000}"   # Render injects PORT automatically

while [[ $# -gt 0 ]]; do
  case $1 in
    --full)          FULL_TRAIN=true   ;;
    --skip-train)    SKIP_TRAIN=true   ;;
    --skip-ablation) SKIP_ABLATION=true ;;
    --port)          PORT="$2"; shift  ;;
    -h|--help) grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) warn "Unknown argument: $1" ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DATA_DIR="$SCRIPT_DIR/data"
OUT_DIR="$SCRIPT_DIR/outputs"

header "Fear-Free Night Navigator"
info "Root: $SCRIPT_DIR  |  Port: $PORT"

# ── Step 1: Python ─────────────────────────────────────────────────────────
header "Step 1 · Python version"
# Support pyenv if available
if command -v pyenv &>/dev/null; then
  PYTHON=$(pyenv which python3 2>/dev/null || command -v python3)
else
  PYTHON=$(command -v python3 || command -v python || error "python3 not found")
fi
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
[[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]] && \
  error "Python 3.10+ required. Found: $PY_VERSION"
success "Python $PY_VERSION"

# ── Step 2: Data files ──────────────────────────────────────────────────────
header "Step 2 · Data files"
MISSING=0
check_file() {
  if [[ -f "$1" ]]; then
    success "$(basename "$1")  ($(du -sh "$1" 2>/dev/null | cut -f1))  — $2"
  else
    echo -e "${RED}[MISS]${RESET}  $(basename "$1")  — $2"; MISSING=$((MISSING+1))
  fi
}
check_file "$DATA_DIR/adjacency_matrix.npz"  "Sparse road graph"
check_file "$DATA_DIR/nodes_features.csv"    "Node coordinates"

# edges_list.csv only needed for training, not for API
if [[ "$SKIP_TRAIN" == false ]]; then
  if [[ ! -f "$DATA_DIR/edges_list.csv" ]] && [[ ! -f "$DATA_DIR/compressed_data_csv.gz" ]]; then
    echo -e "${RED}[MISS]${RESET}  edges_list.csv (or compressed_data_csv.gz) — training data"
    MISSING=$((MISSING+1))
  fi
fi
[[ "$MISSING" -gt 0 ]] && error "$MISSING data file(s) missing in $DATA_DIR/"

# ── Step 3: Virtual environment ─────────────────────────────────────────────
header "Step 3 · Virtual environment"
VENV_DIR="$SCRIPT_DIR/.venv"

# On Render, pip install already ran during build — skip venv creation
if [[ "${RENDER:-}" == "true" ]]; then
  info "Running on Render — using system Python (build step installed deps)"
  PYTHON_BIN="python3"
  PIP_BIN="pip3"
else
  # Local: create/reuse venv
  if [[ -d "$VENV_DIR" ]]; then
    EXISTING_PY=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    if [[ "$EXISTING_PY" != "$PY_VERSION" ]]; then
      warn "Removing old .venv (Python $EXISTING_PY → $PY_VERSION)"
      rm -rf "$VENV_DIR"
    fi
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating .venv with Python $PY_VERSION ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Created $VENV_DIR"
  else
    success "Existing .venv found (Python $EXISTING_PY)"
  fi
  source "$VENV_DIR/bin/activate"
  info "Installing dependencies ..."
  "$VENV_DIR/bin/pip" install --upgrade pip -q
  "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q
  success "Dependencies installed"
  PYTHON_BIN="$VENV_DIR/bin/python"
  PIP_BIN="$VENV_DIR/bin/pip"
fi

# ── Step 4: Training ─────────────────────────────────────────────────────────
header "Step 4 · ML Training"

if [[ "$SKIP_TRAIN" == true ]]; then
  warn "Skipping training (--skip-train)"
  MISSING_OUT=0
  # Must have either NPZ or CSV cache, plus the model
  if [[ ! -f "$OUT_DIR/safety_model.pkl" ]]; then
    echo -e "${RED}[MISS]${RESET}  outputs/safety_model.pkl"; MISSING_OUT=$((MISSING_OUT+1))
  else
    success "outputs/safety_model.pkl  ($(du -sh "$OUT_DIR/safety_model.pkl" | cut -f1))"
  fi
  if [[ -f "$OUT_DIR/css_cache.npz" ]]; then
    success "outputs/css_cache.npz  ($(du -sh "$OUT_DIR/css_cache.npz" | cut -f1))  ← fast path"
  elif [[ -f "$OUT_DIR/css_cache.csv" ]]; then
    warn "outputs/css_cache.csv found — consider converting to NPZ for faster startup"
    warn "Run: python3 src/compress_cache.py"
  else
    echo -e "${RED}[MISS]${RESET}  outputs/css_cache.npz (or css_cache.csv)"; MISSING_OUT=$((MISSING_OUT+1))
  fi
  [[ "$MISSING_OUT" -gt 0 ]] && error "Missing outputs. Run without --skip-train first."
else
  mkdir -p "$OUT_DIR"
  TRAIN_ARGS=""
  [[ "$FULL_TRAIN"    == true ]] && TRAIN_ARGS="$TRAIN_ARGS --full"
  [[ "$SKIP_ABLATION" == true ]] && TRAIN_ARGS="$TRAIN_ARGS --skip-ablation"
  [[ "$FULL_TRAIN"    == true ]] && info "Mode: FULL dataset" || info "Mode: 80k sample (use --full for complete dataset)"
  START_T=$SECONDS
  "$PYTHON_BIN" "$SRC_DIR/train_fast.py" $TRAIN_ARGS
  success "Training complete in $((SECONDS - START_T))s"
  # Verify NPZ was produced
  [[ -f "$OUT_DIR/css_cache.npz" ]] && \
    success "css_cache.npz ready: $(du -sh "$OUT_DIR/css_cache.npz" | cut -f1)" || \
    warn "css_cache.npz not found — run: python3 src/compress_cache.py"
fi

# ── Step 5: Test suite ───────────────────────────────────────────────────────
header "Step 5 · Test suite"
info "Running 26 tests ..."
if "$PYTHON_BIN" "$SCRIPT_DIR/tests/test_router.py"; then
  success "All tests passed"
else
  error "Tests failed. Fix errors before starting the server."
fi

# ── Step 6: Start API ─────────────────────────────────────────────────────────
header "Step 6 · Starting FastAPI"
info "Listening on http://0.0.0.0:$PORT"
info "Docs:    http://localhost:$PORT/docs"
info "Health:  http://localhost:$PORT/health"
echo ""

cd "$SRC_DIR"
exec "$PYTHON_BIN" -m uvicorn main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 65 \
  --log-level info
