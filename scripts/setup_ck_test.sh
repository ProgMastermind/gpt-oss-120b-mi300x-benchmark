#!/bin/bash
# =============================================================================
# setup_ck_test.sh — Install aiter v0.1.17-rc0 + ATOM v0.1.6-rc0 for CK kernel
#                     testing on MI300X (gfx942).
#
# Reuses the SAME container. Auto-finds and preserves model weights before
# wiping aiter/ATOM installations.
#
# Usage on the droplet:
#   bash setup_ck_test.sh          # find weights + install
#   bash setup_ck_test.sh --force  # skip weight-find confirmation
# =============================================================================
set -euo pipefail

LOG() { echo -e "\n\033[1;34m[setup_ck]\033[0m $*"; }
WARN() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
ERR() { echo -e "\033[1;31m[ERR]\033[0m $*" >&2; }

# ---------------------------------------------------------------------------
# 0. Preflight: confirm we're on the right box
# ---------------------------------------------------------------------------
LOG "Preflight checks..."
GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx[0-9]+' | head -1 || echo "unknown")
LOG "Detected GPU arch: ${GPU_ARCH}"
if [[ "$GPU_ARCH" != "gfx942" ]]; then
    WARN "Expected gfx942 (MI300X), got ${GPU_ARCH}. Proceeding anyway."
fi

PY_VER=$(python3 --version 2>&1)
LOG "Python: ${PY_VER}"

# ---------------------------------------------------------------------------
# 1. FIND and PRESERVE model weights (CRITICAL — do not re-download 240GB)
# ---------------------------------------------------------------------------
LOG "Searching for gpt-oss-120b model weights..."

WEIGHTS_DIR=""
SEARCH_PATHS=(
    "/models"
    "/root/models"
    "/data/models"
    "/workspace/models"
    "/opt/models"
    "/root/.cache/huggingface/hub"
    "/root/.cache/huggingface"
    "$HOME/models"
)

for p in "${SEARCH_PATHS[@]}"; do
    if [[ -d "$p" ]]; then
        # Look for gpt-oss in directory names or snapshot dirs
        HIT=$(find "$p" -maxdepth 4 -type d \( \
                -iname "*gpt*oss*120b*" -o \
                -iname "*gpt-oss*" \) 2>/dev/null | head -5)
        if [[ -n "$HIT" ]]; then
            WEIGHTS_DIR="$p"
            LOG "Found gpt-oss weights under: $p"
            LOG "  Matches: $HIT"
            break
        fi
    fi
done

# Also check for any large safetensors files as a fallback
if [[ -z "$WEIGHTS_DIR" ]]; then
    LOG "No gpt-oss dir name match. Searching for large safetensors files..."
    BIG_ST=$(find / -maxdepth 6 -name "*.safetensors" -size +1G 2>/dev/null | head -5 || true)
    if [[ -n "$BIG_ST" ]]; then
        LOG "Found large safetensors:"
        echo "$BIG_ST"
        WARN "Please confirm the parent directory of these files is your model dir."
        WARN "Set WEIGHTS_DIR manually and re-run if needed."
    fi
fi

if [[ -z "$WEIGHTS_DIR" ]]; then
    WARN "Could not auto-locate model weights."
    WARN "If you know the path, export WEIGHTS_DIR=/path/to/models and re-run."
    WARN "Continuing anyway — but you may need to re-download weights later."
    if [[ "$1" != "--force" ]]; then
        read -p "Continue without preserving weights? (y/N) " yn
        [[ "$yn" =~ ^[Yy]$ ]] || { ERR "Aborted by user."; exit 1; }
    fi
else
    LOG "Preserving weights at: ${WEIGHTS_DIR}"
    # Mark the dir so the wipe step skips it
    export PRESERVE_PATH="$WEIGHTS_DIR"
fi

# ---------------------------------------------------------------------------
# 2. STOP any running ATOM server
# ---------------------------------------------------------------------------
LOG "Stopping any running ATOM server..."
pkill -f "openai_server" 2>/dev/null || true
pkill -f "atom.entrypoints" 2>/dev/null || true
sleep 3
# Free port 8000 if held
fuser -k 8000/tcp 2>/dev/null || true
LOG "Server processes stopped."

# ---------------------------------------------------------------------------
# 3. WIPE old aiter + ATOM installations (preserve weights + base ROCm)
# ---------------------------------------------------------------------------
LOG "Wiping old aiter + ATOM installations..."

# Uninstall pip packages
pip uninstall -y aiter atom amd_aiter 2>/dev/null || true

# Remove source dirs (but NEVER touch PRESERVE_PATH)
SAFE_RM() {
    local target="$1"
    if [[ -z "$target" || "$target" == "/" ]]; then
        ERR "Refusing to rm empty/root path"; return 1
    fi
    if [[ -n "${PRESERVE_PATH:-}" && "$target" == "$PRESERVE_PATH"* ]]; then
        WARN "Skipping (inside preserve path): $target"
        return 0
    fi
    rm -rf "$target"
}

for d in /root/aiter /root/ATOM /opt/aiter /opt/ATOM /workspace/aiter /workspace/ATOM; do
    [[ -d "$d" ]] && SAFE_RM "$d"
done

# Clear compile caches (stale cache causes silent failures)
LOG "Clearing compile caches..."
rm -rf /root/.cache/atom/* 2>/dev/null || true
rm -rf /root/.flydsl/cache/* 2>/dev/null || true
rm -rf /tmp/atom_* 2>/dev/null || true
rm -rf /tmp/flydsl_* 2>/dev/null || true

LOG "Wipe complete."

# ---------------------------------------------------------------------------
# 4. CLONE and INSTALL aiter v0.1.17-rc0
# ---------------------------------------------------------------------------
LOG "Cloning aiter v0.1.17-rc0..."
cd /root
git clone --branch v0.1.17-rc0 --depth 1 https://github.com/ROCm/aiter.git aiter
cd /root/aiter

LOG "Installing aiter v0.1.17-rc0 (this builds HIP kernels, ~20-35 min)..."
# aiter needs to build against the ROCm toolchain in the container
pip install -e . --no-build-isolation 2>&1 | tail -30

LOG "Verifying aiter install..."
python3 -c "import aiter; print('aiter OK:', aiter.__file__)"
python3 -c "from aiter.ops.shuffle import moe_shuffle_scale; print('moe_shuffle_scale import OK')"

# ---------------------------------------------------------------------------
# 5. CLONE and INSTALL ATOM v0.1.6-rc0
# ---------------------------------------------------------------------------
LOG "Cloning ATOM v0.1.6-rc0..."
cd /root
git clone --branch v0.1.6-rc0 --depth 1 https://github.com/ROCm/ATOM.git ATOM
cd /root/ATOM

LOG "Installing ATOM v0.1.6-rc0..."
pip install -e . 2>&1 | tail -20

LOG "Verifying ATOM install..."
python3 -c "import atom; print('atom OK:', atom.__file__)"

# ---------------------------------------------------------------------------
# 6. PORT our input_norm / norm_before_fc fix to v0.1.6-rc0 eagle3_llama.py
# ---------------------------------------------------------------------------
LOG "Porting input_norm / norm_before_fc fix to v0.1.6-rc0..."

EAGLE3_FILE="/root/ATOM/atom/models/eagle3_llama.py"
if [[ ! -f "$EAGLE3_FILE" ]]; then
    ERR "eagle3_llama.py not found at $EAGLE3_FILE"
    exit 1
fi

# Apply the input_norm / norm_before_fc fix using the Python patcher.
# The patcher is idempotent and verifies all 4 edits.
PATCHER="/root/ATOM/scripts/apply_input_norm_fix.py"
if [[ ! -f "$PATCHER" ]]; then
    # Patcher lives in our scripts dir — copy it in if not part of ATOM repo
    PATCHER="/root/apply_input_norm_fix.py"
fi
if [[ -f "$PATCHER" ]]; then
    LOG "Applying input_norm fix via $PATCHER..."
    python3 "$PATCHER" "$EAGLE3_FILE"
else
    WARN "apply_input_norm_fix.py not found. Checking if fix already present..."
    if grep -q "input_norm" "$EAGLE3_FILE"; then
        LOG "input_norm fix already present. OK."
    else
        ERR "input_norm fix NOT applied and patcher missing!"
        ERR "scp apply_input_norm_fix.py to /root/ on the droplet and run:"
        ERR "  python3 /root/apply_input_norm_fix.py ${EAGLE3_FILE}"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
LOG "============================================================"
LOG "SETUP COMPLETE"
LOG "============================================================"
LOG "  aiter:  v0.1.17-rc0  at /root/aiter"
LOG "  ATOM:   v0.1.6-rc0   at /root/ATOM"
LOG "  GPU:    ${GPU_ARCH}"
if [[ -n "${PRESERVE_PATH:-}" ]]; then
    LOG "  Weights preserved at: ${PRESERVE_PATH}"
else
    WARN "  Weights location unknown — may need to re-download or set --model path"
fi
LOG ""
LOG "Next steps:"
LOG "  1. Start server with Triton baseline:"
LOG "     ATOM_USE_TRITON_MOE=1 python -m atom.entrypoints.openai_server \\"
LOG "       --model <path-to-gpt-oss-120b> --kv_cache_dtype bf16 -tp 1"
LOG ""
LOG "  2. Benchmark Triton (expect ~291 tok/s with Eagle3)"
LOG ""
LOG "  3. Test CK path:"
LOG "     ATOM_USE_TRITON_MOE=0 python -m atom.entrypoints.openai_server \\"
LOG "       --model <path-to-gpt-oss-120b> --kv_cache_dtype bf16 -tp 1"
LOG "     (This is the real test — may crash with LLVM error if no tuned"
LOG "      config exists for gfx942/304 CUs)"
LOG "============================================================"
