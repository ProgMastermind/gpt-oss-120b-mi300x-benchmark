#!/bin/bash
# =============================================================================
# rebuild_aiter_gfx942.sh — Rebuild patched AITER v0.1.17-rc0 on MI300X (gfx942)
#
# This script applies the aiter-side changes made for gpt-oss-120b on MI300X:
#   - Triton a4w4 MoE kernel now uses gfx942 tile geometry (was hardcoded to
#     gfx950/gfx1250 512x8).
#   - __Float4_e2m1fn_x2 is no longer gated off on gfx942.
#   - CK-Tile moe_cktile2stages generates pk_fp4 instances for non-gfx950.
#   - gptoss_fp4_tuned_fmoe.csv contains cu_num=304 rows.
#
# IMPORTANT: gfx942 (MI300X) does NOT have the v_cvt_scalef32_pk_fp4_* hardware
# instruction. The CK/FlyDSL fp4x2 paths will compile but the fp4x2 quant/dequant
# and CK-Tile pk_fp4 conversions fall back to zero-stubs in aiter. Those paths
# are NOT expected to produce correct results; the only working fp4 MoE path on
# gfx942 is the Triton a4w4 path fixed above.
#
# Usage on the droplet:
#   bash rebuild_aiter_gfx942.sh
# =============================================================================
set -euo pipefail

LOG() { echo -e "\n\033[1;34m[rebuild]\033[0m $*"; }
WARN() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
ERR() { echo -e "\033[1;31m[ERR]\033[0m $*" >&2; }

# ---------------------------------------------------------------------------
# 0. Preflight / paths
# ---------------------------------------------------------------------------
AITER_DIR="${AITER_DIR:-/root/aiter}"
if [[ ! -d "$AITER_DIR" ]]; then
    ERR "aiter not found at $AITER_DIR; set AITER_DIR and re-run"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG "SCRIPT_DIR: $SCRIPT_DIR"
LOG "GPU arch: $(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx[0-9]+' | head -1 || echo unknown)"

# ---------------------------------------------------------------------------
# 0b. Apply the aiter gfx942 patch if present
# ---------------------------------------------------------------------------
PATCH_FILE="$SCRIPT_DIR/aiter_gfx942.patch"
if [[ -f "$PATCH_FILE" ]]; then
    LOG "Applying aiter patch: $PATCH_FILE"
    cd "$AITER_DIR"
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
        LOG "Patch applied successfully."
    else
        WARN "Patch already applied or does not apply cleanly. Skipping."
    fi
else
    WARN "aiter_gfx942.patch not found. Assuming aiter is already patched."
fi

# ---------------------------------------------------------------------------
# 1. Stop any running server
# ---------------------------------------------------------------------------
LOG "Stopping ATOM server processes..."
pkill -f "openai_server" 2>/dev/null || true
pkill -f "atom.entrypoints" 2>/dev/null || true
sleep 2
fuser -k 8000/tcp 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Clear caches and stale JIT artifacts
# ---------------------------------------------------------------------------
LOG "Clearing compile caches..."
rm -rf /root/.cache/atom/* 2>/dev/null || true
rm -rf /root/.flydsl/cache/* 2>/dev/null || true
rm -rf /tmp/atom_* 2>/dev/null || true
rm -rf /tmp/flydsl_* 2>/dev/null || true

# aiter JIT cache location
AITER_JIT_DIR="${HOME}/.aiter/jit"
if [[ -d "$AITER_JIT_DIR" ]]; then
    LOG "Removing aiter JIT artifacts under $AITER_JIT_DIR ..."
    rm -rf "$AITER_JIT_DIR/build/module_quant" \
           "$AITER_JIT_DIR/build/module_moe_cktile2stages" \
           "$AITER_JIT_DIR/build/module_moe_ck2stages" \
           "$AITER_JIT_DIR/build/module_moe_sorting" \
           "$AITER_JIT_DIR/build/module_moe_asm" \
           "$AITER_JIT_DIR/build/module_dsv4_rotate_quant" 2>/dev/null || true
    rm -f "$AITER_JIT_DIR/module_quant.so" \
           "$AITER_JIT_DIR/module_moe_cktile2stages.so" \
           "$AITER_JIT_DIR/module_moe_ck2stages.so" \
           "$AITER_JIT_DIR/module_moe_sorting.so" \
           "$AITER_JIT_DIR/module_moe_asm.so" \
           "$AITER_JIT_DIR/module_dsv4_rotate_quant.so" 2>/dev/null || true
fi

# Triton autotune/cache (forces re-tune with new tile sizes)
rm -rf "${HOME}/.triton" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Rebuild aiter modules
# ---------------------------------------------------------------------------
LOG "Rebuilding aiter modules with AITER_REBUILD=1..."
cd "$AITER_DIR"

export AITER_REBUILD=1
export AITER_LOG_LEVEL=WARNING

# module_quant picks up -D__Float4_e2m1fn_x2 and fp4x2 quant paths
python3 - <<'PY'
import os
os.environ["AITER_REBUILD"] = "1"
import aiter.ops.quant
print("module_quant rebuilt")
PY

# module_moe_cktile2stages picks up pk_fp4 codegen changes
python3 - <<'PY'
import os
os.environ["AITER_REBUILD"] = "1"
import aiter.ops.moe_op
print("moe_op modules rebuilt")
PY

# Explicit import of the CK-Tile entry points to ensure the .so exists
python3 - <<'PY'
import os
os.environ["AITER_REBUILD"] = "1"
from aiter.ops.moe_op import moe_cktile2stages_gemm1, moe_cktile2stages_gemm2
print("module_moe_cktile2stages rebuilt")
PY

# ---------------------------------------------------------------------------
# 4. Quick sanity checks
# ---------------------------------------------------------------------------
LOG "Running aiter sanity checks..."
python3 - <<'PY'
import torch
import aiter
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import get_kernel_config
from aiter.ops.triton.moe.moe_routing.routing import RoutingData

rd = RoutingData(block_m=64, gate_scal=None, expt_hist=None, n_expts_tot=128, n_expts_act=4, expt_data=None)
cfg = get_kernel_config(4096, 3072, 3072, rd)
print("Triton a4w4 config (gfx942 expected):", cfg)
assert cfg["block_m"] == 64
assert cfg["block_n"] == 128
assert cfg["num_warps"] == 8
print("Triton gfx942 heuristic looks correct")
PY

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
LOG "============================================================"
LOG "REBUILD COMPLETE"
LOG "============================================================"
LOG "Triton a4w4 MoE path has been re-tuned for gfx942."
WARN "CK/FlyDSL fp4x2 paths on gfx942 are hardware-limited (no fp4 pack/unpack instruction)."
WARN "They may compile but will produce zeros. Use ATOM_USE_TRITON_MOE=1 for correct gpt-oss-120b inference."
LOG ""
LOG "Next steps:"
LOG "  1. Triton baseline (should restore previous ~291 tok/s after any tuning):"
LOG "     ATOM_USE_TRITON_MOE=1 python -m atom.entrypoints.openai_server \\"
LOG "       --model <path-to-gpt-oss-120b> --kv_cache_dtype bf16 -tp 8"
LOG ""
LOG "  2. Optional env overrides for Triton a4w4 micro-benchmarking:"
LOG "     AITER_TRITON_MOE_BLOCK_N=128 AITER_TRITON_MOE_NUM_WARPS=4"
LOG ""
LOG "  3. CK/FlyDSL attempt (NOT expected to work on gfx942):"
LOG "     unset ATOM_USE_TRITON_MOE  # use atom/ATOM default CK/FlyDSL path"
LOG "     python -m atom.entrypoints.openai_server \\"
LOG "       --model <path-to-gpt-oss-120b> --kv_cache_dtype bf16 -tp 8"
