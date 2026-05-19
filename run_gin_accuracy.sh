#!/usr/bin/env bash
# =============================================================================
# run_gin_accuracy.sh
#
# Identisch mit run_experiments.sh, aber:
#   - Nur GIN (Phase 3 + Phase 4)
#   - 300 train-epochs statt 20
#   - Alle 5 compile modes unverÃ¤ndert
#   - Alle anderen Parameter identisch zur originalen run_experiments.sh
#
# Usage:
#   bash run_gin_accuracy.sh
#   bash run_gin_accuracy.sh --dry-run
#   bash run_gin_accuracy.sh --dynamic=true                        # dynamic=True  (always symbolic)
#   bash run_gin_accuracy.sh --dynamic=false                       # dynamic=False (always static)
#   bash run_gin_accuracy.sh --dynamic=auto                        # dynamic=None  (automatic, default)
#   bash run_gin_accuracy.sh --resume=3:pyg:gin:ogbn-arxiv
#   bash run_gin_accuracy.sh --script=gnn_compile_benchmark_v29_workaround.py
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="$BASE_DIR/data"
OUT="$BASE_DIR/results_gin_300epochs"

MODES=(eager default reduce-overhead max-autotune max-autotune-no-cudagraphs)

BASE_ARGS=(
    --repeats 30
    --warmup 5
    --train-epochs 300
    --train-warmup 5
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
RESUME_FROM=""
SCRIPT_NAME="gnn_compile_benchmark_v29_workaround.py"
DYNAMIC="auto"   # default: dynamic=None (automatic dynamic shapes)

for arg in "$@"; do
  case $arg in
    --dry-run)    DRY_RUN=1 ;;
    --resume=*)   RESUME_FROM="${arg#*=}" ;;
    --script=*)   SCRIPT_NAME="${arg#*=}" ;;
    --dynamic=*)  DYNAMIC="${arg#*=}" ;;
    *) echo "X Unknown argument: $arg"; exit 1 ;;
  esac
done

# Validate --dynamic value.
case "$DYNAMIC" in
  auto|true|false) ;;
  *) echo "X --dynamic must be 'auto', 'true', or 'false' (got: '$DYNAMIC')"; exit 1 ;;
esac

# Build the --dynamic flag to forward to the Python script.
DYNAMIC_ARG=(--dynamic "$DYNAMIC")

if [[ "$SCRIPT_NAME" = /* ]]; then
    SCRIPT="$SCRIPT_NAME"
else
    SCRIPT="$BASE_DIR/$SCRIPT_NAME"
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -f "$SCRIPT" ]] || { echo "X Script not found: $SCRIPT"; exit 1; }
mkdir -p "$DATA" "$OUT"

echo "============================================================"
echo "  Script : $SCRIPT"
echo "  Data   : $DATA"
echo "  Output : $OUT"
echo "  Modes   : ${MODES[*]}"
echo "  Dynamic : $DYNAMIC  (torch.compile dynamic= argument)"
echo "  Resume  : ${RESUME_FROM:-<none>}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
RESUME_ACTIVE=1
_RESUME_FOUND=0

should_run() {
    local key="$1"
    if [[ -z "$RESUME_FROM" ]]; then return 0; fi
    if [[ $RESUME_ACTIVE -eq 1 ]]; then
        if [[ "$key" == "$RESUME_FROM" ]]; then
            echo "Resuming from: $key"
            RESUME_ACTIVE=0
            _RESUME_FOUND=1
            return 0
        else
            return 1
        fi
    fi
    return 0
}

run() {
    local key="$1";   shift
    local label="$1"; shift

    if ! should_run "$key"; then
        echo "Skipping $key"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  [$key] $label"
    echo "============================================================"
    echo "  python $SCRIPT ${DYNAMIC_ARG[*]} $*"

    if [[ $DRY_RUN -eq 0 ]]; then
        python "$SCRIPT" "${DYNAMIC_ARG[@]}" "$@"
    fi
}

# ---------------------------------------------------------------------------
# Phase 3 - GIN auf ogbn-arxiv (node classification)
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 3 - GIN / ogbn-arxiv"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    run "3:$FRAMEWORK:gin:ogbn-arxiv" "$FRAMEWORK GIN / ogbn-arxiv / all modes" \
        --framework "$FRAMEWORK" --model-name gin --dataset ogbn-arxiv \
        --hidden 256 --num-layers 3 --dropout 0.5 \
        --modes "${MODES[@]}" \
        --data-root "$DATA" --out-dir "$OUT" \
        "${BASE_ARGS[@]}"
done

# ---------------------------------------------------------------------------
# Phase 4 - GIN auf ogbl-collab (link prediction)
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 4 - GIN / ogbl-collab"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    run "4:$FRAMEWORK:gin:ogbl-collab" "$FRAMEWORK GIN / ogbl-collab / all modes" \
        --framework "$FRAMEWORK" --model-name gin --dataset ogbl-collab \
        --hidden 256 --num-layers 3 --dropout 0.5 \
        --modes "${MODES[@]}" \
        --data-root "$DATA" --out-dir "$OUT" \
        "${BASE_ARGS[@]}"
done

# ---------------------------------------------------------------------------
# Warn if resume key was never matched
# ---------------------------------------------------------------------------
if [[ -n "$RESUME_FROM" && $_RESUME_FOUND -eq 0 ]]; then
    echo ""
    echo "WARNING: --resume key '$RESUME_FROM' was never matched."
    echo "  Valid keys:"
    echo "    3:pyg:gin:ogbn-arxiv"
    echo "    3:dgl:gin:ogbn-arxiv"
    echo "    4:pyg:gin:ogbl-collab"
    echo "    4:dgl:gin:ogbl-collab"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  All experiments complete.  Results -> $OUT"
echo "################################################################"
echo ""
echo "Runs completed:"
echo "  Phase 3 - 2 runs  (PyG+DGL GIN x ogbn-arxiv,  all modes, 300 epochs)"
echo "  Phase 4 - 2 runs  (PyG+DGL GIN x ogbl-collab, all modes, 300 epochs)"
echo "  Total   - 4 runs"
echo "  Dynamic : $DYNAMIC  (torch.compile dynamic= used for all runs)"
