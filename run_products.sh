#!/usr/bin/env bash
# =============================================================================
# run_products.sh
#
# Identical structure and parameters to run_experiments.sh.
# ogbn-products: ~2.4M nodes, 61.9M edges, 47 classes, requires --use-sampling.
#
# Models  : gcn, gat  x  pyg, dgl  =  4 runs
# Modes   : eager, default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
# Output  : results_products/
#
# Usage:
#   bash run_products.sh
#   bash run_products.sh --dry-run
#   bash run_products.sh --dynamic=true                        # dynamic=True 
#   bash run_products.sh --dynamic=false                       # dynamic=False 
#   bash run_products.sh --dynamic=auto                        # dynamic=None  
#   bash run_products.sh --resume=5:dgl:gcn:ogbn-products
#   bash run_products.sh --script=gnn_compile_benchmark_v29_dynamic.py
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="$BASE_DIR/data"
OUT="$BASE_DIR/results_products"

MODES=(eager default reduce-overhead max-autotune max-autotune-no-cudagraphs)

BASE_ARGS=(
    --repeats 30
    --warmup 5
    --train-epochs 20
    --train-warmup 5
    --use-sampling
    --timeout 18000
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
RESUME_FROM=""
SCRIPT_NAME="gnn_compile_benchmark_v29_parametric.py"
DYNAMIC="auto"   # default: dynamic=None (automatic dynamic shapes)

for arg in "$@"; do
  case $arg in
    --dry-run)    DRY_RUN=1 ;;
    --resume=*)   RESUME_FROM="${arg#*=}" ;;
    --script=*)   SCRIPT_NAME="${arg#*=}" ;;
    --dynamic=*)  DYNAMIC="${arg#*=}" ;;
    *) echo "âœ— Unknown argument: $arg"; exit 1 ;;
  esac
done

# Validate --dynamic value.
case "$DYNAMIC" in
  auto|true|false) ;;
  *) echo "âœ— --dynamic must be 'auto', 'true', or 'false' (got: '$DYNAMIC')"; exit 1 ;;
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
[[ -f "$SCRIPT" ]] || { echo "âœ— Script not found: $SCRIPT"; exit 1; }
mkdir -p "$DATA" "$OUT"

echo "============================================================"
echo "  Script  : $SCRIPT"
echo "  Data    : $DATA"
echo "  Output  : $OUT"
echo "  Modes   : ${MODES[*]}"
echo "  Dynamic : $DYNAMIC  (torch.compile dynamic= argument)"
echo "  Resume  : ${RESUME_FROM:-<none>}"
echo "  Runs    : 4  (gcn/gat x pyg/dgl)"
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
            echo "â†» Resuming from: $key"
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
        echo "â†·  Skipping $key"
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
# ogbn-products â€” gcn/gat x both frameworks
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  ogbn-products â€” gcn/gat x both frameworks"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "5:$FRAMEWORK:$MODEL:ogbn-products" "$FRAMEWORK ${MODEL^^} / ogbn-products / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbn-products \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Warn if resume key was never matched
# ---------------------------------------------------------------------------
if [[ -n "$RESUME_FROM" && $_RESUME_FOUND -eq 0 ]]; then
    echo ""
    echo "âœ— WARNING: --resume key '$RESUME_FROM' was never matched."
    echo "  Valid keys:"
    for FRAMEWORK in pyg dgl; do
        for MODEL in gcn gat; do
            echo "    5:$FRAMEWORK:$MODEL:ogbn-products"
        done
    done
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  All experiments complete.  Results â†’ $OUT"
echo "################################################################"
echo ""
echo "Runs completed:"
echo "  4 runs  (gcn/gat x pyg/dgl, all modes, ogbn-products)"
echo "  Dynamic : $DYNAMIC  (torch.compile dynamic= used for all runs)"
