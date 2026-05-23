# =============================================================================
# run_products.sh
# ogbn-products benchmark script
#
# Experiments:
#   Phase 6  4 runs  (products: GCN/GAT x PyG/DGL, all modes)
#   Total    4 runs
#
# Usage:
#   bash run_products.sh                                      # run everything
#   bash run_products.sh --dry-run                            # print commands only
#   bash run_products.sh --dynamic=true                       # dynamic=True
#   bash run_products.sh --dynamic=false                      # dynamic=False
#   bash run_products.sh --dynamic=auto                       # dynamic=None
#   bash run_products.sh --resume=6:dgl:gcn:ogbn-products     # resume from key
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
SCRIPT_NAME="gnn_compile_benchmark_v29_dynamic.py"   # default script filename
DYNAMIC="auto"   # default: dynamic=None (automatic dynamic shapes)

for arg in "$@"; do
  case $arg in
    --dry-run)    DRY_RUN=1 ;;
    --resume=*)   RESUME_FROM="${arg#*=}" ;;
    --script=*)   SCRIPT_NAME="${arg#*=}" ;;
    --dynamic=*)  DYNAMIC="${arg#*=}" ;;
    *) echo "----- Unknown argument: $arg"; exit 1 ;;
  esac
done

# Validate --dynamic value.
case "$DYNAMIC" in
  auto|true|false) ;;
  *) echo "----- --dynamic must be 'auto', 'true', or 'false' (got: '$DYNAMIC')"; exit 1 ;;
esac

# Build the --dynamic flag to forward to the Python script.
DYNAMIC_ARG=(--dynamic "$DYNAMIC")

# Resolve full path: accept an absolute path or a bare filename relative to BASE_DIR.
if [[ "$SCRIPT_NAME" = /* ]]; then
    SCRIPT="$SCRIPT_NAME"
else
    SCRIPT="$BASE_DIR/$SCRIPT_NAME"
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -f "$SCRIPT" ]] || { echo "----- Script not found: $SCRIPT"; exit 1; }
mkdir -p "$DATA" "$OUT"

echo "============================================================"
echo "  Script  : $SCRIPT"
echo "  Data    : $DATA"
echo "  Output  : $OUT"
echo "  Modes   : ${MODES[*]}"
echo "  Dynamic : $DYNAMIC  (torch.compile dynamic= argument)"
echo "  Resume  : ${RESUME_FROM:-<none>}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
# RESUME_ACTIVE=1 means "we have not yet reached the resume point; skip runs".
# RESUME_ACTIVE=0 means "resume point has been passed; run everything".
# If RESUME_FROM is empty, should_run always returns 0 (run all).
#
# _RESUME_FOUND tracks whether the key was matched; if not, we warn at the end
# (prevents all runs being silently skipped on a typo).
RESUME_ACTIVE=1
_RESUME_FOUND=0

should_run() {
    local key="$1"
    if [[ -z "$RESUME_FROM" ]]; then return 0; fi
    if [[ $RESUME_ACTIVE -eq 1 ]]; then
        if [[ "$key" == "$RESUME_FROM" ]]; then
            echo "------- Resuming from: $key"
            RESUME_ACTIVE=0
            _RESUME_FOUND=1
            return 0
        else
            return 1
        fi
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Helper: run one experiment
# $1 = stable resume key  (phase:framework:model:dataset)
# $2 = human-readable label
# $3+ = arguments forwarded verbatim to the Python benchmark script
# ---------------------------------------------------------------------------
run() {
    local key="$1";   shift
    local label="$1"; shift

    if ! should_run "$key"; then
        echo "-------  Skipping $key"
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
# Phase 6 ogbn-products
# GCN and GAT on both frameworks, all modes.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 6 - ogbn-products benchmark"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "6:$FRAMEWORK:$MODEL:ogbn-products" "$FRAMEWORK ${MODEL^^} / ogbn-products / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbn-products \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Warn if a --resume key was given but never matched.
# Without this check, a typo in --resume silently skips every single run.
# ---------------------------------------------------------------------------
if [[ -n "$RESUME_FROM" && $_RESUME_FOUND -eq 0 ]]; then
    echo ""
    echo "----- WARNING: --resume key '$RESUME_FROM' was never matched."
    echo "  All runs were skipped.  Check the key against the list above."
    echo "  Valid key format: <phase>:<framework>:<model>:<dataset>"
    echo "  Example:          6:dgl:gcn:ogbn-products"
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
echo "  Phase 6  - 4 runs  (products: GCN/GAT x PyG/DGL, all modes)"
echo "  Total    - 4 runs"
echo "  Dynamic  - $DYNAMIC  (torch.compile dynamic= used for all runs)"
