# =============================================================================
# run_no_fixes.sh
# Main experiment run script
#
# Experiments:
#   Phase 1  4 runs  (smoke test: PyG+DGL GCN+GAT on Cora, all modes)
#   Phase 2  4 runs  (scale: PyG+DGL GCN+GAT on PubMed)
#   Phase 3  4 runs  (arxiv: GCN/GAT PyG/DGL)
#   Phase 4  4 runs  (collab: GCN/GAT PyG/DGL)
#   Phase 5  2 runs  (GCN/ogbn-mag, GCN/ogbl-biokg)
#   Total   18 runs
#
# Usage:
#   bash run_no_fixes.sh                                  # run everything 
#   bash run_no_fixes.sh --dry-run                        # print commands only
#   bash run_no_fixes.sh --dynamic=true                   # dynamic=True 
#   bash run_no_fixes.sh --dynamic=false                  # dynamic=False 
#   bash run_no_fixes.sh --dynamic=auto                   # dynamic=None 
#   bash run_no_fixes.sh --resume=3:dgl:gcn:ogbn-arxiv    # resume from key
#   bash run_no_fixes.sh --script=gnn_compile_benchmark_v29_no_fixes.py
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="$BASE_DIR/data"
OUT="$BASE_DIR/results"

MODES=(eager default reduce-overhead max-autotune max-autotune-no-cudagraphs)

BASE_ARGS=(
    --repeats 30
    --warmup 5
    --train-epochs 20
    --train-warmup 5
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
RESUME_FROM=""
SCRIPT_NAME="gnn_compile_benchmark_v29_no_fixes.py"   # default script filename
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
# Phase 1 Validation (smoke test)
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 1 - Validation (smoke test)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "1:$FRAMEWORK:$MODEL:cora" "$FRAMEWORK ${MODEL^^} / Cora / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset cora \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 2 Scale data points (RQ4)
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 2 - Scale data points (RQ4)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "2:$FRAMEWORK:$MODEL:pubmed" "$FRAMEWORK ${MODEL^^} / PubMed / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset pubmed \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 3 Primary benchmark on ogbn-arxiv
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 3 - Primary benchmark: ogbn-arxiv (RQ1/2/3/6/7)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "3:$FRAMEWORK:$MODEL:ogbn-arxiv" "$FRAMEWORK $MODEL / ogbn-arxiv / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbn-arxiv \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 4 Link prediction on ogbl-collab
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 4 - Link prediction: ogbl-collab (RQ5)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn gat; do
        run "4:$FRAMEWORK:$MODEL:ogbl-collab" "$FRAMEWORK $MODEL / ogbl-collab / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbl-collab \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 5 Heterogeneous graphs
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 5 - Heterogeneous graphs"
echo "################################################################"

run "5:pyg:gcn:ogbn-mag" "PyG GCN / ogbn-mag / all modes" \
    --framework pyg --model-name gcn --dataset ogbn-mag \
    --hidden 256 --num-layers 3 --dropout 0.5 \
    --modes "${MODES[@]}" \
    --data-root "$DATA" --out-dir "$OUT" \
    "${BASE_ARGS[@]}"

run "5:pyg:gcn:ogbl-biokg" "PyG GCN / ogbl-biokg / all modes" \
    --framework pyg --model-name gcn --dataset ogbl-biokg \
    --hidden 256 --num-layers 3 --dropout 0.5 \
    --modes "${MODES[@]}" \
    --data-root "$DATA" --out-dir "$OUT" \
    "${BASE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Warn if a --resume key was given but never matched.
# ---------------------------------------------------------------------------
if [[ -n "$RESUME_FROM" && $_RESUME_FOUND -eq 0 ]]; then
    echo ""
    echo "----- WARNING: --resume key '$RESUME_FROM' was never matched."
    echo "  All runs were skipped.  Check the key against the list above."
    echo "  Valid key format: <phase>:<framework>:<model>:<dataset>"
    echo "  Example:          3:dgl:gcn:ogbn-arxiv"
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
echo "  Phase 1  - 4 runs  (smoke test: PyG+DGL GCN+GAT x Cora, all modes)"
echo "  Phase 2  - 4 runs  (scale: PyG+DGL GCN+GAT x PubMed)"
echo "  Phase 3  - 4 runs  (arxiv: GCN/GAT x PyG/DGL)"
echo "  Phase 4  - 4 runs  (collab: GCN/GAT x PyG/DGL)"
echo "  Phase 5  - 2 runs  (GCN/ogbn-mag, GCN/ogbl-biokg)"
echo "  Total    - 18 runs"
echo "  Dynamic  - $DYNAMIC  (torch.compile dynamic= used for all runs)"
