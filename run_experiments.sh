# =============================================================================
# run_experiments.sh
# Focused experiment runner for the GNN torch.compile benchmark thesis.
#
# RQ coverage:
#   RQ1 â€” Inference latency on ogbn-arxiv          â†’ Phase 3
#   RQ2 â€” Framework Ã— architecture effect          â†’ Phase 3
#   RQ3 â€” Latencyâ€“cost trade-off across modes      â†’ Phase 3 (compile_time_s + speedup)
#   RQ4 â€” Scaling effects on speedup               â†’ Phase 2 (PyG + DGL GCN on Cora/PubMed)
#                                                     + Phase 3 (arxiv)
#          Note: training throughput scaling uses Coraâ†’PubMedâ†’arxiv (all full-batch).
#          Inference speedup IS comparable across all three scales.
#   RQ5 â€” Generalisation to link prediction        â†’ Phase 4 (GCN + GAT only;
#          GraphSAGE/GIN omitted as they behave similarly to GCN for this comparison)
#   RQ6 â€” GPU memory overhead                      â†’ Phase 3
#   RQ7 â€” Training throughput under compilation    â†’ Phase 3
#
# Experiment counts:
#   Phase 1  â€” 4 runs  (smoke test: PyG+DGL GCN+GAT on Cora, all modes)
#   Phase 2  â€” 4 runs  (scale: PyG+DGL GCN+GAT on PubMed)
#   Phase 3  â€” 8 runs  (arxiv: GCN/SAGE/GAT/GIN Ã— PyG/DGL)
#   Phase 4  â€” 8 runs  (collab: GCN/SAGE/GAT/GIN Ã— PyG/DGL)
#   Phase 5  â€” 4 runs  (GCN/ogbn-mag, GCN/ogbl-biokg, R-GCN/ogbn-mag, DistMult/ogbl-biokg)
#   Total    â€” 28 runs
#
# Usage:
#   bash run_experiments.sh                                  # run everything (dynamic=auto)
#   bash run_experiments.sh --dry-run                        # print commands only
#   bash run_experiments.sh --dynamic=true                   # dynamic=True  (always symbolic)
#   bash run_experiments.sh --dynamic=false                  # dynamic=False (always static)
#   bash run_experiments.sh --dynamic=auto                   # dynamic=None  (automatic, default)
#   bash run_experiments.sh --resume=3:dgl:gcn:ogbn-arxiv   # resume from key
#   bash run_experiments.sh --script=gnn_compile_benchmark_v29_parametric.py
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
SCRIPT_NAME="gnn_compile_benchmark_v29_parametric.py"   # default script filename
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
# 'auto' is the Python default so no flag is needed, but we pass it explicitly
# for reproducibility and so the echo output is always self-contained.
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
[[ -f "$SCRIPT" ]] || { echo "âœ— Script not found: $SCRIPT"; exit 1; }
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
        echo "â†·  Skipping $key"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  [$key] $label"
    echo "============================================================"
    # Print the full command for logging / reproducibility.
    echo "  python $SCRIPT ${DYNAMIC_ARG[*]} $*"

    if [[ $DRY_RUN -eq 0 ]]; then
        python "$SCRIPT" "${DYNAMIC_ARG[@]}" "$@"
    fi
}

# ---------------------------------------------------------------------------
# Phase 1 â€” Validation (smoke test)
# Both frameworks, both models (GCN + GAT), all modes, smallest dataset (Cora).
# Verifies loading, subprocess isolation, and JSON output across the full mode
# matrix for both architectures.
# Produces its own result dir â€” not used for RQ analysis.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 1 â€” Validation (smoke test)"
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
# Phase 2 â€” Scale data points (RQ4)
# PyG AND DGL GCN+GAT on PubMed with all modes.
# Cora is already covered by Phase 1; this adds the next scale point.
# Both frameworks and both models are needed so the scale trend can be
# plotted consistently across Coraâ†’PubMedâ†’arxiv for both GCN and GAT.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 2 â€” Scale data points (RQ4)"
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
# Phase 3 â€” Primary benchmark on ogbn-arxiv (RQ1, RQ2, RQ3, RQ6, RQ7)
# All 4 models Ã— 2 frameworks, all 5 modes.
# This is the third (and primary) scale point for RQ4.
#
# Note on GAT: the Python script auto-enables --use-sampling for GAT on
# ogbn-arxiv (and ogbl-collab) to prevent OOM during full-graph attention
# computation. No --use-sampling flag needed here.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 3 â€” Primary benchmark: ogbn-arxiv (RQ1/2/3/6/7)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn graphsage gat gin; do
        run "3:$FRAMEWORK:$MODEL:ogbn-arxiv" "$FRAMEWORK $MODEL / ogbn-arxiv / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbn-arxiv \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 4 â€” Link prediction on ogbl-collab (RQ5)
# All 4 models Ã— both frameworks.
# GCN = clean-compile baseline. GAT = graph-break risk case.
# GraphSAGE and GIN included for complete coverage.
#
# --collab-lr (0.001) and --collab-dropout (0.0) are applied automatically
# by the Python script when dataset=ogbl-collab; no extra flags needed.
#
# val/test evaluation uses the official OGB fixed negatives
# (split_edge["valid/test"]["edge_neg"], 100K per split) instead of randomly
# sampled negatives.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 4 â€” Link prediction: ogbl-collab (RQ5)"
echo "################################################################"

for FRAMEWORK in pyg dgl; do
    for MODEL in gcn graphsage gat gin; do
        run "4:$FRAMEWORK:$MODEL:ogbl-collab" "$FRAMEWORK $MODEL / ogbl-collab / all modes" \
            --framework "$FRAMEWORK" --model-name "$MODEL" --dataset ogbl-collab \
            --hidden 256 --num-layers 3 --dropout 0.5 \
            --modes "${MODES[@]}" \
            --data-root "$DATA" --out-dir "$OUT" \
            "${BASE_ARGS[@]}"
    done
done

# ---------------------------------------------------------------------------
# Phase 5 â€” Heterogeneous graphs (breadth demonstration)
#
# GCN and GAT on ogbn-mag / ogbl-biokg (homogeneous baseline, PyG-only):
#   GCN only â€” homogeneous baseline for comparison against R-GCN / DistMult.
#
# R-GCN on ogbn-mag:
#   --hidden 64  --num-layers 2  per the OGB R-GCN baseline.
#   Larger hidden OOMs due to four separate embedding tables (paper, author,
#   institution, field_of_study).
#   PyG-only pipeline (no DGL path implemented for R-GCN).
#   Source: https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/mag
#
# DistMult on ogbl-biokg:
#   --hidden and --num-layers are ignored by the DistMult pipeline; the model
#   uses --emb-dim for embedding size and --batch-size / --train-batch-size for
#   the inference and training batch sizes.
#   PyG-only pipeline (no DGL path implemented for DistMult).
#   Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
#
# The Python script silently downgrades reduce-overhead â†’ default and
# max-autotune â†’ max-autotune-no-cudagraphs for sparse operators (RGCNConv)
# that are incompatible with CUDA Graph capture. No extra flags needed here.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "#  PHASE 5 â€” Heterogeneous graphs"
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

run "5:pyg:rgcn:ogbn-mag" "PyG R-GCN / ogbn-mag / all modes" \
    --framework pyg --model-name rgcn --dataset ogbn-mag \
    --hidden 64 --num-layers 2 --dropout 0.5 \
    --modes "${MODES[@]}" \
    --data-root "$DATA" --out-dir "$OUT" \
    "${BASE_ARGS[@]}"

# --hidden and --num-layers are passed explicitly even though they are ignored
# by the DistMult pipeline, so the run() echo output shows a complete,
# reproducible command. The Python script accepts them without error.
run "5:pyg:distmult:ogbl-biokg" "PyG DistMult / ogbl-biokg / all modes" \
    --framework pyg --model-name distmult --dataset ogbl-biokg \
    --hidden 256 --num-layers 2 \
    --emb-dim 128 --batch-size 8192 --train-batch-size 8192 \
    --modes "${MODES[@]}" \
    --data-root "$DATA" --out-dir "$OUT" \
    "${BASE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Warn if a --resume key was given but never matched.
# Without this check, a typo in --resume silently skips every single run.
# ---------------------------------------------------------------------------
if [[ -n "$RESUME_FROM" && $_RESUME_FOUND -eq 0 ]]; then
    echo ""
    echo "âœ— WARNING: --resume key '$RESUME_FROM' was never matched."
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
echo "#  All experiments complete.  Results â†’ $OUT"
echo "################################################################"
echo ""
echo "Runs completed:"
echo "  Phase 1  â€” 4 runs  (smoke test: PyG+DGL GCN+GAT Ã— Cora, all modes)"
echo "  Phase 2  â€” 4 runs  (scale: PyG+DGL GCN+GAT Ã— PubMed)"
echo "  Phase 3  â€” 8 runs  (arxiv: GCN/SAGE/GAT/GIN Ã— PyG/DGL)"
echo "  Phase 4  â€” 8 runs  (collab: GCN/SAGE/GAT/GIN Ã— PyG/DGL)"
echo "  Phase 5  â€” 4 runs  (GCN/ogbn-mag, GCN/ogbl-biokg, R-GCN/ogbn-mag, DistMult/ogbl-biokg)"
echo "  Total    â€” 28 runs"
echo "  Dynamic  â€” $DYNAMIC  (torch.compile dynamic= used for all runs)"
