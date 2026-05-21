# GNN torch.compile Benchmark

**Author:** Sonia Vetter · March–May 2026

A benchmark suite that measures the effect of `torch.compile()` on Graph Neural Networks across PyTorch Geometric (PyG) and Deep Graph Library (DGL).

---

## Contents

### Benchmark scripts

| File | Description |
|------|-------------|
| `gnn_compile_benchmark_v29_dynamic.py` | Main benchmark script. Supports parametric `dynamic=` shapes (`auto`/`True`/`False`). Use this for all standard runs. |
| `gnn_compile_benchmark_v29_no_fixes.py` | Identical to the above but **without** the self-loop and GCN-normalisation workaround — used to isolate the effect of those code changes. |
| `analyze_results_v12.py` | Post-processing script that reads all result folders into the multi-sheet XLSX workbook. |

Each benchmark run writes five files into `<out-dir>/<framework>_<model>_<dataset>_<timestamp>/`:

- `config.json` — full run configuration and system info
- `results.json` — per-mode result dicts 
- `tables.tex` — LaTeX tables
- `run.log` — execution log
- `recommendations.txt` — plain-text practical recommendations synthesised from the per-mode results

### Experiment run scripts

| File | Description |
|------|-------------|
| `run_experiments.sh` | Full experiment suite.|
| `run_gin_accuracy.sh` | GIN-only re-run on ogbn-arxiv with 300 training epochs instead of 20, to obtain convergence-quality accuracy numbers for GIN. |
| `run_products.sh` | GCN + GAT × PyG + DGL on ogbn-products. Requires mini-batch sampling; run separately due to long wall-time. |

### Results archives

Each archive contains one subfolder per experiment run, with the five output files listed above.

| Archive | `dynamic=` setting | Description |
|---------|-------------------|-------------|
| `results_auto.zip` | `auto` (default) | Primary results — `torch.compile` dynamic shapes set to `None` (PyTorch chooses automatically). Baseline for all comparisons. |
| `results_true.zip` | `True` | Re-run with `dynamic=True` (always symbolic; compiled graph is always reused regardless of shape changes). |
| `results_false.zip` | `False` | Re-run with `dynamic=False` (always static; re-specialises on every new shape). |
| `results_no_fixes.zip` | `auto` | Re-run with `gnn_compile_benchmark_v29_no_fixes.py` — no self-loop / normalisation workarounds applied. Isolates the impact of those fixes. |
| `results_gin_300epochs.zip` | `auto` | GIN on ogbn-arxiv trained for 300 epochs (PyG GIN + DGL GIN). |

### Results workbook

`gnn_benchmark_results.xlsx` — 508 successful runs out of 580 total (56 OOM, 16 other errors).

| Sheet | Description |
|-------|-------------|
| Research Questions | Overview of all research questions with key result figures and colour legend. |
| Eager (Baseline) | Per-experiment eager-mode metrics across all four result variants (Auto, True, False, No-Fixes) with delta columns. |
| Default | Same layout as Eager for the `default` compile mode. |
| Reduce Overhead | Same layout for `reduce-overhead`. |
| Max Autotune | Same layout for `max-autotune`. |
| Max Autotune No Cudagraphs | Same layout for `max-autotune-no-cudagraphs`. |
| Mode Summary – Auto | Median inference latency and speedup per mode for the `dynamic=auto` runs, broken down by framework and model. |
| Mode Summary – True | Same summary for `dynamic=True` runs. |
| Mode Summary – False | Same summary for `dynamic=False` runs. |
| Mode Summary – No-Fixes | Same summary for the no-fixes runs. |
| DGL Summary | Full per-experiment breakdown for DGL across all compile modes. |
| PyG Summary | Full per-experiment breakdown for PyG across all compile modes. |
| DGL vs PyG (Eager) | Direct DGL vs PyG comparison at eager baseline — inference latency, throughput, and memory. |
| DGL vs PyG (All Modes) | DGL vs PyG comparison extended across all five compile modes. |
| DGL Eager vs PyG Compiled | DGL eager used as DGL's practical best; compared against all PyG compiled modes. |
| Compile Modes vs Eager | Speedup of each compiled mode over eager, per framework and model. |
| Mem 1 DGL vs PyG | Peak GPU memory: DGL vs PyG at eager, per model and dataset. |
| Mem 2 Mode vs Eager | Memory overhead of each compiled mode relative to eager. |
| Mem 3 PyG | Full memory breakdown for PyG across all modes. |
| Mem 4 DGL | Full memory breakdown for DGL across all modes. |
| Mem 5 Best Mode | Per-experiment best mode by memory efficiency. |
| Auto vs No-Fixes | Side-by-side comparison of `dynamic=auto` with and without the self-loop / normalisation workarounds. |
| GIN 300 vs 20 Epochs | Accuracy and throughput comparison for GIN trained for 300 vs 20 epochs. |
| Error Summary | All OOM and error entries with framework, model, dataset, mode, and error type. |

### Environment files

| File | Description |
|------|-------------|
| `environment.yml` | Conda environment export (`gnn_bench`). Reproducible install via `conda env create -f environment.yml`. |
| `pip_packages.txt` | `pip freeze` output. Alternative install via `pip install -r pip_packages.txt` (requires Python 3.11 to be set up beforehand). |
| `packages.txt` | Full `conda list` output including both conda and pip packages with build hashes. |
| `system_info.txt` | Hardware and software snapshot at experiment time (GPU, driver, PyTorch, CUDA, cuDNN, PyG, DGL versions). |

---


## Reproducing the experiments
### 1. Set up the environment
Install [Miniconda](https://docs.anaconda.com/miniconda/install/), then:
```bash
conda env create -f environment.yml
conda activate gnn_bench
```

### 2. Run the full benchmark suite

```bash
bash run_experiments.sh --script=<filename>
```

Results are written to `./results/`.

**Flags:**

| Flag | Description |
|------|-------------|
| `--script=<filename>` | Benchmark script to use. Use `gnn_compile_benchmark_v29_dynamic.py` for standard results or `gnn_compile_benchmark_v29_no_fixes.py` to reproduce the no-fixes results. |
| `--dry-run` | Print the commands that would be executed without running them. |
| `--dynamic=auto\|true\|false` | Set the `dynamic=` argument passed to `torch.compile()` for every run. Default is `auto` (`None` is automatically). `true` forces always-symbolic compilation; `false` forces always-static specialisation. |
| `--resume=<key>` | Skip all runs before the given key and start from it. Useful after an interruption. Keys follow the format `<phase>:<framework>:<model>:<dataset>`, e.g. `3:dgl:gcn:ogbn-arxiv`. |

### 3. Run the GIN accuracy experiment (300 epochs)

```bash
bash run_gin_accuracy.sh --script=<filename>
```

Accepts the same `--dry-run`, `--dynamic=`, `--resume=`, and `--script=` flags. Valid resume keys are `3:pyg:gin:ogbn-arxiv` and `3:dgl:gin:ogbn-arxiv`.

### 4. Run the ogbn-products stress test

```bash
bash run_products.sh --script=<filename>
```

Accepts the same optional flags. Valid resume keys follow `5:<framework>:<model>:ogbn-products` (models: `gcn`, `gat`; frameworks: `pyg`, `dgl`). Mini-batch sampling is enabled automatically via `--use-sampling`.

### 5. Run a single experiment manually

```bash
python gnn_compile_benchmark_v29_dynamic.py \
    --framework pyg \
    --model-name gcn \
    --dataset ogbn-arxiv \
    --hidden 256 \
    --num-layers 3 \
    --dropout 0.5 \
    --modes eager default reduce-overhead max-autotune max-autotune-no-cudagraphs \
    --repeats 30 \
    --warmup 5 \
    --train-epochs 20 \
    --train-warmup 5 \
    --dynamic auto
```

**All optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--framework` | `pyg` | GNN framework. Choices: `pyg`, `dgl`. |
| `--model-name` | `gcn` | Model architecture. Choices: `gcn`, `graphsage`, `gat`, `gin`, `rgcn` (PyG + ogbn-mag only), `distmult` (PyG + ogbl-biokg only). |
| `--dataset` | `ogbn-arxiv` | Dataset name. Node classification: `cora`, `pubmed`, `ogbn-arxiv`, `ogbn-products`, `ogbn-mag`. Link prediction: `ogbl-collab`. KG completion: `ogbl-biokg`. |
| `--data-root` | `data/` | Root directory for dataset downloads and caching. |
| `--hidden` | `256` | Hidden dimensionality. For GAT this is the per-head dimension; effective width = `hidden × gat-heads`. |
| `--num-layers` | `3` | Number of GNN message-passing layers. |
| `--dropout` | `0.5` | Dropout probability. |
| `--gat-heads` | `8` | Number of attention heads for GAT. |
| `--emb-dim` | `128` | Embedding dimension for DistMult (ogbl-biokg only). |
| `--batch-size` | `8192` | Inference batch size for DistMult (ogbl-biokg only). |
| `--train-batch-size` | `8192` | Training batch size for DistMult (ogbl-biokg only). |
| `--use-sampling` | off | Enable mini-batch neighbour sampling. Required for ogbn-products and ogbl-citation2 to avoid OOM; GAT enables it automatically for ogbn-arxiv and ogbl-collab. |
| `--dynamic` | `auto` | `dynamic=` argument for `torch.compile()`. `auto` = PyTorch decides, `true` = always symbolic, `false` = always static. |
| `--gat-chunk-size` | `None` | Chunk `edge_index` into batches of this size during full-graph GAT attention to reduce peak memory. No chunking by default. |
| `--repeats` | `30` | Number of timed inference repetitions per mode (after warmup). |
| `--warmup` | `5` | Number of warmup inference passes before timing begins. |
| `--train-epochs` | `20` | Number of training epochs to time. |
| `--train-warmup` | `5` | Number of warmup training epochs before timing begins. |
| `--lr` | `0.01` | Learning rate for standard node classification. |
| `--collab-lr` | `0.001` | Learning rate override for ogbl-collab (OGB baseline). Applied automatically when `--dataset ogbl-collab` is set; no need to pass explicitly. |
| `--collab-dropout` | `0.0` | Dropout override for ogbl-collab (OGB baseline). Applied automatically. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--timeout` | `3600` | Per-mode subprocess timeout in seconds. Increase for very large datasets or slow machines. |
| `--modes` | all five | Space-separated list of compile modes to run. Defaults to all five: `eager default reduce-overhead max-autotune max-autotune-no-cudagraphs`. |
| `--out-dir` | `experiments/` | Root directory for output. Each run creates a timestamped subdirectory inside. |

### 6. Analyse results

Before running the analysis, copy the ogbn-products results (from `run_products.sh`) into the `results_auto` folder so they are picked up together with the main auto results:

```bash
cp -r results_products/* results_auto/
```

Then run the analysis script:

```bash
python analyze_results_v12.py \
    -a results_auto \
    -t results_true \
    -f results_false \
    -n results_no_fixes \
    -g results_gin_300epochs \
    -o gnn_benchmark_results.xlsx
```

All flags are optional and default to the directory names shown above.

---
