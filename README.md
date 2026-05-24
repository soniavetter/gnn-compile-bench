# GNN torch.compile Benchmark

**Author:** Sonia Vetter · March–May 2026

A benchmark suite that measures the effect of `torch.compile()` on Graph Neural Networks across PyTorch Geometric (PyG) and Deep Graph Library (DGL).

---

## Table of Contents

1. [Contents](#contents)
   - [Benchmark Scripts](#benchmark-scripts)
   - [Experiment Run Scripts](#experiment-run-scripts)
   - [Results Archives](#results-archives)
   - [Results Workbook](#results-workbook)
2. [Reproducing the Experiments](#reproducing-the-experiments)
   - [Environment Files](#environment-files)
   - [1. Set Up the Environment](#1-set-up-the-environment)
   - [2. Run the Full Benchmark Suite](#2-run-the-full-benchmark-suite)
   - [3. Run the GIN Accuracy Experiment (300 Epochs)](#3-run-the-gin-accuracy-experiment-300-epochs)
   - [4. Run the No-Workaround Test](#4-run-the-no-workaround-test)
   - [5. Run the ogbn-products Stress Test](#5-run-the-ogbn-products-stress-test)
   - [6. Run a Single Experiment Manually](#6-run-a-single-experiment-manually)
   - [7. Analyse Results](#7-analyse-results)

---

## Contents

### Benchmark Scripts

| File | Description |
|------|-------------|
| `gnn_compile_benchmark_v29_dynamic.py` | Main benchmark script. Supports parametric `dynamic=` shapes (`auto`/`True`/`False`). Use this for all standard runs. |
| `gnn_compile_benchmark_v29_no_fixes.py` | Identical to the above but **without** the self-loop and GCN-normalisation workaround — used to isolate the effect of those code changes. |
| `analyze_results_v16.py` | Post-processing script that reads all result folders into the multi-sheet XLSX workbook. |

Each benchmark run writes five files into `<out-dir>/<framework>_<model>_<dataset>_<timestamp>/`:

- `config.json` — full run configuration and system info
- `results.json` — per-mode result dicts
- `tables.tex` — LaTeX tables
- `run.log` — execution log
- `recommendations.txt` — plain-text practical recommendations synthesised from the per-mode results

---

### Experiment Run Scripts

| File | Description |
|------|-------------|
| `run_experiments.sh` | Full experiment suite. |
| `run_gin_accuracy.sh` | GIN-only re-run on ogbn-arxiv with 300 training epochs instead of 20, to obtain convergence-quality accuracy numbers for GIN. |
| `run_no_fixes.sh` | Full experiment suite (Phases 1–5, GCN + GAT only) re-run with `gnn_compile_benchmark_v29_no_fixes.py`. No workaround applied. Isolates the effect of those code changes. |
| `run_products.sh` | GCN + GAT × PyG + DGL on ogbn-products. Requires mini-batch sampling; run separately due to long wall-time. |

---

### Results Archives

Each archive contains one subfolder per experiment run, with the five output files listed above.

| Archive | `dynamic=` setting | Description |
|---------|-------------------|-------------|
| `results_auto.zip` | `auto` (default) | Primary results — `torch.compile` dynamic shapes set to `None` (PyTorch chooses automatically). Baseline for all comparisons. |
| `results_true.zip` | `True` | Re-run with `dynamic=True` (always symbolic; compiled graph is always reused regardless of shape changes). |
| `results_false.zip` | `False` | Re-run with `dynamic=False` (always static; re-specialises on every new shape). |
| `results_no_fixes.zip` | `auto` | Re-run with `gnn_compile_benchmark_no_fixes.py` — no self-loop / normalisation workarounds applied. Isolates the impact of those fixes. |
| `results_gin_300epochs.zip` | `auto` | GIN on ogbn-arxiv trained for 300 epochs (PyG GIN + DGL GIN). |

---

### Results Workbook

`gnn_benchmark_results.xlsx`

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

---

## Reproducing the Experiments

### Environment Files

| File | Description | Export Command |
|------|-------------|----------------|
| `environment.yml` | Curated install recipe with custom wheel URLs for PyTorch, PyG, DGL, and all dependencies. Custom wheel URLs and index URLs added manually. Versions not fully pinned for all packages. | adjusted manually |
| `gnn_bench.yml` | Full conda environment snapshot without build hashes and without local `prefix` path. All versions exactly pinned. Custom wheel URLs missing — GPU builds of `torch`, `dgl`, and PyG extensions may fail to resolve without them. | `conda env export --no-builds \| grep -v "^prefix" > gnn_bench.yml` |
| `pip_packages.txt` | All exact pip package versions. Requires Python 3.11 and the correct wheel URLs to resolve GPU builds. Custom wheel URLs missing — must be passed manually when installing. | `pip freeze > pip_packages.txt` |
| `packages.txt` | Full `conda list` output with all conda and pip packages including build hashes. Documents exact packages the experiments were run on. | `conda list > packages.txt` |
| `system_info.txt` | Hardware and software snapshot at experiment time (platform, Python, PyTorch, CUDA, cuDNN, PyG, DGL versions) followed by full `nvidia-smi` output. | `platform.platform()`, `platform.python_version()`, `torch.__version__`, `torch.version.cuda`, `torch.backends.cudnn.version()`, `torch_geometric.__version__`, `dgl.__version__`, `nvidia-smi` |

---

### 1. Set Up the Environment

Install [Miniconda](https://docs.anaconda.com/miniconda/install/).

---

**Option 1 — `environment.yml`**

Contains all packages with correct GPU wheel URLs for CUDA 12.4: PyTorch, PyG (`torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, `pyg-lib`), DGL, OGB, and all other dependencies.

```bash
conda env create -f environment.yml
conda activate gnn_bench
```

---

**Option 2 — `gnn_bench.yml` (fallback if Option 1 fails)**

Full conda snapshot with all exact pinned versions. Does not include custom wheel URLs, so GPU builds must be resolved manually after creating the environment.

```bash
conda env create -f gnn_bench.yml
conda activate gnn_bench
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install dgl --find-links https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
```

---

**Option 3 — `pip_packages.txt` (fallback if Option 2 fails)**

Contains all exact pip versions. Requires Python 3.11 to be set up first via Miniconda, then install with the correct wheel URLs:

```bash
conda create -n gnn_bench python=3.11
conda activate gnn_bench
pip install -r pip_packages.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html \
  --find-links https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
```

---

### 2. Run the Full Benchmark Suite

```bash
# results_auto (dynamic=None)
bash run_experiments.sh
mv results results_auto

# results_true (dynamic=True)
bash run_experiments.sh --dynamic=true
mv results results_true

# results_false (dynamic=False)
bash run_experiments.sh --dynamic=false
mv results results_false
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--script=<filename>` | Benchmark script to use. Use `gnn_compile_benchmark_v29_dynamic.py` for standard results or `gnn_compile_benchmark_v29_no_fixes.py` to reproduce the no-fixes results. |
| `--dry-run` | Print the commands that would be executed without running them. |
| `--dynamic=auto\|true\|false` | Set the `dynamic=` argument passed to `torch.compile()` for every run. Default is `auto` (`None` is passed automatically). `true` forces always-symbolic compilation; `false` forces always-static specialisation. |
| `--resume=<key>` | Skip all runs before the given key and start from it. Useful after an interruption. Keys follow the format `<phase>:<framework>:<model>:<dataset>`, e.g. `3:dgl:gcn:ogbn-arxiv`. |

---

### 3. Run the GIN Accuracy Experiment (300 Epochs)

```bash
bash run_gin_accuracy.sh
```

Accepts the same `--dry-run`, `--dynamic=`, `--resume=`, and `--script=` flags. Valid resume keys are `3:pyg:gin:ogbn-arxiv` and `3:dgl:gin:ogbn-arxiv`.

---

### 4. Run the No-Workaround Test

```bash
bash run_no_fixes.sh
```

Accepts the same `--dry-run`, `--dynamic=`, `--resume=`, and `--script=` flags.

---

### 5. Run the ogbn-products Stress Test

```bash
bash run_products.sh
```

Accepts the same optional flags. Valid resume keys follow `6:<framework>:<model>:ogbn-products` (models: `gcn`, `gat`; frameworks: `pyg`, `dgl`). Mini-batch sampling is enabled automatically via `--use-sampling`.

---

> **Expected time (with the hardware used for these results — see `system_info.txt`):**
>
> | Script | Runs | Time |
> |--------|------|-----------|
> | `run_experiments.sh` (auto/true/false, each) | 26 | total ~90min |
> | `run_no_fixes.sh` | 18 | total ~60min |
> | `run_gin_accuracy.sh` | 2 | total ~10min |
> | `run_products.sh` | 4 | individual runs up to  ~8h 30min |

### 6. Run a Single Experiment Manually

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

---

### 7. Analyse Results

Before running the analysis, copy the ogbn-products results (from `run_products.sh`) into the `results_auto` folder so they are picked up together with the main auto results:

```bash
cp -r results_products/* results_auto/
```

Then run the analysis script:

```bash
python analyze_results_v16.py
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `-a` | `results_auto` | Directory containing the `dynamic=auto` results. |
| `-t` | `results_true` | Directory containing the `dynamic=True` results. |
| `-f` | `results_false` | Directory containing the `dynamic=False` results. |
| `-n` | `results_no_fixes` | Directory containing the no-fixes results. |
| `-g` | `results_gin_300epochs` | Directory containing the GIN 300-epoch results. |
| `-o` | `gnn_benchmark_results.xlsx` | Output path for the results workbook. |
