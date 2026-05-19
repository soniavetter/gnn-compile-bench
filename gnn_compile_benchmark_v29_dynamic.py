# -*- coding: utf-8 -*-
"""
Benchmark script for:
    "Evaluation of JIT Compilation and related Optimization Techniques
     in PyTorch-based Graph Neural Network Frameworks"
    Author: Sonia Vetter, March 2026
    Version: v29  (with self-loop and GCN-normalisation workaround, parametric dynamic shapes)

Overview
--------
Measures the effect of torch.compile() on GNN inference latency, training
throughput, GPU memory, and model accuracy across five compile modes:
    eager                      -- plain PyTorch, no compilation
    default                    -- balanced fusion, fast compile time
    reduce-overhead            -- minimises kernel-launch overhead via CUDA Graphs
    max-autotune               -- exhaustive kernel search; slowest to compile
    max-autotune-no-cudagraphs -- max-autotune without CUDA Graph capture

Each compile mode runs in an isolated subprocess to prevent CUDA state or
compiled-kernel-cache leakage between modes.

Supported frameworks : PyG, DGL
Supported models     : GCN, GraphSAGE, GAT, GIN (all with dropout + BatchNorm)
                       R-GCN (heterogeneous, ogbn-mag only, PyG)
                       DistMult (KG completion, ogbl-biokg only, PyG)
Node classification  : Cora, CiteSeer, PubMed, ogbn-arxiv, ogbn-products,
                       ogbn-mag (heterogeneous, R-GCN only)
Link prediction      : ogbl-collab (Hits@50), ogbl-citation2 (MRR)
KG completion        : ogbl-biokg (MRR, DistMult)

Dataset tier structure:
    Tier 1 (small,  fast dev): Cora, CiteSeer, PubMed
    Tier 2 (medium, primary) : ogbn-arxiv, ogbl-collab, ogbn-mag, ogbl-biokg
    Tier 3 (large,  stress)  : ogbn-products (*), ogbl-citation2 (*)
    (*) ogbn-products and ogbl-citation2 require --use-sampling to avoid OOM.

Usage
-----
Node classification (homogeneous):
    python gnn_compile_benchmark_v29.py --framework pyg --model-name gcn --dataset ogbn-arxiv --hidden 256 --num-layers 3 --dropout 0.5 --repeats 30 --warmup 5 --train-epochs 20 --train-warmup 5 --modes eager default reduce-overhead max-autotune --out-dir ./results

Link prediction (homogeneous):
    python gnn_compile_benchmark_v29.py --framework pyg --model-name graphsage --dataset ogbl-collab --hidden 256 --modes eager default --out-dir ./results

Heterogeneous node classification (R-GCN, ogbn-mag):
    python gnn_compile_benchmark_v29.py --framework pyg --model-name rgcn --dataset ogbn-mag --hidden 64 --num-layers 2 --dropout 0.5 --modes eager default reduce-overhead --out-dir ./results

KG completion (DistMult, ogbl-biokg):
    python gnn_compile_benchmark_v29.py --framework pyg --model-name distmult --dataset ogbl-biokg --emb-dim 128 --batch-size 8192 --modes eager default --out-dir ./results

Output
------
Each run writes to <out-dir>/<framework>_<model>_<dataset>_<timestamp>/:
    config.json                -- run configuration and system information
    results.json               -- per-mode result dicts
    tables.tex                 -- publication-ready LaTeX tables
    run.log                    -- timestamped execution log
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import functools
import io
import json
import logging
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Third-party: CLI, numerics, system
# ---------------------------------------------------------------------------
import argparse
import numpy as np
import psutil

# ---------------------------------------------------------------------------
# PyTorch core
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DGL (Deep Graph Library) -- framework B in the benchmark
# DGL provides graph convolutional primitives (GraphConv, SAGEConv, GATConv,
# GINConv) with its own graph object (dgl.DGLGraph). Each DGL layer takes
# (graph, feat) rather than (x, edge_index) as in PyG.
#
# DGL paper: Wang et al. 2019 -- https://arxiv.org/abs/1909.01315
# DGL documentation: https://docs.dgl.ai
# ---------------------------------------------------------------------------
import dgl
# Kipf & Welling 2017 GCN layer
# Paper  : https://arxiv.org/abs/1609.02907
# DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GraphConv
from dgl.nn import GraphConv
# Hamilton et al. 2017 GraphSAGE layer (mean aggregation)
# Paper  : https://arxiv.org/abs/1706.02216
# DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.SAGEConv
from dgl.nn import SAGEConv as DGLSAGEConv
# Velickovic et al. 2018 Graph Attention Network layer
# Paper  : https://arxiv.org/abs/1710.10903
# DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GATConv
from dgl.nn import GATConv as DGLGATConv
# Xu et al. 2019 Graph Isomorphism Network layer
# Paper  : https://arxiv.org/abs/1810.00826
# DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GINConv
from dgl.nn import GINConv as DGLGINConv

# ---------------------------------------------------------------------------
# OGB (Open Graph Benchmark) -- datasets and official evaluation protocol
#
# All dataset splits and evaluation metrics follow the OGB standard protocol.
#
# Node property prediction (ogbn-arxiv, ogbn-products):
#   Dataset descriptions : https://ogb.stanford.edu/docs/nodeprop/
#   PyG loader snippet   : https://ogb.stanford.edu/docs/nodeprop/#pyg
#     from ogb.nodeproppred import PygNodePropPredDataset
#     dataset   = PygNodePropPredDataset(name=d_name)
#     split_idx = dataset.get_idx_split()
#     graph     = dataset[0]
#   DGL loader snippet   : https://ogb.stanford.edu/docs/nodeprop/#dgl
#     from ogb.nodeproppred import DglNodePropPredDataset
#     dataset      = DglNodePropPredDataset(name=d_name)
#     split_idx    = dataset.get_idx_split()
#     graph, label = dataset[0]
#   Evaluator snippet    : https://ogb.stanford.edu/docs/nodeprop/#eval
#     from ogb.nodeproppred import Evaluator
#     evaluator   = Evaluator(name=d_name)
#     result_dict = evaluator.eval(input_dict)   # input: {y_true, y_pred}
#   Leaderboard          : https://ogb.stanford.edu/docs/leader_nodeprop/
#
# Link property prediction (ogbl-collab, ogbl-citation2):
#   Dataset descriptions : https://ogb.stanford.edu/docs/linkprop/
#   PyG loader snippet   : https://ogb.stanford.edu/docs/linkprop/#pyg
#     from ogb.linkproppred import PygLinkPropPredDataset
#     dataset    = PygLinkPropPredDataset(name=d_name)
#     split_edge = dataset.get_edge_split()
#     graph      = dataset[0]
#   DGL loader snippet   : https://ogb.stanford.edu/docs/linkprop/#dgl
#     from ogb.linkproppred import DglLinkPropPredDataset
#     dataset    = DglLinkPropPredDataset(name=d_name)
#     split_edge = dataset.get_edge_split()
#     graph      = dataset[0]
#   Evaluator snippet    : https://ogb.stanford.edu/docs/linkprop/#eval
#     from ogb.linkproppred import Evaluator
#     evaluator   = Evaluator(name=d_name)
#     result_dict = evaluator.eval(input_dict)
#       ogbl-collab  -> Hits@50  (input: {y_pred_pos, y_pred_neg})
#       ogbl-citation2 -> MRR   (input: {y_pred_pos, y_pred_neg})
#   Leaderboard          : https://ogb.stanford.edu/docs/leader_linkprop/
#
# Leaderboard submission rules (10 seeds, torch.mean / torch.std):
#   https://ogb.stanford.edu/docs/leader_rules/
# ---------------------------------------------------------------------------
import torch_geometric
# Source: https://ogb.stanford.edu/docs/linkprop/#pyg  (PyG loader code snippet)
from ogb.linkproppred import PygLinkPropPredDataset
# Source: https://ogb.stanford.edu/docs/linkprop/#dgl  (DGL loader code snippet)
from ogb.linkproppred import DglLinkPropPredDataset
# Source: https://ogb.stanford.edu/docs/linkprop/#eval  (Evaluator code snippet)
from ogb.linkproppred import Evaluator as LinkEvaluator   # Hits@K / MRR
# Source: https://ogb.stanford.edu/docs/nodeprop/#pyg  (PyG loader code snippet)
from ogb.nodeproppred import PygNodePropPredDataset
# Source: https://ogb.stanford.edu/docs/nodeprop/#dgl  (DGL loader code snippet)
from ogb.nodeproppred import DglNodePropPredDataset
# Source: https://ogb.stanford.edu/docs/nodeprop/#eval  (Evaluator code snippet)
from ogb.nodeproppred import Evaluator as NodeEvaluator   # accuracy (acc)

# For PyTorch >= 2.6 with weights_only=True still enforced in some paths,
# allowlist the PyG data classes that OGB serialises (needed for ogbn-mag).
try:
    import torch.serialization as _ts
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    _ts.add_safe_globals([DataEdgeAttr, DataTensorAttr,
                          GlobalStorage, NodeStorage, EdgeStorage])
except Exception:
    pass

# ---------------------------------------------------------------------------
# PyG (PyTorch Geometric) -- framework A in the benchmark
# PyG represents graphs as (x, edge_index) tensors.
#
# PyG paper: Fey & Lenssen 2019 -- https://arxiv.org/abs/1903.02428
# PyG documentation: https://pytorch-geometric.readthedocs.io
#
# Planetoid: covers Tier 1 datasets Cora, CiteSeer, PubMed.
#   Introduced by Yang et al. 2016 -- https://arxiv.org/abs/1603.08861
#   PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
#
# NeighborLoader: mini-batch neighbor sampling for Tier 3 (ogbn-products).
#   Sampling algorithm from Hamilton et al. 2017 -- https://arxiv.org/abs/1706.02216
#   PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.loader.NeighborLoader.html
#
# Layer sources:
#   GCNConv  : Kipf & Welling, ICLR 2017 -- https://arxiv.org/abs/1609.02907
#              PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
#   SAGEConv : Hamilton et al., NeurIPS 2017 -- https://arxiv.org/abs/1706.02216
#              PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
#   GATConv  : Velickovic et al., ICLR 2018 -- https://arxiv.org/abs/1710.10903
#              PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
#   GINConv  : Xu et al., ICLR 2019 -- https://arxiv.org/abs/1810.00826
#              PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html
# ---------------------------------------------------------------------------
from torch_geometric.datasets import Planetoid    # Cora / CiteSeer / PubMed loader
from torch_geometric.loader import NeighborLoader # mini-batch neighbor sampling (Tier 3)
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
# R-GCN layer for heterogeneous ogbn-mag (Schlichtkrull et al. 2018)
# Paper  : https://arxiv.org/abs/1703.06103
# PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html
from torch_geometric.nn import RGCNConv
# Utility for making citation edges undirected (used in ogbn-mag loading)
from torch_geometric.utils import to_undirected
# Graph transforms applied once at data-load time to eliminate per-forward-pass
# self-loop insertion and GCN normalisation, which cause device synchronisations
# and graph breaks under torch.compile.
#   AddSelfLoops: adds missing self-loop edges to edge_index (no edge_weight change).
#                 Used for GAT and as a no-op guard for SAGE/GIN.
#   GCNNorm:      adds self-loops then writes D^{-1/2}(A+I)D^{-1/2} coefficients
#                 into data.edge_weight; consumed by GCNConv(normalize=False).
#   ToUndirected: symmetrises a directed graph before GCNNorm so that the degree
#                 computation is correct for undirected normalisation.
from torch_geometric.transforms import AddSelfLoops, GCNNorm, ToUndirected

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ogbn-products has very large feature tensors that exceed Triton's default
# tile size of 1024, causing: AssertionError: increase TRITON_MAX_BLOCK['X'].
# Setting 8192 covers all GNN workloads in this benchmark.
os.environ.setdefault("TRITON_MAX_BLOCK_X", "8192")

# ---------------------------------------------------------------------------
# Suppress irrelevant downstream library warnings.
# weights_only=False: PyTorch >=2.0 changed the default; this restores old
# behaviour so saved state-dicts (if any) can be loaded without errors.
# ---------------------------------------------------------------------------
torch.load = functools.partial(torch.load, weights_only=False)
warnings.filterwarnings("ignore", category=UserWarning,       module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="outdated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.serialization")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# ===========================================================================
# DEBUG UTILITIES
# ===========================================================================
# Set DBG=True to activate verbose debug output. Each section is written to
# the run.log via log.debug() so it appears alongside the normal INFO lines.
# The logging level is set to DEBUG when DBG=True (see __main__ setup below).
# ===========================================================================

DBG = True   # <- flip to False to silence all debug output

def _dbg_sep(title: str, width: int = 72) -> None:
    """Write a clearly visible section separator with a title to the log."""
    if not DBG:
        return
    bar = "=" * width
    pad = max(0, (width - len(title) - 2) // 2)
    log.debug("\n%s", bar)
    log.debug("%s  %s", " " * pad, title)
    log.debug("%s", bar)


def _dbg(label: str, value: Any = "", indent: int = 0) -> None:
    """Write a single labelled debug line to the log."""
    if not DBG:
        return
    prefix = "  " * indent
    if value == "":
        log.debug("  %s[DBG] %s", prefix, label)
    else:
        log.debug("  %s[DBG] %-45s %s", prefix, label, value)


def _dbg_tensor(name: str, t: "torch.Tensor | None", indent: int = 0) -> None:
    """Log shape, dtype, device, min/max for a tensor (safe if None)."""
    if not DBG:
        return
    prefix = "  " * indent
    if t is None:
        log.debug("  %s[DBG] %-45s None", prefix, name)
        return
    stats = ""
    try:
        if t.is_floating_point():
            stats = f"  min={t.min().item():.4f}  max={t.max().item():.4f}"
    except Exception:
        pass
    log.debug("  %s[DBG] %-45s shape=%s  dtype=%s  device=%s%s",
              prefix, name, list(t.shape), t.dtype, t.device, stats)


def _dbg_model(model: "nn.Module", indent: int = 0) -> None:
    """Log layer names, shapes, and total trainable parameter count."""
    if not DBG:
        return
    prefix = "  " * indent
    log.debug("  %s[DBG] Model architecture: %s", prefix, type(model).__name__)
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        log.debug("  %s[DBG]   layer=%-40s shape=%s  params=%s",
                  prefix, name, list(param.shape), f"{n:,}")
    log.debug("  %s[DBG]   --> Total trainable parameters: %s", prefix, f"{total:,}")


def _dbg_graph_stats(x: "torch.Tensor | None",
                     edge_index: "torch.Tensor | None",
                     label: str = "Graph") -> None:
    """Log node count, feature dim, edge count, and average node degree."""
    if not DBG:
        return
    if x is None or edge_index is None:
        _dbg(f"{label}: x or edge_index is None")
        return
    n_nodes = x.shape[0]
    n_feats = x.shape[1] if x.dim() > 1 else 1
    n_edges = edge_index.shape[1] if edge_index.dim() > 1 else 0
    avg_deg = n_edges / n_nodes if n_nodes > 0 else 0.0
    log.debug("  [DBG] %s:", label)
    log.debug("  [DBG]   Nodes         : %s", f"{n_nodes:,}")
    log.debug("  [DBG]   Node features : %d  (input feature dimension)", n_feats)
    log.debug("  [DBG]   Edges         : %s  (directed; includes self-loops if added)", f"{n_edges:,}")
    log.debug("  [DBG]   Avg degree    : %.2f  edges per node", avg_deg)


def _dbg_mask(name: str, mask: "torch.Tensor | None") -> None:
    """Log how many nodes are in a True/False split mask."""
    if not DBG or mask is None:
        return
    total = mask.numel()
    true_n = int(mask.sum().item())
    pct = 100.0 * true_n / total if total > 0 else 0.0
    log.debug("  [DBG] %-20s: %s / %s nodes (%.1f %%)", name, f"{true_n:,}", f"{total:,}", pct)


def _dbg_latencies(latencies_ms: list, label: str = "Inference latencies") -> None:
    """Log summary statistics for a list of latency measurements."""
    if not DBG or not latencies_ms:
        return
    arr = np.array(latencies_ms)
    log.debug("  [DBG] %s (%d runs):", label, len(arr))
    log.debug("  [DBG]   median = %.3f ms", np.median(arr))
    log.debug("  [DBG]   mean   = %.3f ms  +/- %.3f",
              np.mean(arr), np.std(arr, ddof=1) if len(arr) > 1 else 0)
    log.debug("  [DBG]   IQR    = %.3f ms  (P25=%.3f  P75=%.3f)",
              np.percentile(arr, 75) - np.percentile(arr, 25),
              np.percentile(arr, 25), np.percentile(arr, 75))
    log.debug("  [DBG]   min    = %.3f ms   max = %.3f ms", arr.min(), arr.max())


def _dbg_epoch_times(epoch_times: list, warmup_times: list, label: str = "Training") -> None:
    """Log per-epoch timing for both warmup and measured epochs."""
    if not DBG:
        return
    log.debug("  [DBG] %s epoch times:", label)
    for i, t in enumerate(warmup_times):
        log.debug("  [DBG]   warmup epoch %2d: %.1f ms  (discarded)", i, t * 1000)
    for i, t in enumerate(epoch_times):
        marker = "  <-- first measured (may include compile overhead)" if i == 0 else ""
        log.debug("  [DBG]   epoch %2d: %.1f ms%s", i, t * 1000, marker)
    if epoch_times:
        arr = np.array(epoch_times)
        if len(arr) > 1:
            log.debug("  [DBG]   --> mean=%.1f ms  std=%.1f ms",
                      arr.mean() * 1000, arr.std(ddof=1) * 1000)
        else:
            log.debug("  [DBG]   --> only one epoch measured")

# ===========================================================================
# END DEBUG UTILITIES
# ===========================================================================

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Tier 1 -- small, fast dev (Experiment Plan Phase 1 & 2):
#   Cora, CiteSeer, PubMed -- citation network benchmarks.
#   Original paper introducing these splits (Planetoid):
#     Yang et al. 2016, "Revisiting Semi-Supervised Learning with Graph Embeddings"
#     https://arxiv.org/abs/1603.08861
#   Underlying datasets:
#     Cora      : McCallum et al. 2000, machine learning papers, 7 classes
#                 https://linqs.org/datasets/#cora
#     CiteSeer  : Giles et al. 1998, scientific publications, 6 classes
#                 https://linqs.org/datasets/#citeseer-doc-classification
#     PubMed    : Namata et al. 2012, diabetes papers, 3 classes
#                 https://linqs.org/datasets/#pubmed-diabetes
#   PyG loader : torch_geometric.datasets.Planetoid (public split, no OGB)
#     https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
#
# Tier 2 -- medium, primary benchmark (Phase 3 & 4):
#   ogbn-arxiv  : ~169K nodes, 1.2M edges, 40 classes, temporal split.
#                 Metric: accuracy (40-class classification).
#                 Dataset description : https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
#                 OGB leaderboard     : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
#                 OGB baseline code   : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
#   ogbl-collab : ~235K nodes, 1.2M edges (train), link prediction.
#                 Metric: Hits@50.
#                 Dataset description : https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
#                 OGB leaderboard     : https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-collab
#                 OGB baseline code   : https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
#
# Tier 3 -- large, stress / optional (Phase 5):
#   ogbn-products: ~2.4M nodes, 61M edges, 47 classes, sales-rank split.
#                  Requires --use-sampling (NeighborLoader) to avoid OOM.
#                  Dataset description : https://ogb.stanford.edu/docs/nodeprop/#ogbn-products
#                  OGB leaderboard     : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products
#                  OGB baseline code   : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
#   ogbl-citation2: ~2.5M nodes, 30M edges, directed citation network.
#                  Metric: MRR (Mean Reciprocal Rank over 1000 negatives).
#                  Dataset description : https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2
#                  OGB leaderboard     : https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-citation2
#                  OGB baseline code   : https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/citation2
# ---------------------------------------------------------------------------
_PLANETOID_NAME_MAP = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
PLANETOID_DATASETS  = set(_PLANETOID_NAME_MAP.keys())
OGB_NODE_DATASETS  = {"ogbn-arxiv", "ogbn-products", "ogbn-mag"}
OGB_LINK_DATASETS  = {"ogbl-collab", "ogbl-citation2"}
# Knowledge-graph completion datasets (DistMult model, separate pipeline)
OGB_KG_DATASETS    = {"ogbl-biokg"}

DATASET_TIER: dict[str, int] = {
    "cora":           1,  # Phase 1 & 2: smoke test / initial compilation experiments
    "citeseer":       1,  # Phase 2i: repeat of Phase 2a-2h on CiteSeer
    "pubmed":         1,  # Phase 2j: repeat of Phase 2a-2h on PubMed
    "ogbn-arxiv":     2,  # Phase 3: primary benchmark -- all 5 modes x 4 models x 2 frameworks
    "ogbl-collab":    2,  # Phase 4: link prediction generalisation experiments
    "ogbn-mag":       2,  # Phase 6: heterogeneous R-GCN benchmark (PyG only)
    "ogbl-biokg":     2,  # Phase 7: KG completion DistMult benchmark (PyG only)
    "ogbn-products":  3,  # Phase 5 (optional): stress test; requires --use-sampling
    "ogbl-citation2": 3,  # Phase 5 (optional): stress test MRR; requires --use-sampling
}

# ---------------------------------------------------------------------------
# Graph-break taxonomy
# ---------------------------------------------------------------------------
# torch.compile uses torch._dynamo to trace the model as a computation graph.
# Whenever Dynamo encounters an operation it cannot trace statically, it emits
# a "graph break" -- the compiled region ends, eager execution resumes, and a
# new graph begins. Many graph breaks reduce or eliminate the compilation
# speedup, so documenting their root cause is a primary thesis metric.
#
# These categories follow the torch.compile documentation taxonomy:
#   https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html
#
# scatter_add (used by PyG for message passing):
#   PyG uses torch_scatter.scatter_add / torch.scatter_add for neighbourhood
#   aggregation. This may cause Dynamo graph breaks on certain PyTorch versions.
#   Source: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
#
# SpMM / SDDMM (used by DGL for message passing):
#   DGL's C++ backend dispatches sparse ops (SpMM = sparse-dense matrix multiply,
#   SDDMM = sampled dense-dense matrix multiply) via custom ATen extensions.
#   These are not always symbolically traceable by Dynamo.
#   DGL ops reference: https://docs.dgl.ai/api/python/dgl.ops.html
#
# GAT / GATConv (both PyG and DGL) is marked HIGH RISK in the experiment plan
# because attention scatter operations commonly trigger sparse_op breaks.
# ---------------------------------------------------------------------------
BREAK_CATEGORIES: dict[str, list[str]] = {
    # Sparse tensor ops (scatter, segment_csr) used by PyG internally.
    # GAT and GIN message-passing in PyG relies on scatter_add which may
    # cause breaks depending on the PyTorch / TorchInductor version.
    "sparse_op":      ["torch.sparse", "SparseTensor", "scatter", "segment_csr",
                       "sparse", "coalesce"],

    # Data-dependent shapes: Dynamo cannot statically determine output size.
    # Common in graphs with variable node degrees.
    "data_dependent": ["data-dependent", "dynamic shape", "unbacked symint",
                       "dynamic", "SymInt"],

    # Custom C++ / CUDA extensions (e.g., DGL kernels) that are not
    # symbolically traceable by Dynamo.
    "custom_kernel":  ["custom op", "fallback", "C++ extension",
                       "_torch_dispatch", "torch_function"],

    # Python control flow (if/while/for/assert) that depends on tensor values.
    # Cannot be compiled into a static graph; Dynamo falls back to eager.
    "control_flow":   ["if ", "while ", "for ", "assert", "aten::where"],

    # In-place operations that modify tensors in-place; problematic under
    # CUDA Graphs (reduce-overhead, max-autotune) because captured graphs
    # require static memory addresses.
    "in_place":       ["inplace", "in-place", "._"],
}


def categorise_graph_breaks(break_reasons: list[str]) -> dict[str, int]:
    """
    Categorise Dynamo graph-break reasons by root cause.

    Called after _get_dynamo_metrics() to produce a structured breakdown
    of break categories for LaTeX Table 3 (usability / diagnostics).
    """
    counts: dict[str, int] = {k: 0 for k in BREAK_CATEGORIES}
    counts["other"] = 0
    for reason in break_reasons:
        matched = False
        for cat, keywords in BREAK_CATEGORIES.items():
            if any(kw.lower() in reason.lower() for kw in keywords):
                counts[cat] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# The OGB leaderboard protocol requires results over 10 random seeds
# (seeds 0-9 or any 10 fixed seeds).
# Source: https://ogb.stanford.edu/docs/leader_rules/
# This benchmark uses a single seed (default: 42) because the focus is on
# latency / compilation behaviour rather than accuracy. The seed controls
# model weight initialisation, data shuffling, and CUDA non-determinism.

def _count_params(model: nn.Module) -> int:
    """
    Return the total number of trainable parameters.

    Used to populate the #Params row in LaTeX Table 3 (usability).
    Matches the parameter counting convention used by OGB baseline code:
      https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducible inference and training timings.

    Sets:
      - torch.manual_seed         : weight initialisation and dropout masks
      - np.random.seed            : numpy ops (data loading, IQR computation)
      - random.seed               : Python built-in RNG (negative sampling)
      - torch.cuda.manual_seed_all: per-GPU RNG state
      - cudnn.deterministic=True  : forces deterministic CUDA kernels
      - cudnn.benchmark=False     : disables auto-tuner which changes across runs

    Best practice: cudnn.benchmark=False is essential when benchmarking compile
    modes because the auto-tuner would otherwise run its own kernel search and
    conflate with torch.compile overhead.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

def get_system_info() -> dict[str, Any]:
    """Collect hardware and software environment details."""
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    cpu_model = platform.processor()
    if not cpu_model:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            cpu_model = "unknown"

    return {
        "hostname":            socket.gethostname(),
        "python_version":      platform.python_version(),
        "pytorch_version":     torch.__version__,
        "pyg_version":         torch_geometric.__version__,
        "dgl_version":         dgl.__version__,
        "cuda_available":      torch.cuda.is_available(),
        "cuda_version":        torch.version.cuda,
        "gpu_name":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                               if torch.cuda.is_available() else None,
        "cpu_model":           cpu_model,
        "total_ram_gb":        total_ram_gb,
        "platform":            platform.platform(),
    }


# ---------------------------------------------------------------------------
# Models - PyG
# ---------------------------------------------------------------------------
# All PyG models follow the architecture used by the official OGB baselines:
#   - Stacked GNN layers with BatchNorm1d + ReLU/ELU after each hidden layer
#   - Dropout applied before each layer (input dropout)
#   - Final layer has no BN or activation (raw logits for CrossEntropyLoss)
#
# Architecture reference: OGB baseline code for node property prediction
#   https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
#
# BatchNorm1d: Ioffe & Szegedy 2015, "Batch Normalization: Accelerating Deep
#   Network Training by Reducing Internal Covariate Shift"
#   https://arxiv.org/abs/1502.03167
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
#
# Dropout: Srivastava et al. 2014, "Dropout: A Simple Way to Prevent Neural
#   Networks from Overfitting", JMLR 15(56):1929-1958
#   https://jmlr.org/papers/v15/srivastava14a.html
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html
#
# Original model papers:
#   GCN      : Kipf & Welling, ICLR 2017 -- https://arxiv.org/abs/1609.02907
#   GraphSAGE: Hamilton et al., NeurIPS 2017 -- https://arxiv.org/abs/1706.02216
#   GAT      : Velickovic et al., ICLR 2018 -- https://arxiv.org/abs/1710.10903
#   GIN      : Xu et al., ICLR 2019 -- https://arxiv.org/abs/1810.00826
#
# Experiment plan coverage (Phase 2-5):
#   All four architectures x PyG + DGL x all datasets.
#   GAT is marked HIGH RISK (Phases 2e, 2f, 4e, 4f, 5e, 5f) because attention
#   scatter operations commonly produce graph breaks under torch.compile.

class PyGGCN(nn.Module):
    """
    Multi-layer GCN (Graph Convolutional Network) for PyG.

    Architecture:
        [dropout -> GCNConv -> BatchNorm1d -> ReLU] x (num_layers - 1)
        dropout -> GCNConv  (final layer, no BN/activation)

    GCNConv implements the spectral graph convolution from Kipf & Welling 2017:
        H' = sigma(D^{-1/2} A_hat D^{-1/2} H W)
    where A_hat = A + I (adjacency with added self-loops).
    cached=False is used so the normalised adjacency is recomputed each forward
    pass; this is required for correctness with mini-batch (NeighborLoader) where
    edge_index changes between batches.
    Paper  : Kipf & Welling, ICLR 2017 -- https://arxiv.org/abs/1609.02907
    PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html

    Used in Phase 2a (PyG/Cora), Phase 3a (PyG/ogbn-arxiv eager baseline),
    Phase 3a (PyG/ogbn-arxiv all 5 modes), Phase 4a (PyG/ogbl-collab).
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden_feats] * (num_layers - 1) + [out_feats]
        for i in range(num_layers):
            # cached=False: caching the normalised adjacency is only valid for a
            # fixed graph. Mini-batch (NeighborLoader) changes edge_index every
            # batch, so cached=True silently returns stale values after the first
            # batch. Keep False for correctness across all dataset tiers.
            #
            # normalize=False and add_self_loops=False: self-loops and the
            # symmetric D^{-1/2}(A+I)D^{-1/2} normalisation coefficients are
            # pre-computed once at data-load time by GCNNorm and stored in
            # data.edge_weight. The default normalize=True and add_self_loops=True
            # recompute these every forward pass via remove_self_loops() and
            # add_remaining_self_loops(), which mask edge_index and trigger a
            # device synchronisation to determine the output shape -- a source of
            # graph breaks under torch.compile. Disabling them here and consuming
            # the pre-computed coefficients via the edge_weight argument eliminates
            # that overhead entirely.
            self.convs.append(GCNConv(dims[i], dims[i + 1], cached=False,
                                      normalize=False, add_self_loops=False))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        # edge_weight carries the pre-computed GCNNorm coefficients
        # (D^{-1/2}(A+I)D^{-1/2}) from data.edge_weight. Passing it here
        # lets GCNConv(normalize=False) apply the correct symmetric normalisation
        # without recomputing the degree scatter on every forward call.
        # For datasets where no GCNNorm was applied (e.g. DGL path, link pred),
        # edge_weight=None is safe -- GCNConv treats every edge as weight 1.0.
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)


class PyGGraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE (Sample and AGgregate) for PyG.

    Uses mean aggregation: h_v = W * CONCAT(h_v, mean({h_u : u in N(v)}))
    Paper  : Hamilton et al., NeurIPS 2017 -- https://arxiv.org/abs/1706.02216
    PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html

    OGB baseline reference: ogbn-arxiv GraphSAGE (71.49 +/- 0.27%):
      https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
    OGB baseline code:
      https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py

    Used in Phase 2c (PyG/Cora), Phase 3c (PyG/ogbn-arxiv),
    Phase 3c (full compile), Phase 4c (PyG/ogbl-collab link pred).
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden_feats] * (num_layers - 1) + [out_feats]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class PyGGAT(nn.Module):
    """
    Multi-layer GAT (Graph Attention Network) for PyG.

    Uses multi-head attention: h_v = concat/mean(alpha_{vu} * W * h_u)
    where alpha_{vu} are learned attention coefficients normalised by softmax
    over each node's neighbourhood.
    Paper  : Velickovic et al., ICLR 2018 -- https://arxiv.org/abs/1710.10903
    PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html

    OGB baseline: ogbn-arxiv GAT (73.65 +/- 0.16%):
      https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv

    HIGH RISK for torch.compile: The attention scatter operations
    (softmax over variable-degree neighbourhoods) commonly trigger
    sparse_op or data-dependent graph breaks. Experiment plan Phases
    2e, 4e, 5e are flagged for careful graph-break documentation.

    Hidden layer output dim is hidden_feats * heads (concatenated).
    Final layer uses heads=1 and concat=False (mean over single head).
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int,
                 num_layers: int = 2, heads: int = 8, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        # add_self_loops=False on every GATConv layer: self-loops are pre-added to
        # edge_index once at data-load time by AddSelfLoops(). The default
        # add_self_loops=True calls remove_self_loops() and add_remaining_self_loops()
        # on every forward pass, masking edge_index and triggering a device
        # synchronisation that causes graph breaks under torch.compile.
        self.convs.append(GATConv(in_feats, hidden_feats, heads=heads,
                                  dropout=dropout, add_self_loops=False))
        self.bns.append(nn.BatchNorm1d(hidden_feats * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_feats * heads, hidden_feats,
                                      heads=heads, dropout=dropout,
                                      add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_feats * heads))
        self.convs.append(GATConv(hidden_feats * heads, out_feats,
                                  heads=1, concat=False, dropout=dropout,
                                  add_self_loops=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                chunk_size: int | None = None) -> torch.Tensor:
        # chunk_size is retained in the signature so existing CLI calls that pass
        # --gat-chunk-size raise a clear RuntimeError rather than crashing with an
        # unexpected-argument error.
        if chunk_size is not None:
            raise RuntimeError(
                "GAT chunked-forward (--gat-chunk-size) has been removed because "
                "it produced mathematically incorrect results: GATConv softmax "
                "normalisation requires the full neighbourhood and cannot be "
                "reconstructed by summing partial-graph outputs.  "
                "Use --use-sampling instead (auto-enabled for GAT on large datasets)."
            )
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            out = conv(x, edge_index)
            x = F.elu(self.bns[i](out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class PyGGIN(nn.Module):
    """
    Multi-layer GIN (Graph Isomorphism Network) for PyG.

    Aggregation: h_v = MLP((1 + epsilon) * h_v + sum({h_u : u in N(v)}))
    train_eps=True allows learning epsilon per the GIN paper.
    Paper  : Xu et al., ICLR 2019 -- https://arxiv.org/abs/1810.00826
    PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html

    No official OGB GIN baseline on ogbn-arxiv.
    Used in Phase 2g (PyG/Cora), Phase 3g (PyG/ogbn-arxiv all modes).
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden_feats] * (num_layers - 1) + [out_feats]
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if is_last:
                # Final layer: plain linear so the GINConv output is raw logits.
                # BN+ReLU inside the MLP would zero all negative class scores,
                # preventing multi-class classification from converging.
                mlp = nn.Linear(dims[i], dims[i + 1])
            else:
                mlp = nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                    nn.Linear(dims[i + 1], dims[i + 1]),
                )
            self.convs.append(GINConv(mlp, train_eps=True))
            if not is_last:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ---------------------------------------------------------------------------
# Models - DGL
# ---------------------------------------------------------------------------
# DGL models mirror the PyG implementations but use the DGL graph object
# (dgl.DGLGraph) and DGL layer API: layer(g, x) instead of layer(x, edge_index).
#
# DGL paper: Wang et al. 2019 -- https://arxiv.org/abs/1909.01315
#
# DGL layers used (all from dgl.nn.pytorch.conv):
#   GraphConv  : Kipf & Welling GCN
#     Paper  : https://arxiv.org/abs/1609.02907
#     DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GraphConv
#   DGLSAGEConv: Hamilton GraphSAGE (mean aggregation)
#     Paper  : https://arxiv.org/abs/1706.02216
#     DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.SAGEConv
#   DGLGATConv : Velickovic GAT (multi-head attention)
#     Paper  : https://arxiv.org/abs/1710.10903
#     DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GATConv
#   DGLGINConv : Xu GIN (sum aggregation)
#     Paper  : https://arxiv.org/abs/1810.00826
#     DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GINConv
#
# IMPORTANT for torch.compile: DGL uses its own C++ sparse kernels (SpMM,
# SDDMM) which may not be symbolically traceable by torch.Dynamo. This is
# why wrapping DGL models in torch.compile (Phases 4b, 4d, 4f, 4h) is
# flagged as requiring careful graph-break documentation in the experiment plan.

class DGLGCN(nn.Module):
    """
    Multi-layer GCN for DGL.

    Uses dgl.nn.GraphConv which internally calls dgl.ops.u_mul_e (SpMM)
    for the message-passing step. The norm='both' default applies symmetric
    normalisation D^{-1/2} A D^{-1/2}, matching the PyG GCNConv behaviour.
    Paper  : Kipf & Welling, ICLR 2017 -- https://arxiv.org/abs/1609.02907
    DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GraphConv

    Used in Phase 2b (DGL/Cora), Phase 3b (DGL/ogbn-arxiv),
    Phase 3b (DGL/ogbn-arxiv all 5 compile modes).
    """
    def __init__(self, in_feats: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.convs.append(GraphConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(g, x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](g, x)


class DGLGraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE for DGL.

    Uses 'mean' aggregator: h_v = W * CONCAT(h_v, mean_neighbour).
    Matches the PyGGraphSAGE architecture for fair framework comparison.
    Paper  : Hamilton et al., NeurIPS 2017 -- https://arxiv.org/abs/1706.02216
    DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.SAGEConv

    Used in Phase 2d (DGL/Cora), Phase 3d (DGL/ogbn-arxiv),
    Phase 3d (DGL/ogbn-arxiv all 5 compile modes), Phase 4d (ogbl-collab).
    """
    def __init__(self, in_feats: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.convs.append(DGLSAGEConv(dims[i], dims[i + 1], "mean"))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(g, x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](g, x)


class DGLGAT(nn.Module):
    """
    Multi-layer GAT for DGL.

    Uses dgl.nn.GATConv which calls SDDMM (sampled dense-dense matrix
    multiply) for attention coefficient computation. This kernel is a
    known source of graph breaks under torch.compile because it dispatches
    through DGL's custom C++ dispatch mechanism.
    Paper  : Velickovic et al., ICLR 2018 -- https://arxiv.org/abs/1710.10903
    DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GATConv

    HIGH RISK: Phases 2f, 4f, 5f in experiment plan.
    The forward output is (N, heads, out_feats); hidden layers flatten to
    (N, heads * out_feats) before BatchNorm. Final layer squeezes to (N, num_classes).
    """
    def __init__(self, in_feats: int, hidden: int, num_classes: int,
                 num_layers: int = 2, heads: int = 8, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.convs.append(DGLGATConv(in_feats, hidden, num_heads=heads))
        self.bns.append(nn.BatchNorm1d(hidden * heads))
        for _ in range(num_layers - 2):
            self.convs.append(DGLGATConv(hidden * heads, hidden, num_heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden * heads))
        self.convs.append(DGLGATConv(hidden * heads, num_classes, num_heads=1))

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.bns[i](conv(g, x).flatten(1)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](g, x).squeeze(1)


class DGLGIN(nn.Module):
    """
    Multi-layer GIN for DGL.

    Uses 'sum' aggregation (sum of neighbour features) matching the original
    GIN paper. The MLP is a 2-layer feedforward network per layer.
    train_eps=True allows learning epsilon per the GIN paper.
    Paper  : Xu et al., ICLR 2019 -- https://arxiv.org/abs/1810.00826
    DGL API: https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GINConv

    Used in Phase 2h (DGL/Cora), Phase 3h (DGL/ogbn-arxiv all 5 modes).
    """
    def __init__(self, in_feats: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        dims = [in_feats] + [hidden] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if is_last:
                # Final layer: plain linear so the GINConv output is raw logits.
                # BN+ReLU inside the MLP would zero all negative class scores,
                # preventing multi-class classification from converging.
                mlp = nn.Linear(dims[i], dims[i + 1])
            else:
                mlp = nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                    nn.Linear(dims[i + 1], dims[i + 1]),
                )
            self.convs.append(DGLGINConv(mlp, "sum", learn_eps=True))
            if not is_last:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bns[i](conv(g, x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](g, x)


# ---------------------------------------------------------------------------
# Models - R-GCN (heterogeneous, ogbn-mag only)
# ---------------------------------------------------------------------------
# R-GCN handles multiple relation types natively via RGCNConv (num_relations
# parameter). Non-paper nodes have no input features in ogbn-mag; their
# embeddings are initialised as learnable parameters (nn.Embedding tables).
# This mirrors the official OGB R-GCN baseline:
#   https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/rgcn.py
#
# R-GCN paper: Schlichtkrull et al. 2018 -- https://arxiv.org/abs/1703.06103
# PyG RGCNConv: https://pytorch-geometric.readthedocs.io/en/latest/
#               generated/torch_geometric.nn.conv.RGCNConv.html
#
# ogbn-mag edge types (7 total, including 3 reversed for bidirectional passing):
#   (paper, cites, paper)           -- made undirected via to_undirected()
#   (author, writes, paper)         + reverse (paper, to, author)
#   (author, affiliated_with, institution) + reverse (institution, to, author)
#   (paper, has_topic, field_of_study)    + reverse (field_of_study, to, paper)
#
# Phase 6 in the experiment plan.

# Relation-type mapping for RGCNConv on ogbn-mag.
# Order determines the integer relation indices used in edge_type tensors.
_MAG_EDGE_TYPES = [
    ("paper",          "cites",           "paper"),
    ("author",         "writes",          "paper"),
    ("author",         "affiliated_with", "institution"),
    ("paper",          "has_topic",       "field_of_study"),
    # reversed edges (added for bidirectional message passing)
    ("paper",          "to",              "author"),
    ("institution",    "to",              "author"),
    ("field_of_study", "to",              "paper"),
]
_MAG_NUM_RELATIONS = len(_MAG_EDGE_TYPES)


class RGCN(nn.Module):
    """
    Multi-layer R-GCN for heterogeneous ogbn-mag.

    All node types are flattened into a single node space. Non-paper node
    features are represented by learnable embedding tables (same approach as
    the official OGB baseline). The forward pass returns only paper-node logits.

    Forward signature: forward(paper_feat, edge_index, edge_type, n_paper)
      paper_feat : [N_paper, in_channels]
      edge_index  : [2, E_total] flattened homogeneous edge index
      edge_type   : [E_total]    relation index per edge
      n_paper     : int          number of paper nodes (for output slicing)

    Paper  : Schlichtkrull et al. 2018 -- https://arxiv.org/abs/1703.06103
    PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/
             torch_geometric.nn.conv.RGCNConv.html

    Used in Phase 6 (PyG/ogbn-mag, all 5 compile modes).
    """

    def __init__(
        self,
        in_channels:      int,   # paper feature dim (128 for ogbn-mag)
        hidden:           int,
        out_channels:     int,   # num_classes = 349 for ogbn-mag
        num_layers:       int,
        dropout:          float,
        num_relations:    int,
        num_authors:      int,
        num_institutions: int,
        num_fields:       int,
    ):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers

        # Learnable embeddings for node types without input features
        self.emb_author      = nn.Embedding(num_authors,      in_channels)
        self.emb_institution = nn.Embedding(num_institutions, in_channels)
        self.emb_field       = nn.Embedding(num_fields,       in_channels)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(RGCNConv(dims[i], dims[i + 1],
                                       num_relations=num_relations,
                                       aggr="mean"))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_author.weight)
        nn.init.xavier_uniform_(self.emb_institution.weight)
        nn.init.xavier_uniform_(self.emb_field.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def _build_x(self, paper_feat: torch.Tensor) -> torch.Tensor:
        """Concatenate paper features and non-paper embeddings into one matrix."""
        return torch.cat([
            paper_feat,
            self.emb_author.weight,
            self.emb_institution.weight,
            self.emb_field.weight,
        ], dim=0)

    def forward(
        self,
        paper_feat: torch.Tensor,
        edge_index:  torch.Tensor,
        edge_type:   torch.Tensor,
        n_paper:     int,
    ) -> torch.Tensor:
        x = self._build_x(paper_feat)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x[:n_paper]   # return only paper-node logits


# ---------------------------------------------------------------------------
# Models - DistMult (KG completion, ogbl-biokg only)
# ---------------------------------------------------------------------------
# DistMult scoring function:
#   score(h, r, t) = <e_h, w_r, e_t>  (element-wise product, then sum)
#
# Each entity type has its own nn.Embedding table.
# Each relation type has its own diagonal weight vector (nn.Embedding of emb_dim).
#
# ogbl-biokg has 5 entity types and 51 relation types.
# Evaluation metric: MRR (same-type negatives, 500 per head/tail replacement).
# Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
#
# DistMult paper: Yang et al. 2015 -- https://arxiv.org/abs/1412.6575
# OGB biokg baseline:
#   https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/biokg/
#
# Phase 7 in the experiment plan.

# Entity types defined by ogbl-biokg (order determines embedding-table indices).
_BIOKG_ENTITY_TYPES = ["disease", "protein", "drug", "sideeffect", "function"]


class DistMult(nn.Module):
    """
    DistMult KG-completion model for ogbl-biokg.

    Separate embedding tables per entity type; one diagonal weight vector
    per relation type.

    Forward signature: forward(h_type, h_idx, r_idx, t_type, t_idx) -> [B]
      h_type, t_type : str   -- entity type names
      h_idx, t_idx   : [B]   -- entity indices (type-local)
      r_idx          : [B]   -- relation indices

    Paper  : Yang et al. 2015 -- https://arxiv.org/abs/1412.6575
    Dataset: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg

    Used in Phase 7 (PyG/ogbl-biokg, all 5 compile modes).
    """

    def __init__(self, num_nodes_dict: dict[str, int],
                 num_relations: int, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        # Per-type entity embeddings
        self.entity_emb = nn.ModuleDict({
            etype: nn.Embedding(n, emb_dim)
            for etype, n in num_nodes_dict.items()
        })
        # Per-relation diagonal weights
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.entity_emb.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(self, h_emb: torch.Tensor, r_emb: torch.Tensor,
              t_emb: torch.Tensor) -> torch.Tensor:
        """DistMult scoring: sum of element-wise product h * r * t."""
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def forward(self, h_type: str, h_idx: torch.Tensor,
                r_idx: torch.Tensor,
                t_type: str, t_idx: torch.Tensor) -> torch.Tensor:
        h_emb = self.entity_emb[h_type](h_idx)
        r_emb = self.rel_emb(r_idx)
        t_emb = self.entity_emb[t_type](t_idx)
        return self.score(h_emb, r_emb, t_emb)


# ---------------------------------------------------------------------------
# Link prediction encoder wrappers
# ---------------------------------------------------------------------------
# For link prediction (Phase 4: ogbl-collab), the GNN backbone computes node
# embeddings, and a dot-product decoder scores each candidate edge.
#
# Scoring function: score(u, v) = z_u . z_v (dot product of node embeddings)
# This is the standard decoder used in all OGB link prediction baselines.
# Source: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
#   (see LinkPredictor class and the decode step in the training loop)
# Also referenced from: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
#
# The PyGLinkEncoder and DGLLinkEncoder are thin wrappers that add a decode()
# method to any GNN backbone, keeping the inference benchmark path identical
# to node classification (same benchmark_inference function is reused).

class PyGLinkEncoder(nn.Module):
    """
    Wraps a PyG GNN backbone for link prediction.

    forward(x, edge_index, edge_weight=None) -> node embeddings z [N, hidden]
    decode(z, edge_pairs)  -> edge scores [E] via dot product z_u . z_v

    Dot-product decoder pattern from OGB collab baseline:
      https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
    Dataset: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab

    Used in Phase 4a, 4c, 4e, 4g (PyG link prediction on ogbl-collab).
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x, edge_index, edge_weight=None):
        # edge_weight (GCNNorm coefficients) is only supported by PyGGCN.
        # PyGGraphSAGE, PyGGAT, and PyGGIN only accept (x, edge_index).
        if edge_weight is not None and isinstance(self.base, PyGGCN):
            return self.base(x, edge_index, edge_weight)
        return self.base(x, edge_index)

    def decode(self, z, edge_pairs):
        return (z[edge_pairs[0]] * z[edge_pairs[1]]).sum(dim=-1)


class DGLLinkEncoder(nn.Module):
    """
    Wraps a DGL backbone for link prediction.

    Same dot-product decoder as PyGLinkEncoder.
    Source: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
    Dataset: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, g, x):
        return self.base(g, x)

    def decode(self, z, edge_pairs):
        return (z[edge_pairs[0]] * z[edge_pairs[1]]).sum(dim=-1)


# ---------------------------------------------------------------------------
# Model factory and forward adapter
# ---------------------------------------------------------------------------
# build_model() is the single entry point for constructing any of the 8
# (framework x model) combinations used across all experiment plan phases.
# All combinations are registered in a dict keyed by (framework, model_name)
# to ensure no combination is accidentally omitted.
#
# Supported combinations (Experiment Plan Phases 2-5):
#   (pyg, gcn)       -- Phase 2a, 3a, 4a, 5a
#   (pyg, graphsage) -- Phase 2c, 3c, 4c, 5c
#   (pyg, gat)       -- Phase 2e, 3e, 4e, 5e  [HIGH RISK]
#   (pyg, gin)       -- Phase 2g, 3g, 4g, 5g
#   (dgl, gcn)       -- Phase 2b, 3b, 4b, 5b
#   (dgl, graphsage) -- Phase 2d, 3d, 4d, 5d
#   (dgl, gat)       -- Phase 2f, 3f, 4f, 5f  [HIGH RISK]
#   (dgl, gin)       -- Phase 2h, 3h, 4h, 5h

def build_model(framework, model_name, in_feats, hidden, num_classes, device,
                num_layers=2, dropout=0.5, gat_heads=8):
    """
    Build and return a GNN model for the given (framework, model_name) pair.

    All 8 framework x model combinations are registered in a dict so any
    combination from the experiment plan can be run without code changes.
    Raises ValueError for unknown combinations (fail-fast design).
    """
    fw = framework.lower()
    mn = model_name.lower()

    registry = {
        ("pyg", "gcn"):       lambda: PyGGCN(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout),
        ("pyg", "graphsage"): lambda: PyGGraphSAGE(in_feats, hidden, num_classes,
                                                   num_layers=num_layers, dropout=dropout),
        ("pyg", "gat"):       lambda: PyGGAT(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout,
                                             heads=gat_heads),
        ("pyg", "gin"):       lambda: PyGGIN(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout),
        ("dgl", "gcn"):       lambda: DGLGCN(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout),
        ("dgl", "graphsage"): lambda: DGLGraphSAGE(in_feats, hidden, num_classes,
                                                   num_layers=num_layers, dropout=dropout),
        ("dgl", "gat"):       lambda: DGLGAT(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout,
                                             heads=gat_heads),
        ("dgl", "gin"):       lambda: DGLGIN(in_feats, hidden, num_classes,
                                             num_layers=num_layers, dropout=dropout),
    }
    key = (fw, mn)
    if key not in registry:
        raise ValueError(f"Unknown combination framework='{framework}' model='{model_name}'.")
    return registry[key]().to(device)


def build_link_model(framework, model_name, in_feats, hidden, device,
                     num_layers=2, dropout=0.5, gat_heads=8):
    """Build a link-prediction encoder."""
    base = build_model(framework, model_name, in_feats, hidden, hidden, device,
                       num_layers=num_layers, dropout=dropout, gat_heads=gat_heads)
    if framework.lower() == "pyg":
        return PyGLinkEncoder(base).to(device)
    return DGLLinkEncoder(base).to(device)


def link_model_forward(framework, model, x, edge_index, dgl_graph=None, edge_weight=None):
    """Node embedding forward pass for link prediction."""
    if framework.lower() == "pyg":
        return model(x, edge_index, edge_weight)
    if dgl_graph is None:
        raise ValueError("DGL link forward requires dgl_graph.")
    return model(dgl_graph, x)


def model_forward(framework, model, x, edge_index, dgl_graph=None,
                  edge_weight=None):
    """Unified forward pass dispatcher.

    edge_weight is threaded through to PyGGCN.forward() so that the
    pre-computed GCNNorm coefficients (stored in data.edge_weight) reach
    GCNConv(normalize=False). For all other models the argument is accepted
    but unused (they do not take edge_weight in their forward signature).
    """
    if framework.lower() == "pyg":
        # PyGGCN accepts edge_weight; all other PyG models ignore it.
        try:
            return model(x, edge_index, edge_weight)
        except TypeError:
            # Fallback for models whose forward() does not accept edge_weight
            # (e.g. link-encoder wrappers, SAGE, GIN, GAT).
            return model(x, edge_index)
    if dgl_graph is None:
        raise ValueError("DGL forward requires dgl_graph.")
    return model(dgl_graph, x)


def _to_dgl_graph(edge_index, num_nodes, device):
    """Convert a PyG-style edge_index to a DGL graph with self-loops."""
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    return g.to(device)


# ---------------------------------------------------------------------------
# CPU utilization sampler
# ---------------------------------------------------------------------------

class _CpuSampler:
    """Background thread that samples psutil.cpu_percent every `interval` seconds."""
    def __init__(self, interval: float = 0.1):
        self._interval = interval
        self._samples: list[float] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.clear()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self._interval)

    def stop(self) -> list[float]:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return list(self._samples)


# ---------------------------------------------------------------------------
# Dynamo / graph-capture metrics
# ---------------------------------------------------------------------------
# torch._dynamo.explain() is the official API to analyse what torch.compile
# will do with a model. It traces the model and reports:
#   - graphs: the FX graphs Dynamo will compile
#   - break_reasons: why each graph boundary was created
#   - ops_per_graph: number of ATen ops in each compiled subgraph
#   - total_ops: total ATen ops in the traced model
#
# graph_capture_rate_pct = sum(ops_per_graph) / total_ops * 100
#
# 100% capture = all ops are inside a single compiled graph (no breaks).
# <100% = some ops fell back to eager, limiting potential speedup.
#
# Sources:
#   torch.compile troubleshooting guide (graph breaks, Dynamo taxonomy):
#     https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html
#   torch._dynamo.explain() API:
#     https://pytorch.org/docs/stable/torch.compiler_api.html#torch._dynamo.explain
#   FX graph representation used by Dynamo:
#     https://pytorch.org/docs/stable/fx.html
#   TorchInductor (the default torch.compile backend):
#     https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html
#
# NOTE: torch._dynamo.explain() is an internal/semi-public API. The exact
# attributes available on the ExplainOutput object changed between PyTorch
# versions. The code handles both the exact (ops-based) and approximate
# (graphs/breaks count heuristic) cases.
#
# For DGL models the explain() call is made directly as explain(model, g, x)
# rather than via a lambda. A lambda is opaque to Dynamo -- it sees no graphs
# and reports zero breaks even when many exist (e.g. DGL GAT SDDMM ops).
# torch._dynamo.reset() is called before each explain() to clear the trace
# cache from any previous compile() calls in the same subprocess.
# A binary-safe stdout wrapper is used because DGL's C++ backend can emit
# non-UTF-8 bytes into exception text; the wrapper replaces them with the
# Unicode replacement character instead of letting the codec crash.

def _get_dynamo_metrics(framework, model, x, edge_index, dgl_graph=None,
                        train_loader=None, edge_weight=None) -> dict[str, Any]:
    """
    Compute graph-capture rate by running torch._dynamo.explain() on the model.

    Returns graph_capture_rate_pct, break_reasons, break_categories, and
    ops_per_graph. These populate Table 3 (usability) in the LaTeX output.

    For sampled datasets (train_loader is not None), a single mini-batch is
    used instead of the full graph to avoid OOM on large datasets like
    ogbn-products (120 GiB allocation for full-graph GCN forward).

    Falls back to a heuristic (n_graphs / (n_graphs + n_breaks)) if
    ops_per_graph is not available in the installed PyTorch version.
    """
    try:
        model.eval()
        torch._dynamo.reset()

        # For sampled datasets use a single batch; full graph OOMs on large datasets.
        if train_loader is not None:
            _batch = next(iter(train_loader))
            _batch = _batch.to(x.device)
            _x          = _batch.x
            _edge_index = _batch.edge_index
            _dgl_graph  = (_to_dgl_graph(_batch.edge_index, _batch.x.shape[0], x.device)
                           if framework.lower() == "dgl" else None)
            _ew         = getattr(_batch, "edge_weight", None)
        else:
            _x, _edge_index, _dgl_graph, _ew = x, edge_index, dgl_graph, edge_weight

        class _ByteSafeCapture:
            """stdout wrapper that accepts both str and bytes writes without raising."""
            def __init__(self):
                self._buf = io.StringIO()
            def write(self, s):
                if isinstance(s, (bytes, bytearray)):
                    s = s.decode("utf-8", errors="replace")
                return self._buf.write(s)
            def flush(self):
                self._buf.flush()
            def getvalue(self):
                return self._buf.getvalue()

        buf = _ByteSafeCapture()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            if framework.lower() == "pyg":
                expl = torch._dynamo.explain(model)(_x, _edge_index, _ew) \
                       if _ew is not None else \
                       torch._dynamo.explain(model)(_x, _edge_index)
            else:
                expl = torch._dynamo.explain(model)(_dgl_graph, _x)
        finally:
            sys.stdout = old_stdout

        is_exact = False
        rate     = None
        ops_per_graph_list = None

        if hasattr(expl, "graphs"):
            break_reasons = [str(r) for r in getattr(expl, "break_reasons", [])]
            total_ops     = getattr(expl, "total_ops", None)
            ops_per_graph = getattr(expl, "ops_per_graph", None)

            if ops_per_graph is not None and hasattr(ops_per_graph, "__iter__"):
                ops_per_graph_list = list(ops_per_graph)

            if total_ops and ops_per_graph and total_ops > 0:
                captured = sum(ops_per_graph_list) if ops_per_graph_list else ops_per_graph
                rate     = round(captured / total_ops * 100.0, 2)
                is_exact = True
            if rate is None:
                n_graphs = len(getattr(expl, "graphs", []))
                n_breaks = len(break_reasons)
                if n_graphs > 0 and n_breaks == 0:
                    rate = 100.0
                elif n_graphs > 0:
                    rate = round(n_graphs / (n_graphs + n_breaks) * 100.0, 2)
                else:
                    rate = 0.0
                log.warning("graph_capture_rate_pct is an approximation "
                            "(graphs=%d, breaks=%d).", n_graphs, n_breaks)
        else:
            text          = buf.getvalue()
            break_reasons = re.findall(r"Graph Break.*?(?=\n|$)", text)

        n_breaks = len(break_reasons)
        if rate is None:
            text_val = buf.getvalue()
            n_graphs = len(re.findall(r"Graph \d+", text_val))
            if n_graphs > 0 and n_breaks == 0:
                rate = 100.0
            elif n_graphs > 0:
                rate = round(n_graphs / (n_graphs + n_breaks) * 100.0, 2)
            else:
                rate = 0.0

        if ops_per_graph_list is not None:
            ops_per_graph_list = [int(x) for x in ops_per_graph_list
                                  if isinstance(x, (int, float, np.integer, np.floating))]
            if not ops_per_graph_list:
                ops_per_graph_list = None

        return {
            "graph_capture_rate_pct":      rate,
            "graph_capture_rate_is_exact": is_exact,
            "graph_breaks":                break_reasons,
            "unsupported_op_count":        n_breaks,
            "ops_per_graph":               ops_per_graph_list,
            "break_categories":            categorise_graph_breaks(break_reasons),
        }
    except UnicodeDecodeError as exc:
        log.warning(
            "_dynamo.explain raised UnicodeDecodeError "
            "(byte 0x%02x at position %d, codec=%r): %s",
            exc.object[exc.start] if exc.object and exc.start < len(exc.object) else 0,
            exc.start, exc.encoding, exc)
        return {
            "graph_capture_rate_pct":      None,
            "graph_capture_rate_is_exact": False,
            "graph_breaks":                [str(exc)],
            "unsupported_op_count":        None,
            "ops_per_graph":               None,
            "break_categories":            None,
        }
    except Exception as exc:
        log.warning("_dynamo.explain failed: %s", exc)
        return {
            "graph_capture_rate_pct":      None,
            "graph_capture_rate_is_exact": False,
            "graph_breaks":                [str(exc)],
            "unsupported_op_count":        None,
            "ops_per_graph":               None,
            "break_categories":            None,
        }


def _get_cuda_kernel_count(framework, model, x, edge_index, dgl_graph=None,
                           train_loader=None, edge_weight=None) -> int | None:
    """
    Count distinct CUDA kernels launched during a single forward pass.

    Uses torch.profiler.profile with ProfilerActivity.CUDA to capture the
    device-side kernel timeline, then counts entries with non-zero CUDA time.
    Source: https://pytorch.org/docs/stable/profiler.html

    For sampled datasets (train_loader is not None), a single mini-batch is
    used instead of the full graph to avoid OOM on large datasets.
    """
    if not torch.cuda.is_available():
        return None
    try:
        if train_loader is not None:
            _batch      = next(iter(train_loader))
            _batch      = _batch.to(x.device)
            _x          = _batch.x
            _edge_index = _batch.edge_index
            _dgl_graph  = (_to_dgl_graph(_batch.edge_index, _batch.x.shape[0], x.device)
                           if framework.lower() == "dgl" else None)
            _ew         = getattr(_batch, "edge_weight", None)
        else:
            _x, _edge_index, _dgl_graph, _ew = x, edge_index, dgl_graph, edge_weight

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            with torch.no_grad():
                if framework.lower() == "dgl":
                    model(_dgl_graph, _x)
                else:
                    model_forward(framework, model, _x, _edge_index, _dgl_graph,
                                  edge_weight=_ew)

        def _cuda_time(e):
            if hasattr(e, "self_device_time_total"):
                return e.self_device_time_total
            return getattr(e, "self_cuda_time_total", 0)

        return len([e for e in prof.key_averages() if _cuda_time(e) > 0])
    except Exception as exc:
        log.warning("CUDA kernel profiling failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Accuracy and numerical equivalence
# ---------------------------------------------------------------------------
# Accuracy is evaluated using the official OGB NodeEvaluator / LinkEvaluator.
# These evaluators ensure the metric matches exactly what the OGB leaderboard
# uses, making our (short-run) accuracy numbers a directional sanity check.
#
# OGB NodeEvaluator -- node property prediction:
#   Source  : https://ogb.stanford.edu/docs/nodeprop/#eval
#   Snippet from that page:
#     from ogb.nodeproppred import Evaluator
#     evaluator = Evaluator(name=d_name)
#     print(evaluator.expected_input_format)
#     print(evaluator.expected_output_format)
#     result_dict = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
#     # y_true / y_pred : torch tensors of shape (num_nodes, 1)
#     # result_dict["acc"] -> float accuracy in [0, 1]
#
# OGB LinkEvaluator -- link property prediction:
#   Source  : https://ogb.stanford.edu/docs/linkprop/#eval
#   Snippet from that page:
#     from ogb.linkproppred import Evaluator
#     evaluator = Evaluator(name=d_name)
#     result_dict = evaluator.eval(input_dict)
#     # ogbl-collab    : input_dict = {"y_pred_pos": Tensor[E],
#     #                               "y_pred_neg": Tensor[E_neg]}
#     #                  result_dict["hits@50"] -> float in [0, 1]
#     # ogbl-citation2 : result_dict["mrr_list"] -> Tensor[E], per-edge MRR
#
# Numerical equivalence check: For compiled modes, we verify that the
# compiled model produces logits within atol=rtol=1e-3 of the eager model
# (same weights, same input). This catches rare cases where compilation
# changes numerical behaviour (e.g., fused ops with different rounding).

def _make_node_evaluator(dataset_name: str):
    """
    Return a node-classification evaluator for the given dataset.

    For OGB datasets instantiates ogb.nodeproppred.Evaluator, which wraps
    the official per-dataset metric (accuracy for all ogbn-* datasets).

    Instantiation pattern from OGB docs:
      https://ogb.stanford.edu/docs/nodeprop/#eval
        evaluator = Evaluator(name=d_name)
        result_dict = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    Uses a local accuracy computation for Planetoid datasets (Cora/CiteSeer/PubMed)
    which don't have an OGB evaluator but use the same acc formula.
    """
    n = dataset_name.lower()
    if n in OGB_NODE_DATASETS:
        ev = NodeEvaluator(name=n)
        log.debug("OGB NodeEvaluator '%s' input format: %s", n, ev.expected_input_format)
        log.debug("OGB NodeEvaluator '%s' output format: %s", n, ev.expected_output_format)
        return ev

    class _PlanetoidEvaluator:
        @staticmethod
        def eval(input_dict):
            y_true = input_dict["y_true"].squeeze()
            y_pred = input_dict["y_pred"].squeeze()
            correct = (y_pred == y_true).sum().item()
            return {"acc": correct / y_true.size(0)}
    return _PlanetoidEvaluator()


@torch.no_grad()
def evaluate_node(framework, model, x, edge_index, labels, mask,
                  evaluator, dgl_graph=None, train_loader=None,
                  edge_weight=None) -> float:
    """
    Evaluate node classification accuracy.

    Full-batch path (train_loader=None): single forward pass on the entire graph.
    Mini-batch path (train_loader provided): a temporary NeighborLoader is built
    with the same sampling parameters as train_loader but scoped to the nodes
    selected by mask. This avoids both the full-graph OOM and the incorrect
    use of train_loader (which only covers training seed nodes).
    """
    model.eval()
    if train_loader is not None:
        # Guard: train_loader.data.y and the separately passed `labels` tensor must
        # agree in node count. We always index via `labels` and never touch
        # train_loader.data.y for ground-truth lookup.
        n_nodes_loader = train_loader.data.num_nodes
        if labels.shape[0] != n_nodes_loader:
            raise ValueError(
                f"evaluate_node: labels.shape[0]={labels.shape[0]} does not match "
                f"train_loader.data.num_nodes={n_nodes_loader}.  "
                "Ensure the same labels tensor is used for training and evaluation."
            )
        # Derive sampling hyperparameters from the existing train_loader so the
        # evaluation uses the same neighbourhood depth/width.
        eval_loader = NeighborLoader(
            train_loader.data,
            num_neighbors=(
                getattr(train_loader, "num_neighbors", None)
                or getattr(getattr(train_loader, "sampler", None), "num_neighbors", None)
                or [-1]
            ),
            batch_size=train_loader.batch_size,
            input_nodes=mask.nonzero(as_tuple=False).squeeze(1),
            shuffle=False,
        )
        all_preds = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        for batch in eval_loader:
            batch = batch.to(x.device)
            if framework.lower() == "dgl":
                _b_dgl = _to_dgl_graph(batch.edge_index, batch.x.shape[0], batch.x.device)
                out = model(_b_dgl, batch.x)
            else:
                ew  = getattr(batch, "edge_weight", None)
                out = model_forward(framework, model, batch.x, batch.edge_index,
                                    None, edge_weight=ew)
            preds = out[:batch.batch_size].argmax(dim=-1)
            all_preds[batch.n_id[:batch.batch_size]] = preds
        eval_mask = mask & (all_preds >= 0)
        if eval_mask.sum() == 0:
            return 0.0
        y_pred = all_preds[eval_mask].unsqueeze(1)
        y_true = labels[eval_mask]   # always use the passed-in labels, not data.y
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return round(result["acc"] * 100.0, 4)

    logits = model_forward(framework, model, x, edge_index, dgl_graph,
                           edge_weight=edge_weight)
    y_pred = logits[mask].argmax(dim=-1, keepdim=True)
    y_true = labels[mask]
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return round(result["acc"] * 100.0, 4)


def _infer_model_name(model) -> str:
    """
    Infer model name from instance, unwrapping torch.compile and link-encoder
    wrappers if present.

    Unwraps torch.compile's _orig_mod and PyGLinkEncoder / DGLLinkEncoder's
    .base attribute so that compiled link encoders resolve to their backbone
    GNN class name rather than raising ValueError.
    """
    # Unwrap torch.compile wrapper
    src = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Unwrap link-encoder wrapper (PyGLinkEncoder / DGLLinkEncoder store backbone in .base)
    if hasattr(src, "base"):
        src = src.base
        # Also unwrap a second compile layer that may wrap the base
        if hasattr(src, "_orig_mod"):
            src = src._orig_mod
    cls = type(src).__name__.lower()
    for key in ("graphsage", "gcn", "gat", "gin"):
        if key in cls:
            return key
    raise ValueError(f"Cannot infer model name from class '{cls}'")


@torch.no_grad()
def check_numerical_equivalence(
    framework, trained_model, x, edge_index, device,
    in_feats, hidden, num_classes, dgl_graph=None,
    atol=1e-3, rtol=1e-3,
    num_layers=2, dropout=0.5,
    gat_heads=8,
    train_loader=None,
    edge_weight=None,
) -> dict[str, Any]:
    """
    Verify that a compiled model produces numerically equivalent outputs to eager.

    When train_loader is provided (Tier 3 mini-batch), the check is performed on
    a single batch from the loader instead of the full graph to avoid OOM.
    """
    trained_model.eval()
    model_name = _infer_model_name(trained_model)
    eager_ref  = build_model(framework, model_name, in_feats, hidden, num_classes, device,
                             num_layers=num_layers, dropout=dropout, gat_heads=gat_heads)
    src_state  = (trained_model._orig_mod.state_dict()
                  if hasattr(trained_model, "_orig_mod")
                  else trained_model.state_dict())
    eager_ref.load_state_dict(src_state)
    eager_ref.eval()

    if train_loader is not None:
        # Use a single batch for the equivalence check to avoid full-graph OOM.
        batch = next(iter(train_loader))
        batch = batch.to(device)
        if framework.lower() == "dgl":
            _b_dgl = _to_dgl_graph(batch.edge_index, batch.x.shape[0], device)
            logits_eager    = eager_ref(_b_dgl, batch.x)
            logits_compiled = trained_model(_b_dgl, batch.x)
        else:
            # Pass edge_weight from the batch (set by GCNNorm for PyG GCN;
            # None for all other models). Without this, GCN runs without
            # normalisation, causing logit divergence vs the trained compiled model
            # which was trained with normalisation via batch.edge_weight.
            _b_ew = getattr(batch, "edge_weight", None)
            logits_eager    = model_forward(framework, eager_ref,
                                            batch.x, batch.edge_index, None,
                                            edge_weight=_b_ew)
            logits_compiled = model_forward(framework, trained_model,
                                            batch.x, batch.edge_index, None,
                                            edge_weight=_b_ew)
    else:
        logits_eager    = model_forward(framework, eager_ref,     x, edge_index, dgl_graph,
                                        edge_weight=edge_weight)
        if framework.lower() == "dgl":
            logits_compiled = trained_model(dgl_graph, x)
        else:
            logits_compiled = model_forward(framework, trained_model, x, edge_index, dgl_graph,
                                            edge_weight=edge_weight)

    passed   = bool(torch.allclose(logits_eager, logits_compiled, atol=atol, rtol=rtol))
    max_diff = float((logits_eager - logits_compiled).abs().max().item())
    return {
        "quality_check_passed": passed,
        "max_logit_abs_diff":   round(max_diff, 6),
        "logit_allclose_atol":  atol,
        "logit_allclose_rtol":  rtol,
    }


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------
# Training throughput is a SECONDARY metric per the experiment plan
# (primary metric is inference latency). The training benchmark measures:
#   - mean epoch time (steady-state, excluding first compiled epoch)
#   - first_measured_epoch_time_s (includes compilation overhead for compiled modes)
#   - train_compile_overhead_pct = (first - mean) / mean * 100
#   - throughput_train_nodes_per_s (or edges for link prediction)
#   - peak GPU memory during training
#
# Optimiser: Adam (Adaptive Moment Estimation)
#   Kingma & Ba 2015, "Adam: A Method for Stochastic Optimization"
#   https://arxiv.org/abs/1412.6980
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
#
# LR scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
#   Reduces learning rate when a metric stops improving.
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
#
# Loss: CrossEntropyLoss on training nodes only (train_mask).
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#
# This setup matches the OGB baseline training code for node property prediction:
#   ogbn-arxiv  : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
#   ogbn-products: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
#
# Note: 20 training epochs (--train-epochs 20) produces accuracy well below
# OGB leaderboard numbers. This is intentional -- the benchmark cares about
# timing, not accuracy. The accuracy check is a sanity test only.

def run_training_epochs(
    framework, model, x, edge_index, labels, train_mask,
    n_epochs=20, warmup_epochs=5, lr=0.01, dgl_graph=None,
    is_compiled=False, train_loader=None, edge_weight=None,
) -> dict[str, Any]:
    """
    Run training for warmup_epochs + n_epochs and return timing stats.

    Two paths:
      - train_loader is None  -> full-batch training on the entire graph.
      - train_loader provided -> mini-batch training via NeighborLoader.
        The compiled model object is reused across all batches; torch.compile
        caches compiled kernels by input shape, so the cache warms up after
        the first few unique subgraph shapes and all subsequent batches run
        at full compiled speed.

    For compiled models: epoch 0 of the measured epochs typically includes
    residual compilation overhead; this is captured separately as
    first_measured_epoch_time_s. Steady-state mean/std uses epochs 1..n_epochs.

    Calls torch.cuda.synchronize() after each epoch to ensure the GPU has
    finished executing before stopping the host timer.
    """
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    criterion  = nn.CrossEntropyLoss()
    # n_nodes is computed differently for mini-batch (accumulated per epoch).
    n_nodes    = 0 if train_loader is not None else int(train_mask.sum().item())

    # DEBUG: derive everything from the actual objects just constructed
    _dbg_sep(f"TRAINING BENCHMARK  |  {type(model).__name__}")
    _dbg("model",          type(model).__name__)
    _dbg("loss",           type(criterion).__name__)
    _dbg("optimizer",      f"{type(optimizer).__name__}  lr={lr}")
    _dbg("lr_scheduler",   f"{type(scheduler).__name__}  "
         f"factor={scheduler.factor}  patience={scheduler.patience}  min_lr={scheduler.min_lrs[0]}")
    _dbg("warmup_epochs",  str(warmup_epochs))
    _dbg("n_epochs",       str(n_epochs))
    _dbg("is_compiled",    str(is_compiled))
    if train_loader is not None:
        _dbg("loader",             type(train_loader).__name__)
        _dbg("  batch_size",       str(train_loader.batch_size))
        _dbg("  NOTE",             "epoch_times include neighbor sampling + data transfer time, "
             "not GPU compute only. Full end-to-end wall-clock.", indent=2)
        _num_nb = (
            getattr(train_loader, "num_neighbors", None)
            or getattr(getattr(train_loader, "sampler", None), "num_neighbors", None)
            or "(unknown)"
        )
        _dbg("  num_neighbors",    str(_num_nb))
        _dbg("  shuffle",          str(getattr(train_loader, "shuffle", "(unknown)")))
    else:
        total_nodes = x.shape[0] if x is not None else "?"
        _dbg("loader",                    "full-batch")
        _dbg("  total_nodes",             f"{total_nodes:,}" if isinstance(total_nodes, int) else total_nodes)
        _dbg("  train_nodes (mask=True)", f"{n_nodes:,}")
        _dbg("  non-train_nodes",         f"{(total_nodes - n_nodes):,}" if isinstance(total_nodes, int) else "?")
    _dbg_model(model, indent=1)

    warmup_times: list[float] = []
    epoch_times:  list[float] = []
    all_cpu_samples: list[float] = []
    warmup_cpu_samples: list[float] = []
    cpu_sampler = _CpuSampler(interval=0.1)
    _last_epoch_loss: float | None = None

    model.train()

    # Reset peak memory immediately before the first epoch so that
    # peak_gpu_memory_train_mb reflects forward+backward tensor allocations
    # only -- not torch.compile()'s Dynamo/Inductor compilation buffers
    # (which were already freed before we reach this point).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for ep in range(warmup_epochs + n_epochs):
        is_warmup_ep = ep < warmup_epochs
        ep_label     = (f'warmup {ep:>2d}' if is_warmup_ep
                        else f'epoch  {ep - warmup_epochs:>2d}')
        cpu_sampler.start()
        t0 = time.perf_counter()

        if train_loader is not None:
            # ------------------------------------------------------------------
            # Mini-batch path (Tier 3: ogbn-products, ogbl-citation2).
            # Each batch is a subgraph; only the batch_size seed nodes
            # contribute to the loss (standard NeighborLoader convention).
            # The compiled model is reused unchanged ? torch.compile handles
            # variable-shape subgraphs with a cudagraph-free compile mode.
            # ------------------------------------------------------------------
            epoch_seed_nodes = 0
            epoch_loss_sum   = torch.tensor(0.0, device=x.device)
            n_batches        = 0
            for batch in train_loader:
                batch = batch.to(x.device)
                optimizer.zero_grad()
                # DGL models expect (dgl_graph, x); PyG expects (x, edge_index).
                if framework.lower() == "dgl":
                    _b_dgl = _to_dgl_graph(batch.edge_index, batch.x.shape[0], batch.x.device)
                    out = model(_b_dgl, batch.x)
                else:
                    # edge_weight is propagated into each batch by NeighborLoader
                    # (it is a Data attribute set by GCNNorm at load time).
                    _ew = getattr(batch, "edge_weight", None)
                    out = model_forward(framework, model, batch.x, batch.edge_index,
                                       None, edge_weight=_ew)
                loss = criterion(
                    out[:batch.batch_size],
                    labels[batch.n_id[:batch.batch_size]].squeeze().to(x.device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_seed_nodes += batch.batch_size
                epoch_loss_sum   += loss.detach()
                n_batches        += 1
            # Step scheduler on mean epoch loss, not just the last batch loss.
            if n_batches > 0:
                scheduler.step(epoch_loss_sum / n_batches)
            # Accumulate seed-node count for throughput calculation
            if ep >= warmup_epochs:
                n_nodes += epoch_seed_nodes
            # DEBUG: mini-batch epoch summary (first 3 + last)
            if DBG and (ep < 3 or ep == warmup_epochs + n_epochs - 1):
                avg_loss = (epoch_loss_sum / n_batches).item() if n_batches else float('nan')
                _dbg(f'  [{ep_label}] mini-batch',
                     f'batches={n_batches}  seed_nodes={epoch_seed_nodes:,}  avg_loss={avg_loss:.4f}',
                     indent=1)
        else:
            # ------------------------------------------------------------------
            # Full-batch path (Tier 1 / Tier 2).
            # ------------------------------------------------------------------
            optimizer.zero_grad()
            out  = model_forward(framework, model, x, edge_index, dgl_graph,
                                 edge_weight=edge_weight)
            loss = criterion(out[train_mask], labels[train_mask].squeeze())
            loss.backward()
            # Gradient clipping stabilises GAT on small-training-set datasets
            # (e.g. PubMed public split: only 60 training nodes) where
            # large-head attention can produce exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            # DEBUG: full-batch epoch summary (first 3 + last)
            if DBG and (ep < 3 or ep == warmup_epochs + n_epochs - 1):
                _dbg(f'  [{ep_label}] full-batch',
                     f'loss={loss.item():.4f}  '
                     f'out.shape={list(out.shape)}  '
                     f'(all {x.shape[0]:,} nodes fwd; '
                     f'loss from {int(train_mask.sum()):,} train nodes)',
                     indent=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed  = time.perf_counter() - t0
        ep_cpu   = cpu_sampler.stop()

        if ep < warmup_epochs:
            warmup_times.append(elapsed)
            warmup_cpu_samples.extend(ep_cpu)
        else:
            epoch_times.append(elapsed)
            all_cpu_samples.extend(ep_cpu)
            # track loss for last measured epoch
            if train_loader is not None and n_batches > 0:
                _last_epoch_loss = (epoch_loss_sum / n_batches).item()
            elif train_loader is None:
                _last_epoch_loss = loss.item()
        # DEBUG: per-epoch timing
        if DBG and (ep < 3 or ep == warmup_epochs + n_epochs - 1):
            marker = '  <- first measured (may carry compile overhead)' if (not is_warmup_ep and ep == warmup_epochs) else ''
            _dbg(f'  [{ep_label}] elapsed', f'{elapsed*1000:.1f} ms{marker}', indent=1)

    # DEBUG: training summary
    _dbg_sep('TRAINING RESULTS')
    _dbg_epoch_times(epoch_times, warmup_times, label='Node classification training')
    if epoch_times:
        import numpy as _np2
        arr_dbg = _np2.array(epoch_times)
        _dbg('Mean epoch time',  f'{arr_dbg.mean()*1000:.1f} ms')
        _dbg('Std  epoch time',  f'{arr_dbg.std(ddof=1)*1000:.1f} ms' if len(arr_dbg)>1 else 'n/a (only 1 epoch)')
        if n_nodes > 0 and arr_dbg.mean() > 0:
            _dbg('Throughput (approx)', f'{n_nodes / arr_dbg.mean():,.0f} nodes/s')

    # For mini-batch: n_nodes is the total seed nodes across all measured epochs;
    # divide by n_epochs to get per-epoch average for throughput.
    if train_loader is not None and n_epochs > 0:
        n_nodes = n_nodes // n_epochs

    arr   = np.array(epoch_times)
    first = epoch_times[0] if epoch_times else None
    # mean_epoch_time_s uses ALL measured epochs for comparability between modes.
    mean  = float(np.mean(arr)) if len(arr) > 0 else 0.0
    std   = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    # --- Steady-state stats (epochs 1..N, excluding first compiled epoch) ---
    # For compiled modes epoch 0 may still carry residual Dynamo overhead from
    # the backward graph compilation. Steady-state excludes it so the table
    # shows the cost once the compiler has fully settled.
    if is_compiled and len(epoch_times) > 1:
        ss_arr            = arr[1:]
        ss_mean           = float(np.mean(ss_arr))
        ss_median         = float(np.median(ss_arr))
        ss_std            = float(np.std(ss_arr, ddof=1)) if len(ss_arr) > 1 else 0.0
        train_oh          = (first - ss_mean) / ss_mean * 100.0 if ss_mean > 0 else 0.0
    else:
        ss_mean = ss_median = ss_std = None
        train_oh = None

    # --- First warmup epoch cost (dominated by backward-graph compile) ---
    # For compiled modes: warmup_epoch[0] contains zero_grad + forward +
    # backward + optimizer.step. The backward graph is compiled lazily on the
    # first loss.backward() call here, making this epoch 10-100x slower than
    # steady state. This is NOT a pure backward-compile cost -- it includes the
    # full epoch. It is stored as train_backward_compile_s for historical
    # reasons but the correct interpretation is "first training epoch wall-clock,
    # which includes the backward-graph compilation spike".
    # NOTE: this is a SEPARATE compile from compile_time_s. compile_time_s
    # times the inference forward graph on infer_model (no gradients). This
    # times the training forward+backward graph on train_model (with gradients).
    # Do NOT add them as "total compile cost" without clarifying the use-case.
    train_backward_compile_s = (
        round(warmup_times[0], 6)
        if (is_compiled and warmup_times)
        else None
    )

    # DEBUG: surface the new timing fields clearly
    _dbg_sep("  TRAINING TIMING SUMMARY  (new v24 fields)")
    _dbg("warmup_epoch_0 (raw)",
         f"{warmup_times[0]*1000:.1f} ms" if warmup_times else "n/a")
    _dbg("train_backward_compile_s",
         f"{train_backward_compile_s:.3f} s  "
         "(backward-graph compile; compiled modes only)"
         if train_backward_compile_s is not None else "n/a (eager)")
    _dbg("steady_state_mean_epoch_s",
         f"{ss_mean*1000:.2f} ms  (epochs 1..N, compiler settled)"
         if ss_mean is not None else "n/a (eager or only 1 measured epoch)")
    _dbg("steady_state_median_epoch_s",
         f"{ss_median*1000:.2f} ms" if ss_median is not None else "n/a")
    _dbg("first_measured_epoch vs steady-state",
         f"{first*1000:.2f} ms  vs  {ss_mean*1000:.2f} ms  "
         f"(overhead = {train_oh:.1f} %)"
         if (first and ss_mean) else "n/a")

    peak_gpu_mem_train = None
    if torch.cuda.is_available():
        peak_gpu_mem_train = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    return {
        # --- per-epoch steady-state (primary training metric) ---
        "mean_epoch_time_s":                 mean,
        "final_train_loss":                  round(_last_epoch_loss, 6) if _last_epoch_loss is not None else None,
        "n_measured_epochs":                 len(epoch_times),
        "n_warmup_epochs":                   len(warmup_times),
        "std_epoch_time_s":                  std,
        "median_epoch_time_s":               float(np.median(arr)) if len(arr) > 0 else None,
        "first_measured_epoch_time_s":       first,
        "max_epoch_time_s":                  float(np.max(arr)) if len(arr) > 0 else None,
        # --- steady-state (epochs 1..N, after compiler settled) ---
        # Use these for the per-epoch speedup table; they are uncontaminated
        # by residual backward-graph compilation in epoch 0.
        "steady_state_mean_epoch_s":         ss_mean,
        "steady_state_median_epoch_s":       ss_median,
        "steady_state_std_epoch_s":          ss_std,
        # --- compile overhead on training side ---
        # train_backward_compile_s: wall-clock cost of compiling the backward
        # graph (warmup epoch 0 for compiled modes). Add this to compile_time_s
        # from the inference benchmark to get the total one-time compile tax.
        "train_backward_compile_s":          train_backward_compile_s,
        "train_compile_overhead_pct":        round(train_oh, 4) if train_oh is not None else None,
        # --- raw arrays (kept for plotting / reproducibility) ---
        "all_epoch_times_s":                 epoch_times,
        "all_warmup_times_s":                warmup_times,
        # --- throughput / memory / cpu ---
        "throughput_train_nodes_per_s":      float(n_nodes / mean) if mean > 0 else None,
        "cpu_utilization_train_pct_avg":     round(float(np.mean(all_cpu_samples)), 2)
                                             if all_cpu_samples else None,
        "cpu_utilization_train_pct_max":     round(float(np.max(all_cpu_samples)), 2)
                                             if all_cpu_samples else None,
        "all_cpu_samples_train":             list(all_cpu_samples),
        "all_warmup_cpu_samples_train":      list(warmup_cpu_samples),
        "peak_gpu_memory_train_mb":          peak_gpu_mem_train,
    }


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------
# benchmark_inference() measures per-mode inference latency, which is the
# PRIMARY metric for this thesis (experiment plan: "latency median+IQR").
#
# Compile modes (torch.compile mode= argument):
#   eager                      : No compilation. Plain PyTorch. Baseline.
#   default                    : TorchInductor default. Fuses ops, balanced
#                                compile time vs. speedup.
#   reduce-overhead            : Enables CUDA Graphs to eliminate kernel-launch
#                                overhead. Requires static input shapes.
#   max-autotune               : Exhaustive Triton kernel search (autotuning).
#                                Slowest to compile, best steady-state latency.
#   max-autotune-no-cudagraphs : max-autotune without CUDA Graph capture.
#                                Safer for variable-shape inputs.
# Source: https://pytorch.org/docs/stable/generated/torch.compile.html
# torch.compile internals (TorchInductor, Dynamo, AOTAutograd):
#   https://pytorch.org/docs/stable/torch.compiler.html
#
# Timing method: CUDA Events (torch.cuda.Event) rather than time.perf_counter.
# CUDA is asynchronous -- host-side timers measure kernel launch, not completion.
# torch.cuda.synchronize() + CUDA Events measure actual GPU execution time.
# This is best practice for GPU latency benchmarking.
# CUDA Events API: https://pytorch.org/docs/stable/generated/torch.cuda.Event.html
#
# CUDA kernel count uses torch.profiler.profile:
#   https://pytorch.org/docs/stable/profiler.html
#
# Protocol:
#   1. warmup=5 forward passes (discarded) to let the GPU reach steady state
#      and warm up any JIT compilation caches.
#   2. For compiled modes: the first forward pass triggers compilation;
#      this cost is recorded separately as compile_time_s.
#   3. repeats=30 timed forward passes. Median and IQR reported.
#      Median is more robust to outliers than mean for GPU benchmarking.

def benchmark_inference(
    framework, model, x, edge_index, mode="eager",
    repeats=30, warmup=5, n_nodes=0, dgl_graph=None,
    link_mode=False, cfg: dict | None = None, model_name: str = "",
    train_loader=None, edge_weight=None,
) -> dict[str, Any]:
    """
    Benchmark inference latency for one compile mode.

    For compiled modes (mode != 'eager'):
      - Calls torch.compile(model, mode=mode) to compile the model.
      - Times the first compiled forward pass as compile_time_s.
      - Then runs (warmup-1) + repeats forward passes.

    When train_loader is provided (Tier 3 datasets like ogbn-products that use
    neighbor sampling), inference is also run in mini-batch mode using a
    NeighborLoader over ALL nodes to avoid OOM from full-graph forward passes.
    Each timed "pass" iterates the full node set in batches; the reported
    latency is the median total time to score every node once.

    Returns a dict including median/IQR/mean latency, compile time,
    peak GPU memory, throughput, and the compiled model object.

    The compiled_model is returned so run_mode() can reuse it for the
    CUDA kernel count step (_get_cuda_kernel_count), which should measure
    kernel counts under compilation. Dynamo metrics (_get_dynamo_metrics)
    intentionally runs on the original uncompiled model to analyse what
    compilation will do to the graph.
    """
    # For Tier 3 datasets (ogbn-products etc.) a NeighborLoader was used for
    # training to avoid OOM.  Full-batch inference on 2.4M nodes x 128 features
    # requires ~18 GB just for the intermediate GAT tensors, which exceeds GPU
    # memory even on a 48 GB card when other state is resident.  Build a
    # dedicated inference loader that iterates ALL nodes in mini-batches.
    _use_sampled_infer = (train_loader is not None) and (not link_mode)
    if _use_sampled_infer:
        _num_nb = (
            getattr(train_loader, "num_neighbors", None)
            or getattr(getattr(train_loader, "sampler", None), "num_neighbors", None)
            or [-1]
        )
        from torch_geometric.loader import NeighborLoader as _NL
        _all_nodes = torch.arange(x.shape[0], device=x.device)
        _infer_loader = _NL(
            train_loader.data,
            num_neighbors=_num_nb,
            batch_size=train_loader.batch_size,
            input_nodes=_all_nodes,
            shuffle=False,
        )

    def _fwd():
        if link_mode:
            return link_model_forward(framework, model, x, edge_index, dgl_graph, edge_weight)
        # For DGL compiled models, dgl_graph must be passed as a direct argument
        # rather than closed over. When torch.compile() traces _fwd(), the closure
        # variable dgl_graph becomes None inside the compiled graph because Dynamo
        # does not capture external Python object references that are not tensors.
        # Calling model(dgl_graph, x) directly (bypassing model_forward) keeps the
        # dgl_graph reference alive and visible to the compiled function.
        if framework.lower() == "dgl":
            return model(dgl_graph, x)
        return model_forward(framework, model, x, edge_index, dgl_graph,
                             edge_weight=edge_weight)

    def _fwd_sampled_ms() -> float:
        """Run one full inference pass over all nodes in mini-batches; return ms."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in _infer_loader:
                batch = batch.to(x.device)
                if framework.lower() == "dgl":
                    # Build a DGL subgraph on-the-fly; call compiled model directly
                    # (same pattern as _fwd()) to keep dgl_graph as a live argument.
                    _b_dgl = _to_dgl_graph(batch.edge_index, batch.x.shape[0],
                                           batch.x.device)
                    model(_b_dgl, batch.x)
                else:
                    ew = getattr(batch, "edge_weight", None)
                    model_forward(framework, model, batch.x, batch.edge_index, None,
                                  edge_weight=ew)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def _timed_perf_fwd() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _fwd()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _timed_cuda_event_fwd() -> float:
        if _use_sampled_infer:
            return _fwd_sampled_ms()
        if not torch.cuda.is_available():
            return _timed_perf_fwd() * 1000.0
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        _fwd()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # ------------------------------------------------------------------
    # DEBUG: what are we about to benchmark?
    # ------------------------------------------------------------------
    _dbg_sep(f"INFERENCE BENCHMARK  |  mode='{mode}'  model={model_name or type(model).__name__}")
    _dbg("Framework", framework)
    _dbg("Compile mode", mode,
         indent=1)
    _dbg("Warmup passes (discarded)", warmup,
         indent=1)
    _dbg("Measured passes (repeats)", repeats,
         indent=1)
    _dbg("Node count used for throughput", f"{n_nodes:,}",
         indent=1)
    if _use_sampled_infer:
        _dbg("Inference mode", "mini-batch (sampled) -- full-batch would OOM", indent=1)
        _dbg("  batch_size", str(train_loader.batch_size), indent=1)
        _dbg("  num_neighbors", str(_num_nb), indent=1)
    _dbg_tensor("  x (node features)", x, indent=1)
    if edge_index is not None:
        _dbg_tensor("  edge_index", edge_index, indent=1)
        n_edges_dbg = edge_index.shape[1] if edge_index.dim() > 1 else 0
        _dbg("  edges -> avg degree", f"{n_edges_dbg/n_nodes:.2f}" if n_nodes else "n/a", indent=1)
    _dbg_model(model, indent=1)
    # ------------------------------------------------------------------

    compile_time_s        = 0.0
    latencies_ms:          list[float] = []
    warmup_latencies_ms:   list[float] = []
    infer_cpu_samples:     list[float] = []
    peak_memory_mb_infer   = None
    compilation_success    = None
    compiled_model         = model
    cpu_sampler = _CpuSampler(interval=0.1)

    def _failure_dict(error_msg: str) -> dict[str, Any]:
        return {
            "inference_latency_median_ms":       None,
            "inference_latency_iqr_ms":          None,
            "inference_latency_mean_ms":         None,
            "inference_latency_std_ms":          None,
            "inference_latency_min_ms":          None,
            "inference_latency_max_ms":          None,
            "all_latencies_ms":                  [],
            "all_warmup_latencies_ms":           [],
            "all_passes_ms":                     [],
            "pass_labels":                       [],
            "compile_and_first_fwd_ms":          None,
            "first_compiled_inference_ms":       None,
            "warmup_pass_0_ms":                  None,
            "warmup_pass_1_ms":                  None,
            "jit_overhead_ms":                   None,
            "jit_ramp_n_passes":                 None,
            "compile_time_s":                    compile_time_s,
            "compile_overhead_equivalent_calls": None,
            "compile_overhead_factor":           None,
            "peak_gpu_memory_inference_mb":      None,
            "throughput_inference_nodes_per_s":  None,
            "cpu_utilization_inference_pct_avg": None,
            "cpu_utilization_inference_pct_max": None,
            "all_cpu_samples_inference":         [],
            "compilation_success":               False,
            "inference_sampled":                 _use_sampled_infer,
            "compiled_model":                    None,
            "error":                             error_msg,
        }

    try:
        model.eval()

        # Declare timeline lists before the compiled/eager branch so that
        # compiled mode can append to them (passes A & B) without hitting
        # an UnboundLocalError from Python seeing the later annotation.
        all_passes_ms:  list[float] = []
        pass_labels:    list[str]   = []   # "compile", "warmup_N", "rep_N"

        if mode != "eager":
            compilation_success = True
            torch._dynamo.config.suppress_errors = True
            # Automatic dynamic shapes (dynamic=None, the default) are used.
            # Override with --dynamic true (always symbolic) or --dynamic false (always static).
            # DGL's SpMM/SDDMM kernels use internal CUDA stream operations that
            # are incompatible with CUDA Graph capture: reduce-overhead and
            # max-autotune both fail with "curr_block->next == nullptr". For DGL,
            # downgrade to the nearest safe equivalent that still autotuning where
            # possible but without CUDA Graph capture.
            _compile_mode = mode
            if framework.lower() == "dgl":
                if mode == "reduce-overhead":
                    _compile_mode = "default"
                    log.info("DGL + reduce-overhead: downgraded to 'default' "
                             "to avoid CUDA Graph / SpMM incompatibility.")
                elif mode == "max-autotune":
                    _compile_mode = "max-autotune-no-cudagraphs"
                    log.info("DGL + max-autotune: downgraded to "
                             "'max-autotune-no-cudagraphs' to avoid CUDA Graph "
                             "/ SpMM incompatibility.")
            # DEBUG: report what torch.compile was actually called with
            _dynamic = (cfg or {}).get("dynamic", None)   # None=auto, True, or False
            _dynamic_label = {None: "automatic (None)", True: "True", False: "False"}.get(
                _dynamic, str(_dynamic))
            _dbg_sep(f"  torch.compile()  |  mode='{_compile_mode}'  dynamic={_dynamic_label}")
            _cuda_graphs = _compile_mode in ("reduce-overhead", "max-autotune")
            _autotuning  = _compile_mode in ("max-autotune", "max-autotune-no-cudagraphs")
            _dbg("mode",           _compile_mode, indent=1)
            _dbg("dynamic",        _dynamic_label, indent=1)
            _dbg("suppress_errors",str(torch._dynamo.config.suppress_errors), indent=1)
            _dbg("cuda_graphs",    str(_cuda_graphs), indent=1)
            _dbg("autotuning",     str(_autotuning), indent=1)

            model          = torch.compile(model, mode=_compile_mode, dynamic=_dynamic)
            compiled_model = model

            # -------------------------------------------------------
            # Separate compile cost from inference latency.
            #
            # torch.compile() returns immediately; actual compilation
            # (Dynamo tracing + Inductor codegen) is triggered lazily
            # on the first forward call.
            #
            # We measure two consecutive passes:
            #   pass A: first forward  = compile + inference (fused)
            #   pass B: second forward = pure inference (kernels cached)
            #
            # Then:
            #   compile_and_first_fwd_s = wall-clock of pass A
            #   first_compiled_inference_ms = CUDA-event time of pass B
            #   pure_compile_time_s = compile_and_first_fwd_s
            #                        - first_compiled_inference_ms / 1000
            #
            # compile_time_s is stored as pure_compile_time_s.
            # first_compiled_inference_ms is stored separately and
            # prepended to warmup_latencies_ms as warmup pass 0,
            # so the warmup count stays symmetric with eager mode.
            # -------------------------------------------------------
            if _use_sampled_infer:
                # Trigger compilation with mini-batch shapes (not full-graph).
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t_a0 = time.perf_counter()
                with torch.no_grad():
                    for _b in _infer_loader:
                        _b = _b.to(x.device)
                        if framework.lower() == "dgl":
                            _b_dgl = _to_dgl_graph(_b.edge_index, _b.x.shape[0], _b.x.device)
                            model(_b_dgl, _b.x)
                        else:
                            _ew_b = getattr(_b, "edge_weight", None)
                            model_forward(framework, model, _b.x, _b.edge_index, None, edge_weight=_ew_b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _compile_and_fwd_s = time.perf_counter() - _t_a0
                # Second pass: pure inference latency (same mini-batch path)
                _first_infer_ms = _timed_cuda_event_fwd()
            else:
                _compile_and_fwd_s = _timed_perf_fwd()   # pass A: compile + inference
                _first_infer_ms    = _timed_cuda_event_fwd()  # pass B: pure inference

            compile_time_s = max(0.0, _compile_and_fwd_s - _first_infer_ms / 1000.0)

            # For CUDAGraphs modes (reduce-overhead, max-autotune) with NeighborLoader,
            # the first post-compile inference (pass B) includes CUDA Graph capture for
            # every unique mini-batch shape, which can take substantially longer than the
            # Dynamo+Inductor compilation step itself (pass A). When B > A the subtraction
            # is negative and compile_time_s collapses to 0.0, causing break-even analysis
            # to report no compilation cost. Fall back to compile_and_first_fwd_s as a
            # conservative lower bound so the break-even estimate remains meaningful.
            if compile_time_s == 0.0 and _compile_and_fwd_s > 0.0:
                compile_time_s = _compile_and_fwd_s
                log.warning(
                    "compile_time_s: pass B (%.1f ms) exceeded pass A (%.1f ms); "
                    "CUDAGraph capture dominated first inference. "
                    "Using compile_and_first_fwd_s=%.3f s as compile_time_s.",
                    _first_infer_ms, _compile_and_fwd_s * 1000.0, _compile_and_fwd_s)

            _dbg("Pass A (compile + first fwd)", f"{_compile_and_fwd_s:.3f} s", indent=1)
            _dbg("Pass B (first pure inference)", f"{_first_infer_ms:.3f} ms", indent=1)
            _dbg("compile_time_s (A - B)",
                 f"{compile_time_s:.3f} s  = {_compile_and_fwd_s:.3f} - {_first_infer_ms/1000:.3f}",
                 indent=1)

            # Pass B is the first post-compile inference. Treat it as warmup
            # pass 0 so warmup count is symmetric with eager (both have `warmup`
            # passes before the measured window).
            warmup_latencies_ms.append(_first_infer_ms)
            all_passes_ms.append(round(_compile_and_fwd_s * 1000.0, 3))
            pass_labels.append("compile+fwd_A")
            all_passes_ms.append(_first_infer_ms)
            pass_labels.append("first_pure_infer_B")

            remaining_warmup = max(0, warmup - 1)  # warmup-1 more passes after pass B
            log.info("Compile finished in %.3fs  (pure, excl. first fwd).", compile_time_s)
        else:
            _dbg_sep(f"  EAGER mode  (mode='{mode}')")
            _dbg("torch.compile", "not called")
            _dbg("cuda_graphs",   "False")
            _dbg("autotuning",    "False")
            _compile_and_fwd_s = 0.0  # no compilation in eager mode
            _first_infer_ms    = 0.0
            remaining_warmup = warmup

        # ---------------------------------------------------------------
        # FULL PASS TIMELINE  -- no time is discarded.
        #
        # all_passes_ms is a contiguous list of every forward-pass time
        # from cold start to the end of the measured window:
        #
        #   Eager:
        #     pass 0            = first forward  (ATen JIT + cold CUDA launch)
        #     passes 1..W-1     = cache/GPU ramp-up (legacy "warmup")
        #     passes W..W+R-1   = steady-state (legacy "measured")
        #
        #   Compiled (torch.compile):
        #     pass 0            = compile_time_s * 1000  (Dynamo + Inductor,
        #                         measured with perf_counter during compilation)
        #     passes 1..W       = post-compile cache warmup (Triton residuals,
        #                         CUDA Graph capture, L2 cache fill)
        #     passes W+1..W+R   = steady-state
        #
        # The split between warmup and measured is preserved in
        # all_warmup_latencies_ms / all_latencies_ms for backward compat.
        # Use all_passes_ms + pass_labels for the full trajectory plot.
        # ---------------------------------------------------------------
        # For compiled modes: passes A and B (compile+fwd and first pure infer)
        # are already in all_passes_ms/pass_labels from the compile block above.
        # For eager mode: all_passes_ms starts empty here.
        _dbg_sep(f"  FULL PASS TIMELINE  (warmup={remaining_warmup}  measured={repeats}  no discards)")
        _dbg("note",
             "compile=Dynamo+Inductor; warmup=cache ramp; rep=steady-state" if mode != "eager"
             else "pass_0=ATen JIT spike; warmup=cache ramp; rep=steady-state",
             indent=1)

        with torch.no_grad():
            for wi in range(remaining_warmup):
                wt = _timed_cuda_event_fwd()
                warmup_latencies_ms.append(wt)
                all_passes_ms.append(wt)
                pass_labels.append(f"warmup_{wi}")
                _dbg(f"  warmup pass {wi}", f"{wt:.3f} ms", indent=1)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # DEBUG: measured inference loop -- derive everything from the actual
            # model and input tensors rather than hardcoding assumptions.
            _dbg_sep(f"  MEASURED INFERENCE  ({repeats} passes)")

            # --- introspect model structure ---
            _src_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            _base      = getattr(_src_model, "base", _src_model)  # unwrap link encoder
            _base      = _base._orig_mod if hasattr(_base, "_orig_mod") else _base
            _layer_cls = type(_base).__name__
            _convs     = getattr(_base, "convs", None)
            _n_layers  = len(_convs) if _convs is not None else "?"
            _bns       = getattr(_base, "bns", [])
            _dropout   = getattr(_base, "dropout", "?")
            _act_names = []
            # detect activation from forward source if possible
            import inspect as _inspect
            try:
                _fwd_src = _inspect.getsource(_base.forward)
                for _act in ("elu", "relu", "leaky_relu", "selu", "tanh", "sigmoid"):
                    if f"F.{_act}" in _fwd_src or f"torch.{_act}" in _fwd_src:
                        _act_names.append(_act)
            except Exception:
                pass
            _act_str = ", ".join(_act_names) if _act_names else "unknown"

            # --- introspect input/output dims ---
            _n_feats    = x.shape[1] if x.dim() > 1 else 1
            _n_edges    = edge_index.shape[1] if edge_index is not None and edge_index.dim() > 1 else 0
            _avg_deg    = _n_edges / n_nodes if n_nodes > 0 else 0.0
            # derive output dim from last conv layer
            _last_conv  = _convs[-1] if _convs is not None and len(_convs) > 0 else None
            _out_dim    = "?"
            for _attr in ("out_channels", "out_feats", "_out_feats"):
                if _last_conv is not None and hasattr(_last_conv, _attr):
                    _out_dim = getattr(_last_conv, _attr)
                    break

            _dbg("What one inference call is",
                 f"model.forward(x, edge_index)  --  no loss, no backward, no argmax")
            _dbg("")
            _dbg("Model class",        _layer_cls)
            _dbg("  Num conv layers",  str(_n_layers))
            _dbg("  Conv layer type",  type(_convs[0]).__name__ if _convs else "?")
            _dbg("  BatchNorm layers", str(len(_bns)))
            _dbg("  Activation(s)",    _act_str)
            _dbg("  Dropout p",        str(_dropout))
            _dbg("")
            _dbg("Input  x",           f"[{n_nodes:,} nodes  x  {_n_feats} features]  -- every node in the graph")
            _dbg("Input  edge_index",  f"[2  x  {_n_edges:,} edges]  -- avg degree {_avg_deg:.1f}")
            _dbg("Output logits",      f"[{n_nodes:,} nodes  x  {_out_dim} "
                                        + ("embedding dims]  -- node embeddings (link-encoder)" if link_mode
                                           else "classes]  -- raw pre-softmax scores"))
            _dbg("")
            _dbg("scope",  "link-encoder" if link_mode else ("node-classifier, mini-batch (sampled)" if _use_sampled_infer else "node-classifier, full-batch"))
            _dbg("timer",  f"torch.cuda.Event  gpu={torch.cuda.is_available()}")

            cpu_sampler.start()
            for ri in range(repeats):
                t_ms = _timed_cuda_event_fwd()
                latencies_ms.append(t_ms)
                all_passes_ms.append(t_ms)
                pass_labels.append(f"rep_{ri}")
                if DBG and (ri < 3 or ri == repeats - 1):
                    tag = " <- last" if ri == repeats - 1 else ""
                    _dbg(f"  rep {ri:>2d}", f"{t_ms:.3f} ms{tag}", indent=1)
                elif DBG and ri == 3:
                    _dbg("  ...", "(intermediate reps omitted)", indent=1)
            infer_cpu_samples = cpu_sampler.stop()

            if torch.cuda.is_available():
                peak_memory_mb_infer = torch.cuda.max_memory_allocated() / (1024 ** 2)

        arr    = np.array(latencies_ms)
        median = float(np.median(arr))
        iqr    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        mean   = float(np.mean(arr))
        std    = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        if mode != "eager" and median > 0:
            steady_s                = median / 1000.0
            compile_overhead_factor = compile_time_s / steady_s
            # compile_overhead_equivalent_calls: how many inference calls the
            # compile cost equals (same quantity as break_even_runs before speedup).
            compile_overhead_equivalent_calls = int(round(compile_overhead_factor))
        else:
            compile_overhead_factor           = None
            compile_overhead_equivalent_calls = None

        throughput = float(n_nodes / (median / 1000.0)) if median > 0 and n_nodes > 0 else None

        # DEBUG: final inference summary
        _dbg_sep(f"  INFERENCE RESULTS  |  mode='{mode}'")
        _dbg_latencies(latencies_ms, label=f"mode='{mode}'")
        if throughput:
            _dbg("Throughput", f"{throughput:,.0f} nodes/s  "
                 "(= node_count / median_latency_s)", indent=1)
        if peak_memory_mb_infer:
            _dbg("Peak GPU memory", f"{peak_memory_mb_infer:.1f} MB", indent=1)
        if compile_overhead_factor is not None:
            _dbg("Compile overhead factor",
                 f"{compile_overhead_factor:.1f}x  "
                 f"(= compile_time_s / steady_state_latency_s)", indent=1)
            _dbg("Compile overhead equivalent calls",
                 f"{compile_overhead_equivalent_calls:,}  "
                 "(how many eager inference calls the compile cost equals; "
                 "see also break_even_runs which accounts for speedup savings)", indent=1)
        if infer_cpu_samples:
            _dbg("CPU utilisation (avg/max)",
                 f"{np.mean(infer_cpu_samples):.1f}% / {np.max(infer_cpu_samples):.1f}%"
                 "  (sampled at 100 ms intervals)", indent=1)

        log.info("mode='%s' | median=%.3fms | IQR=%.3fms | peak_gpu=%.1fMB",
                 mode, median, iqr, peak_memory_mb_infer or 0.0)

    except KeyboardInterrupt:
        raise

    except UnicodeDecodeError as exc:
        # DGL's C++ backend can surface non-UTF-8 bytes (e.g. a Windows-1252
        # em-dash) through Dynamo's tracing machinery when torch.compile()
        # triggers the first forward pass. The byte offset is logged so it can
        # be confirmed in run.log.
        error_msg = (
            f"UnicodeDecodeError during torch.compile(mode='{mode}'): "
            f"codec={exc.encoding!r} byte=0x{exc.object[exc.start]:02x} "
            f"position={exc.start}. "
            "Non-UTF-8 bytes in DGL C++ exception text propagated through "
            "Dynamo tracing. Model cannot be compiled with this combination."
        )
        log.error("mode='%s' raised UnicodeDecodeError: %s", mode, error_msg)
        return _failure_dict(error_msg)

    except Exception as exc:
        error_msg = str(exc)
        log.error("mode='%s' raised: %s", mode, error_msg)
        return _failure_dict(error_msg)

    return {
        "inference_latency_median_ms":       median,
        "inference_latency_iqr_ms":          iqr,
        "inference_latency_mean_ms":         mean,
        "inference_latency_std_ms":          std,
        "inference_latency_min_ms":          float(np.min(arr)),
        "inference_latency_max_ms":          float(np.max(arr)),
        "all_latencies_ms":                  latencies_ms,
        "all_warmup_latencies_ms":           warmup_latencies_ms,
        # --- Full cold-start-to-steady-state trajectory ---
        # all_passes_ms / pass_labels: every timed event in chronological order.
        #   Compiled: ["compile+fwd_A", "first_pure_infer_B",
        #              "warmup_0".."warmup_W-2", "rep_0".."rep_R-1"]
        #     compile+fwd_A       = perf_counter wall-clock of pass A (ms)
        #                           = compile overhead + 1 inference latency
        #     first_pure_infer_B  = CUDA-event time of pass B (ms)
        #                           = pure inference, kernels already compiled
        #     compile_time_s      = (A_ms - B_ms) / 1000  (stored separately)
        #   Eager:    ["warmup_0".."warmup_W-1", "rep_0".."rep_R-1"]
        # FOR TRAJECTORY PLOTS ONLY. Do NOT compute median/mean from this array.
        # Use inference_latency_median_ms for the steady-state latency metric.
        "all_passes_ms":                     all_passes_ms,
        "pass_labels":                       pass_labels,
        # Separate compile cost components (compiled modes only, else None):
        "compile_and_first_fwd_ms":  round(_compile_and_fwd_s * 1000.0, 3) if mode != "eager" else None,
        "first_compiled_inference_ms": round(_first_infer_ms, 3) if mode != "eager" else None,
        # --- JIT / cache warmup analysis ---
        # warmup_pass_0_ms: first warmup call. For eager mode this is the
        # dominant JIT spike (ATen op compilation + first CUDA kernel launch).
        # For compiled modes this follows compile_time_s and reflects any
        # residual Triton autotuning or CUDA Graph capture overhead.
        "warmup_pass_0_ms":                  warmup_latencies_ms[0] if warmup_latencies_ms else None,
        "warmup_pass_1_ms":                  warmup_latencies_ms[1] if len(warmup_latencies_ms) > 1 else None,
        # jit_overhead_ms: extra cost of the first warmup pass vs steady-state.
        # For eager: quantifies ATen kernel JIT + GPU cache cold-start cost.
        # For compiled: residual post-compilation overhead (e.g. CUDA Graph
        # capture, Triton autotuning residuals).
        # Formula: warmup_pass_0_ms - steady_state_median_ms
        # jit_overhead_source: what warmup_pass_0 actually contains.
        #   eager    -> ATen kernel JIT + cold CUDA launch overhead
        #   compiled -> post-Inductor residuals (Triton autotuning, CUDA Graph
        #               capture). compile_time_s already captured Dynamo+Inductor.
        "jit_overhead_source": (
            "post_inductor_residuals" if mode != "eager" else "aten_jit_cold_start"
        ) if warmup_latencies_ms else None,
        "jit_overhead_ms": (
            round(warmup_latencies_ms[0] - median, 3)
            if warmup_latencies_ms and median > 0
            else None
        ),
        # jit_ramp_n_passes: number of warmup passes until latency is within
        # 2% of steady-state median. 0 means pass 0 was already at steady state;
        # equals len(warmup_latencies_ms) if never converged within warmup.
        "jit_ramp_n_passes": (
            next(
                (i for i, t in enumerate(warmup_latencies_ms)
                 if median > 0 and abs(t - median) / median <= 0.02),
                len(warmup_latencies_ms)
            )
            if warmup_latencies_ms and median > 0
            else None
        ),
        "compile_time_s":                    compile_time_s,
        "compile_overhead_equivalent_calls": compile_overhead_equivalent_calls,
        "compile_overhead_factor":           round(compile_overhead_factor, 1)
                                             if compile_overhead_factor is not None else None,
        "peak_gpu_memory_inference_mb":      round(peak_memory_mb_infer, 2)
                                             if peak_memory_mb_infer is not None else None,
        "throughput_inference_nodes_per_s":  throughput,
        "cpu_utilization_inference_pct_avg": round(float(np.mean(infer_cpu_samples)), 2)
                                             if infer_cpu_samples else None,
        "cpu_utilization_inference_pct_max": round(float(np.max(infer_cpu_samples)), 2)
                                             if infer_cpu_samples else None,
        "all_cpu_samples_inference":         list(infer_cpu_samples),
        "compilation_success":               compilation_success,
        "inference_sampled":                 _use_sampled_infer,
        "compiled_model":                    compiled_model,
        "error":                             "",
    }


# ---------------------------------------------------------------------------
# Break-even analysis and usability scoring
# ---------------------------------------------------------------------------
# The break-even analysis answers the practical question:
#   "How many inference calls are needed before torch.compile pays off?"
#
# Formula:
#   saved_per_run = (eager_latency - compiled_latency) / 1000  [seconds]
#   break_even_n  = compile_time_s / saved_per_run
#
# This is a usability metric (Table 3 in the LaTeX output) that helps
# practitioners decide whether torch.compile is worthwhile for their
# deployment scenario (e.g., a short batch job vs. a long-running service).
#
# Note: compile_time_s is measured once per process (not per call). In
# production, a service that handles thousands of requests per day will
# easily amortise even a 5-minute max-autotune compilation.

def compute_breakeven(compile_time_s: float,
                      eager_median_ms: float | None,
                      compiled_median_ms: float | None) -> dict[str, Any]:
    """
    Compute the number of inference calls needed to recover compilation cost.

    Returns break_even_runs = compile_time_s / (saved_ms_per_run / 1000).
    Returns None if there is no speedup (compilation overhead, no benefit).
    """
    if compile_time_s is None or compile_time_s <= 0:
        return {"break_even_runs": None,
                "break_even_note": "N/A (no compilation cost)."}
    if eager_median_ms is None or compiled_median_ms is None:
        return {"break_even_runs": None,
                "break_even_note": "Insufficient data for break-even calculation."}
    saved_per_run_s = (eager_median_ms - compiled_median_ms) / 1000.0
    if saved_per_run_s <= 0:
        return {
            "break_even_runs": None,
            "break_even_note": (
                "No speedup -- compilation adds overhead. "
                "Disadvantageous for all run counts."),
        }
    bev = int(compile_time_s / saved_per_run_s)
    return {
        "break_even_runs": bev,
        "break_even_note": (
            f"Compile cost recovered after {bev:,} inference calls. "
            f"Advantageous for long-running services (>{bev:,} calls); "
            f"disadvantageous for short or one-off inference."),
    }


def _count_code_changes(mode: str, framework: str) -> int:
    """Count the number of source-level modifications needed to apply a compile mode."""
    if mode == "eager":
        return 0
    # 1 change: add torch.compile(model, mode=...) call
    changes = 1
    # 1 change: set TORCHINDUCTOR_CACHE_DIR or equivalent env var
    changes += 1
    if mode in ("reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        # 1 change: ensure static input shapes (required by TorchInductor autotuning
        # and CUDA Graphs for PyG; max-autotune-no-cudagraphs still requires static
        # shapes for the exhaustive kernel search).
        # Note: for DGL, reduce-overhead and max-autotune are silently downgraded
        # to default and max-autotune-no-cudagraphs respectively (DGL SpMM kernels
        # are incompatible with CUDA Graph capture). No extra user change required.
        changes += 1
    return changes


# ---------------------------------------------------------------------------
# Full per-mode benchmark (inference + training + quality checks)
# ---------------------------------------------------------------------------
# run_mode() is the inner function that runs inside each isolated subprocess.
# It orchestrates the complete measurement pipeline for one compile mode:
#
#   1. Inference benchmark (benchmark_inference) -- PRIMARY METRIC
#   2. Training benchmark (run_training_epochs)  -- SECONDARY METRIC
#   3. Accuracy evaluation (evaluate_node)
#   4. Numerical equivalence check (check_numerical_equivalence)
#   5. Dynamo graph-break metrics (_get_dynamo_metrics)
#   6. CUDA kernel count (_get_cuda_kernel_count)
#   7. Break-even analysis (compute_breakeven)
#
# Each mode is run in an isolated subprocess (_run_single_mode) to prevent
# CUDA state or compiled-kernel-cache leakage between modes. This is critical
# because torch.compile caches compiled kernels in process memory, and a
# prior compilation (e.g., eager mode) can affect subsequent mode timings.
#
# Subprocess isolation design: The parent process spawns itself with
# --single-mode, collects results as JSON from stdout, and aggregates them.

# Empty training result used when training fails (mode failure is documented
# in the "train_error" field of the result dict; inference metrics are still
# valid and will appear in the report).
_EMPTY_TRAIN_RES: dict = {
    "mean_epoch_time_s":                None,
    "final_train_loss":                 None,
    "n_measured_epochs":                0,
    "n_warmup_epochs":                  0,
    "std_epoch_time_s":                 None,
    "median_epoch_time_s":              None,
    "first_measured_epoch_time_s":      None,
    "max_epoch_time_s":                 None,
    "steady_state_mean_epoch_s":        None,
    "steady_state_median_epoch_s":      None,
    "steady_state_std_epoch_s":         None,
    "train_backward_compile_s":         None,
    "train_compile_overhead_pct":       None,
    "all_epoch_times_s":                [],
    "all_warmup_times_s":               [],
    "throughput_train_nodes_per_s":     None,
    "cpu_utilization_train_pct_avg":    None,
    "cpu_utilization_train_pct_max":    None,
    "all_cpu_samples_train":            [],
    "all_warmup_cpu_samples_train":     [],
    "throughput_train_edges_per_s":     None,
    "break_even_runs":                  None,
    "break_even_note":                  None,
    "peak_gpu_memory_train_mb":         None,
}


def run_mode(
    mode, framework, model_name, in_feats, hidden, num_classes,
    x, edge_index, labels, train_mask, val_mask, test_mask,
    device, eager_median_ms, cfg, node_evaluator, dgl_graph=None,
    train_loader=None, eager_train_epoch_s=None, edge_weight=None,
) -> dict[str, Any]:
    """
    Run the complete benchmark pipeline for one compile mode.

    edge_weight: pre-computed GCNNorm coefficients (data.edge_weight).
    Passed through to benchmark_inference, run_training_epochs, and evaluate_node
    for the PyG GCN full-batch path. None for all other models and for DGL.

    Builds fresh model instances for inference and training (separate instances
    to avoid state leakage from compilation affecting training measurements).
    Returns a flat result dict that is JSON-serialised and passed back to
    the parent orchestrator via stdout.

    eager_train_epoch_s: steady_state_mean_epoch_s from the eager run, used to
    compute train_speedup_vs_eager for compiled modes.
    """
    n_nodes      = x.shape[0]
    is_compiled  = mode != "eager"
    num_layers   = cfg.get("num_layers", 2)
    gat_heads    = cfg.get("gat_heads", 8)
    dataset      = cfg.get("dataset", "").lower()
    is_collab    = dataset == "ogbl-collab"
    dropout      = cfg.get("collab_dropout", 0.0) if is_collab else cfg.get("dropout", 0.5)
    lr           = cfg.get("collab_lr", 0.001)    if is_collab else cfg.get("lr", 0.01)

    # ------------------------------------------------------------------ DEBUG
    _dbg_sep(f"RUN MODE  |  mode='{mode}'  framework={framework}  model={model_name}  dataset={dataset}")
    _dbg("Pipeline steps (in order):", "")
    _dbg("  1", f"benchmark_inference()       repeats={cfg['repeats']}  warmup={cfg['warmup']}  [PRIMARY]")
    _dbg("  2", f"run_training_epochs()        epochs={cfg['train_epochs']}  warmup={cfg['train_warmup']}  lr={lr}")
    _dbg("  3", f"evaluate_node()  evaluator={type(node_evaluator).__name__ if node_evaluator else chr(110)+chr(47)+chr(97)}")
    _dbg("  4", f"check_numerical_equivalence()  skip_if_eager={not is_compiled}")
    _dbg("  5", "torch._dynamo.explain()")
    _dbg("  6", f"torch.profiler.profile()  gpu={torch.cuda.is_available()}")
    _dbg("  7", "compute_breakeven()")
    _dbg("")
    _dbg("mode",         mode)
    _dbg("is_compiled",  str(is_compiled), indent=1)
    _dbg("framework",    framework)
    _dbg("model_name",   model_name)
    _dbg("dataset",      dataset)
    _dbg("num_layers",   str(num_layers))
    _dbg("hidden dim",   str(hidden))
    _dbg("dropout",      str(dropout))
    _dbg("lr",           str(lr))
    _dbg("gat_heads",    str(gat_heads))
    _dbg("")
    _dbg("Graph dimensions:")
    _dbg("  Total nodes  (n_nodes)",      f"{n_nodes:,}")
    _dbg("  Node feature dim (in_feats)", str(in_feats))
    _dbg("  Num classes",                 str(num_classes))
    _dbg_graph_stats(x, edge_index, label="  Dataset graph")
    if train_mask is not None:
        _dbg_mask("  train_mask", train_mask)
        _dbg_mask("  val_mask",   val_mask)
        _dbg_mask("  test_mask",  test_mask)
    _dbg("eager_median_ms",
         f"{eager_median_ms:.3f} ms" if eager_median_ms else "None  (this IS the eager run)")
    # ------------------------------------------------------------------

    # --- Inference ---
    # Seed before building the inference model so that weight initialisation
    # is identical across all compile modes. Each mode runs in an isolated
    # subprocess, but explicit seeding here guards against future refactors
    # that might move model construction earlier in the call stack.
    set_seed(cfg.get("seed", 42))
    infer_model = build_model(framework, model_name, in_feats, hidden, num_classes,
                              device, num_layers=num_layers, dropout=dropout,
                              gat_heads=gat_heads)
    num_params  = _count_params(infer_model)
    infer_res   = benchmark_inference(
        framework=framework, model=infer_model, x=x, edge_index=edge_index,
        mode=mode, repeats=cfg["repeats"], warmup=cfg["warmup"],
        n_nodes=n_nodes, dgl_graph=dgl_graph, cfg=cfg, model_name=model_name,
        train_loader=train_loader, edge_weight=edge_weight)
    compiled_infer_model = infer_res.pop("compiled_model", None)

    median_ms = infer_res["inference_latency_median_ms"]
    speedup   = (1.0 if not is_compiled
                 else round(eager_median_ms / median_ms, 4)
                 if eager_median_ms and median_ms else None)

    # --- Training ---
    # Re-seed before building the training model so that weight initialisation
    # and all training-time dropout masks are identical across all compile modes.
    # torch._dynamo may advance the global RNG during inference tracing;
    # without this re-seed, train_model weights and dropout sequences
    # would differ between eager and compiled modes.
    _train_seed = cfg.get("seed", 42) + 1  # +1 to distinguish from inference seed
    set_seed(_train_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_model = build_model(framework, model_name, in_feats, hidden, num_classes,
                              device, num_layers=num_layers, dropout=dropout,
                              gat_heads=gat_heads)
    # Training uses the same compile mode as inference. If training fails
    # (e.g. CUDA-Graph backward crash), the error is stored in 'train_error';
    # inference metrics remain valid and will appear in the report.
    train_compile_mode = mode
    if framework.lower() == "dgl":
        if train_compile_mode == "reduce-overhead":
            train_compile_mode = "default"
        elif train_compile_mode == "max-autotune":
            train_compile_mode = "max-autotune-no-cudagraphs"
    train_error = ""
    if is_compiled:
        # suppress_errors=False ensures compilation failures surface as exceptions
        # and are recorded in train_error rather than falling back to eager silently.
        # A silent fallback would produce training results that appear valid but
        # reflect a mixed compiled/eager execution, making mode comparison unreliable.
        torch._dynamo.config.suppress_errors = False
        _dynamic = cfg.get("dynamic", None)
        train_model = torch.compile(train_model, mode=train_compile_mode, dynamic=_dynamic)
    try:
        train_res = run_training_epochs(
            framework=framework, model=train_model, x=x, edge_index=edge_index,
            labels=labels, train_mask=train_mask, n_epochs=cfg["train_epochs"],
            warmup_epochs=cfg["train_warmup"], lr=lr, dgl_graph=dgl_graph,
            is_compiled=is_compiled, train_loader=train_loader,
            edge_weight=edge_weight)
    except Exception as _train_exc:
        train_error = str(_train_exc)
        log.warning("run_mode mode='%s': training failed: %s", mode, _train_exc)
        train_res = dict(_EMPTY_TRAIN_RES)

    # --- Training speedup vs eager ---
    _ss_epoch = train_res.get("steady_state_mean_epoch_s") or train_res.get("mean_epoch_time_s")
    train_speedup = (
        1.0 if not is_compiled
        else round(eager_train_epoch_s / _ss_epoch, 4)
        if (eager_train_epoch_s and _ss_epoch and _ss_epoch > 0)
        else None
    )

    # --- Accuracy ---
    # Guard: if training failed (CUDA-Graph crash or other error), train_model
    # is in a corrupted state and cannot be used for forward passes.  Calling
    # evaluate_node() on it would crash again with the same CUDA runtime error.
    # Skip evaluation and leave test_acc/val_acc as None; inference metrics
    # (latency, compile_time, etc.) are still fully valid.
    if train_error:
        log.warning("run_mode mode='%s': skipping accuracy evaluation because "
                    "training failed (%s)", mode, train_error[:120])
        test_acc = None
        val_acc  = None
    else:
        train_model.eval()
        test_acc = evaluate_node(framework, train_model, x, edge_index, labels,
                                 test_mask, node_evaluator, dgl_graph,
                                 train_loader=train_loader, edge_weight=edge_weight)
        val_acc  = evaluate_node(framework, train_model, x, edge_index, labels,
                                 val_mask, node_evaluator, dgl_graph,
                                 train_loader=train_loader, edge_weight=edge_weight)

    # DEBUG: accuracy results
    _dbg_sep(f"  ACCURACY  |  mode='{mode}'")
    _dbg("evaluator",     type(node_evaluator).__name__ if node_evaluator else "n/a")
    _dbg("metric",        getattr(node_evaluator, "eval_metric", "acc") if node_evaluator else "n/a")
    _dbg("test_accuracy", f"{test_acc:.2f} %" if test_acc is not None else "n/a")
    _dbg("val_accuracy",  f"{val_acc:.2f} %"  if val_acc  is not None else "n/a")

    # --- Numerical equivalence ---
    if not is_compiled or train_error:
        quality_res = {
            "quality_check_passed": None,
            "max_logit_abs_diff":   None,
            "logit_allclose_atol":  1e-3,
            "logit_allclose_rtol":  1e-3,
        }
        reason = f"mode='{mode}' is eager -- no compiled model" if not is_compiled \
                 else f"training failed -- model state corrupted"
        _dbg("Numerical equivalence", f"skipped  ({reason})")
    else:
        quality_res = check_numerical_equivalence(
            framework=framework, trained_model=train_model, x=x,
            edge_index=edge_index, device=device, in_feats=in_feats,
            hidden=hidden, num_classes=num_classes, dgl_graph=dgl_graph,
            num_layers=num_layers, dropout=dropout, gat_heads=gat_heads,
            train_loader=train_loader, edge_weight=edge_weight)
        _dbg("torch.allclose",
             f"atol={quality_res['logit_allclose_atol']}  rtol={quality_res['logit_allclose_rtol']}")
        _dbg("  passed",         str(quality_res['quality_check_passed']))
        _dbg("  max_logit_diff", str(quality_res['max_logit_abs_diff']))

    # --- Dynamo metrics ---
    dynamo_res = _get_dynamo_metrics(framework, infer_model, x, edge_index, dgl_graph,
                                     train_loader=train_loader, edge_weight=edge_weight)

    # DEBUG: dynamo results -- all values read from dynamo_res dict
    _dbg_sep(f"  DYNAMO METRICS  |  mode='{mode}'")
    _dbg("tool",                   "torch._dynamo.explain")
    _dbg("graph_breaks",           str(dynamo_res.get("graph_breaks")))
    _dbg("graph_capture_rate_pct", str(dynamo_res.get("graph_capture_rate_pct")), indent=1)
    _dbg("  is_exact",             str(dynamo_res.get("graph_capture_rate_is_exact")), indent=1)
    _dbg("ops_per_graph",          str(dynamo_res.get("ops_per_graph")))
    _dbg("unsupported_op_count",   str(dynamo_res.get("unsupported_op_count")))
    _dbg("break_categories",       str(dynamo_res.get("break_categories")))

    # --- CUDA kernel count ---
    kernel_model = compiled_infer_model if compiled_infer_model is not None else infer_model
    cuda_kernels = _get_cuda_kernel_count(framework, kernel_model, x, edge_index, dgl_graph,
                                           train_loader=train_loader, edge_weight=edge_weight)
    _dbg("tool",              f"torch.profiler.profile  gpu={torch.cuda.is_available()}")
    _dbg("cuda_kernel_count", str(cuda_kernels))
    _dbg("  kernel model",    f"{'compiled' if compiled_infer_model is not None else 'uncompiled'} infer_model")

    # --- Break-even ---
    # Inference break-even: how many inference calls to amortise the
    # inference forward compile cost (compile_time_s).
    bev_res = compute_breakeven(
        compile_time_s     = infer_res.get("compile_time_s"),
        eager_median_ms    = eager_median_ms,
        compiled_median_ms = median_ms)
    # Training break-even: how many training epochs to amortise the
    # training backward compile cost (train_backward_compile_s, which is
    # the first warmup epoch and includes the backward-graph compilation spike).
    _eager_epoch_ms = (eager_train_epoch_s * 1000.0) if eager_train_epoch_s else None
    _ss_epoch_ms    = (train_res.get("steady_state_mean_epoch_s") or
                       train_res.get("mean_epoch_time_s") or 0.0) * 1000.0
    bev_train_res = compute_breakeven(
        compile_time_s     = train_res.get("train_backward_compile_s"),
        eager_median_ms    = _eager_epoch_ms,
        compiled_median_ms = _ss_epoch_ms if _ss_epoch_ms > 0 else None)

    # DEBUG: break-even -- show the actual numbers plugged into the formula
    _ct  = infer_res.get("compile_time_s") or 0.0
    _em  = eager_median_ms or 0.0
    _cm  = median_ms or 0.0
    _dbg_sep(f"  BREAK-EVEN  |  mode='{mode}'")
    _dbg("formula",            f"ct={_ct:.2f}s / ((em={_em:.3f}ms - cm={_cm:.3f}ms) / 1000)")
    _dbg("compile_time_s",     f"{_ct:.2f} s")
    _dbg("eager_median_ms",    f"{_em:.3f} ms" if eager_median_ms else "n/a (this is the eager run)")
    _dbg("compiled_median_ms", f"{_cm:.3f} ms" if median_ms else "n/a")
    _dbg("saved_per_call",     f"{(_em - _cm)/1000:.6f} s" if _em and _cm else "n/a")
    _dbg("break_even_runs",    str(bev_res.get("break_even_runs")))

    code_changes = _count_code_changes(mode, framework)

    # DEBUG: final mode summary -- all values from actual result dicts
    _bw   = train_res.get("train_backward_compile_s")
    _tax  = (_ct + (_bw or 0.0)) if _ct is not None else None
    _ss   = train_res.get("steady_state_mean_epoch_s")
    _n    = len(train_res.get("all_epoch_times_s") or [])
    _wc   = (
        round(_tax + _n * (_ss or train_res["mean_epoch_time_s"] or 0.0), 3)
        if (_tax is not None and _n > 0)
        else None
    )
    _dbg_sep(f"  MODE SUMMARY  |  mode='{mode}'")
    _dbg("inference_latency_median_ms",   f"{median_ms:.3f} ms" if median_ms else "n/a")
    _dbg("speedup_vs_eager",              f"{speedup:.4f}x"     if speedup    else "1.0x (baseline)")
    _dbg("compile_time_s (fwd graph)",    f"{_ct:.3f} s")
    _dbg("train_backward_compile_s",      f"{_bw:.3f} s" if _bw is not None else "n/a (eager)")
    _dbg("total_compile_tax_s",           f"{_tax:.3f} s" if _tax is not None else "n/a")
    _dbg("mean_epoch_time_ms (all)",      f"{train_res['mean_epoch_time_s']*1000:.2f} ms" if train_res['mean_epoch_time_s'] is not None else "n/a (training failed)")
    _dbg("steady_state_mean_epoch_ms",    f"{_ss*1000:.2f} ms" if _ss is not None else "n/a (eager)")
    _dbg("total_wallclock_N_epochs_s",    f"{_wc:.3f} s  (tax + {_n}xss_epoch)" if _wc is not None else "n/a")
    _dbg("test_accuracy",                 f"{test_acc:.2f} %"   if test_acc   else "n/a")
    _dbg("break_even_runs",               str(bev_res.get("break_even_runs")))
    _dbg("cuda_kernel_count",             str(cuda_kernels))
    _dbg("graph_capture_rate_pct",        str(dynamo_res.get("graph_capture_rate_pct")))

    return {
        # ------------------------------------------------------------------ #
        # INFERENCE LATENCY                                                   #
        # ------------------------------------------------------------------ #
        "inference_latency_median_ms":       infer_res["inference_latency_median_ms"],
        "inference_latency_iqr_ms":          infer_res["inference_latency_iqr_ms"],
        "inference_latency_mean_ms":         infer_res["inference_latency_mean_ms"],
        "inference_latency_std_ms":          infer_res["inference_latency_std_ms"],
        "inference_latency_min_ms":          infer_res["inference_latency_min_ms"],
        "inference_latency_max_ms":          infer_res["inference_latency_max_ms"],
        "all_latencies_ms":                  infer_res["all_latencies_ms"],
        "all_warmup_latencies_ms":           infer_res["all_warmup_latencies_ms"],
        "all_passes_ms":                     infer_res.get("all_passes_ms", []),
        "pass_labels":                       infer_res.get("pass_labels", []),
        "compile_and_first_fwd_ms":          infer_res.get("compile_and_first_fwd_ms"),
        "first_compiled_inference_ms":       infer_res.get("first_compiled_inference_ms"),
        # ------------------------------------------------------------------ #
        # TRAINING EPOCH TIMING                                               #
        # ------------------------------------------------------------------ #
        # mean/std/median over ALL measured epochs (epochs 0..N-1).
        # Use for overall comparability across modes.
        "mean_epoch_time_s":                 train_res["mean_epoch_time_s"],
        "std_epoch_time_s":                  train_res["std_epoch_time_s"],
        "median_epoch_time_s":               train_res["median_epoch_time_s"],
        "first_measured_epoch_time_s":       train_res["first_measured_epoch_time_s"],
        "max_epoch_time_s":                  train_res["max_epoch_time_s"],
        # Steady-state: epochs 1..N only (backward-graph compile settled).
        # Use these for per-epoch speedup tables in the thesis -- they reflect
        # the cost a practitioner pays for every epoch after the first.
        "steady_state_mean_epoch_s":         train_res["steady_state_mean_epoch_s"],
        "steady_state_median_epoch_s":       train_res["steady_state_median_epoch_s"],
        "steady_state_std_epoch_s":          train_res["steady_state_std_epoch_s"],
        "all_epoch_times_s":                 train_res["all_epoch_times_s"],
        "all_warmup_times_s":                train_res["all_warmup_times_s"],
        # ------------------------------------------------------------------ #
        # COMPILE COSTS  (one-time, paid before steady-state begins)         #
        # ------------------------------------------------------------------ #
        # compile_time_s: forward-graph + Dynamo/Inductor tracing cost.
        #   Measured on the first inference call (outside the training loop).
        "compile_time_s":                    infer_res["compile_time_s"],
        # --- JIT / cache warmup analysis (from benchmark_inference) ---
        "warmup_pass_0_ms":                  infer_res.get("warmup_pass_0_ms"),
        "warmup_pass_1_ms":                  infer_res.get("warmup_pass_1_ms"),
        "jit_overhead_ms":                   infer_res.get("jit_overhead_ms"),
        "jit_ramp_n_passes":                 infer_res.get("jit_ramp_n_passes"),
        # train_backward_compile_s: backward-graph compile cost.
        #   Measured as warmup_epoch[0] for compiled modes; None for eager.
        #   This is the cost that appears as the giant spike in training warmup.
        "train_backward_compile_s":          train_res["train_backward_compile_s"],
        # total_compile_tax_s: inference forward compile + first training epoch.
        #   compile_time_s          = Dynamo+Inductor forward graph (on infer_model,
        #                             no gradients, separate from training graph).
        #   train_backward_compile_s = first training warmup epoch (dominated by
        #                             backward-graph compile on train_model).
        #   These are two independent compiles of different graphs.
        #   Use this sum only if you do BOTH inference and training in sequence.
        #   For inference-only: use compile_time_s.
        #   For training-only:  use train_backward_compile_s.
        "total_compile_tax_s": (
            round(
                (infer_res["compile_time_s"] or 0.0)
                + (train_res["train_backward_compile_s"] or 0.0),
                4,
            )
            if (infer_res["compile_time_s"] is not None
                or train_res["train_backward_compile_s"] is not None)
            else None
        ),
        # train_compile_overhead_pct: how much slower epoch 0 is vs steady-state.
        "train_compile_overhead_pct":        train_res["train_compile_overhead_pct"],
        "compile_overhead_equivalent_calls": infer_res["compile_overhead_equivalent_calls"],
        "compile_overhead_factor":           infer_res["compile_overhead_factor"],
        "train_compile_mode":                train_compile_mode if is_compiled else "eager",
        # Field kept for backward compatibility with compare_baseline xlsx.
        "train_mode_differs_from_infer":     (train_compile_mode != mode) if is_compiled else False,
        # ------------------------------------------------------------------ #
        # PRACTITIONER WALL-CLOCK ESTIMATE                                   #
        # ------------------------------------------------------------------ #
        # total_train_wallclock_N_epochs_s: estimated wall-clock to run
        #   exactly N training epochs from a compiled model cold start.
        #   Formula: train_backward_compile_s + N * steady_state_mean_epoch_s
        #   This is what a training-only practitioner should compare across modes.
        #   Does NOT include compile_time_s (inference forward compile is separate).
        "total_train_wallclock_N_epochs_s": (
            round(
                (train_res["train_backward_compile_s"] or 0.0)
                + len(train_res["all_epoch_times_s"])
                * (train_res["steady_state_mean_epoch_s"]
                   or train_res["mean_epoch_time_s"]
                   or 0.0),
                4,
            )
            if train_res["all_epoch_times_s"]
            else None
        ),
        # total_wallclock_N_epochs_s: retained for backward compatibility.
        # Prefer total_train_wallclock_N_epochs_s for training-only cost, or
        # add compile_time_s to that field for the combined inference+training estimate.
        "total_wallclock_N_epochs_s": (
            round(
                (train_res["train_backward_compile_s"] or 0.0)
                + len(train_res["all_epoch_times_s"])
                * (train_res["steady_state_mean_epoch_s"]
                   or train_res["mean_epoch_time_s"]
                   or 0.0),
                4,
            )
            if train_res["all_epoch_times_s"]
            else None
        ),
        # ------------------------------------------------------------------ #
        # SPEEDUP / BREAK-EVEN                                                #
        # ------------------------------------------------------------------ #
        "speedup_vs_eager":                  speedup,
        "train_speedup_vs_eager":            train_speedup,
        # break_even_runs: inference calls needed to amortise compile_time_s.
        "break_even_runs":                   bev_res["break_even_runs"],
        "break_even_note":                   bev_res["break_even_note"],
        # break_even_train_epochs: training epochs needed to amortise
        # train_backward_compile_s (the backward-graph compile spike).
        "break_even_train_epochs":           bev_train_res["break_even_runs"],
        "break_even_train_note":             bev_train_res["break_even_note"],
        # ------------------------------------------------------------------ #
        # THROUGHPUT / MEMORY / CPU                                           #
        # ------------------------------------------------------------------ #
        "throughput_inference_nodes_per_s":  infer_res["throughput_inference_nodes_per_s"],
        "throughput_train_nodes_per_s":      train_res["throughput_train_nodes_per_s"],
        "peak_gpu_memory_inference_mb":      infer_res["peak_gpu_memory_inference_mb"],
        "peak_gpu_memory_train_mb":          train_res["peak_gpu_memory_train_mb"],
        "cpu_utilization_inference_pct_avg": infer_res["cpu_utilization_inference_pct_avg"],
        "cpu_utilization_inference_pct_max": infer_res["cpu_utilization_inference_pct_max"],
        "all_cpu_samples_inference":         infer_res["all_cpu_samples_inference"],
        "cpu_utilization_train_pct_avg":     train_res["cpu_utilization_train_pct_avg"],
        "cpu_utilization_train_pct_max":     train_res["cpu_utilization_train_pct_max"],
        "all_cpu_samples_train":             train_res["all_cpu_samples_train"],
        "all_warmup_cpu_samples_train":      train_res["all_warmup_cpu_samples_train"],
        # ------------------------------------------------------------------ #
        # ACCURACY / NUMERICAL CHECKS                                         #
        # ------------------------------------------------------------------ #
        "test_accuracy_pct":                 test_acc,
        "val_accuracy_pct":                  val_acc,
        "quality_check_passed":              quality_res["quality_check_passed"],
        "max_logit_abs_diff":                quality_res["max_logit_abs_diff"],
        "logit_allclose_atol":               quality_res["logit_allclose_atol"],
        "logit_allclose_rtol":               quality_res["logit_allclose_rtol"],
        # ------------------------------------------------------------------ #
        # DYNAMO / USABILITY                                                  #
        # ------------------------------------------------------------------ #
        "compilation_success":               infer_res["compilation_success"],
        "graph_capture_rate_pct":            dynamo_res["graph_capture_rate_pct"],
        "graph_capture_rate_is_exact":       dynamo_res["graph_capture_rate_is_exact"],
        "graph_breaks":                      dynamo_res["graph_breaks"],
        "unsupported_op_count":              dynamo_res["unsupported_op_count"],
        "ops_per_graph":                     dynamo_res["ops_per_graph"],
        "break_categories":                  dynamo_res["break_categories"],
        # cuda_kernel_count: CUDA kernels fired during one INFERENCE forward pass
        # (compiled_infer_model in eval mode, no_grad). NOT training kernels.
        # Compiled modes typically show fewer unique kernels due to op fusion.
        "cuda_kernel_count":                 cuda_kernels,
        "num_params":                        num_params,
        "required_code_changes":             code_changes,
        "inference_sampled":                 infer_res.get("inference_sampled", False),
        # final_train_loss / epoch counts forwarded from training
        "final_train_loss":                  train_res.get("final_train_loss"),
        "n_measured_epochs":                 train_res.get("n_measured_epochs"),
        "n_warmup_epochs":                   train_res.get("n_warmup_epochs"),
        "error":                             infer_res["error"],
        # Non-empty when training failed; inference metrics remain valid.
        "train_error":                       train_error,
    }


# ---------------------------------------------------------------------------
# Link prediction evaluation and training
# ---------------------------------------------------------------------------
# Phase 4 (ogbl-collab, ogbl-citation2) uses link prediction, which differs
# from node classification in three key ways:
#
#   1. Loss: Binary cross-entropy on (positive, negative) edge pairs.
#      Negative edges are sampled uniformly (not provided by OGB split for train).
#
#   2. Metric:
#      ogbl-collab  -> Hits@50
#        "we rank each true collaboration among a set of 100,000 randomly-
#        sampled negative collaborations, and count the ratio of positive
#        edges that are ranked at K-place or above (Hits@K). K=50."
#        Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
#        Leaderboard: https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-collab
#      ogbl-citation2 -> MRR (Mean Reciprocal Rank over 1000 negatives)
#        "for each source paper, two of its references are randomly dropped...
#        rank the missing two references higher than 1,000 negative candidates"
#        Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2
#        Leaderboard: https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-citation2
#
#   3. Evaluation: The OGB LinkEvaluator requires y_pred_pos and y_pred_neg.
#      Source: https://ogb.stanford.edu/docs/linkprop/#eval
#      Snippet from that page:
#        from ogb.linkproppred import Evaluator
#        evaluator   = Evaluator(name=d_name)
#        result_dict = evaluator.eval(input_dict)
#      For ogbl-collab, OGB provides 100K fixed negatives for val/test.
#      For training, we sample random negatives (not provided by OGB).
#
# run_link_training_epochs() implements Phase 4i:
#   "Add link pred training loop: BCE loss + Hits@K evaluator"

class _LinkEvalResult:
    """Thin container normalising the link metric name across datasets."""
    __slots__ = ("metric_name", "value")
    def __init__(self, metric_name: str, value: float):
        self.metric_name = metric_name
        self.value       = value


def _make_link_evaluator(dataset_name: str):
    """
    Return a dataset-aware link evaluator wrapper around ogb.linkproppred.Evaluator.

    Instantiation pattern from OGB docs:
      https://ogb.stanford.edu/docs/linkprop/#eval
        from ogb.linkproppred import Evaluator
        evaluator   = Evaluator(name=d_name)
        result_dict = evaluator.eval(input_dict)

    ogbl-collab  -> K=50; result_dict["hits@50"] in [0,1]
      Dataset description: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
    ogbl-citation2 -> result_dict["mrr_list"] (per-edge MRR Tensor)
      Dataset description: https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2
    """
    n = dataset_name.lower()
    base_evaluator = LinkEvaluator(name=n)

    if n == "ogbl-collab":
        class _CollabEvaluator:
            neg_mode = "shared_pool"
            K        = 50
            _ev      = base_evaluator

            def eval(self, input_dict: dict) -> _LinkEvalResult:
                self._ev.K = self.K
                result = self._ev.eval(input_dict)
                return _LinkEvalResult(
                    metric_name=f"hits@{self.K}",
                    value=result[f"hits@{self.K}"],
                )
        return _CollabEvaluator()

    if n == "ogbl-citation2":
        class _Citation2Evaluator:
            neg_mode = "per_edge"
            _ev      = base_evaluator

            def eval(self, input_dict: dict) -> _LinkEvalResult:
                result = self._ev.eval(input_dict)
                mrr = result["mrr_list"].mean().item()
                return _LinkEvalResult(metric_name="mrr", value=mrr)
        return _Citation2Evaluator()

    raise ValueError(f"No link evaluator defined for dataset '{dataset_name}'.")


@torch.no_grad()
def evaluate_link(framework, model, x, edge_index, pos_edges, neg_edges,
                  evaluator, dgl_graph=None, edge_weight=None) -> _LinkEvalResult:
    """
    Evaluate link prediction using the OGB metric appropriate for the dataset.

    Calls evaluator.eval({"y_pred_pos": pos_scores, "y_pred_neg": neg_scores}).
    Input format from OGB docs: https://ogb.stanford.edu/docs/linkprop/#eval
      y_pred_pos : Tensor[E_pos]  -- scores for true edges
      y_pred_neg : Tensor[E_neg]  -- scores for negative edges
        ogbl-collab    : shared pool -- one flat [E_neg] tensor for all positives
        ogbl-citation2 : per-edge   -- shape [E_pos, 1000], one row per positive
    """
    model.eval()
    z          = link_model_forward(framework, model, x, edge_index, dgl_graph, edge_weight)
    pos_scores = model.decode(z, pos_edges.t())

    neg_mode = getattr(evaluator, "neg_mode", "shared_pool")
    if neg_mode == "per_edge":
        n_pos      = pos_edges.shape[0]
        neg_scores = model.decode(z, neg_edges.t())
        neg_scores = neg_scores.view(n_pos, -1)
    else:
        neg_scores = model.decode(z, neg_edges.t())

    return evaluator.eval({"y_pred_pos": pos_scores, "y_pred_neg": neg_scores})


# ---------------------------------------------------------------------------
# Link prediction training
# ---------------------------------------------------------------------------
# Loss: BCEWithLogitsLoss on concatenated positive + negative edge scores.
#   Combines sigmoid activation and binary cross-entropy in a single step
#   for numerical stability (log-sum-exp trick).
#   PyTorch API: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# Optimiser and scheduler: same Adam + ReduceLROnPlateau as node classification.
#   Adam    : https://arxiv.org/abs/1412.6980
#   PyTorch : https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

def run_link_training_epochs(
    framework, model, x, edge_index, train_edges, neg_train_edges,
    n_epochs=20, warmup_epochs=5, lr=0.01, dgl_graph=None,
    is_compiled=False, edge_weight=None,
) -> dict[str, Any]:
    """Link-prediction training loop."""
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    criterion  = nn.BCEWithLogitsLoss()
    n_edges    = train_edges.shape[0]

    warmup_times: list[float] = []
    epoch_times:  list[float] = []
    all_cpu_samples: list[float] = []
    warmup_cpu_samples: list[float] = []
    cpu_sampler = _CpuSampler(interval=0.1)
    _last_epoch_loss: float | None = None

    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for ep in range(warmup_epochs + n_epochs):
        cpu_sampler.start()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        z          = link_model_forward(framework, model, x, edge_index, dgl_graph, edge_weight)
        pos_scores = model.decode(z, train_edges.t())
        neg_scores = model.decode(z, neg_train_edges.t())
        labels_lp  = torch.cat([torch.ones(pos_scores.size(0)),
                                 torch.zeros(neg_scores.size(0))]).to(x.device)
        loss = criterion(torch.cat([pos_scores, neg_scores]), labels_lp)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed  = time.perf_counter() - t0
        ep_cpu   = cpu_sampler.stop()

        if ep < warmup_epochs:
            warmup_times.append(elapsed)
            warmup_cpu_samples.extend(ep_cpu)
        else:
            epoch_times.append(elapsed)
            all_cpu_samples.extend(ep_cpu)
            _last_epoch_loss = loss.item()

    arr       = np.array(epoch_times)
    first     = epoch_times[0] if epoch_times else None
    mean      = float(np.mean(arr)) if len(arr) > 0 else 0.0
    std       = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if is_compiled and len(epoch_times) > 1:
        ss_arr    = arr[1:]
        ss_mean   = float(np.mean(ss_arr))
        ss_median = float(np.median(ss_arr))
        ss_std    = float(np.std(ss_arr, ddof=1)) if len(ss_arr) > 1 else 0.0
        train_oh  = (first - ss_mean) / ss_mean * 100.0 if ss_mean > 0 else 0.0
    else:
        ss_mean = ss_median = ss_std = None
        train_oh = None

    train_backward_compile_s = (
        round(warmup_times[0], 6) if (is_compiled and warmup_times) else None
    )

    peak_gpu_mem_train = None
    if torch.cuda.is_available():
        peak_gpu_mem_train = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    return {
        "mean_epoch_time_s":             mean,
        "std_epoch_time_s":              std,
        "median_epoch_time_s":           float(np.median(arr)) if len(arr) > 0 else None,
        "first_measured_epoch_time_s":   first,
        "max_epoch_time_s":              float(np.max(arr)) if len(arr) > 0 else None,
        "steady_state_mean_epoch_s":     ss_mean,
        "steady_state_median_epoch_s":   ss_median,
        "steady_state_std_epoch_s":      ss_std,
        "train_backward_compile_s":      train_backward_compile_s,
        "train_compile_overhead_pct":    round(train_oh, 4) if train_oh is not None else None,
        "final_train_loss":              round(_last_epoch_loss, 6) if _last_epoch_loss is not None else None,
        "n_measured_epochs":             len(epoch_times),
        "n_warmup_epochs":               len(warmup_times),
        "all_epoch_times_s":             epoch_times,
        "all_warmup_times_s":            warmup_times,
        "throughput_train_edges_per_s":  float(n_edges / mean) if mean > 0 else None,
        "cpu_utilization_train_pct_avg": round(float(np.mean(all_cpu_samples)), 2)
                                         if all_cpu_samples else None,
        "cpu_utilization_train_pct_max": round(float(np.max(all_cpu_samples)), 2)
                                         if all_cpu_samples else None,
        "all_cpu_samples_train":         list(all_cpu_samples),
        "all_warmup_cpu_samples_train":  list(warmup_cpu_samples),
        "peak_gpu_memory_train_mb":      peak_gpu_mem_train,
    }


# ---------------------------------------------------------------------------
# Full per-mode benchmark for link prediction
# ---------------------------------------------------------------------------

def run_mode_link(
    mode, framework, model_name, in_feats, hidden,
    x, edge_index, train_edges, val_edges, test_edges,
    device, eager_median_ms, cfg, link_evaluator, dgl_graph=None,
    val_neg_edges=None, test_neg_edges=None, edge_weight=None,
) -> dict[str, Any]:
    """
    Run the complete benchmark for one compile mode (link prediction).

    val_neg_edges / test_neg_edges: official fixed negatives from load_link_dataset.
      If provided (ogbl-collab), they are used for val/test evaluation so that
      Hits@50 is comparable to the OGB leaderboard.
      If None (ogbl-citation2, or training negatives), random sampling is used.
    """
    n_nodes     = x.shape[0]
    is_compiled = mode != "eager"
    num_layers  = cfg.get("num_layers", 2)
    gat_heads   = cfg.get("gat_heads", 8)
    dataset     = cfg.get("dataset", "").lower()
    is_collab   = dataset == "ogbl-collab"
    dropout     = cfg.get("collab_dropout", 0.0) if is_collab else cfg.get("dropout", 0.5)
    lr          = cfg.get("collab_lr", 0.001)    if is_collab else cfg.get("lr", 0.01)

    neg_mode        = getattr(link_evaluator, "neg_mode", "shared_pool")
    num_neg_per_pos = 1000

    def sample_neg_1d(n: int) -> torch.Tensor:
        src = torch.randint(0, n_nodes, (n,), device=device)
        dst = torch.randint(0, n_nodes, (n,), device=device)
        return torch.stack([src, dst], dim=1)

    def sample_neg_per_edge(pos_edges: torch.Tensor) -> torch.Tensor:
        n_pos = pos_edges.shape[0]
        src   = pos_edges[:, 0].repeat_interleave(num_neg_per_pos)
        dst   = torch.randint(0, n_nodes, (n_pos * num_neg_per_pos,), device=device)
        return torch.stack([src, dst], dim=1)

    # Training always uses randomly sampled negatives (no official train negatives
    # are provided by any OGB link dataset).
    if neg_mode == "per_edge":
        neg_train = sample_neg_per_edge(train_edges)
    else:
        neg_train = sample_neg_1d(train_edges.shape[0])

    # For val/test, use official fixed negatives when available
    # (ogbl-collab provides 100K fixed negatives per split in edge_neg).
    # Random sampling makes Hits@50 incomparable to the leaderboard.
    if val_neg_edges is not None:
        neg_val  = val_neg_edges
        neg_test = test_neg_edges
        log.info(
            "run_mode_link mode='%s': using official OGB fixed negatives "
            "(val=%d, test=%d) -- results are leaderboard-comparable.",
            mode, neg_val.shape[0], neg_test.shape[0]
        )
    else:
        # ogbl-citation2: no fixed negatives; sample per-edge at eval time.
        if neg_mode == "per_edge":
            neg_val  = sample_neg_per_edge(val_edges)
            neg_test = sample_neg_per_edge(test_edges)
        else:
            neg_val  = sample_neg_1d(val_edges.shape[0])
            neg_test = sample_neg_1d(test_edges.shape[0])
        log.info(
            "run_mode_link mode='%s': no official negatives available -- "
            "sampled random negatives (val=%d, test=%d).  "
            "Results are NOT directly comparable to the OGB leaderboard.",
            mode, neg_val.shape[0], neg_test.shape[0]
        )

    # Seed before inference model build so that weight initialisation is identical
    # across all compile modes (see run_mode for the same pattern and rationale).
    set_seed(cfg.get("seed", 42))
    infer_model = build_link_model(framework, model_name, in_feats, hidden, device,
                                   num_layers=num_layers,
                                   dropout=dropout, gat_heads=gat_heads)
    num_params  = _count_params(infer_model)
    infer_res   = benchmark_inference(
        framework=framework, model=infer_model, x=x, edge_index=edge_index,
        mode=mode, repeats=cfg["repeats"], warmup=cfg["warmup"],
        n_nodes=n_nodes, dgl_graph=dgl_graph, link_mode=True, cfg=cfg, model_name=model_name,
        edge_weight=edge_weight)
    compiled_infer_model = infer_res.pop("compiled_model", None)

    median_ms = infer_res["inference_latency_median_ms"]
    speedup   = (1.0 if not is_compiled
                 else round(eager_median_ms / median_ms, 4)
                 if eager_median_ms and median_ms else None)

    # Re-seed before building train_model so that weight initialisation and
    # dropout masks are identical across all compile modes.
    set_seed(cfg.get("seed", 42) + 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_model = build_link_model(framework, model_name, in_feats, hidden, device,
                                   num_layers=num_layers,
                                   dropout=dropout, gat_heads=gat_heads)
    train_compile_mode = mode
    if framework.lower() == "dgl":
        if train_compile_mode == "reduce-overhead":
            train_compile_mode = "default"
        elif train_compile_mode == "max-autotune":
            train_compile_mode = "max-autotune-no-cudagraphs"
    train_error = ""
    if is_compiled:
        # suppress_errors=False ensures compilation failures surface as exceptions
        # and are recorded in train_error rather than falling back to eager silently.
        torch._dynamo.config.suppress_errors = False
        _dynamic = cfg.get("dynamic", None)
        train_model = torch.compile(train_model, mode=train_compile_mode, dynamic=_dynamic)
    try:
        train_res = run_link_training_epochs(
            framework=framework, model=train_model, x=x, edge_index=edge_index,
            train_edges=train_edges, neg_train_edges=neg_train,
            n_epochs=cfg["train_epochs"], warmup_epochs=cfg["train_warmup"],
            lr=lr, dgl_graph=dgl_graph, is_compiled=is_compiled,
            edge_weight=edge_weight)
    except Exception as _train_exc:
        train_error = str(_train_exc)
        log.warning("run_mode_link mode='%s': training failed: %s", mode, _train_exc)
        train_res = dict(_EMPTY_TRAIN_RES)

    # Guard: if training failed, train_model is corrupted (CUDA Graph state).
    # evaluate_link would crash with the same error. Skip and record None.
    if train_error:
        log.warning("run_mode_link mode='%s': skipping link evaluation "
                    "because training failed (%s)", mode, train_error[:120])
        val_metric_pct  = None
        test_metric_pct = None
        metric_name     = "n/a"
    else:
        train_model.eval()
        val_result  = evaluate_link(framework, train_model, x, edge_index,
                                    val_edges, neg_val, link_evaluator, dgl_graph=dgl_graph,
                                    edge_weight=edge_weight)
        test_result = evaluate_link(framework, train_model, x, edge_index,
                                    test_edges, neg_test, link_evaluator, dgl_graph=dgl_graph,
                                    edge_weight=edge_weight)
        metric_name     = test_result.metric_name
        val_metric_pct  = round(val_result.value  * 100.0, 4)
        test_metric_pct = round(test_result.value * 100.0, 4)

    if not is_compiled:
        quality_res = {
            "quality_check_passed": None,
            "max_logit_abs_diff":   None,
            "logit_allclose_atol":  1e-3,
            "logit_allclose_rtol":  1e-3,
        }
    else:
        ref_model = build_link_model(framework, model_name, in_feats, hidden, device,
                                     num_layers=num_layers,
                                     dropout=dropout, gat_heads=gat_heads)
        src_state = (train_model._orig_mod.state_dict()
                     if hasattr(train_model, "_orig_mod") else train_model.state_dict())
        ref_model.load_state_dict(src_state)
        ref_model.eval()
        with torch.no_grad():
            z_eager    = link_model_forward(framework, ref_model,   x, edge_index, dgl_graph, edge_weight)
            z_compiled = link_model_forward(framework, train_model, x, edge_index, dgl_graph, edge_weight)
        passed   = bool(torch.allclose(z_eager, z_compiled, atol=1e-3, rtol=1e-3))
        max_diff = float((z_eager - z_compiled).abs().max().item())
        quality_res = {
            "quality_check_passed": passed,
            "max_logit_abs_diff":   round(max_diff, 6),
            "logit_allclose_atol":  1e-3,
            "logit_allclose_rtol":  1e-3,
        }

    dynamo_res   = _get_dynamo_metrics(framework, infer_model, x, edge_index, dgl_graph,
                                        edge_weight=edge_weight)
    kernel_model = compiled_infer_model if compiled_infer_model is not None else infer_model
    cuda_kernels = _get_cuda_kernel_count(framework, kernel_model, x, edge_index, dgl_graph,
                                           edge_weight=edge_weight)
    bev_res      = compute_breakeven(infer_res.get("compile_time_s"), eager_median_ms, median_ms)
    # run_mode_link does not receive eager_train_epoch_s so training break-even
    # cannot be computed here; fields are set to None for consistency with run_mode.
    bev_train_res = {"break_even_runs": None, "break_even_note": None}
    code_changes = _count_code_changes(mode, framework)

    return {
        "inference_latency_median_ms":       infer_res["inference_latency_median_ms"],
        "inference_latency_iqr_ms":          infer_res["inference_latency_iqr_ms"],
        "inference_latency_mean_ms":         infer_res["inference_latency_mean_ms"],
        "inference_latency_std_ms":          infer_res["inference_latency_std_ms"],
        "inference_latency_min_ms":          infer_res["inference_latency_min_ms"],
        "inference_latency_max_ms":          infer_res["inference_latency_max_ms"],
        "all_latencies_ms":                  infer_res["all_latencies_ms"],
        "all_warmup_latencies_ms":           infer_res["all_warmup_latencies_ms"],
        "mean_epoch_time_s":                 train_res["mean_epoch_time_s"],
        "std_epoch_time_s":                  train_res["std_epoch_time_s"],
        "median_epoch_time_s":               train_res.get("median_epoch_time_s"),
        "first_measured_epoch_time_s":       train_res["first_measured_epoch_time_s"],
        "max_epoch_time_s":                  train_res["max_epoch_time_s"],
        "steady_state_mean_epoch_s":         train_res.get("steady_state_mean_epoch_s"),
        "steady_state_median_epoch_s":       train_res.get("steady_state_median_epoch_s"),
        "steady_state_std_epoch_s":          train_res.get("steady_state_std_epoch_s"),
        "train_backward_compile_s":          train_res.get("train_backward_compile_s"),
        "final_train_loss":                  train_res.get("final_train_loss"),
        "n_measured_epochs":                 train_res.get("n_measured_epochs"),
        "n_warmup_epochs":                   train_res.get("n_warmup_epochs"),
        "all_epoch_times_s":                 train_res["all_epoch_times_s"],
        "all_warmup_times_s":                train_res["all_warmup_times_s"],
        "all_passes_ms":                     infer_res.get("all_passes_ms", []),
        "pass_labels":                       infer_res.get("pass_labels", []),
        "warmup_pass_0_ms":                  infer_res.get("warmup_pass_0_ms"),
        "warmup_pass_1_ms":                  infer_res.get("warmup_pass_1_ms"),
        "jit_overhead_ms":                   infer_res.get("jit_overhead_ms"),
        "jit_overhead_source":               infer_res.get("jit_overhead_source"),
        "jit_ramp_n_passes":                 infer_res.get("jit_ramp_n_passes"),
        "inference_sampled":                 infer_res.get("inference_sampled", False),
        "throughput_inference_nodes_per_s":  infer_res["throughput_inference_nodes_per_s"],
        "throughput_train_edges_per_s":      train_res["throughput_train_edges_per_s"],
        "speedup_vs_eager":                  speedup,
        "compile_time_s":                    infer_res["compile_time_s"],
        "total_compile_tax_s": (
            round((infer_res["compile_time_s"] or 0.0)
                  + (train_res.get("train_backward_compile_s") or 0.0), 4)
            if infer_res["compile_time_s"] is not None else None
        ),
        "total_train_wallclock_N_epochs_s": (
            round((train_res.get("train_backward_compile_s") or 0.0)
                  + len(train_res["all_epoch_times_s"])
                  * (train_res.get("steady_state_mean_epoch_s")
                     or train_res["mean_epoch_time_s"] or 0.0), 4)
            if train_res["all_epoch_times_s"] else None
        ),
        "compile_overhead_equivalent_calls": infer_res["compile_overhead_equivalent_calls"],
        "compile_overhead_factor":           infer_res["compile_overhead_factor"],
        "train_compile_overhead_pct":        train_res["train_compile_overhead_pct"],
        "train_compile_mode":                train_compile_mode if is_compiled else "eager",
        "train_mode_differs_from_infer":     (train_compile_mode != mode) if is_compiled else False,
        "peak_gpu_memory_inference_mb":      infer_res["peak_gpu_memory_inference_mb"],
        "peak_gpu_memory_train_mb":          train_res["peak_gpu_memory_train_mb"],
        "cpu_utilization_inference_pct_avg": infer_res["cpu_utilization_inference_pct_avg"],
        "cpu_utilization_inference_pct_max": infer_res["cpu_utilization_inference_pct_max"],
        "all_cpu_samples_inference":         infer_res["all_cpu_samples_inference"],
        "cpu_utilization_train_pct_avg":     train_res["cpu_utilization_train_pct_avg"],
        "cpu_utilization_train_pct_max":     train_res["cpu_utilization_train_pct_max"],
        "all_cpu_samples_train":             train_res["all_cpu_samples_train"],
        "all_warmup_cpu_samples_train":      train_res["all_warmup_cpu_samples_train"],
        "link_metric_name":                  metric_name,
        "test_link_metric_pct":              test_metric_pct,
        "val_link_metric_pct":               val_metric_pct,
        "quality_check_passed":              quality_res["quality_check_passed"],
        "max_logit_abs_diff":                quality_res["max_logit_abs_diff"],
        "logit_allclose_atol":               quality_res["logit_allclose_atol"],
        "logit_allclose_rtol":               quality_res["logit_allclose_rtol"],
        "compilation_success":               infer_res["compilation_success"],
        "graph_capture_rate_pct":            dynamo_res["graph_capture_rate_pct"],
        "graph_capture_rate_is_exact":       dynamo_res["graph_capture_rate_is_exact"],
        "graph_breaks":                      dynamo_res["graph_breaks"],
        "unsupported_op_count":              dynamo_res["unsupported_op_count"],
        "ops_per_graph":                     dynamo_res["ops_per_graph"],
        "break_categories":                  dynamo_res["break_categories"],
        "cuda_kernel_count":                 cuda_kernels,
        "num_params":                        num_params,
        "break_even_runs":                   bev_res["break_even_runs"],
        "break_even_note":                   bev_res["break_even_note"],
        "break_even_train_epochs":           bev_train_res["break_even_runs"],
        "break_even_train_note":             bev_train_res["break_even_note"],
        "required_code_changes":             code_changes,
        "error":                             infer_res["error"],
        "train_error":                       train_error,
    }



# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_dataset(name, root, device, framework="pyg", model_name="graphsage"):
    """
    Load a node-classification dataset (Tier 1 or Tier 2) and return tensors.

    model_name is used to select the correct per-model graph transform applied
    once at load time:
      - "gcn"  : ToUndirected() + GCNNorm(add_self_loops=True) -> data.edge_weight
      - others : AddSelfLoops() (no-op for SAGE/GIN; required for GAT)
    The transform result is reflected in the returned edge_index (and edge_weight
    for GCN), so GCNConv(normalize=False, add_self_loops=False) and
    GATConv(add_self_loops=False) receive correctly pre-augmented inputs.

    Dispatches to:
      - Planetoid (Cora/CiteSeer/PubMed, Tier 1, Phase 1 & 2):
          Uses torch_geometric.datasets.Planetoid (public train/val/test split).
          Planetoid split introduced by Yang et al. 2016:
            "Revisiting Semi-Supervised Learning with Graph Embeddings"
            https://arxiv.org/abs/1603.08861  (20 labelled nodes/class for train)
          PyG API: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html

      - OGB node datasets (ogbn-arxiv / ogbn-products, Tier 2/3, Phase 3-5):
          PyG path -- PygNodePropPredDataset:
            Source : https://ogb.stanford.edu/docs/nodeprop/#pyg
            Snippet:
              dataset   = PygNodePropPredDataset(name=d_name)
              split_idx = dataset.get_idx_split()
              train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
              graph = dataset[0]   # pyg graph object; graph.y is shape (num_nodes, num_tasks)
          DGL path -- DglNodePropPredDataset:
            Source : https://ogb.stanford.edu/docs/nodeprop/#dgl
            Snippet:
              dataset      = DglNodePropPredDataset(name=d_name)
              split_idx    = dataset.get_idx_split()
              graph, label = dataset[0]   # graph: dgl graph object
          get_idx_split() returns dict with 'train', 'valid', 'test' node indices
          (torch tensors of shape (num_nodes,)).
          ogbn-arxiv uses a temporal split (train=papers <=2017, val=2018, test>=2019);
          this is why val_acc > test_acc is expected.
            Source: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv

    Returns: (x, edge_index, labels, train_mask, val_mask, test_mask,
               in_feats, num_classes, evaluator, dgl_graph, edge_weight)
    dgl_graph is None for PyG path; built later by _to_dgl_graph() for Planetoid+DGL.
    edge_weight holds GCNNorm coefficients for PyG GCN; None for all other paths.
    """
    n   = name.lower()
    fw  = framework.lower()
    mn  = model_name.lower()

    # DEBUG: explain what is being loaded
    _dbg_sep(f"LOAD DATASET  |  '{name}'  framework={framework}")
    _dbg("Dataset tier", str(DATASET_TIER.get(n, "?")),
         indent=1)
    _dbg("root",    root, indent=1)

    if DATASET_TIER.get(n, 2) == 3:
        raise ValueError(
            f"Dataset '{name}' is Tier 3 (large). Full-batch loading will OOM. "
            "Pass --use-sampling to use NeighborLoader instead."
        )

    if n in PLANETOID_DATASETS:
        ds   = Planetoid(root=root, name=_PLANETOID_NAME_MAP[n])
        data = ds[0].to(device)
        # Apply per-model transform at load time. Planetoid graphs are already
        # undirected so ToUndirected() is a no-op here but included for
        # consistency with the OGB path.
        if fw == "pyg":
            if mn == "gcn":
                data = ToUndirected()(data)
                data = GCNNorm(add_self_loops=True)(data)
            else:  # GAT, SAGE, GIN
                data = AddSelfLoops()(data)
        evaluator = _make_node_evaluator(n)
        _dbg("Loader",    f"{type(ds).__name__}  name={ds.name}  root={root}")
        _dbg("split_type",  "Planetoid public  (train={}/val={}/test={})".format(
                          int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum())))
        _dbg("Loaded!",   f"nodes={data.x.shape[0]:,}  feats={data.x.shape[1]}  "
             f"classes={ds.num_classes}  edges={data.edge_index.shape[1]:,}")
        _dbg_mask("  train_mask", data.train_mask)
        _dbg_mask("  val_mask",   data.val_mask)
        _dbg_mask("  test_mask",  data.test_mask)
        _ew = getattr(data, "edge_weight", None)  # set by GCNNorm; None for other models
        return (data.x, data.edge_index, data.y,
                data.train_mask, data.val_mask, data.test_mask,
                data.x.shape[1], ds.num_classes, evaluator, None, _ew)

    if n in OGB_NODE_DATASETS:
        if fw == "dgl":
            ds          = DglNodePropPredDataset(name=n, root=root)
            split_idx   = ds.get_idx_split()
            g, labels   = ds[0]
            g           = dgl.add_self_loop(g)
            g           = g.to(device)
            x           = g.ndata["feat"].to(device)
            labels      = labels.to(device)
            src, dst    = g.edges()
            edge_index  = torch.stack([src, dst], dim=0)
        else:
            ds          = PygNodePropPredDataset(name=n, root=root)
            split_idx   = ds.get_idx_split()
            data        = ds[0]
            # Apply per-model transform at load time.
            # GCN: ToUndirected() first (ogbn-arxiv is directed; symmetric GCN norm
            # requires an undirected graph), then GCNNorm writes the normalisation
            # coefficients to data.edge_weight.
            # GAT: AddSelfLoops() needed for attention over self.
            # SAGE, GIN: AddSelfLoops() is a no-op guard.
            if mn == "gcn":
                data = ToUndirected()(data)
                data = GCNNorm(add_self_loops=True)(data)
            else:
                data = AddSelfLoops()(data)
            data        = data.to(device)
            x           = data.x
            labels      = data.y
            edge_index  = data.edge_index
            g           = None

        nn_  = x.shape[0]
        tm   = torch.zeros(nn_, dtype=torch.bool, device=device)
        vm   = torch.zeros(nn_, dtype=torch.bool, device=device)
        tsm  = torch.zeros(nn_, dtype=torch.bool, device=device)
        tm[split_idx["train"]]  = True
        vm[split_idx["valid"]]  = True
        tsm[split_idx["test"]]  = True
        evaluator   = _make_node_evaluator(n)
        num_classes = ds.num_classes
        _dbg("Loader",    f"{type(ds).__name__}  name={n}  framework={fw}  root={root}")
        _dbg("Split keys", str(list(split_idx.keys())))
        _dbg("Loaded!",   f"nodes={x.shape[0]:,}  feats={x.shape[1]}  "
             f"classes={num_classes}  edges={edge_index.shape[1]:,}")
        _dbg_mask("  train_mask", tm)
        _dbg_mask("  val_mask",   vm)
        _dbg_mask("  test_mask",  tsm)
        # edge_weight: GCNNorm coefficients for PyG GCN; None for DGL/other models
        _ew = getattr(data, "edge_weight", None) if fw == "pyg" else None
        return (x, edge_index, labels, tm, vm, tsm, x.shape[1], num_classes, evaluator, g, _ew)

    if n in OGB_LINK_DATASETS:
        raise ValueError(
            f"'{name}' is a link-prediction dataset. Use load_link_dataset().")
    raise ValueError(
        f"Unknown dataset '{name}'. "
        f"Node classification: {sorted(OGB_NODE_DATASETS | PLANETOID_DATASETS)}. "
        f"Link prediction: {sorted(OGB_LINK_DATASETS)}.")


def load_link_dataset(name, root, device, framework="pyg", model_name="graphsage"):
    """
    Load an OGB link-prediction dataset (Tier 2/3) and return edge split tensors.

    Used in Phase 4 (ogbl-collab) and optionally Phase 5 (ogbl-citation2).

    OGB link datasets provide official edge splits via get_edge_split().
    PyG path -- PygLinkPropPredDataset:
      Source : https://ogb.stanford.edu/docs/linkprop/#pyg
    DGL path -- DglLinkPropPredDataset:
      Source : https://ogb.stanford.edu/docs/linkprop/#dgl

    ogbl-collab specifics:
      - Undirected co-authorship network (MAG subset)
      - Metric: Hits@50 -- rank each true collaboration among 100K fixed negatives,
        count fraction that rank in top 50.
      - The OGB split provides 100K FIXED negatives in split_edge["valid"]["edge_neg"]
        and split_edge["test"]["edge_neg"].  These MUST be used for results to be
        comparable to the OGB leaderboard.
      - Year-based split: train edges up to 2017, val=2018, test=2019
      - Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab

    ogbl-citation2 specifics:
      - Directed citation network (~2.5M nodes, 128-dim word2vec node features)
      - Metric: MRR (Mean Reciprocal Rank, 1000 per-edge negatives)
      - No fixed negatives in the split; negatives are sampled per positive at
        eval time.
      - Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2

    Returns official fixed negatives for val/test when the dataset provides them
    (ogbl-collab), or None when it does not (ogbl-citation2, training).
    Callers use official negatives when present and fall back to random sampling
    only when needed.

    Returns: (x, edge_index, train_edges, val_edges, test_edges,
               in_feats, evaluator, dgl_graph,
               val_neg_edges, test_neg_edges, edge_weight)
      val_neg_edges  : official fixed negatives for val  (or None -> sample randomly)
      test_neg_edges : official fixed negatives for test (or None -> sample randomly)
      edge_weight    : GCNNorm coefficients for PyG GCN (or None for all other models/DGL)
    """
    n  = name.lower()
    fw = framework.lower()
    mn = model_name.lower()
    if n not in OGB_LINK_DATASETS:
        raise ValueError(f"'{name}' not supported. Choose from {sorted(OGB_LINK_DATASETS)}.")

    if fw == "dgl":
        ds         = DglLinkPropPredDataset(name=n, root=root)
        split_idx  = ds.get_edge_split()
        g          = ds[0]
        g          = dgl.add_self_loop(g)
        g          = g.to(device)
        x          = g.ndata["feat"].to(device)
        src, dst   = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        _ew        = None
    else:
        ds         = PygLinkPropPredDataset(name=n, root=root)
        split_idx  = ds.get_edge_split()
        data       = ds[0]
        if mn == "gcn":
            # ogbl-collab stores edge_weight as torch.long (year attribute).
            # GCNNorm calls deg.pow_(-0.5) which requires float; cast first.
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                data.edge_weight = data.edge_weight.float()
            data = GCNNorm(add_self_loops=True)(data)
        data       = data.to(device)
        x          = data.x
        edge_index = data.edge_index
        # Only GCN uses edge_weight (GCNNorm coefficients). For all other models
        # the raw OGB data may have an edge_weight attribute (e.g. ogbl-collab
        # stores the collaboration year as edge_weight), which must not be passed
        # to models that don't accept it.
        _ew        = getattr(data, "edge_weight", None) if mn == "gcn" else None
        g          = None

    def _get_edges(split):
        s = split_idx[split]
        if "edge" in s:
            return s["edge"].to(device)
        src = s["source_node"].to(device)
        dst = s["target_node"].to(device)
        return torch.stack([src, dst], dim=1)

    def _get_official_neg_edges(split):
        """Return official fixed negatives if provided by the dataset, else None."""
        s = split_idx[split]
        if "edge_neg" in s:
            # ogbl-collab: shape [100000, 2] -- 100K fixed negatives for the split
            return s["edge_neg"].to(device)
        return None  # ogbl-citation2 has no fixed negatives; caller must sample

    val_neg_edges  = _get_official_neg_edges("valid")
    test_neg_edges = _get_official_neg_edges("test")

    if val_neg_edges is not None:
        log.info(
            "load_link_dataset('%s'): using official OGB fixed negatives for val "
            "(%d) and test (%d).  Results are leaderboard-comparable.",
            n, val_neg_edges.shape[0], test_neg_edges.shape[0]
        )
    else:
        log.info(
            "load_link_dataset('%s'): no fixed negatives in split -- "
            "run_mode_link will sample per-edge negatives at eval time.", n
        )

    evaluator = _make_link_evaluator(n)
    return (x, edge_index,
            _get_edges("train"),
            _get_edges("valid"),
            _get_edges("test"),
            x.shape[1],
            evaluator,
            g,
            val_neg_edges,
            test_neg_edges,
            _ew)


def load_dataset_sampled(name: str, root: str, device, framework: str = "pyg",
                         num_neighbors: tuple = (15, 10),
                         batch_size: int = 1024,
                         model_name: str = "graphsage"):
    """
    Mini-batch NeighborLoader for Tier 3 datasets (ogbn-products, Phase 5).

    model_name is used to select the correct per-model graph transform applied
    before constructing NeighborLoader. PyG propagates all Data attributes
    (including edge_weight added by GCNNorm) into every sampled mini-batch, so
    this one-time application at load time covers the full training and inference
    loops without any per-batch transform overhead.

    Full-batch training on ogbn-products (~2.4M nodes, 61M edges) requires
    >40GB GPU memory. Instead this loader uses neighbor sampling to build
    mini-batches -- the same strategy used by OGB's official products baseline.

    Neighbor sampling parameters:
      num_neighbors=(15, 10): sample 15 neighbours at hop 1, 10 at hop 2.
      This matches the OGB GraphSAGE NeighborSampler default.
      Source: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py

    Dataset loading for ogbn-products:
      Source: https://ogb.stanford.edu/docs/nodeprop/#ogbn-products
      Snippet (PyG path):
        dataset   = PygNodePropPredDataset(name="ogbn-products")
        split_idx = dataset.get_idx_split()
        graph     = dataset[0]
      Snippet (DGL path):
        dataset      = DglNodePropPredDataset(name="ogbn-products")
        split_idx    = dataset.get_idx_split()
        graph, label = dataset[0]

    NeighborLoader is from torch_geometric.loader.NeighborLoader (PyG).
    DGL equivalent (dgl.dataloading.NodeDataLoader) is not used here;
    the PyG loader is reused for both frameworks for consistency.

    Enabled via --use-sampling flag. Required for ogbn-products (Tier 3).
    """
    n  = name.lower()
    fw = framework.lower()
    mn = model_name.lower()
    if n not in OGB_NODE_DATASETS:
        raise ValueError(f"Sampled loader only implemented for {OGB_NODE_DATASETS}.")
    if n == "ogbn-mag":
        raise ValueError(
            "ogbn-mag is heterogeneous and cannot be loaded via load_dataset_sampled(). "
            "Use --dataset ogbn-mag without --use-sampling; the heterogeneous "
            "pipeline (load_mag / run_mode_mag) handles it separately."
        )

    if fw == "dgl":
        dgl_ds      = DglNodePropPredDataset(name=n, root=root)
        split_idx   = dgl_ds.get_idx_split()
        g, _labels  = dgl_ds[0]
        g           = dgl.add_self_loop(g)
        g           = g.to(device)
        pyg_ds      = PygNodePropPredDataset(name=n, root=root)
        data        = pyg_ds[0]
        num_classes = dgl_ds.num_classes
    else:
        dataset     = PygNodePropPredDataset(name=n, root=root)
        data        = dataset[0]
        split_idx   = dataset.get_idx_split()
        num_classes = dataset.num_classes
        g           = None

    # Apply the same per-model transform used in load_dataset (full-batch path)
    # before constructing NeighborLoader, so that every sampled mini-batch
    # already contains the correct edge_index (with self-loops) and edge_weight
    # (GCN normalisation coefficients). PyG propagates all Data attributes,
    # including edge_weight, into subgraph batches automatically.
    if fw == "pyg":
        if mn == "gcn":
            data = ToUndirected()(data)
            data = GCNNorm(add_self_loops=True)(data)
            log.info("load_dataset_sampled: applied ToUndirected + GCNNorm for GCN.")
        else:
            data = AddSelfLoops()(data)
            log.info("load_dataset_sampled: applied AddSelfLoops for %s.", mn)

    train_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        input_nodes=split_idx["train"],
        shuffle=True,
    )
    log.info("NeighborLoader: dataset=%s, framework=%s, num_neighbors=%s, batch_size=%d",
             name, framework, num_neighbors, batch_size)
    return train_loader, data.to(device), split_idx, num_classes, g


# ---------------------------------------------------------------------------
# Dataset loader: ogbn-mag (heterogeneous, R-GCN)
# ---------------------------------------------------------------------------
# ogbn-mag contains 4 node types and 4 directed edge types. To use RGCNConv
# (which operates on a flattened homogeneous graph), all node types are
# mapped into a single node space with offsets, and all edges (plus their
# reverses for bidirectional message passing) are concatenated into a single
# edge_index tensor with a parallel edge_type tensor.
#
# Two PyG data formats are handled:
#   HeteroData (PyG >= 2.0): data["paper"], data["author"], etc.
#   Old homogeneous Data   : data.stores[0]["num_nodes_dict"], etc.
#
# OGB dataset description: https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag
# OGB leaderboard        : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag
# Official R-GCN baseline: https://github.com/snap-stanford/ogb/blob/master/
#                          examples/nodeproppred/mag/rgcn.py
#
# Returns a dict ("mag bundle") consumed by run_mode_mag().

def load_mag(data_root: str, device) -> dict[str, Any]:
    """
    Load ogbn-mag and build the flattened homogeneous representation for RGCNConv.

    Handles both PyG HeteroData (PyG >= 2.0) and the older homogeneous
    Data format where OGB pre-flattens node types into a single node space.

    Returns a dict with keys:
        paper_feat  : [N_paper, 128]   paper word2vec features
        edge_index  : [2, E_total]     flattened homogeneous edge index
        edge_type   : [E_total]        relation index per edge
        labels      : [N_paper]        venue class labels (0-348)
        train_mask  : [N_paper] bool
        val_mask    : [N_paper] bool
        test_mask   : [N_paper] bool
        n_paper     : int
        n_author    : int
        n_inst      : int
        n_field     : int
        in_feats    : int   (128)
        num_classes : int   (349)
        evaluator   : OGB NodeEvaluator
    """
    log.info("Loading ogbn-mag ...")
    dataset   = PygNodePropPredDataset(name="ogbn-mag", root=data_root)
    split_idx = dataset.get_idx_split()
    data      = dataset[0]
    evaluator = NodeEvaluator(name="ogbn-mag")
    num_classes = dataset.num_classes

    is_hetero = hasattr(data, "node_types")          # HeteroData (PyG >= 2.0)

    rel_order = _MAG_EDGE_TYPES
    rel_to_idx = {r: i for i, r in enumerate(rel_order)}
    reverse_map = {
        ("author", "writes",          "paper"):          ("paper",          "to", "author"),
        ("author", "affiliated_with", "institution"):    ("institution",    "to", "author"),
        ("paper",  "has_topic",       "field_of_study"): ("field_of_study", "to", "paper"),
    }

    if is_hetero:
        n_paper  = data["paper"].num_nodes
        n_author = data["author"].num_nodes
        n_inst   = data["institution"].num_nodes
        n_field  = data["field_of_study"].num_nodes

        offsets = {
            "paper":          0,
            "author":         n_paper,
            "institution":    n_paper + n_author,
            "field_of_study": n_paper + n_author + n_inst,
        }

        paper_feat = data["paper"].x.to(device)
        labels     = data["paper"].y.squeeze().to(device)

        ei_dict_raw: dict[tuple, torch.Tensor] = {}
        for stype, rtype, dtype in [
            ("paper",  "cites",           "paper"),
            ("author", "writes",          "paper"),
            ("author", "affiliated_with", "institution"),
            ("paper",  "has_topic",       "field_of_study"),
        ]:
            ei_dict_raw[(stype, rtype, dtype)] = data[(stype, rtype, dtype)].edge_index

        # Make paper-cites-paper undirected (same as OGB baseline)
        cites_key = ("paper", "cites", "paper")
        ei_dict_raw[cites_key] = to_undirected(ei_dict_raw[cites_key])

        all_src, all_dst, all_rel = [], [], []
        for orig_key, ei in ei_dict_raw.items():
            stype, _, dtype = orig_key
            all_src.append(ei[0] + offsets[stype])
            all_dst.append(ei[1] + offsets[dtype])
            all_rel.append(torch.full((ei.shape[1],), rel_to_idx[orig_key],
                                      dtype=torch.long))
        for orig_key, rev_key in reverse_map.items():
            ei = ei_dict_raw[orig_key]
            stype, _, dtype = orig_key
            all_src.append(ei[1] + offsets[dtype])
            all_dst.append(ei[0] + offsets[stype])
            all_rel.append(torch.full((ei.shape[1],), rel_to_idx[rev_key],
                                      dtype=torch.long))

    else:
        # Old PyG format: all data in data.stores[0]
        store  = data.stores[0]
        nnd    = store["num_nodes_dict"]
        n_paper  = nnd["paper"]
        n_author = nnd["author"]
        n_inst   = nnd["institution"]
        n_field  = nnd["field_of_study"]

        offsets = {
            "paper":          0,
            "author":         n_paper,
            "institution":    n_paper + n_author,
            "field_of_study": n_paper + n_author + n_inst,
        }

        paper_feat = store["x_dict"]["paper"].to(device)
        labels     = store["y_dict"]["paper"].squeeze().to(device)

        ei_dict = store["edge_index_dict"]
        cites_ei = to_undirected(ei_dict[("paper", "cites", "paper")])

        all_src, all_dst, all_rel = [], [], []
        for (stype, rtype, dtype), ei in ei_dict.items():
            if (stype, rtype, dtype) == ("paper", "cites", "paper"):
                ei = cites_ei
            rel_idx = rel_to_idx.get((stype, rtype, dtype))
            if rel_idx is None:
                continue
            all_src.append(ei[0] + offsets[stype])
            all_dst.append(ei[1] + offsets[dtype])
            all_rel.append(torch.full((ei.shape[1],), rel_idx, dtype=torch.long))
        for orig_key, rev_key in reverse_map.items():
            ei = ei_dict[orig_key]
            stype, _, dtype = orig_key
            all_src.append(ei[1] + offsets[dtype])
            all_dst.append(ei[0] + offsets[stype])
            all_rel.append(torch.full((ei.shape[1],), rel_to_idx[rev_key],
                                      dtype=torch.long))

    edge_index = torch.stack(
        [torch.cat(all_src), torch.cat(all_dst)], dim=0).to(device)
    edge_type  = torch.cat(all_rel).to(device)

    train_idx = split_idx["train"]["paper"].to(device)
    val_idx   = split_idx["valid"]["paper"].to(device)
    test_idx  = split_idx["test"]["paper"].to(device)

    train_mask = torch.zeros(n_paper, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n_paper, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n_paper, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    n_total = n_paper + n_author + n_inst + n_field
    log.info(
        "ogbn-mag loaded (%s format): %d paper, %d total nodes, "
        "%d edges, %d relation types, %d classes",
        "hetero" if is_hetero else "old-homogeneous",
        n_paper, n_total, edge_index.shape[1], _MAG_NUM_RELATIONS, num_classes,
    )

    return {
        "paper_feat":  paper_feat,
        "edge_index":  edge_index,
        "edge_type":   edge_type,
        "labels":      labels,
        "train_mask":  train_mask,
        "val_mask":    val_mask,
        "test_mask":   test_mask,
        "n_paper":     n_paper,
        "n_author":    n_author,
        "n_inst":      n_inst,
        "n_field":     n_field,
        "in_feats":    paper_feat.shape[1],
        "num_classes": num_classes,
        "evaluator":   evaluator,
    }


# ---------------------------------------------------------------------------
# Dataset loader: ogbl-biokg (KG completion, DistMult)
# ---------------------------------------------------------------------------
# ogbl-biokg stores triplets as separate head/relation/tail arrays per split.
# Entity types are identified by string labels ("disease", "protein", etc.);
# each type has its own index space.
#
# OGB dataset description: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
# OGB leaderboard        : https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-biokg
# Official baseline      : https://github.com/snap-stanford/ogb/blob/master/
#                          examples/linkproppred/biokg/
#
# Evaluation: MRR with same-type negatives, 500 per head/tail replacement.
# OGB provides pre-sampled negatives (head_neg / tail_neg) for val/test splits.
#
# Returns a dict ("biokg bundle") consumed by run_mode_biokg().

def load_biokg(data_root: str, device) -> dict[str, Any]:
    """
    Load ogbl-biokg and prepare all tensors needed for training/evaluation.

    Returns a dict with keys:
        split_edge    : raw OGB split dict (train/valid/test)
        num_nodes_dict: {entity_type: count}
        num_relations : int
        evaluator     : OGB LinkEvaluator
        device        : torch.device
    """
    log.info("Loading ogbl-biokg ...")
    dataset    = PygLinkPropPredDataset(name="ogbl-biokg", root=data_root)
    split_edge = dataset.get_edge_split()
    data       = dataset[0]

    if hasattr(data, "num_nodes_dict"):
        num_nodes_dict: dict[str, int] = {
            k: int(v) for k, v in data.num_nodes_dict.items()
        }
    else:
        # Fallback to known fixed sizes if attribute is absent
        num_nodes_dict = {
            "disease":    10687,
            "protein":    17499,
            "drug":       10533,
            "sideeffect":  9969,
            "function":   45085,
        }
    log.info("ogbl-biokg entity type counts: %s", num_nodes_dict)

    num_relations = int(split_edge["train"]["relation"].max().item()) + 1
    log.info("ogbl-biokg relation types: %d", num_relations)

    evaluator = LinkEvaluator(name="ogbl-biokg")

    log.info(
        "ogbl-biokg loaded: %d entity types, %d relations, "
        "%d train / %d val / %d test triplets",
        len(num_nodes_dict), num_relations,
        len(split_edge["train"]["head"]),
        len(split_edge["valid"]["head"]),
        len(split_edge["test"]["head"]),
    )

    return {
        "split_edge":     split_edge,
        "num_nodes_dict": num_nodes_dict,
        "num_relations":  num_relations,
        "evaluator":      evaluator,
        "device":         device,
    }




def generate_recommendations(all_results: dict, cfg: dict) -> str:
    """
    Synthesise per-mode benchmark results into plain-text practical recommendations.

    Verdict logic (in priority order):
        AVOID       -- compilation failed, or output numerically incorrect
        AVOID       -- regression (compiled is slower than eager)
        CAUTION     -- graph breaks detected (partial capture, benefit limited)
        RECOMMENDED -- speedup >= 1.5x with no quality issues
        MARGINAL    -- speedup > 1.0x but < 1.5x
        UNKNOWN     -- no latency data recorded (subprocess error)
        NEUTRAL     -- catchall
    """
    lines = [
        "=" * 70,
        "Practical Recommendations",
        f"  Framework : {cfg.get('framework', '?')}",
        f"  Model     : {cfg.get('model', '?')}",
        f"  Dataset   : {cfg.get('dataset', '?')}",
        f"  Device    : {cfg.get('device', '?')}",
        "=" * 70,
        "",
    ]
    eager     = all_results.get("eager", {})
    eager_lat = eager.get("inference_latency_median_ms")

    for mode, res in all_results.items():
        if mode == "eager":
            eager_lat_str = f"{eager_lat:.3f} ms" if eager_lat is not None else "N/A"
            lines.append(f"[eager] Baseline -- {eager_lat_str} median inference latency.")
            if eager.get("error"):
                lines.append(f"  WARNING: eager mode reported an error: {eager['error']}")
            lines.append("")
            continue

        lat     = res.get("inference_latency_median_ms")
        speedup = res.get("speedup_vs_eager")
        breaks  = res.get("unsupported_op_count") or 0
        success = res.get("compilation_success", False)
        quality = res.get("quality_check_passed")
        bev     = res.get("break_even_runs")
        cats    = res.get("break_categories") or {}
        dom_cat = max(cats, key=cats.get) if cats and any(cats.values()) else None

        if not success:
            verdict = "AVOID -- compilation failed entirely."
        elif quality is False:
            verdict = ("AVOID -- compiled output is numerically incorrect "
                       "(torch.allclose failed at atol=rtol=1e-3). "
                       "Do not use in production.")
        elif speedup is not None and speedup < 1.0:
            overhead_pct = round((1.0 / speedup - 1.0) * 100, 1) if speedup > 0 else 999
            verdict = (f"AVOID -- {overhead_pct:.1f}% overhead vs eager. "
                       "Compilation cost not amortised.")
        elif breaks > 0:
            cat_note     = f" Dominant break category: {dom_cat}." if dom_cat else ""
            speedup_note = f" Speedup limited to {speedup:.3f}x." if speedup is not None else ""
            verdict      = f"CAUTION -- {breaks} graph break(s) detected.{cat_note}{speedup_note}"
        elif speedup is not None and speedup >= 1.5:
            verdict = (f"RECOMMENDED -- {speedup:.3f}x speedup, no quality issues, "
                       "0 graph breaks.")
        elif speedup is not None and speedup > 1.0:
            verdict = (f"MARGINAL -- {speedup:.3f}x speedup. "
                       "Only worthwhile for long-running services.")
        elif speedup is None:
            err_msg = res.get("error", "no error detail recorded")
            verdict = f"UNKNOWN -- no latency data. Error: {err_msg}"
        else:
            verdict = f"NEUTRAL -- speedup={speedup}, breaks={breaks}."

        lines.append(f"[{mode}] {verdict}")

        # Only show break-even when the mode is actually usable â€” not when
        # compilation failed or the output is numerically incorrect.
        show_bev = success and quality is not False
        if show_bev:
            if bev is not None:
                lines.append(f"  Break-even: {bev:,} inference calls to recover compile cost.")
            elif res.get("break_even_note"):
                lines.append(f"  Break-even: {res['break_even_note']}")

        if res.get("error"):
            lines.append(f"  Error recorded: {res['error']}")

        lines.append("")

    lines += [
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Forward helpers and evaluation for ogbn-mag (R-GCN)
# ---------------------------------------------------------------------------

def _build_mag_model(cfg: dict[str, Any], mag: dict[str, Any], device) -> RGCN:
    """Construct and initialise a fresh RGCN for ogbn-mag."""
    model = RGCN(
        in_channels=mag["in_feats"],
        hidden=cfg.get("hidden", 64),
        out_channels=mag["num_classes"],
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.5),
        num_relations=_MAG_NUM_RELATIONS,
        num_authors=mag["n_author"],
        num_institutions=mag["n_inst"],
        num_fields=mag["n_field"],
    ).to(device)
    model.reset_parameters()
    return model


def _fwd_mag(model: RGCN, mag: dict[str, Any]) -> torch.Tensor:
    """Single full-graph forward pass for ogbn-mag."""
    return model(
        mag["paper_feat"],
        mag["edge_index"],
        mag["edge_type"],
        mag["n_paper"],
    )


@torch.no_grad()
def _evaluate_mag(model: RGCN, mag: dict[str, Any], mask: torch.Tensor) -> float:
    """Accuracy on paper nodes selected by mask, using OGB NodeEvaluator."""
    model.eval()
    logits = _fwd_mag(model, mag)
    y_pred = logits[mask].argmax(dim=-1, keepdim=True)
    y_true = mag["labels"][mask]
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    result = mag["evaluator"].eval({"y_true": y_true, "y_pred": y_pred})
    return round(result["acc"] * 100.0, 4)


def run_mode_mag(
    mode: str, cfg: dict[str, Any], mag: dict[str, Any],
    device, eager_median_ms: float | None,
) -> dict[str, Any]:
    """
    Run the complete benchmark pipeline for one compile mode on ogbn-mag.

    Uses the same measurement protocol as run_mode() (inference + training +
    accuracy + numerical equivalence + Dynamo + CUDA kernels + break-even)
    but with the RGCN model and the mag bundle instead of the homogeneous
    GNN models and (x, edge_index) tensors.
    """
    is_compiled = mode != "eager"
    n_nodes     = mag["n_paper"]

    # --- Inference ---
    infer_model = _build_mag_model(cfg, mag, device)
    num_params  = _count_params(infer_model)

    # Reuse benchmark_inference by wrapping the forward via link_mode=False;
    # we adapt it by passing a dummy edge_index/x and using dgl_graph=None,
    # but benchmark_inference calls model_forward() which we cannot override.
    # Instead we time directly here, mirroring the benchmark_inference logic.
    def _timed_perf_mag() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _fwd_mag(infer_model, mag)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _timed_cuda_mag() -> float:
        if not torch.cuda.is_available():
            return _timed_perf_mag() * 1000.0
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _fwd_mag(infer_model, mag)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    compile_time_s      = 0.0
    latencies_ms:        list[float] = []
    warmup_latencies_ms: list[float] = []
    infer_cpu_samples:   list[float] = []
    peak_memory_mb_infer = None
    compilation_success  = None
    compiled_infer_model = infer_model
    cpu_sampler = _CpuSampler(interval=0.1)

    repeats = cfg.get("repeats", 30)
    warmup  = cfg.get("warmup",  5)

    try:
        infer_model.eval()
        if mode != "eager":
            compilation_success = True
            torch._dynamo.config.suppress_errors = True
            # PyG GCN/RGCN on ogbn-mag: reduce-overhead and max-autotune both
            # trigger CUDA Graph capture, which crashes with curr_block->next ==
            # nullptr because the sparse RGCNConv propagation uses internal CUDA
            # stream ops incompatible with graph capture.
            _infer_compile_mode = mode
            if mode == "reduce-overhead":
                _infer_compile_mode = "default"
                log.info("mag: reduce-overhead downgraded to 'default' "
                         "to avoid CUDA Graph / sparse-op incompatibility.")
            elif mode == "max-autotune":
                _infer_compile_mode = "max-autotune-no-cudagraphs"
                log.info("mag: max-autotune downgraded to "
                         "'max-autotune-no-cudagraphs' to avoid CUDA Graph "
                         "/ sparse-op incompatibility.")
            compiled_infer_model = infer_model
            compile_time_s       = _timed_perf_mag()
            _dynamic = cfg.get("dynamic", None)
            infer_model          = torch.compile(infer_model, mode=_infer_compile_mode, dynamic=_dynamic)
            log.info("mag compile finished in %.2fs.", compile_time_s)
        else:
            remaining_warmup = warmup

        with torch.no_grad():
            for _ in range(remaining_warmup):
                warmup_latencies_ms.append(_timed_cuda_mag())
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            cpu_sampler.start()
            for _ in range(repeats):
                latencies_ms.append(_timed_cuda_mag())
            infer_cpu_samples = cpu_sampler.stop()
            if torch.cuda.is_available():
                peak_memory_mb_infer = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2))

    except Exception as exc:
        log.error("mag inference benchmark failed (mode=%s): %s", mode, exc)
        return _failure_result_mag(str(exc))

    arr    = np.array(latencies_ms)
    median = float(np.median(arr))
    iqr    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    mean_l = float(np.mean(arr))
    std_l  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    speedup = (1.0 if not is_compiled
               else round(eager_median_ms / median, 4)
               if eager_median_ms and median else None)

    if mode != "eager" and median > 0:
        compile_overhead_factor = compile_time_s / (median / 1000.0)
        compile_overhead_equivalent_calls = int(round(compile_overhead_factor))
    else:
        compile_overhead_factor = None
        compile_overhead_equivalent_calls = None

    throughput = float(n_nodes / (median / 1000.0)) if median > 0 else None
    log.info("mag mode='%s' | median=%.3fms | IQR=%.3fms | peak_gpu=%.1fMB",
             mode, median, iqr, peak_memory_mb_infer or 0.0)

    # --- Training ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Re-seed before training model build so that weight initialisation and
    # dropout masks are identical across all compile modes.
    set_seed(cfg.get("seed", 42) + 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_model      = _build_mag_model(cfg, mag, device)
    train_compile_mode = mode
    if train_compile_mode == "reduce-overhead":
        train_compile_mode = "default"
    elif train_compile_mode == "max-autotune":
        train_compile_mode = "max-autotune-no-cudagraphs"
    if is_compiled:
        # suppress_errors=False ensures compilation failures surface as exceptions
        # rather than falling back to eager silently.
        torch._dynamo.config.suppress_errors = False
        # RGCNConv.propagate is compiled by Dynamo as a separate function frame.
        # Switching between eval() (inference) and train() causes GLOBAL_STATE
        # grad_mode changes that trigger recompilation. The default cache_size_limit
        # of 8 is exhausted by the inference warmup + measured passes switching
        # between no_grad and grad contexts, causing Dynamo to fall back to eager
        # for all subsequent calls and crash with OOM on the full-graph forward.
        # Resetting Dynamo clears the inference cache before training starts, and
        # raising the cache limit to 64 accommodates the grad_mode variants.
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 64
        _dynamic = cfg.get("dynamic", None)
        try:
            train_model = torch.compile(train_model, mode=train_compile_mode, dynamic=_dynamic)
        except Exception as _compile_exc:
            log.warning("mode='%s': training compile failed: %s", mode, _compile_exc)
    optimizer  = torch.optim.Adam(train_model.parameters(),
                                  lr=cfg.get("lr", 0.01))
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    criterion  = nn.CrossEntropyLoss()
    n_train_epochs  = cfg.get("train_epochs", 10)
    n_train_warmup  = cfg.get("train_warmup", 3)
    warmup_times_tr: list[float] = []
    epoch_times_tr:  list[float] = []
    train_cpu_samples: list[float] = []
    train_warmup_cpu:  list[float] = []

    train_model.train()
    for ep in range(n_train_warmup + n_train_epochs):
        cpu_sampler.start()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = _fwd_mag(train_model, mag)
        loss   = criterion(logits[mag["train_mask"]],
                           mag["labels"][mag["train_mask"]])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ep_cpu  = cpu_sampler.stop()
        if ep < n_train_warmup:
            warmup_times_tr.append(elapsed)
            train_warmup_cpu.extend(ep_cpu)
        else:
            epoch_times_tr.append(elapsed)
            train_cpu_samples.extend(ep_cpu)

    arr_tr    = np.array(epoch_times_tr)
    first_tr  = epoch_times_tr[0] if epoch_times_tr else None
    mean_tr   = float(np.mean(arr_tr)) if len(arr_tr) > 0 else 0.0
    std_tr    = float(np.std(arr_tr, ddof=1)) if len(arr_tr) > 1 else 0.0
    if is_compiled and len(epoch_times_tr) > 1:
        mean_rest = float(np.mean(arr_tr[1:]))
        train_oh  = (first_tr - mean_rest) / mean_rest * 100.0 if mean_rest > 0 else 0.0
    else:
        train_oh = None
    peak_gpu_train = None
    if torch.cuda.is_available():
        peak_gpu_train = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    # --- Accuracy ---
    train_model.eval()
    val_acc  = _evaluate_mag(train_model, mag, mag["val_mask"])
    test_acc = _evaluate_mag(train_model, mag, mag["test_mask"])

    # --- Numerical equivalence ---
    if not is_compiled:
        quality_res = {"quality_check_passed": None, "max_logit_abs_diff": None,
                       "logit_allclose_atol": 1e-3, "logit_allclose_rtol": 1e-3}
    else:
        ref_model = _build_mag_model(cfg, mag, device)
        src_state = (train_model._orig_mod.state_dict()
                     if hasattr(train_model, "_orig_mod")
                     else train_model.state_dict())
        ref_model.load_state_dict(src_state)
        ref_model.eval()
        with torch.no_grad():
            z_eager    = _fwd_mag(ref_model, mag)
            z_compiled = _fwd_mag(train_model, mag)
        passed   = bool(torch.allclose(z_eager, z_compiled, atol=1e-3, rtol=1e-3))
        max_diff = float((z_eager - z_compiled).abs().max().item())
        quality_res = {"quality_check_passed": passed,
                       "max_logit_abs_diff":   round(max_diff, 6),
                       "logit_allclose_atol":  1e-3,
                       "logit_allclose_rtol":  1e-3}

    # --- Dynamo metrics ---
    # Use the uncompiled infer_model so explain() sees the raw graph structure.
    # For compiled modes infer_model is already wrapped; rebuild a fresh copy.
    dynamo_model = _build_mag_model(cfg, mag, device)
    dynamo_model.eval()
    try:
        torch._dynamo.reset()
        expl = torch._dynamo.explain(dynamo_model)(
            mag["paper_feat"], mag["edge_index"],
            mag["edge_type"],  mag["n_paper"])
        dynamo_res = _parse_dynamo_explain(expl)
    except Exception as exc:
        log.warning("mag _dynamo.explain failed: %s", exc)
        dynamo_res = _dynamo_fail_dict(str(exc))

    # --- CUDA kernel count ---
    cuda_kernels: int | None = None
    if torch.cuda.is_available():
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                with torch.no_grad():
                    _fwd_mag(compiled_infer_model, mag)

            def _ct(e):
                return getattr(e, "self_device_time_total",
                               getattr(e, "self_cuda_time_total", 0))
            cuda_kernels = len([e for e in prof.key_averages() if _ct(e) > 0])
        except Exception as exc:
            log.warning("mag CUDA kernel profiling failed: %s", exc)

    bev_res      = compute_breakeven(compile_time_s, eager_median_ms, median)
    code_changes = _count_code_changes(mode, "pyg")

    return {
        "inference_latency_median_ms":       median,
        "inference_latency_iqr_ms":          iqr,
        "inference_latency_mean_ms":         mean_l,
        "inference_latency_std_ms":          std_l,
        "inference_latency_min_ms":          float(np.min(arr)),
        "inference_latency_max_ms":          float(np.max(arr)),
        "all_latencies_ms":                  latencies_ms,
        "all_warmup_latencies_ms":           warmup_latencies_ms,
        "mean_epoch_time_s":                 mean_tr,
        "std_epoch_time_s":                  std_tr,
        "first_measured_epoch_time_s":       first_tr,
        "max_epoch_time_s":                  float(np.max(arr_tr)) if len(arr_tr) > 0 else None,
        "all_epoch_times_s":                 epoch_times_tr,
        "all_warmup_times_s":                warmup_times_tr,
        "throughput_inference_nodes_per_s":  throughput,
        "throughput_train_nodes_per_s":      float(n_nodes / mean_tr) if mean_tr > 0 else None,
        "speedup_vs_eager":                  speedup,
        "compile_time_s":                    compile_time_s,
        "compile_overhead_equivalent_calls": compile_overhead_equivalent_calls,
        "compile_overhead_factor":           round(compile_overhead_factor, 1)
                                             if compile_overhead_factor is not None else None,
        "train_compile_overhead_pct":        round(train_oh, 4) if train_oh is not None else None,
        "train_compile_mode":                train_compile_mode if is_compiled else "eager",
        "train_mode_differs_from_infer":     (train_compile_mode != mode) if is_compiled else False,
        "peak_gpu_memory_inference_mb":      round(peak_memory_mb_infer, 2)
                                             if peak_memory_mb_infer is not None else None,
        "peak_gpu_memory_train_mb":          peak_gpu_train,
        "cpu_utilization_inference_pct_avg": round(float(np.mean(infer_cpu_samples)), 2)
                                             if infer_cpu_samples else None,
        "cpu_utilization_inference_pct_max": round(float(np.max(infer_cpu_samples)), 2)
                                             if infer_cpu_samples else None,
        "all_cpu_samples_inference":         list(infer_cpu_samples),
        "cpu_utilization_train_pct_avg":     round(float(np.mean(train_cpu_samples)), 2)
                                             if train_cpu_samples else None,
        "cpu_utilization_train_pct_max":     round(float(np.max(train_cpu_samples)), 2)
                                             if train_cpu_samples else None,
        "all_cpu_samples_train":             list(train_cpu_samples),
        "all_warmup_cpu_samples_train":      list(train_warmup_cpu),
        "test_accuracy_pct":                 test_acc,
        "val_accuracy_pct":                  val_acc,
        "quality_check_passed":              quality_res["quality_check_passed"],
        "max_logit_abs_diff":                quality_res["max_logit_abs_diff"],
        "logit_allclose_atol":               quality_res["logit_allclose_atol"],
        "logit_allclose_rtol":               quality_res["logit_allclose_rtol"],
        "compilation_success":               compilation_success,
        "graph_capture_rate_pct":            dynamo_res["graph_capture_rate_pct"],
        "graph_capture_rate_is_exact":       dynamo_res["graph_capture_rate_is_exact"],
        "graph_breaks":                      dynamo_res["graph_breaks"],
        "unsupported_op_count":              dynamo_res["unsupported_op_count"],
        "ops_per_graph":                     dynamo_res["ops_per_graph"],
        "break_categories":                  dynamo_res["break_categories"],
        "cuda_kernel_count":                 cuda_kernels,
        "num_params":                        num_params,
        "break_even_runs":                   bev_res["break_even_runs"],
        "break_even_note":                   bev_res["break_even_note"],
        "required_code_changes":             code_changes,
        "error":                             "",
    }


def _failure_result_mag(error_msg: str) -> dict[str, Any]:
    """Return a minimal failure result dict for run_mode_mag errors."""
    return {
        "inference_latency_median_ms": None, "inference_latency_iqr_ms": None,
        "inference_latency_mean_ms":   None, "inference_latency_std_ms":  None,
        "inference_latency_min_ms":    None, "inference_latency_max_ms":  None,
        "all_latencies_ms": [], "all_warmup_latencies_ms": [],
        "mean_epoch_time_s": None, "std_epoch_time_s": None,
        "first_measured_epoch_time_s": None, "max_epoch_time_s": None,
        "all_epoch_times_s": [], "all_warmup_times_s": [],
        "throughput_inference_nodes_per_s": None,
        "throughput_train_nodes_per_s": None,
        "speedup_vs_eager": None, "compile_time_s": 0.0,
        "compile_overhead_equivalent_calls": None, "compile_overhead_factor": None,
        "train_compile_overhead_pct": None, "train_compile_mode": None,
        "train_mode_differs_from_infer": False,
        "peak_gpu_memory_inference_mb": None, "peak_gpu_memory_train_mb": None,
        "cpu_utilization_inference_pct_avg": None,
        "cpu_utilization_inference_pct_max": None,
        "all_cpu_samples_inference": [],
        "cpu_utilization_train_pct_avg": None,
        "cpu_utilization_train_pct_max": None,
        "all_cpu_samples_train": [], "all_warmup_cpu_samples_train": [],
        "test_accuracy_pct": None, "val_accuracy_pct": None,
        "quality_check_passed": None, "max_logit_abs_diff": None,
        "logit_allclose_atol": 1e-3, "logit_allclose_rtol": 1e-3,
        "compilation_success": False,
        "graph_capture_rate_pct": None, "graph_capture_rate_is_exact": False,
        "graph_breaks": [error_msg], "unsupported_op_count": None,
        "ops_per_graph": None, "break_categories": None,
        "cuda_kernel_count": None, "num_params": None,
        "break_even_runs": None, "break_even_note": None,
        "required_code_changes": None, "error": error_msg,
    }


def _parse_dynamo_explain(expl) -> dict[str, Any]:
    """
    Extract graph-capture metrics from a torch._dynamo.explain() result.

    Shared helper used by run_mode_mag and run_mode_biokg so both routes
    use exactly the same extraction logic as _get_dynamo_metrics() in the
    main pipeline, without duplicating the try/except boilerplate.
    """
    is_exact          = False
    rate              = None
    ops_per_graph_list = None
    break_reasons:    list[str] = []

    if hasattr(expl, "graphs"):
        break_reasons = [str(r) for r in getattr(expl, "break_reasons", [])]
        total_ops     = getattr(expl, "total_ops", None)
        ops_per_graph = getattr(expl, "ops_per_graph", None)
        if ops_per_graph is not None and hasattr(ops_per_graph, "__iter__"):
            ops_per_graph_list = list(ops_per_graph)
        if total_ops and ops_per_graph and total_ops > 0:
            captured = sum(ops_per_graph_list) if ops_per_graph_list else ops_per_graph
            rate     = round(captured / total_ops * 100.0, 2)
            is_exact = True
        if rate is None:
            n_graphs = len(getattr(expl, "graphs", []))
            n_breaks = len(break_reasons)
            if n_graphs > 0 and n_breaks == 0:
                rate = 100.0
            elif n_graphs > 0:
                rate = round(n_graphs / (n_graphs + n_breaks) * 100.0, 2)
            else:
                rate = 0.0

    n_breaks = len(break_reasons)
    if rate is None:
        rate = 0.0

    if ops_per_graph_list is not None:
        ops_per_graph_list = [int(x) for x in ops_per_graph_list
                              if isinstance(x, (int, float, np.integer, np.floating))]
        if not ops_per_graph_list:
            ops_per_graph_list = None

    return {
        "graph_capture_rate_pct":      rate,
        "graph_capture_rate_is_exact": is_exact,
        "graph_breaks":                break_reasons,
        "unsupported_op_count":        n_breaks,
        "ops_per_graph":               ops_per_graph_list,
        "break_categories":            categorise_graph_breaks(break_reasons),
    }


def _dynamo_fail_dict(error_msg: str) -> dict[str, Any]:
    """Return a zeroed dynamo metrics dict for cases where explain() throws."""
    return {
        "graph_capture_rate_pct":      None,
        "graph_capture_rate_is_exact": False,
        "graph_breaks":                [error_msg],
        "unsupported_op_count":        None,
        "ops_per_graph":               None,
        "break_categories":            None,
    }


# ---------------------------------------------------------------------------
# Forward helpers and evaluation for ogbl-biokg (DistMult)
# ---------------------------------------------------------------------------

def _build_biokg_model(cfg: dict[str, Any], biokg: dict[str, Any],
                       device) -> DistMult:
    """Construct and initialise a fresh DistMult for ogbl-biokg."""
    model = DistMult(
        num_nodes_dict=biokg["num_nodes_dict"],
        num_relations=biokg["num_relations"],
        emb_dim=cfg.get("emb_dim", 128),
    ).to(device)
    model.reset_parameters()
    return model


def _make_biokg_infer_batch(biokg: dict[str, Any],
                            batch_size: int = 1024) -> dict[str, Any]:
    """
    Build a fixed type-homogeneous inference batch from training triplets.

    Uses the most common (head_type, tail_type) pair so the forward call
    is a single, deterministic call that torch.compile can trace cleanly
    without encountering data-dependent string branching.
    """
    device = biokg["device"]
    s      = biokg["split_edge"]["train"]
    pairs  = list(zip(s["head_type"], s["tail_type"]))
    (ht, tt), _ = Counter(pairs).most_common(1)[0]
    mask = torch.tensor(
        [i for i, (h, t) in enumerate(pairs) if h == ht and t == tt],
        dtype=torch.long)[:batch_size]
    return {
        "head_type": ht,
        "head":      s["head"][mask].to(device),
        "relation":  s["relation"][mask].to(device),
        "tail_type": tt,
        "tail":      s["tail"][mask].to(device),
    }


def _fwd_biokg(model: DistMult, batch: dict[str, Any]) -> torch.Tensor:
    """Single forward pass for ogbl-biokg inference timing."""
    return model(
        batch["head_type"], batch["head"],
        batch["relation"],
        batch["tail_type"], batch["tail"],
    )


@torch.no_grad()
def _evaluate_biokg_mrr(model: DistMult, biokg: dict[str, Any],
                        split: str = "valid",
                        max_triplets: int = 4096) -> float:
    """
    Compute MRR on a subsample of the given split using OGB LinkEvaluator.

    Uses the pre-sampled same-type negatives provided by OGB (head_neg /
    tail_neg, 500 per triplet). Returns mean MRR over head and tail
    corruptions, expressed as a percentage.

    Source: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
    """
    model.eval()
    device     = biokg["device"]
    split_data = biokg["split_edge"][split]
    evaluator  = biokg["evaluator"]

    n   = min(max_triplets, len(split_data["head"]))
    idx = torch.randperm(len(split_data["head"]))[:n]

    heads      = split_data["head"][idx].to(device)
    rels       = split_data["relation"][idx].to(device)
    tails      = split_data["tail"][idx].to(device)
    head_types = [split_data["head_type"][i] for i in idx.tolist()]
    tail_types = [split_data["tail_type"][i] for i in idx.tolist()]
    neg_heads  = split_data["head_neg"][idx].to(device)   # [n, 500]
    neg_tails  = split_data["tail_neg"][idx].to(device)   # [n, 500]

    mrr_vals: list[float] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (ht, tt) in enumerate(zip(head_types, tail_types)):
        groups[(ht, tt)].append(i)

    for (ht, tt), grp_idx in groups.items():
        gi  = torch.tensor(grp_idx, device=device)
        h   = heads[gi];  r = rels[gi];  t = tails[gi]
        nh  = neg_heads[gi];  nt = neg_tails[gi]

        h_emb  = model.entity_emb[ht](h)
        t_emb  = model.entity_emb[tt](t)
        r_emb  = model.rel_emb(r)
        pos_scores = (h_emb * r_emb * t_emb).sum(dim=-1)

        nh_emb       = model.entity_emb[ht](nh)
        t_exp        = t_emb.unsqueeze(1).expand_as(nh_emb)
        r_exp        = r_emb.unsqueeze(1).expand_as(nh_emb)
        neg_h_scores = (nh_emb * r_exp * t_exp).sum(dim=-1)
        res_h = evaluator.eval({"y_pred_pos": pos_scores,
                                "y_pred_neg": neg_h_scores})
        mrr_vals.extend(res_h["mrr_list"].tolist())

        nt_emb       = model.entity_emb[tt](nt)
        h_exp        = h_emb.unsqueeze(1).expand_as(nt_emb)
        r_exp2       = r_emb.unsqueeze(1).expand_as(nt_emb)
        neg_t_scores = (h_exp * r_exp2 * nt_emb).sum(dim=-1)
        res_t = evaluator.eval({"y_pred_pos": pos_scores,
                                "y_pred_neg": neg_t_scores})
        mrr_vals.extend(res_t["mrr_list"].tolist())

    return round(float(np.mean(mrr_vals)) * 100.0, 4)


def run_mode_biokg(
    mode: str, cfg: dict[str, Any], biokg: dict[str, Any],
    device, eager_median_ms: float | None,
) -> dict[str, Any]:
    """
    Run the complete benchmark pipeline for one compile mode on ogbl-biokg.

    Uses the same measurement protocol as run_mode_link() but with the
    DistMult model and the biokg bundle. Inference is timed on a fixed
    type-homogeneous batch; training uses BCE loss on pos + random negatives.
    Evaluation metric is MRR (OGB LinkEvaluator, same-type negatives).
    """
    is_compiled = mode != "eager"
    batch_size  = cfg.get("batch_size", 1024)
    infer_batch = _make_biokg_infer_batch(biokg, batch_size=batch_size)
    n_triplets  = len(infer_batch["head"])

    # --- Inference timing (mirrors benchmark_inference structure) ---
    infer_model = _build_biokg_model(cfg, biokg, device)
    num_params  = _count_params(infer_model)

    def _timed_perf_biokg() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _fwd_biokg(infer_model, infer_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _timed_cuda_biokg() -> float:
        if not torch.cuda.is_available():
            return _timed_perf_biokg() * 1000.0
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _fwd_biokg(infer_model, infer_batch)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    compile_time_s       = 0.0
    latencies_ms:         list[float] = []
    warmup_latencies_ms:  list[float] = []
    infer_cpu_samples:    list[float] = []
    peak_memory_mb_infer  = None
    compilation_success   = None
    compiled_infer_model  = infer_model
    cpu_sampler = _CpuSampler(interval=0.1)

    repeats = cfg.get("repeats", 30)
    warmup  = cfg.get("warmup",  5)

    try:
        infer_model.eval()
        if mode != "eager":
            compilation_success  = True
            torch._dynamo.config.suppress_errors = True
            _dynamic = cfg.get("dynamic", None)
            infer_model          = torch.compile(infer_model, mode=mode, dynamic=_dynamic)
            compiled_infer_model = infer_model
            remaining_warmup     = max(0, warmup - 1)
            log.info("biokg compile finished in %.2fs.", compile_time_s)
        else:
            remaining_warmup = warmup

        with torch.no_grad():
            for _ in range(remaining_warmup):
                warmup_latencies_ms.append(_timed_cuda_biokg())
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            cpu_sampler.start()
            for _ in range(repeats):
                latencies_ms.append(_timed_cuda_biokg())
            infer_cpu_samples = cpu_sampler.stop()
            if torch.cuda.is_available():
                peak_memory_mb_infer = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2))

    except Exception as exc:
        log.error("biokg inference benchmark failed (mode=%s): %s", mode, exc)
        return _failure_result_biokg(str(exc))

    arr    = np.array(latencies_ms)
    median = float(np.median(arr))
    iqr    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    mean_l = float(np.mean(arr))
    std_l  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    speedup = (1.0 if not is_compiled
               else round(eager_median_ms / median, 4)
               if eager_median_ms and median else None)

    if mode != "eager" and median > 0:
        compile_overhead_factor = compile_time_s / (median / 1000.0)
        compile_overhead_equivalent_calls = int(round(compile_overhead_factor))
    else:
        compile_overhead_factor = None
        compile_overhead_equivalent_calls = None

    throughput = float(n_triplets / (median / 1000.0)) if median > 0 else None
    log.info("biokg mode='%s' | median=%.3fms | IQR=%.3fms | peak_gpu=%.1fMB",
             mode, median, iqr, peak_memory_mb_infer or 0.0)

    # --- Training ---
    # Re-seed before training model build so that weight initialisation and
    # dropout masks are identical across all compile modes.
    set_seed(cfg.get("seed", 42) + 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_model      = _build_biokg_model(cfg, biokg, device)
    train_compile_mode = mode
    if is_compiled:
        # suppress_errors=False ensures compilation failures surface as exceptions
        # rather than falling back to eager silently.
        torch._dynamo.config.suppress_errors = False
        _dynamic = cfg.get("dynamic", None)
        try:
            train_model = torch.compile(train_model, mode=train_compile_mode, dynamic=_dynamic)
        except Exception as _compile_exc:
            log.warning("mode='%s': biokg training compile failed: %s", mode, _compile_exc)

    optimizer   = torch.optim.Adam(train_model.parameters(),
                                   lr=cfg.get("lr", 0.01))
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    criterion   = nn.BCEWithLogitsLoss()
    s_train     = biokg["split_edge"]["train"]
    tbatch_size = cfg.get("train_batch_size", 1024)
    pairs       = list(zip(s_train["head_type"], s_train["tail_type"]))
    (ht, tt), _ = Counter(pairs).most_common(1)[0]
    mask_tr     = torch.tensor(
        [i for i, (h, t) in enumerate(pairs) if h == ht and t == tt],
        dtype=torch.long)[:tbatch_size]
    h_pos = s_train["head"][mask_tr].to(device)
    r_pos = s_train["relation"][mask_tr].to(device)
    t_pos = s_train["tail"][mask_tr].to(device)
    n_ent_h = biokg["num_nodes_dict"][ht]
    n_ent_t = biokg["num_nodes_dict"][tt]
    n_tr    = len(h_pos)

    n_train_epochs = cfg.get("train_epochs", 5)
    n_train_warmup = cfg.get("train_warmup", 2)
    warmup_times_tr:  list[float] = []
    epoch_times_tr:   list[float] = []
    train_cpu_samples: list[float] = []
    train_warmup_cpu:  list[float] = []

    train_model.train()
    for ep in range(n_train_warmup + n_train_epochs):
        h_neg = torch.randint(0, n_ent_h, (n_tr,), device=device)
        t_neg = torch.randint(0, n_ent_t, (n_tr,), device=device)
        cpu_sampler.start()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        pos_scores  = train_model(ht, h_pos, r_pos, tt, t_pos)
        neg_h_scores = train_model(ht, h_neg, r_pos, tt, t_pos)
        neg_t_scores = train_model(ht, h_pos, r_pos, tt, t_neg)
        scores = torch.cat([pos_scores, neg_h_scores, neg_t_scores])
        labels_lp = torch.cat([
            torch.ones(n_tr,  device=device),
            torch.zeros(n_tr, device=device),
            torch.zeros(n_tr, device=device),
        ])
        loss = criterion(scores, labels_lp)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ep_cpu  = cpu_sampler.stop()
        if ep < n_train_warmup:
            warmup_times_tr.append(elapsed)
            train_warmup_cpu.extend(ep_cpu)
        else:
            epoch_times_tr.append(elapsed)
            train_cpu_samples.extend(ep_cpu)

    arr_tr   = np.array(epoch_times_tr)
    first_tr = epoch_times_tr[0] if epoch_times_tr else None
    mean_tr  = float(np.mean(arr_tr)) if len(arr_tr) > 0 else 0.0
    std_tr   = float(np.std(arr_tr, ddof=1)) if len(arr_tr) > 1 else 0.0
    if is_compiled and len(epoch_times_tr) > 1:
        mean_rest = float(np.mean(arr_tr[1:]))
        train_oh  = (first_tr - mean_rest) / mean_rest * 100.0 if mean_rest > 0 else 0.0
    else:
        train_oh = None
    peak_gpu_train = None
    if torch.cuda.is_available():
        peak_gpu_train = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    # --- MRR evaluation ---
    train_model.eval()
    val_mrr  = _evaluate_biokg_mrr(train_model, biokg, split="valid")
    test_mrr = _evaluate_biokg_mrr(train_model, biokg, split="test")

    # --- Numerical equivalence ---
    if not is_compiled:
        quality_res = {"quality_check_passed": None, "max_logit_abs_diff": None,
                       "logit_allclose_atol": 1e-3, "logit_allclose_rtol": 1e-3}
    else:
        ref_model = _build_biokg_model(cfg, biokg, device)
        src_state = (train_model._orig_mod.state_dict()
                     if hasattr(train_model, "_orig_mod")
                     else train_model.state_dict())
        ref_model.load_state_dict(src_state)
        ref_model.eval()
        with torch.no_grad():
            z_eager    = _fwd_biokg(ref_model, infer_batch)
            z_compiled = _fwd_biokg(train_model, infer_batch)
        passed   = bool(torch.allclose(z_eager, z_compiled, atol=1e-3, rtol=1e-3))
        max_diff = float((z_eager - z_compiled).abs().max().item())
        quality_res = {"quality_check_passed": passed,
                       "max_logit_abs_diff":   round(max_diff, 6),
                       "logit_allclose_atol":  1e-3,
                       "logit_allclose_rtol":  1e-3}

    # --- Dynamo metrics ---
    dynamo_model = _build_biokg_model(cfg, biokg, device)
    dynamo_model.eval()
    try:
        torch._dynamo.reset()
        expl = torch._dynamo.explain(dynamo_model)(
            infer_batch["head_type"], infer_batch["head"],
            infer_batch["relation"],
            infer_batch["tail_type"], infer_batch["tail"])
        dynamo_res = _parse_dynamo_explain(expl)
    except Exception as exc:
        log.warning("biokg _dynamo.explain failed: %s", exc)
        dynamo_res = _dynamo_fail_dict(str(exc))

    # --- CUDA kernel count ---
    cuda_kernels: int | None = None
    if torch.cuda.is_available():
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                with torch.no_grad():
                    _fwd_biokg(compiled_infer_model, infer_batch)

            def _ct(e):
                return getattr(e, "self_device_time_total",
                               getattr(e, "self_cuda_time_total", 0))
            cuda_kernels = len([e for e in prof.key_averages() if _ct(e) > 0])
        except Exception as exc:
            log.warning("biokg CUDA kernel profiling failed: %s", exc)

    bev_res      = compute_breakeven(compile_time_s, eager_median_ms, median)
    code_changes = _count_code_changes(mode, "pyg")

    return {
        "inference_latency_median_ms":         median,
        "inference_latency_iqr_ms":            iqr,
        "inference_latency_mean_ms":           mean_l,
        "inference_latency_std_ms":            std_l,
        "inference_latency_min_ms":            float(np.min(arr)),
        "inference_latency_max_ms":            float(np.max(arr)),
        "all_latencies_ms":                    latencies_ms,
        "all_warmup_latencies_ms":             warmup_latencies_ms,
        "mean_epoch_time_s":                   mean_tr,
        "std_epoch_time_s":                    std_tr,
        "first_measured_epoch_time_s":         first_tr,
        "max_epoch_time_s":                    float(np.max(arr_tr)) if len(arr_tr) > 0 else None,
        "all_epoch_times_s":                   epoch_times_tr,
        "all_warmup_times_s":                  warmup_times_tr,
        "throughput_inference_triplets_per_s": throughput,
        "throughput_train_triplets_per_s":     float(n_tr / mean_tr) if mean_tr > 0 else None,
        "speedup_vs_eager":                    speedup,
        "compile_time_s":                      compile_time_s,
        "compile_overhead_equivalent_calls":   compile_overhead_equivalent_calls,
        "compile_overhead_factor":             round(compile_overhead_factor, 1)
                                               if compile_overhead_factor is not None else None,
        "train_compile_overhead_pct":          round(train_oh, 4) if train_oh is not None else None,
        "train_compile_mode":                  train_compile_mode if is_compiled else "eager",
        "peak_gpu_memory_inference_mb":        round(peak_memory_mb_infer, 2)
                                               if peak_memory_mb_infer is not None else None,
        "peak_gpu_memory_train_mb":            peak_gpu_train,
        "cpu_utilization_inference_pct_avg":   round(float(np.mean(infer_cpu_samples)), 2)
                                               if infer_cpu_samples else None,
        "cpu_utilization_inference_pct_max":   round(float(np.max(infer_cpu_samples)), 2)
                                               if infer_cpu_samples else None,
        "all_cpu_samples_inference":           list(infer_cpu_samples),
        "cpu_utilization_train_pct_avg":       round(float(np.mean(train_cpu_samples)), 2)
                                               if train_cpu_samples else None,
        "cpu_utilization_train_pct_max":       round(float(np.max(train_cpu_samples)), 2)
                                               if train_cpu_samples else None,
        "all_cpu_samples_train":               list(train_cpu_samples),
        "all_warmup_cpu_samples_train":        list(train_warmup_cpu),
        "link_metric_name":                    "mrr",
        "val_link_metric_pct":                 val_mrr,
        "test_link_metric_pct":                test_mrr,
        "quality_check_passed":                quality_res["quality_check_passed"],
        "max_logit_abs_diff":                  quality_res["max_logit_abs_diff"],
        "logit_allclose_atol":                 quality_res["logit_allclose_atol"],
        "logit_allclose_rtol":                 quality_res["logit_allclose_rtol"],
        "compilation_success":                 compilation_success,
        "graph_capture_rate_pct":              dynamo_res["graph_capture_rate_pct"],
        "graph_capture_rate_is_exact":         dynamo_res["graph_capture_rate_is_exact"],
        "graph_breaks":                        dynamo_res["graph_breaks"],
        "unsupported_op_count":                dynamo_res["unsupported_op_count"],
        "ops_per_graph":                       dynamo_res["ops_per_graph"],
        "break_categories":                    dynamo_res["break_categories"],
        "cuda_kernel_count":                   cuda_kernels,
        "num_params":                          num_params,
        "break_even_runs":                     bev_res["break_even_runs"],
        "break_even_note":                     bev_res["break_even_note"],
        "required_code_changes":               code_changes,
        "error":                               "",
    }


def _failure_result_biokg(error_msg: str) -> dict[str, Any]:
    """Return a minimal failure result dict for run_mode_biokg errors."""
    return {
        "inference_latency_median_ms": None, "inference_latency_iqr_ms": None,
        "inference_latency_mean_ms":   None, "inference_latency_std_ms":  None,
        "inference_latency_min_ms":    None, "inference_latency_max_ms":  None,
        "all_latencies_ms": [], "all_warmup_latencies_ms": [],
        "mean_epoch_time_s": None, "std_epoch_time_s": None,
        "first_measured_epoch_time_s": None, "max_epoch_time_s": None,
        "all_epoch_times_s": [], "all_warmup_times_s": [],
        "throughput_inference_triplets_per_s": None,
        "throughput_train_triplets_per_s": None,
        "speedup_vs_eager": None, "compile_time_s": 0.0,
        "compile_overhead_equivalent_calls": None, "compile_overhead_factor": None,
        "train_compile_overhead_pct": None, "train_compile_mode": None,
        "peak_gpu_memory_inference_mb": None, "peak_gpu_memory_train_mb": None,
        "cpu_utilization_inference_pct_avg": None,
        "cpu_utilization_inference_pct_max": None,
        "all_cpu_samples_inference": [],
        "cpu_utilization_train_pct_avg": None,
        "cpu_utilization_train_pct_max": None,
        "all_cpu_samples_train": [], "all_warmup_cpu_samples_train": [],
        "link_metric_name": "mrr",
        "val_link_metric_pct": None, "test_link_metric_pct": None,
        "quality_check_passed": None, "max_logit_abs_diff": None,
        "logit_allclose_atol": 1e-3, "logit_allclose_rtol": 1e-3,
        "compilation_success": False,
        "graph_capture_rate_pct": None, "graph_capture_rate_is_exact": False,
        "graph_breaks": [error_msg], "unsupported_op_count": None,
        "ops_per_graph": None, "break_categories": None,
        "cuda_kernel_count": None, "num_params": None,
        "break_even_runs": None, "break_even_note": None,
        "required_code_changes": None, "error": error_msg,
    }


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------
# The LaTeX tables are the primary output for the thesis paper.
# generate_latex_tables() produces three tables:
#
#   Table 1 (tab:perf_*): Performance metrics
#     - Inference latency: median +/- IQR (30 runs, CUDA event timing)
#     - Training: steady-state mean +/- std epoch time
#     - Compile overhead (N/A for eager), throughput, peak GPU memory
#     - Speedup vs. eager (x)
#
#   Table 2 (tab:quality_*): Accuracy / numerical equivalence
#     - Test/Val accuracy via OGB Evaluator (acc for nodes, Hits@K for links)
#     - max |delta| logits from torch.allclose check (atol=rtol=1e-3)
#
#   Table 3 (tab:usability_*): Usability / diagnostics
#     - Compilation success, graph-capture rate % (torch._dynamo.explain)
#     - Graph-break taxonomy (BREAK_CATEGORIES), CUDA kernel count
#     - #Parameters, break-even calls, required code changes
# ---------------------------------------------------------------------------

def _fmt(value, decimals=3, suffix=""):
    """Format a scalar value for a LaTeX table cell. None -> \\text{N/A}.

    All numeric and boolean values are wrapped in $...$ so they sit in math
    mode consistently alongside the row_pm cells and _fmt_large cells.
    """
    if value is None:
        return r"$\text{N/A}$"
    if isinstance(value, bool):
        return r"$\checkmark$" if value else r"$\times$"
    try:
        return f"${float(value):.{decimals}f}{suffix}$"
    except (TypeError, ValueError):
        return str(value)


def _fmt_large(value, decimals=0) -> str:
    """Format large integers with LaTeX thousands separator ({,})."""
    if value is None:
        return r"\text{N/A}"
    try:
        formatted = f"{int(round(float(value))):,}".replace(",", "{,}")
        return f"${formatted}$"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_tables(base_config, system_info, all_results, out_dir):
    """Generate publication-ready LaTeX tables from benchmark results."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(out_dir, "tables.tex")

    modes   = list(all_results.keys())
    dataset = base_config.get("dataset",   "unknown")
    model   = base_config.get("model",     "GNN")
    fw      = base_config.get("framework", "PyG")

    MODE_ABBREV = {
        "eager":                       "Eager",
        "default":                     "Default",
        "reduce-overhead":             "Red.-OH",
        "max-autotune":                "Max-AT",
        "max-autotune-no-cudagraphs":  "Max-AT-noCG",
    }
    n         = len(modes)
    col_spec  = "l" + "r" * n
    mode_hdrs = " & ".join(r"\textbf{" + MODE_ABBREV.get(m, m) + "}" for m in modes)

    def tbl_open(caption, label):
        return [r"\begin{table}[htbp]", r"  \centering",
                r"  \caption{" + caption + "}", r"  \label{" + label + "}",
                r"  \resizebox{\textwidth}{!}{%"]

    def tbl_close():
        return [r"  }% end resizebox", r"\end{table}", ""]

    def row(label, key, dec=3, sfx=""):
        vals = " & ".join(_fmt(all_results[m].get(key), dec, sfx) for m in modes)
        return f"    {label} & {vals} \\\\"

    def row_large(label, key):
        vals = " & ".join(_fmt_large(all_results[m].get(key)) for m in modes)
        return f"    {label} & {vals} \\\\"

    def row_pm(label, k_mid, k_spr, dec=3, sfx=""):
        cells = []
        for m in modes:
            mid, spr = all_results[m].get(k_mid), all_results[m].get(k_spr)
            if mid is None:
                cells.append(r"$\text{N/A}$")
            elif spr is None:
                cells.append(_fmt(mid, dec, sfx))
            else:
                cells.append(f"${float(mid):.{dec}f} \\pm {float(spr):.{dec}f}${sfx}")
        return f"    {label} & " + " & ".join(cells) + r" \\"

    def section(text):
        return r"    \multicolumn{" + str(1 + n) + r"}{l}{\textit{" + text + r"}} \\"

    out = [
        "% Auto-generated by gnn_compile_benchmark_v29.py",
        f"% Model: {model}  |  Dataset: {dataset}  |  Framework: {fw}",
        f"% Generated: {timestamp}", "",
        r"% Required packages: \usepackage{booktabs,graphicx,amsmath,amssymb,xcolor,siunitx}", "",
    ]

    # Table 1: Performance
    out += tbl_open(
        f"Performance metrics for \\textbf{{{model}}} on \\texttt{{{dataset}}} ({fw}). "
        r"Inference latency: median\,$\pm$\,IQR of 30 runs. "
        r"Training: steady-state mean\,$\pm$\,std. "
        r"GPU memory: inference = activation memory only; training = peak during full loop.",
        f"tab:perf_{dataset.replace('-','_')}_{model.lower()}")
    out += [
        f"  \\begin{{tabular}}{{{col_spec}}}", r"    \toprule",
        f"    \\textbf{{Metric}} & {mode_hdrs} \\\\", r"    \midrule",
        section("Inference latency (30 runs, GPU timeline)"),
        row_pm(r"~~Median\,$\pm$\,IQR (ms)",
               "inference_latency_median_ms", "inference_latency_iqr_ms"),
        row(r"~~Min (ms)", "inference_latency_min_ms"),
        row(r"~~Max (ms)", "inference_latency_max_ms"),
        r"    \midrule",
        section("Training epoch time (steady-state)"),
        row_pm(r"~~Mean\,$\pm$\,Std (s)", "mean_epoch_time_s", "std_epoch_time_s", dec=4),
        r"    \midrule",
        section("Compile overhead (N/A for eager)"),
        row(r"~~Inference compile time (s)",               "compile_time_s",                       dec=2),
        row(r"~~Overhead factor ($\times$)",               "compile_overhead_factor",              dec=1),
        # This column shows how many inference calls the compile cost is equivalent to.
        row_large(r"~~Overhead (equiv.\ inference calls)", "compile_overhead_equivalent_calls"),
        row(r"~~Training 1st-epoch overhead (\%)",         "train_compile_overhead_pct",           dec=2),
        r"    \midrule",
        section(r"Throughput (full-graph forward pass)"),
        row_large(r"~~Inference (nodes\,s$^{-1}$)", "throughput_inference_nodes_per_s"),
        row_large(
            r"~~Training  (nodes\,s$^{-1}$)" if dataset.lower() not in OGB_LINK_DATASETS
            else r"~~Training  (edges\,s$^{-1}$)",
            "throughput_train_nodes_per_s" if dataset.lower() not in OGB_LINK_DATASETS
            else "throughput_train_edges_per_s"),
        r"    \midrule",
        section("GPU memory (separate inference / training)"),
        row(r"~~Peak -- inference (MB)", "peak_gpu_memory_inference_mb", dec=1),
        row(r"~~Peak -- training (MB)",  "peak_gpu_memory_train_mb",     dec=1),
        r"    \midrule",
        section("CPU utilisation (separate inference / training)"),
        row(r"~~Inference (\%)", "cpu_utilization_inference_pct_avg", dec=1),
        row(r"~~Training (\%)",  "cpu_utilization_train_pct_avg",     dec=1),
        r"    \midrule",
        row(r"\textbf{Speedup vs.\ eager} ($\times$)", "speedup_vs_eager", dec=3),
        r"    \bottomrule", r"  \end{tabular}",
    ]
    out += tbl_close()

    # Table 2: Quality / accuracy
    is_link_dataset = dataset.lower() in OGB_LINK_DATASETS
    is_kg_dataset   = dataset.lower() in OGB_KG_DATASETS
    if is_link_dataset or is_kg_dataset:
        sample_mode  = all_results.get(list(all_results.keys())[0], {})
        metric_label = sample_mode.get("link_metric_name", "mrr" if is_kg_dataset else "link metric").upper()
        acc_rows = [
            row(f"Test {metric_label} (\\%)", "test_link_metric_pct", dec=2),
            row(f"Val {metric_label} (\\%)",  "val_link_metric_pct",  dec=2),
        ]
        if is_kg_dataset:
            quality_caption = (
                f"Evaluation metric for \\textbf{{{model}}} on \\texttt{{{dataset}}}. "
                f"MRR computed with same-type negatives (500 per triplet) via OGB LinkEvaluator.")
        else:
            quality_caption = (
                f"Evaluation metric for \\textbf{{{model}}} on \\texttt{{{dataset}}}. "
                f"ogbl-collab uses Hits@50; ogbl-citation2 uses MRR.")
    else:
        acc_rows = [
            row(r"Test accuracy (\%)", "test_accuracy_pct", dec=2),
            row(r"Val accuracy (\%)",  "val_accuracy_pct",  dec=2),
        ]
        quality_caption = (
            f"Accuracy and numerical equivalence for \\textbf{{{model}}} on "
            f"\\texttt{{{dataset}}}. "
            r"Accuracy via OGB \texttt{Evaluator.eval()}. "
            r"Numerical check: \texttt{torch.allclose} at atol=rtol=1e-3. "
            r"\text{N/A} for eager (reference model).")

    out += tbl_open(quality_caption,
                    f"tab:quality_{dataset.replace('-','_')}_{model.lower()}")
    out += [
        f"  \\begin{{tabular}}{{{col_spec}}}", r"    \toprule",
        f"    \\textbf{{Metric}} & {mode_hdrs} \\\\", r"    \midrule",
        *acc_rows,
        r"    \midrule",
        row(r"Max embedding $|\Delta|$", "max_logit_abs_diff",   dec=6),
        row(r"Quality check passed",     "quality_check_passed", dec=0),
        r"    \bottomrule", r"  \end{tabular}",
    ]
    out += tbl_close()

    # Table 3: Usability / diagnostics
    out += tbl_open(
        f"Usability and diagnostic metrics for \\textbf{{{model}}} on "
        f"\\texttt{{{dataset}}}. "
        r"Graph-capture rate from \texttt{torch.\_dynamo.explain}. "
        r"CUDA kernel count from \texttt{torch.profiler}. "
        r"Break-even: compile cost amortised after this many inference calls. "
        r"Code changes: usability score.",
        f"tab:usability_{dataset.replace('-','_')}_{model.lower()}")
    out += [
        f"  \\begin{{tabular}}{{{col_spec}}}", r"    \toprule",
        f"    \\textbf{{Metric}} & {mode_hdrs} \\\\", r"    \midrule",
        row(r"Compilation success",     "compilation_success",         dec=0),
        row(r"Graph-capture rate (\%)", "graph_capture_rate_pct",      dec=1),
        row(r"~~Exact measurement",     "graph_capture_rate_is_exact", dec=0),
        row(r"Graph breaks (\#)",       "unsupported_op_count",        dec=0),
        r"    \midrule",
        section("Graph break root causes (Dynamo taxonomy)"),
    ]
    for cat in list(BREAK_CATEGORIES.keys()) + ["other"]:
        cells = []
        for m in modes:
            val = (all_results[m].get("break_categories") or {}).get(cat)
            cells.append(_fmt(val, decimals=0))
        out.append(
            r"    ~~" + cat.replace("_", r"\_") + " & " +
            " & ".join(cells) + r" \\"
        )
    out += [
        r"    \midrule",
        row(r"CUDA kernels (\#)",          "cuda_kernel_count",     dec=0),
        row_large(r"\#Params",             "num_params"),
        row_large(r"Break-even (calls)",   "break_even_runs"),
        row(r"Code changes required (\#)", "required_code_changes", dec=0),
        r"    \bottomrule", r"  \end{tabular}",
    ]
    out += tbl_close()

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    log.info("LaTeX tables written: %s", filename)
    return filename


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _sanitise(obj):
    """
    Recursively convert any non-JSON-serialisable value to a string.

    Handles the case where a dict value is accidentally a Python built-in
    function or method reference (e.g. from torch._dynamo.explain() returning
    unexpected types), which would cause json.dump to raise:
        TypeError: Object of type builtin_function_or_method is not JSON serializable
    """
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # fallback: convert anything else to its string representation
    return str(obj)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitise(data), fh, indent=4)
    log.info("Written: %s", path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark JIT compilation for PyG/DGL GNNs",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework",    default="PyG", choices=["PyG", "DGL", "pyg", "dgl"])
    p.add_argument("--model-name",   default="GCN",
                   choices=["GCN", "GraphSAGE", "GAT", "GIN",
                            "gcn", "graphsage", "gat", "gin",
                            "RGCN", "rgcn",
                            "DistMult", "distmult"])
    p.add_argument("--dataset",      default="ogbn-arxiv")
    p.add_argument("--data-root",    default="data/")
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--num-layers",   type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--gat-heads",    type=int,   default=8,
                   help="Number of attention heads for GAT. hidden dim is per-head, so effective width = hidden * gat_heads.")
    # DistMult / ogbl-biokg specific arguments
    p.add_argument("--emb-dim",          type=int, default=128,
                   help="Embedding dimension for DistMult (ogbl-biokg only).")
    p.add_argument("--batch-size",       type=int, default=8192,
                   help="Inference batch size for DistMult (ogbl-biokg only).")
    p.add_argument("--train-batch-size", type=int, default=8192,
                   help="Training batch size for DistMult (ogbl-biokg only).")
    p.add_argument("--use-sampling", action="store_true")
    p.add_argument("--dynamic", type=str, default="auto",
                   choices=["true", "false", "auto"],
                   help="dynamic= argument passed to torch.compile(). "
                        "'auto' (default) = None (automatic dynamic: first compile static, "
                        "recompile promotes changed dims to dynamic). "
                        "'true' = dynamic=True (always symbolic; not recommended). "
                        "'false' = dynamic=False (always static; disables automatic dynamic).")
    p.add_argument("--gat-chunk-size", type=int, default=None,
                   help="Chunk edge_index into batches of this size during full-graph GAT "
                        "forward (OOM fallback without NeighborLoader). "
                        "E.g. --gat-chunk-size 500000 for ogbn-arxiv on 16 GB GPU.")
    p.add_argument("--repeats",      type=int,   default=30)
    p.add_argument("--warmup",       type=int,   default=5)
    p.add_argument("--train-epochs", type=int,   default=20)
    p.add_argument("--train-warmup", type=int,   default=5)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--collab-lr",    type=float, default=0.001,
                   help="LR for ogbl-collab (OGB baseline: 0.001).")
    p.add_argument("--collab-dropout", type=float, default=0.0,
                   help="Dropout for ogbl-collab (OGB baseline: 0.0).")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--timeout",      type=int,   default=3600,
                   help="Per-mode subprocess timeout in seconds (default: 3600). "
                        "Increase for max-autotune on large models/datasets. "
                        "On expiry the mode is recorded as an error and the next mode runs.")
    p.add_argument("--out-dir",      default="experiments")
    p.add_argument("--modes",        nargs="+",
                   default=["eager", "default", "reduce-overhead",
                            "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--_single-mode", dest="single_mode",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eager-median-ms", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--eager-train-epoch-s", type=float, default=None,
                   help=argparse.SUPPRESS)

    args = p.parse_args()

    if (DATASET_TIER.get(args.dataset.lower(), 2) == 3
            and not getattr(args, "use_sampling", False)
            and not getattr(args, "single_mode", False)):
        warnings.warn(
            f"Dataset '{args.dataset}' is Tier 3 (large). "
            "Full-batch training will OOM at hidden>=128 on <48 GB GPU. "
            "Pass --use-sampling to enable NeighborLoader.",
            stacklevel=2)

    return args


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_single_mode(mode, args, run_dir, eager_median_ms=None, eager_train_epoch_s=None) -> dict[str, Any]:
    """
    Spawn an isolated child process for one compile mode.

    A configurable --timeout (default: 3600 s) is passed to subprocess.run().
    On expiry the child is killed and the mode is recorded as an error, allowing
    remaining modes to still run and partial results to be saved.
    """
    timeout_s = getattr(args, "timeout", 3600)

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--framework",    args.framework,
        "--model-name",   args.model_name,
        "--dataset",      args.dataset,
        "--data-root",    args.data_root,
        "--hidden",       str(args.hidden),
        "--num-layers",   str(args.num_layers),
        "--dropout",      str(args.dropout),
        "--gat-heads",    str(args.gat_heads),
        "--emb-dim",          str(args.emb_dim),
        "--batch-size",       str(args.batch_size),
        "--train-batch-size", str(args.train_batch_size),
        "--collab-lr",    str(args.collab_lr),
        "--collab-dropout", str(args.collab_dropout),
        "--repeats",      str(args.repeats),
        "--warmup",       str(args.warmup),
        "--train-epochs", str(args.train_epochs),
        "--train-warmup", str(args.train_warmup),
        "--lr",           str(args.lr),
        "--seed",         str(args.seed),
        "--out-dir",      run_dir,
        "--modes",        mode,
        "--_single-mode",
    ]
    if getattr(args, "use_sampling", False):
        cmd.append("--use-sampling")
    _dynamic_val = getattr(args, "dynamic", "auto")
    if _dynamic_val != "auto":  # auto is the default, no need to forward
        cmd += ["--dynamic", _dynamic_val]
    if getattr(args, "gat_chunk_size", None) is not None:
        cmd += ["--gat-chunk-size", str(args.gat_chunk_size)]
    if eager_median_ms is not None:
        cmd += ["--eager-median-ms", str(eager_median_ms)]
    if eager_train_epoch_s is not None:
        cmd += ["--eager-train-epoch-s", str(eager_train_epoch_s)]

    # Force the child process to decode all I/O as UTF-8 with replacement
    # so that non-UTF-8 bytes emitted by DGL's C++ backend cannot cause a
    # UnicodeDecodeError before the child has written its JSON result to stdout.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONUTF8"] = "1"   # PEP 540: UTF-8 mode (Python 3.7+)

    def _err(msg):
        # compilation_success is only meaningful for compiled modes; for eager
        # a subprocess failure (e.g. OOM) reflects a data or memory issue,
        # not a compilation outcome, so the field is left as None.
        compile_success = None if mode == "eager" else False
        return {"compilation_success": compile_success, "error": msg,
                "inference_latency_median_ms": None, "speedup_vs_eager": None}

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=timeout_s, 
        )
        if proc.returncode in (-11, 139):
            log.error("mode='%s' SEGFAULT (exit %d)", mode, proc.returncode)
            return _err(f"Segmentation fault (exit {proc.returncode})")
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "")
            log.error("mode='%s' exit %d: %s", mode, proc.returncode, msg)
            # Truncate the stored error string so that large subprocess stderr
            # outputs do not inflate results.json and recommendations.txt.
            # The full output is preserved in run.log via the log.error call above.
            msg_short = msg[:500] + ("..." if len(msg) > 500 else "")
            return _err(f"exit {proc.returncode}: {msg_short}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            log.error("mode='%s' stdout JSON parse failed: %s\nstdout: %.400s",
                      mode, exc, proc.stdout)
            return _err(f"stdout JSON parse failed: {exc}")
    except subprocess.TimeoutExpired:
        log.error(
            "mode='%s' timed out after %d s (--timeout).  "
            "The child process has been killed.  "
            "Increase --timeout for slow modes like max-autotune.",
            mode, timeout_s)
        return _err(f"subprocess timeout after {timeout_s}s")
    except KeyboardInterrupt:
        log.warning("Ctrl+C during mode='%s' - saving results so far", mode)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# The script has two execution paths, selected by the --single-mode flag:
#
#   Orchestrator path (default, --single-mode not set):
#     Spawns one subprocess per compile mode (in MODES_ORDERED order, always
#     starting with 'eager' to establish the baseline latency). Collects JSON
#     results from each subprocess's stdout and writes the aggregated output.
#     Order: eager first (provides eager_median_ms for speedup calculation),
#     then default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs.
#
#   Child path (--single-mode, called by the orchestrator):
#     Runs exactly one mode, prints result as JSON to stdout, exits.
#     This isolation prevents CUDA state / compiled-kernel-cache leakage
#     between modes, ensuring each mode is measured from a cold start.
#
# Dataset routing (all combinations are supported):
#   OGB link datasets (ogbl-*) -> load_link_dataset() + run_mode_link()
#   Tier 3 + --use-sampling    -> load_dataset_sampled() + run_mode()
#   All others                 -> load_dataset() + run_mode()
#   DGL + Planetoid            -> _to_dgl_graph() builds DGL graph from edge_index

if __name__ == "__main__":
    args = parse_args()

    # Auto-enable neighbor sampling for GAT on datasets with >500K edges
    # to prevent OOM during full-graph attention score computation.
    if (args.model_name.lower() == "gat"
            and args.dataset.lower() in {"ogbn-arxiv", "ogbl-collab", "ogbn-products"}
            and not getattr(args, "use_sampling", False)):
        import warnings as _warnings
        _warnings.warn(
            f"GAT + {args.dataset}: automatically enabling --use-sampling to avoid OOM "
            "during full-graph attention score computation. "
            "Pass --use-sampling explicitly to suppress this warning.",
            stacklevel=2)
        args.use_sampling = True

    if args.single_mode:
        run_dir = args.out_dir
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name      = f"{args.framework}_{args.model_name}_{args.dataset}_{run_timestamp}"
        run_dir       = os.path.join(args.out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

    for h in log.root.handlers[:]:
        log.root.removeHandler(h)
    _log_handlers = [logging.FileHandler(os.path.join(run_dir, "run.log"), mode="a")]
    if not args.single_mode:
        _log_handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG if DBG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=_log_handlers,
    )

    set_seed(args.seed)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_info = get_system_info()

    # DEBUG: startup banner -- all values from args and system_info, nothing hardcoded
    _dbg_sep("GNN COMPILE BENCHMARK  --  STARTUP")
    _dbg("script",        __file__)
    _dbg("DBG",           str(DBG))
    _dbg("single_mode",   str(args.single_mode))
    _dbg("")
    _dbg("Framework",     args.framework)
    _dbg("Model",         args.model_name)
    _dbg("Dataset",       args.dataset)
    _dbg("  tier",        str(DATASET_TIER.get(args.dataset.lower(), "?")))
    _dbg("  is_link",     str(args.dataset.lower() in OGB_LINK_DATASETS))
    _dbg("  is_mag",      str(args.dataset.lower() == "ogbn-mag"))
    _dbg("  is_biokg",    str(args.dataset.lower() == "ogbl-biokg"))
    _dbg("  use_sampling",str(getattr(args, "use_sampling", False)))
    _dbg("Modes",         str(args.modes))
    _dbg("  compiled modes", str([m for m in args.modes if m != "eager"]))
    _dbg("  eager mode",     str("eager" in args.modes))
    _dbg("Inference repeats", str(args.repeats))
    _dbg("Inference warmup",  str(args.warmup))
    _dbg("Train epochs",      str(args.train_epochs))
    _dbg("Train warmup",      str(args.train_warmup))
    _dbg("Hidden dim",        str(args.hidden))
    _dbg("Num layers",        str(args.num_layers))
    _dbg("Dropout",           str(args.dropout))
    _dbg("LR",                str(args.lr))
    _dbg("Seed",              str(args.seed))
    _dbg("Timeout (s)",       str(args.timeout))
    _dbg("Device",            str(device))
    _dbg("GPU",               system_info.get("gpu_name", "n/a"))
    _dbg("GPU memory (GB)",   str(system_info.get("gpu_memory_total_gb", "n/a")))
    _dbg("PyTorch version",   system_info.get("pytorch_version", "n/a"))
    _dbg("CUDA version",      system_info.get("cuda_version", "n/a"))
    _dbg("Output dir",        run_dir)
    _dbg("")

    log.info("Framework=%s | Model=%s | Dataset=%s | Device=%s | Tier=%d",
             args.framework, args.model_name, args.dataset, system_info["gpu_name"],
             DATASET_TIER.get(args.dataset.lower(), 2))

    is_link  = args.dataset.lower() in OGB_LINK_DATASETS
    is_mag   = args.dataset.lower() == "ogbn-mag"
    is_biokg = args.dataset.lower() == "ogbl-biokg"

    # base_cfg is defined here ? before any dataset-specific branch ? so that
    # all branches (is_biokg, is_mag, is_link, standard) can reference it.
    base_cfg: dict[str, Any] = {
        "run_name":         os.path.basename(run_dir),
        "framework":        args.framework,
        "model":            args.model_name,
        "dataset":          args.dataset,
        "hidden":           args.hidden,
        "num_layers":       args.num_layers,
        "dropout":          args.dropout,
        "gat_heads":        args.gat_heads,
        "emb_dim":          args.emb_dim,
        "batch_size":       args.batch_size,
        "train_batch_size": args.train_batch_size,
        "collab_lr":        args.collab_lr,
        "collab_dropout":   args.collab_dropout,
        "repeats":          args.repeats,
        "warmup":           args.warmup,
        "train_epochs":     args.train_epochs,
        "train_warmup":     args.train_warmup,
        "lr":               args.lr,
        "seed":             args.seed,
        "device":           str(device),
        "modes":            args.modes,
        "run_dir":          run_dir,
        "dataset_tier":     DATASET_TIER.get(args.dataset.lower(), 2),
        "script_version":   "v29",
        # dynamic: the value passed to torch.compile(dynamic=...).
        # Resolved from --dynamic cli arg: "auto"->None, "true"->True, "false"->False.
        "dynamic":          {"auto": None, "true": True, "false": False}.get(
                                getattr(args, "dynamic", "auto"), None),
        "gat_chunk_size":   getattr(args, "gat_chunk_size", None),
    }

    if is_biokg:
        # ---------------------------------------------------------------
        # ogbl-biokg: DistMult KG completion
        # ---------------------------------------------------------------
        biokg = load_biokg(args.data_root, device)

        if args.single_mode:
            assert len(args.modes) == 1
            mode    = args.modes[0]
            results = run_mode_biokg(mode, {**base_cfg, "mode": mode},
                                     biokg, device,
                                     eager_median_ms=args.eager_median_ms)
            if mode == "eager":
                results["speedup_vs_eager"] = 1.0
            print(json.dumps(_sanitise(results)))
            sys.exit(0)

        _write_json(os.path.join(run_dir, "config.json"),
                    {"timestamp": datetime.now().isoformat(),
                     "config": base_cfg, "system_info": system_info})
        modes_ordered = (["eager"] + [m for m in args.modes if m != "eager"]
                         if "eager" in args.modes else args.modes)
        all_results:     dict[str, dict[str, Any]] = {}
        eager_median_ms: float | None              = None
        try:
            for mode in modes_ordered:
                log.info("=" * 60 + "  [biokg] mode='%s' ...", mode)
                results = _run_single_mode(mode, args, run_dir, eager_median_ms)
                if mode == "eager":
                    eager_median_ms             = results.get("inference_latency_median_ms")
                    results["speedup_vs_eager"] = 1.0
                status = results.get("error") or "ok"
                log.info("mode='%s' | status=%s | latency=%s ms",
                         mode, status[:60], results.get("inference_latency_median_ms"))
                all_results[mode] = results
        except KeyboardInterrupt:
            log.warning("Ctrl+C - saving partial results (%d modes completed)", len(all_results))
        if all_results:
            _write_json(os.path.join(run_dir, "results.json"), {"results": all_results})
            tex_path = generate_latex_tables(base_cfg, system_info, all_results, run_dir)
            rec_text = generate_recommendations(all_results, base_cfg)
            rec_path = os.path.join(run_dir, "recommendations.txt")
            with open(rec_path, "w", encoding="utf-8") as _rf:
                _rf.write(rec_text)
            log.info("Results saved to    : %s", run_dir)
            log.info("LaTeX tables        : %s", tex_path)
            log.info("Recommendations     : %s", rec_path)
            log.info("\n%s", rec_text)
        else:
            log.warning("No results to save.")
        sys.exit(0)

    elif is_mag:
        # ---------------------------------------------------------------
        # ogbn-mag: R-GCN heterogeneous node classification
        # ---------------------------------------------------------------
        mag = load_mag(args.data_root, device)

        if args.single_mode:
            assert len(args.modes) == 1
            mode    = args.modes[0]
            results = run_mode_mag(mode, {**base_cfg, "mode": mode},
                                   mag, device,
                                   eager_median_ms=args.eager_median_ms)
            if mode == "eager":
                results["speedup_vs_eager"] = 1.0
            print(json.dumps(_sanitise(results)))
            sys.exit(0)

        _write_json(os.path.join(run_dir, "config.json"),
                    {"timestamp": datetime.now().isoformat(),
                     "config": base_cfg, "system_info": system_info})
        modes_ordered = (["eager"] + [m for m in args.modes if m != "eager"]
                         if "eager" in args.modes else args.modes)
        all_results     = {}
        eager_median_ms = None
        try:
            for mode in modes_ordered:
                log.info("=" * 60 + "  [mag] mode='%s' ...", mode)
                results = _run_single_mode(mode, args, run_dir, eager_median_ms)
                if mode == "eager":
                    eager_median_ms             = results.get("inference_latency_median_ms")
                    results["speedup_vs_eager"] = 1.0
                status = results.get("error") or "ok"
                log.info("mode='%s' | status=%s | latency=%s ms",
                         mode, status[:60], results.get("inference_latency_median_ms"))
                all_results[mode] = results
        except KeyboardInterrupt:
            log.warning("Ctrl+C - saving partial results (%d modes completed)", len(all_results))
        if all_results:
            _write_json(os.path.join(run_dir, "results.json"), {"results": all_results})
            tex_path = generate_latex_tables(base_cfg, system_info, all_results, run_dir)
            rec_text = generate_recommendations(all_results, base_cfg)
            rec_path = os.path.join(run_dir, "recommendations.txt")
            with open(rec_path, "w", encoding="utf-8") as _rf:
                _rf.write(rec_text)
            log.info("Results saved to    : %s", run_dir)
            log.info("LaTeX tables        : %s", tex_path)
            log.info("Recommendations     : %s", rec_path)
            log.info("\n%s", rec_text)
        else:
            log.warning("No results to save.")
        sys.exit(0)

    elif is_link:
        (x, edge_index, train_edges, val_edges, test_edges,
         in_feats, link_evaluator, dgl_graph,
         val_neg_edges, test_neg_edges,
         _edge_weight) = load_link_dataset(
             args.dataset, args.data_root, device, framework=args.framework,
             model_name=args.model_name)
        num_classes = args.hidden
        labels = train_mask = val_mask = test_mask = None
        node_evaluator = None
    elif getattr(args, "use_sampling", False):
        log.info("Neighbor sampling enabled (Tier 3 dataset).")
        train_loader, data, split_idx, num_classes, dgl_graph = load_dataset_sampled(
            args.dataset, args.data_root, device, framework=args.framework,
            model_name=args.model_name)
        x, edge_index = data.x, data.edge_index
        nn_ = x.shape[0]
        train_mask = torch.zeros(nn_, dtype=torch.bool, device=device)
        val_mask   = torch.zeros(nn_, dtype=torch.bool, device=device)
        test_mask  = torch.zeros(nn_, dtype=torch.bool, device=device)
        train_mask[split_idx["train"]] = True
        val_mask[split_idx["valid"]]   = True
        test_mask[split_idx["test"]]   = True
        labels         = data.y
        in_feats       = x.shape[1]
        node_evaluator = _make_node_evaluator(args.dataset)
        train_edges = val_edges = test_edges = link_evaluator = None
        # edge_weight from GCNNorm (GCN only); baked into NeighborLoader batches.
        _edge_weight = getattr(data, "edge_weight", None)
    else:
        (x, edge_index, labels, train_mask, val_mask, test_mask,
         in_feats, num_classes, node_evaluator, dgl_graph,
         _edge_weight) = load_dataset(
             args.dataset, args.data_root, device, framework=args.framework,
             model_name=args.model_name)
        train_edges = val_edges = test_edges = None
        link_evaluator = None

    if args.framework.lower() == "dgl" and dgl_graph is None:
        log.info("Building DGL graph from edge_index (Planetoid dataset) ...")
        dgl_graph = _to_dgl_graph(edge_index, x.shape[0], device)

    # -----------------------------------------------------------------------
    # Child path: run exactly one mode, print result as JSON, exit.
    # -----------------------------------------------------------------------
    if args.single_mode:
        assert len(args.modes) == 1
        mode = args.modes[0]
        if is_link:
            results = run_mode_link(
                mode=mode, framework=args.framework, model_name=args.model_name,
                in_feats=in_feats, hidden=args.hidden,
                x=x, edge_index=edge_index,
                train_edges=train_edges, val_edges=val_edges, test_edges=test_edges,
                device=device, eager_median_ms=args.eager_median_ms,
                cfg={**base_cfg, "mode": mode}, link_evaluator=link_evaluator,
                dgl_graph=dgl_graph,
                val_neg_edges=val_neg_edges, test_neg_edges=test_neg_edges,
                edge_weight=_edge_weight)
        else:
            results = run_mode(
                mode=mode, framework=args.framework, model_name=args.model_name,
                in_feats=in_feats, hidden=args.hidden, num_classes=num_classes,
                x=x, edge_index=edge_index, labels=labels,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                device=device, eager_median_ms=args.eager_median_ms,
                cfg={**base_cfg, "mode": mode}, node_evaluator=node_evaluator,
                dgl_graph=dgl_graph,
                train_loader=train_loader if getattr(args, "use_sampling", False) else None,
                eager_train_epoch_s=getattr(args, "eager_train_epoch_s", None),
                edge_weight=_edge_weight)
        if mode == "eager":
            results["speedup_vs_eager"]       = 1.0
            results["train_speedup_vs_eager"] = 1.0
        print(json.dumps(_sanitise(results)))
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Orchestrator path: spawn one subprocess per mode, collect results.
    # -----------------------------------------------------------------------
    _write_json(os.path.join(run_dir, "config.json"),
                {"timestamp": datetime.now().isoformat(),
                 "config": base_cfg, "system_info": system_info})

    modes_ordered = (["eager"] + [m for m in args.modes if m != "eager"]
                     if "eager" in args.modes else args.modes)

    all_results:       dict[str, dict[str, Any]] = {}
    eager_median_ms:   float | None              = None
    eager_train_epoch_s: float | None            = None

    try:
        for mode in modes_ordered:
            log.info("=" * 60 + "  mode='%s' ...", mode)
            results = _run_single_mode(mode, args, run_dir, eager_median_ms,
                                       eager_train_epoch_s=eager_train_epoch_s)
            if mode == "eager":
                eager_median_ms               = results.get("inference_latency_median_ms")
                eager_train_epoch_s           = results.get("steady_state_mean_epoch_s")                                                 or results.get("mean_epoch_time_s")
                results["speedup_vs_eager"]   = 1.0
                results["train_speedup_vs_eager"] = 1.0
            status = results.get("error") or "ok"
            log.info("mode='%s' | status=%s | latency=%s ms",
                     mode, status[:60], results.get("inference_latency_median_ms"))
            all_results[mode] = results
    except KeyboardInterrupt:
        log.warning("Ctrl+C - saving partial results (%d modes completed)", len(all_results))

    if all_results:
        _write_json(os.path.join(run_dir, "results.json"), {"results": all_results})
        tex_path = generate_latex_tables(base_cfg, system_info, all_results, run_dir)
        rec_text = generate_recommendations(all_results, base_cfg)
        rec_path = os.path.join(run_dir, "recommendations.txt")
        with open(rec_path, "w", encoding="utf-8") as _rf:
            _rf.write(rec_text)
        log.info("Results saved to    : %s", run_dir)
        log.info("LaTeX tables        : %s", tex_path)
        log.info("Recommendations     : %s", rec_path)
        log.info("\n%s", rec_text)
    else:
        log.warning("No results to save.")
