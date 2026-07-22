#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse

# ----------------------------
# Font
# ----------------------------
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ----------------------------
# Paths
# ----------------------------
rank_csv = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "POST_drop_recluster/03_leiden_T_0.6__rank_genes_groups_wilcoxon_top50.csv"
)
outdir = os.path.dirname(rank_csv)

# 你的 reclustered h5ad（你给的路径）
h5ad_path = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)

cluster_key = "leiden_T_0.6"
top_n = 10

# 选择使用哪个表达层
preferred_layers = ["log1p_norm", "norm_expr"]
layer_to_use = None

# 让 scanpy 的 save= 输出到 outdir
sc.settings.figdir = outdir
sc.settings.autoshow = False

# ----------------------------
# Helpers
# ----------------------------
def _cluster_sort_key(x: str):
    """Sort cluster labels numerically if possible."""
    try:
        return (0, int(x))
    except Exception:
        return (1, x)

def _dedup_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _minmax_0_1_per_gene(X, eps=1e-12):
    """
    X: (n_cells, n_genes) dense ndarray
    Per-gene min-max across cells to [0,1].
    """
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    denom = np.maximum(xmax - xmin, eps)
    return (X - xmin) / denom

def _save_and_rename_scanpy_plot(prefix, suffix, final_name):
    """
    Scanpy save= typically produces files like: {prefix}{suffix} in sc.settings.figdir
    e.g., prefix="heatmap", suffix="_xxx.png" -> "heatmap_xxx.png"
    We then rename it to a clean filename.
    """
    expected = os.path.join(outdir, f"{prefix}{suffix}")
    final = os.path.join(outdir, final_name)
    if os.path.exists(expected):
        os.replace(expected, final)
        print("[INFO] Saved:", final)
    else:
        # 有些版本 scanpy 会带额外前缀/路径差异；至少告诉你它没找到
        print("[WARN] Could not find expected file:", expected)
        print("[WARN] Check outdir for generated files:", outdir)

# ----------------------------
# Load data
# ----------------------------
print("[INFO] rank_csv:", rank_csv)
print("[INFO] h5ad_path:", h5ad_path)

df = pd.read_csv(rank_csv)
adata = sc.read_h5ad(h5ad_path)

# pick layer
for ly in preferred_layers:
    if ly in adata.layers:
        layer_to_use = ly
        break
print("[INFO] Using expression from:", f"layer='{layer_to_use}'" if layer_to_use else "adata.X")

# sanity: cluster_key exists
if cluster_key not in adata.obs.columns:
    raise KeyError(f"{cluster_key} not found in adata.obs. Available keys: {adata.obs.columns.tolist()}")

# ----------------------------
# Get top genes per cluster
# ----------------------------
df["cluster"] = df["cluster"].astype(str)
df["rank"] = df["rank"].astype(int)
df["gene"] = df["gene"].astype(str)

top_df = df[df["rank"] <= top_n].copy()

clusters = sorted(top_df["cluster"].unique(), key=_cluster_sort_key)

genes_ordered = []
for c in clusters:
    genes_ordered.extend(top_df.loc[top_df["cluster"] == c, "gene"].tolist())

# 去重（强烈建议，不然热图会重复基因）
genes_ordered = _dedup_preserve_order(genes_ordered)

# 去掉 adata 里不存在的基因
genes_ordered = [g for g in genes_ordered if g in adata.var_names]
if len(genes_ordered) == 0:
    raise ValueError("None of the selected top genes are found in adata.var_names.")

print("[INFO] #clusters:", len(clusters))
print("[INFO] #genes used (dedup + present):", len(genes_ordered))

# ----------------------------
# Build a small adata subset for selected genes (memory safe)
# ----------------------------
adata_g = adata[:, genes_ordered].copy()

# Get expression matrix (cells x selected_genes) from chosen layer / X
X = adata_g.layers[layer_to_use] if (layer_to_use is not None) else adata_g.X
if sparse.issparse(X):
    X = X.toarray()
else:
    X = np.asarray(X)

# ----------------------------
# (A) Heatmap with per-gene cross-cell min-max normalization
# ----------------------------
X_mm = _minmax_0_1_per_gene(X)
adata_g.layers["minmax01"] = X_mm  # dense OK: genes only ~几十到一百

heat_suffix = f"_minmax01_top{top_n}_percluster_{cluster_key}.png"

sc.pl.heatmap(
    adata_g,
    var_names=adata_g.var_names.tolist(),
    groupby=cluster_key,
    layer="minmax01",
    use_raw=False,
    # 不再 standard_scale="var"（zscore），因为我们已经 min-max 到 0-1
    swap_axes=True,   # genes on rows
    vmin=0, vmax=1,   # 显式控制色条范围
    show=False,
    save=heat_suffix,
)

_save_and_rename_scanpy_plot(
    prefix="heatmap",
    suffix=heat_suffix,
    final_name=f"fig_heatmap_minmax01_top{top_n}_percluster_{cluster_key}.png",
)

# ----------------------------
# (B) Dotplot（建议仍用原始表达层 + zscore），更像 Seurat 的 DotPlot 信息量
# ----------------------------
dot_suffix = f"_top{top_n}_percluster_{cluster_key}.png"

sc.pl.dotplot(
    adata,  # 用原始 adata（包含完整表达层）
    var_names=genes_ordered,
    groupby=cluster_key,
    layer=layer_to_use,
    use_raw=False,
    standard_scale="var",  # 让不同基因更可比（Seurat风格常用）
    show=False,
    save=dot_suffix,
)

_save_and_rename_scanpy_plot(
    prefix="dotplot",
    suffix=dot_suffix,
    final_name=f"fig_dotplot_top{top_n}_percluster_{cluster_key}.png",
)

print("[DONE] Outputs in:", outdir)


