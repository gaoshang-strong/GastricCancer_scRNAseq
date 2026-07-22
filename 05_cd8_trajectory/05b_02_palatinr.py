#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
h5ad_in = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05b_trajectory_revisit/adata_CD8_palantir.h5ad"   # <- 改成你的 palantir 输出文件
)
outdir = os.path.dirname(h5ad_in)
os.makedirs(outdir, exist_ok=True)

out_png = os.path.join(outdir, "palantir_branchprob_scatter_colored_by_cluster.png")

cluster_col = "leiden_T_0.6"   # 你要按这个上色


# =========================
# Load
# =========================
adata = sc.read_h5ad(h5ad_in)
print("[INFO] loaded:", adata)

if cluster_col not in adata.obs.columns:
    raise KeyError(f"Missing adata.obs['{cluster_col}'].")

if "palantir_branch_probs" not in adata.obsm or "palantir_branch_probs_cols" not in adata.uns:
    raise KeyError("Missing palantir results: need adata.obsm['palantir_branch_probs'] and adata.uns['palantir_branch_probs_cols'].")

bp = pd.DataFrame(
    adata.obsm["palantir_branch_probs"],
    index=adata.obs_names,
    columns=[str(x) for x in adata.uns["palantir_branch_probs_cols"]],
)

if bp.shape[1] < 2:
    raise ValueError(f"Need at least 2 terminal states, got {bp.shape[1]}")

# ✅ 默认复现你之前那张图：用前两个 terminal
t1, t2 = bp.columns[0], bp.columns[1]
print("[INFO] using terminals:")
print("  x =", t1)
print("  y =", t2)

x = bp[t1].values
y = bp[t2].values


# =========================
# Color by cluster
# =========================
cl = adata.obs[cluster_col].astype(str).values
cats = np.unique(cl)

# 给 cluster 分配颜色：用 scanpy 的 vega_20（够用且好看）
palette = sc.pl.palettes.vega_20
color_map = {c: palette[i % len(palette)] for i, c in enumerate(sorted(cats))}
colors = np.array([color_map[c] for c in cl])

# =========================
# Point size: emphasize branch arms
#   - bigger when close to either branch end
# =========================
conf = np.maximum(x, y)   # 越接近某个终点越大
sizes = 6 + 80 * (conf ** 1.6)   # 你可以调：底座6，放大80，指数1.2~2.0


# =========================
# Plot
# =========================
plt.figure(figsize=(7.2, 7.2), dpi=250)

plt.scatter(
    x, y,
    s=sizes,
    c=colors,
    alpha=0.75,
    linewidths=0,
)

plt.xlabel(f"Branch prob → {t1}")
plt.ylabel(f"Branch prob → {t2}")
plt.title("Branch bifurcation evidence (Palantir) — colored by cluster")

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)

# Legend（按 cluster）
# 为了不让 legend 太大：按 cluster 类别画“空点”当句柄
handles = []
labels = []
for c in sorted(cats):
    h = plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=color_map[c], markeredgewidth=0)
    handles.append(h)
    labels.append(c)

plt.legend(handles, labels, title=cluster_col, frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(out_png, bbox_inches="tight")
plt.close()

print("[OK] saved:", out_png)
