#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import palantir
import palantir.core as pal_core


# =========================
# Paths
# =========================
h5ad_in = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05b_trajectory_revisit/adata_CD8_harmony_cellrank2.h5ad"
)
outdir = os.path.dirname(h5ad_in)
os.makedirs(outdir, exist_ok=True)

h5ad_out = os.path.join(outdir, "adata_CD8_palantir.h5ad")

fig_dm_pt = os.path.join(outdir, "palantir_diffmap_pseudotime.png")
fig_dm_ent = os.path.join(outdir, "palantir_diffmap_entropy.png")
fig_dm_bp = os.path.join(outdir, "palantir_diffmap_branch_probs_top4.png")
fig_bp_scatter = os.path.join(outdir, "palantir_branchprob_scatter_top2.png")


# =========================
# Load
# =========================
adata = sc.read_h5ad(h5ad_in)
print("[INFO] loaded:", adata)


# =========================
# Ensure we have a low-dim manifold for Palantir
# Palantir expects diffusion components (DCs) as input.
# We'll compute diffmap from your Harmony PCA if present, else PCA.
# =========================
rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
if rep not in adata.obsm:
    raise KeyError("Need adata.obsm['X_pca_harmony'] or ['X_pca'] to proceed.")

# neighbors + diffmap (if missing)
if "X_diffmap" not in adata.obsm:
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=30, n_pcs=min(50, adata.obsm[rep].shape[1]))
    sc.tl.diffmap(adata)

# build diffusion map DF for palantir
dm = pd.DataFrame(adata.obsm["X_diffmap"], index=adata.obs_names)
dm.columns = [f"DC{i+1}" for i in range(dm.shape[1])]
print("[INFO] diffmap shape:", dm.shape)


# =========================
# Choose an "early cell" (root) robustly
# - If you have a known starting cluster (e.g. leiden_T_0.6 == "4"), pick the most central in Harmony PCA.
# - Otherwise pick the global central cell in Harmony PCA.
# =========================
cluster_col = "leiden_T_0.6"
start_cell = None

X = adata.obsm[rep]
if cluster_col in adata.obs.columns:
    mask = (adata.obs[cluster_col].astype(str).values == "4")
    idx = np.where(mask)[0]
    if len(idx) > 0:
        mu = X[idx].mean(axis=0, keepdims=True)
        start_cell = adata.obs_names[idx[np.argmin(((X[idx] - mu) ** 2).sum(axis=1))]]

if start_cell is None:
    mu = X.mean(axis=0, keepdims=True)
    start_cell = adata.obs_names[np.argmin(((X - mu) ** 2).sum(axis=1))]

print("[INFO] start_cell:", start_cell)


# =========================
# Run Palantir (latest API)
# Returns: PResults with pseudotime, entropy, branch_probs, waypoints
# =========================
pr_res = pal_core.run_palantir(dm, early_cell=start_cell)

# Sanity check: required attrs in latest docs
need = ["pseudotime", "entropy", "branch_probs", "waypoints"]
print("[INFO] PResults has:", [k for k in need if hasattr(pr_res, k)])

# Write back to adata.obs (pseudotime + entropy)
adata.obs["palantir_pseudotime"] = pr_res.pseudotime.reindex(adata.obs_names).values
adata.obs["palantir_entropy"] = pr_res.entropy.reindex(adata.obs_names).values

# Branch probabilities is a DataFrame: cells x terminal_states
bp = pr_res.branch_probs.reindex(adata.obs_names)
print("[INFO] branch_probs shape:", bp.shape)
print("[INFO] terminal states:", list(bp.columns))

# Store top terminals into obs (all terminals can be large; keep as obs columns only for top few)
# We keep ALL terminals in obsm as a dense array + column names in uns for reproducibility.
adata.obsm["palantir_branch_probs"] = bp.values
adata.uns["palantir_branch_probs_cols"] = [str(c) for c in bp.columns]

# Also store a few as obs for easy plotting
topk = min(8, bp.shape[1])
for c in list(bp.columns[:topk]):
    adata.obs[f"palantir_bp_{c}"] = bp[c].values


# =========================
# Plot 1: Diffmap colored by Palantir pseudotime
# =========================
sc.pl.diffmap(
    adata,
    components=["1,2"],
    color="palantir_pseudotime",
    show=False,
    title="Palantir pseudotime (Diffmap DC1/DC2)",
    color_map="viridis",
    size=8,
)
plt.savefig(fig_dm_pt, dpi=200, bbox_inches="tight")
plt.close()
print("[OK] saved:", fig_dm_pt)

# Plot 2: Diffmap colored by entropy (plasticity/uncertainty)
sc.pl.diffmap(
    adata,
    components=["1,2"],
    color="palantir_entropy",
    show=False,
    title="Palantir entropy (Diffmap DC1/DC2)",
    color_map="viridis",
    size=8,
)
plt.savefig(fig_dm_ent, dpi=200, bbox_inches="tight")
plt.close()
print("[OK] saved:", fig_dm_ent)

# Plot 3: Diffmap colored by top 4 branch probabilities (if available)
plot_terms = list(bp.columns[:min(4, bp.shape[1])])
if len(plot_terms) > 0:
    sc.pl.diffmap(
        adata,
        components=["1,2"],
        color=[f"palantir_bp_{t}" for t in plot_terms if f"palantir_bp_{t}" in adata.obs.columns],
        show=False,
        title="Palantir branch probabilities (top terminals)",
        color_map="viridis",
        size=8,
    )
    plt.savefig(fig_dm_bp, dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", fig_dm_bp)

# Optional: if there are >=2 terminal states, make the “two-branch evidence” scatter:
if bp.shape[1] >= 2:
    t1, t2 = bp.columns[0], bp.columns[1]
    x = bp[t1].values
    y = bp[t2].values
    plt.figure(figsize=(5, 5), dpi=220)
    plt.scatter(x, y, s=3, alpha=0.35)
    plt.xlabel(f"Branch prob → {t1}")
    plt.ylabel(f"Branch prob → {t2}")
    plt.title("Branch bifurcation evidence (Palantir)")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(fig_bp_scatter, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", fig_bp_scatter)

# Save
adata.write_h5ad(h5ad_out, compression="gzip")
print("[OK] saved:", h5ad_out)
print("[DONE]")
