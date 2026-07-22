#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import scanpy as sc

# ----------------------------
# Input / Output
# ----------------------------
h5ad_in = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)

outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit"
h5ad_out = os.path.join(outdir, "adata_CD8_subset_clusters_0_1_2_4.h5ad")

# CD8 clusters (as strings; most Leiden labels are strings)
cd8_clusters = {"0", "1", "2", "4"}

# ----------------------------
# Load
# ----------------------------
adata = sc.read_h5ad(h5ad_in)

# ----------------------------
# Pick the clustering column automatically
# ----------------------------
candidate_cols = [
    "leiden_T_0.6",      # most likely in your pipeline
    "leiden",            # common
    "leiden_scvi",       # if you clustered in scVI latent
    "over_clustering",   # sometimes used
]

cluster_col = None
for c in candidate_cols:
    if c in adata.obs.columns:
        cluster_col = c
        break

if cluster_col is None:
    raise KeyError(
        f"Cannot find a clustering column in adata.obs. "
        f"Tried: {candidate_cols}. Available columns: {list(adata.obs.columns)}"
    )

# ----------------------------
# Subset CD8 clusters
# ----------------------------
cl = adata.obs[cluster_col].astype(str)
mask = cl.isin(cd8_clusters)

adata_cd8 = adata[mask].copy()

# (Optional) keep the cluster column in a standardized name
adata_cd8.obs["CD8_cluster_source"] = cluster_col
adata_cd8.obs["CD8_cluster_label"] = adata_cd8.obs[cluster_col].astype(str)

# ----------------------------
# Save
# ----------------------------
adata_cd8.write_h5ad(h5ad_out, compression="gzip")

print(f"[OK] cluster_col used: {cluster_col}")
print(f"[OK] Saved CD8 subset: {h5ad_out}")
print(f"[OK] CD8 subset shape: {adata_cd8.n_obs} cells × {adata_cd8.n_vars} genes")
print("[OK] CD8 cluster counts:")
print(adata_cd8.obs[cluster_col].value_counts().sort_index())
