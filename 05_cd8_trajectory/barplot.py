#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# -----------------------------
# User params (edit these 3)
# -----------------------------
H5AD_PATH = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_harmony_cellrank2.h5ad"
CLUSTER_KEY = "leiden_T_0.6"   # e.g. "leiden_T_0.6"
RESP_KEY = "Respond"          # e.g. "Respond" (values like "Responder"/"Non-responder" or "R"/"NR")

OUTDIR = os.path.dirname(H5AD_PATH)
OUTPNG1 = os.path.join(OUTDIR, "barplot_cluster0124_by_Respond_stacked.png")
OUTPNG2 = os.path.join(OUTDIR, "barplot_Respond_by_cluster0124_stacked.png")

CLUSTERS = ["0", "1", "2", "4"]  # clusters of interest, as strings

# -----------------------------
# Load
# -----------------------------
adata = sc.read_h5ad(H5AD_PATH)

if CLUSTER_KEY not in adata.obs.columns:
    raise KeyError(f"Missing {CLUSTER_KEY} in adata.obs. Available: {list(adata.obs.columns)[:30]} ...")
if RESP_KEY not in adata.obs.columns:
    raise KeyError(f"Missing {RESP_KEY} in adata.obs. Available: {list(adata.obs.columns)[:30]} ...")

# Make sure they are strings/categorical (safe)
adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype(str)
adata.obs[RESP_KEY] = adata.obs[RESP_KEY].astype(str)

# Keep only clusters of interest
ad = adata[adata.obs[CLUSTER_KEY].isin(CLUSTERS)].copy()
if ad.n_obs == 0:
    raise ValueError(f"No cells left after filtering clusters {CLUSTERS}. Check CLUSTER_KEY={CLUSTER_KEY} values.")

# -----------------------------
# 1) Within each Respond group: cluster proportions
#    rows = Respond group, cols = cluster, values = proportion
# -----------------------------
ct = pd.crosstab(ad.obs[RESP_KEY], ad.obs[CLUSTER_KEY])
ct = ct.reindex(columns=CLUSTERS, fill_value=0)

prop = ct.div(ct.sum(axis=1), axis=0)  # normalize row-wise
# Sort Respond groups to keep a stable order (optional)
prop = prop.sort_index()

print("[INFO] Counts table (Respond x cluster):")
print(ct)
print("\n[INFO] Proportion table (Respond x cluster):")
print(prop)

# Plot stacked bar: each Respond group is a bar, stacks are clusters
fig, ax = plt.subplots(figsize=(8, 5))
prop.plot(kind="bar", stacked=True, ax=ax, width=0.75, edgecolor="none")
ax.set_xlabel(RESP_KEY)
ax.set_ylabel("Proportion within Respond group")
ax.set_title(f"Cluster composition within each {RESP_KEY} group\n(clusters {','.join(CLUSTERS)}; key={CLUSTER_KEY})")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(OUTPNG1, dpi=300)
plt.close(fig)
print(f"[INFO] Saved: {OUTPNG1}")

# -----------------------------
# 2) Within each cluster: Respond proportions (alternative view)
#    rows = cluster, cols = Respond group, values = proportion
# -----------------------------
ct2 = pd.crosstab(ad.obs[CLUSTER_KEY], ad.obs[RESP_KEY])
ct2 = ct2.reindex(index=CLUSTERS, fill_value=0)

prop2 = ct2.div(ct2.sum(axis=1), axis=0)  # normalize row-wise
# keep Respond columns stable
prop2 = prop2[sorted(prop2.columns)]

print("\n[INFO] Counts table (cluster x Respond):")
print(ct2)
print("\n[INFO] Proportion table (cluster x Respond):")
print(prop2)

fig, ax = plt.subplots(figsize=(8, 5))
prop2.plot(kind="bar", stacked=True, ax=ax, width=0.75, edgecolor="none")
ax.set_xlabel("Cluster")
ax.set_ylabel("Proportion within cluster")
ax.set_title(f"{RESP_KEY} composition within each cluster\n(clusters {','.join(CLUSTERS)}; key={CLUSTER_KEY})")
ax.legend(title=RESP_KEY, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(OUTPNG2, dpi=300)
plt.close(fig)
print(f"[INFO] Saved: {OUTPNG2}")
