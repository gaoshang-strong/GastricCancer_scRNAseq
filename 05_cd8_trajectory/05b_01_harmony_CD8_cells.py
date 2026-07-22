#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import harmonypy as hm
import cellrank as cr

# =========================
# Plot style
# =========================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]

# ----------------------------
# Paths
# ----------------------------
h5ad_in = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05b_trajectory_revisit/adata_CD8_subset_clusters_0_1_2_4.h5ad"
)
outdir = os.path.dirname(h5ad_in)
os.makedirs(outdir, exist_ok=True)

h5ad_out = os.path.join(outdir, "adata_CD8_harmony_cellrank2.h5ad")

# Figures
png_comp_cluster = os.path.join(outdir, "cellrank_macrostates_composition_over_cluster.png")
png_comp_respond = os.path.join(outdir, "cellrank_macrostates_composition_over_Respond.png")
png_umap_macrostates = os.path.join(outdir, "umap_macrostates.png")
png_umap_respond = os.path.join(outdir, "umap_Respond.png")
png_umap_cluster = os.path.join(outdir, "umap_cluster.png")
png_umap_texscore = os.path.join(outdir, "umap_score_TexTRM.png")
png_fate_probs = os.path.join(outdir, "cellrank_fate_probabilities.png")
png_coarse_T = os.path.join(outdir, "cellrank_coarse_T.png")
png_macrostates_all = os.path.join(outdir, "cellrank_macrostates_all.png")

# ----------------------------
# Load
# ----------------------------
adata = sc.read_h5ad(h5ad_in)
print("[INFO] loaded:", adata)

# ----------------------------
# Labels
# ----------------------------
NR_PATIENTS = {"PHD001", "PHD002", "PHD008"}

if "patient" not in adata.obs.columns:
    raise KeyError(f"Missing adata.obs['patient']. Available: {list(adata.obs.columns)}")

adata.obs["Respond"] = pd.Categorical(
    np.where(adata.obs["patient"].isin(NR_PATIENTS), "Non-responder", "Responder"),
    categories=["Responder", "Non-responder"],
    ordered=False
)

resp_col = "Respond"
cluster_col = "leiden_T_0.6"
if cluster_col not in adata.obs.columns:
    raise KeyError(f"Missing adata.obs['{cluster_col}']. Available: {list(adata.obs.columns)}")

# make sure categorical
adata.obs[cluster_col] = pd.Categorical(adata.obs[cluster_col].astype(str))
adata.obs["patient"] = pd.Categorical(adata.obs["patient"].astype(str))

# ----------------------------
# Use counts layer
# ----------------------------
if "counts" not in adata.layers.keys():
    raise KeyError(f"Missing adata.layers['counts']. Available layers: {list(adata.layers.keys())}")

adata.X = adata.layers["counts"].copy()

# ----------------------------
# Preprocess -> PCA
# ----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# IMPORTANT: root scoring (optional) should be done BEFORE HVG subsetting.
# We don't need root for CellRank2 connectivity kernel, so we skip it here.

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=3000,
    batch_key="patient",
    subset=True
)

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

X_pca = adata.obsm["X_pca"]
meta = adata.obs[["patient"]].copy()

# ----------------------------
# Harmony (direct harmonypy)
# ----------------------------
ho = hm.run_harmony(X_pca, meta, vars_use=["patient"])

Z_corr = None
if hasattr(ho, "Z_corr"):
    Z_corr = ho.Z_corr
elif hasattr(ho, "result") and hasattr(ho.result, "Z_corr"):
    Z_corr = ho.result.Z_corr

if Z_corr is None:
    raise RuntimeError("Cannot find Z_corr in Harmony result. harmonypy API may have changed.")

# harmonypy: Z_corr is usually (n_pcs, n_cells) -> transpose
if Z_corr.shape[0] == X_pca.shape[1] and Z_corr.shape[1] == X_pca.shape[0]:
    X_pca_harmony = Z_corr.T
elif Z_corr.shape == X_pca.shape:
    X_pca_harmony = Z_corr
else:
    raise ValueError(
        f"Unexpected Harmony Z_corr shape: {Z_corr.shape}, expected {(X_pca.shape[1], X_pca.shape[0])} or {X_pca.shape}"
    )

adata.obsm["X_pca_harmony"] = X_pca_harmony
print("[INFO] X_pca_harmony:", adata.obsm["X_pca_harmony"].shape)

# ----------------------------
# Neighbors on Harmony
# ----------------------------
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, n_pcs=50)

# ----------------------------
# CellRank2: ConnectivityKernel + GPCCA
# ----------------------------
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

g = cr.estimators.GPCCA(ck)

# n_states: start from 6; if too split/unstable try 4-5
g.fit(cluster_key=cluster_col, n_states=6)

print("macrostates:", list(g.macrostates.cat.categories))
print("initial_states (auto):", g.initial_states)
print("terminal_states (auto):", g.terminal_states)

# Write macrostates into adata.obs for downstream plotting/stats
adata.obs["macrostates"] = pd.Categorical(g.macrostates)

# ----------------------------
# Save macrostate composition plots (cluster / Respond)
# ----------------------------
def save_macrostate_composition(key: str, out_png: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    plt.sca(ax)
    ret = g.plot_macrostate_composition(key, title=title)
    # robustly pick a figure
    if hasattr(ret, "figure") and ret.figure is not None:
        fig_to_save = ret.figure
    elif hasattr(ret, "savefig"):
        fig_to_save = ret
    else:
        fig_to_save = fig
    fig_to_save.tight_layout()
    fig_to_save.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig_to_save)
    print("[OK] saved:", out_png)

save_macrostate_composition(cluster_col, png_comp_cluster, f"Macrostates composition over {cluster_col}")
save_macrostate_composition(resp_col, png_comp_respond, "Macrostates composition over Respond")

# ----------------------------
# Score Tex/TRM markers to decide 2_1 vs 2_2
# ----------------------------
tex_markers = ["ENTPD1", "ITGAE", "TOX", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "CXCL13"]
tex_present = [gname for gname in tex_markers if gname in adata.var_names]
if len(tex_present) >= 2:
    sc.tl.score_genes(adata, tex_present, score_name="score_TexTRM")
    print(
        "[INFO] median score_TexTRM by macrostate:\n",
        adata.obs.groupby("macrostates")["score_TexTRM"].median().sort_values(ascending=False)
    )
else:
    print("[WARN] Too few Tex/TRM markers found in var_names. Found:", tex_present)

# ----------------------------
# Manually set initial/terminal states
# ----------------------------
# Root: 4
g.set_initial_states(states=["4"])

# Terminal: pick cytotoxic (0) + Tex/TRM (2_2)
# If later you find 2_1 is more Tex-like, change to ["0","2_1"] or include both.
g.set_terminal_states(states=["0", "2_2"])

# Compute fate probabilities
g.compute_fate_probabilities()

# ----------------------------
# UMAP for visualization
# ----------------------------
sc.tl.umap(adata)

# basic UMAP plots
sc.pl.umap(adata, color="macrostates", show=False, title="UMAP: macrostates")
plt.savefig(png_umap_macrostates, dpi=200, bbox_inches="tight"); plt.close()
print("[OK] saved:", png_umap_macrostates)

sc.pl.umap(adata, color=resp_col, show=False, title=f"UMAP: {resp_col}")
plt.savefig(png_umap_respond, dpi=200, bbox_inches="tight"); plt.close()
print("[OK] saved:", png_umap_respond)

sc.pl.umap(adata, color=cluster_col, show=False, title=f"UMAP: {cluster_col}")
plt.savefig(png_umap_cluster, dpi=200, bbox_inches="tight"); plt.close()
print("[OK] saved:", png_umap_cluster)

if "score_TexTRM" in adata.obs.columns:
    sc.pl.umap(adata, color="score_TexTRM", show=False, title="UMAP: score_TexTRM")
    plt.savefig(png_umap_texscore, dpi=200, bbox_inches="tight"); plt.close()
    print("[OK] saved:", png_umap_texscore)

# ----------------------------
# CellRank2 plots that are stable across versions
# ----------------------------
# 1) Fate probabilities (multi-panel)
# Some versions return a Figure, others draw on current figure; handle both.
fig = g.plot_fate_probabilities(same_plot=False)
plt.savefig(png_fate_probs, dpi=200, bbox_inches="tight")
plt.close()
print("[OK] saved:", png_fate_probs)

# 2) Coarse-grained transitions among macrostates
fig = g.plot_coarse_T()
plt.savefig(png_coarse_T, dpi=200, bbox_inches="tight")
plt.close()
print("[OK] saved:", png_coarse_T)

# 3) Macrostates plot
fig = g.plot_macrostates(which="all")
plt.savefig(png_macrostates_all, dpi=200, bbox_inches="tight")
plt.close()
print("[OK] saved:", png_macrostates_all)

# ----------------------------
# Optional: plot lineage probabilities on UMAP
# Try to access where they are stored; different versions store differently.
# We'll attempt a few safe ways and only plot if recognized.
# ----------------------------
lineages = None
if "lineages_fwd" in adata.obsm:
    lineages = adata.obsm["lineages_fwd"]

# If lineages is a pandas DataFrame, easiest:
if lineages is not None and hasattr(lineages, "columns"):
    for col in lineages.columns:
        adata.obs[f"fate_{col}"] = np.asarray(lineages[col]).astype(float)
        out_png = os.path.join(outdir, f"umap_fate_{col}.png")
        sc.pl.umap(adata, color=f"fate_{col}", show=False, title=f"UMAP: fate {col}")
        plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()
        print("[OK] saved:", out_png)
else:
    print("[INFO] adata.obsm['lineages_fwd'] is not a DataFrame; skipping per-lineage UMAP plots.")

# ----------------------------
# Save AnnData
# ----------------------------
adata.write_h5ad(h5ad_out, compression="gzip")
print("[OK] saved:", h5ad_out)
print("[DONE]")
