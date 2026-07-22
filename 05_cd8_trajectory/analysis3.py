#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CellRank2 (GPCCA) on CD8 subset clusters 0/1/2/4:
- DPT pseudotime (root in cluster 4)
- PseudotimeKernel -> transition matrix
- GPCCA macrostates + coarse_T heatmap
- terminal/initial states on UMAP
- fate probabilities (no PETSc)
- aggregate fate by cluster heatmap
- Branch analysis (soft = mean fate prob; hard = relative-advantage labels)
- Cluster 1 only: fate bias between R vs NR (heatmap + per-lineage violin)

You said you already have `ad`, but this script can load it too.
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Basic matplotlib config
# ----------------------------
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# User inputs
# ----------------------------
H5AD_PATH = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_subset_clusters_0_1_2_4.h5ad"
OUTDIR = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05c_trajectory_rerevisit/results_cellrank2_subset_0_1_2_4"
os.makedirs(OUTDIR, exist_ok=True)

CLUSTER_KEY = "leiden_T_0.6"     # values: 0/1/2/4 in this subset
RESP_KEY = "Respond"            # "Responder"/"Non-Responder"
EMBED_BASIS = "X_umap"          # use existing UMAP in adata
ROOT_CLUSTER = "4"
ROOT_STRATEGY = "top_naive"   # "min_exhaustion" or "first_cell"
EXH_KEY = "score_Exhaustion"       # only used if ROOT_STRATEGY = "min_exhaustion"
NAIVE_KEY = "score_Naive"   # 你已有的 naive 分数列名；如果没有就用下面方法计算
NAIVE_TOP_FRAC = 0.10       # 前10%
NAIVE_MIN_CAND = 30  

# GPCCA parameters
N_COMPONENTS = 20
N_STATES_RANGE = (4, 10)        # lets minChi pick
N_TERMINAL = 3                  # top_n terminal macrostates

# Branch hard-label parameters (relative advantage)
DELTA = 0.15  # fp_A - fp_B >= DELTA
EPS   = 0.25  # fp_A (or fp_B) >= EPS
 
# ----------------------------
# Helpers
# ----------------------------
def savefig(path: str):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def pick_root_cell(ad: sc.AnnData) -> int:
    """Pick a root cell index (iroot) in the *current* AnnData."""
    mask_root = (ad.obs[CLUSTER_KEY].astype(str) == str(ROOT_CLUSTER)).values
    idxs = np.where(mask_root)[0]
    if len(idxs) == 0:
        raise ValueError(f"No cells found in ROOT_CLUSTER={ROOT_CLUSTER} for {CLUSTER_KEY}.")

    if ROOT_STRATEGY == "min_exhaustion":
        if EXH_KEY not in ad.obs.columns:
            raise ValueError(f"ROOT_STRATEGY=min_exhaustion but {EXH_KEY} not in ad.obs.")
        sub = ad.obs.loc[mask_root, EXH_KEY].astype(float)
        iroot = np.where(ad.obs_names == sub.idxmin())[0][0]
        print(f"[INFO] Root cell selected by min {EXH_KEY}: {sub.idxmin()} (iroot={iroot})")
        return int(iroot)

    # fallback: first cell in root cluster
    iroot = int(idxs[0])
    print(f"[INFO] Root cell selected by first cell in cluster {ROOT_CLUSTER}: {ad.obs_names[iroot]} (iroot={iroot})")
    return iroot

def ensure_neighbors_for_dpt(ad: sc.AnnData):
    """
    DPT needs connectivities. If neighbors missing, compute from X_scVI if present,
    otherwise from PCA.
    """
    if "neighbors" in ad.uns and "connectivities" in ad.obsp:
        return

    if "X_scVI" in ad.obsm:
        print("[INFO] Computing neighbors using X_scVI.")
        sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=30)
    else:
        print("[INFO] Computing PCA + neighbors (X_pca).")
        if "X_pca" not in ad.obsm:
            sc.pp.pca(ad, n_comps=50)
        sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=30)

def safe_scanpy_violin(ad, keys, groupby, out_png, rotation=45):
    """
    scanpy violin can differ by version; do show=False then save with matplotlib.
    """
    sc.pl.violin(ad, keys=keys, groupby=groupby, show=False, rotation=rotation)
    savefig(out_png)

def infer_lineage_for_cluster(fp_df: pd.DataFrame, ad: sc.AnnData, cluster_value: str) -> str:
    """
    Find which lineage (column in fp_df) is most associated with a given cluster,
    by mean fate probability within that cluster.
    """
    mask = (ad.obs[CLUSTER_KEY].astype(str) == str(cluster_value)).values
    means = fp_df.loc[ad.obs_names[mask]].mean(axis=0)
    return str(means.idxmax())

# ----------------------------
# Load / subset
# ----------------------------
ad = sc.read_h5ad(H5AD_PATH)
print(ad)

# If this file is already subset to 0/1/2/4 you can skip. Keep it safe anyway.
keep = ad.obs[CLUSTER_KEY].astype(str).isin(["0", "1", "2", "4"]).values
ad = ad[keep].copy()
print(f"[INFO] Subset shape: {ad.shape}")
print("[INFO] Subset clusters:\n", ad.obs[CLUSTER_KEY].astype(str).value_counts())

# ----------------------------
# DPT pseudotime
# ----------------------------
ensure_neighbors_for_dpt(ad)
iroot = pick_root_cell(ad)
ad.uns["iroot"] = iroot

# Diffmap + DPT
sc.tl.diffmap(ad)
sc.tl.dpt(ad)  # uses ad.uns["iroot"]
if "dpt_pseudotime" not in ad.obs:
    raise RuntimeError("scanpy did not create ad.obs['dpt_pseudotime'].")

# Quick DPT plots
sc.pl.umap(ad, color=[CLUSTER_KEY, "dpt_pseudotime"], show=False)
savefig(os.path.join(OUTDIR, "00_umap_cluster_dpt.png"))

# ----------------------------
# CellRank2: PseudotimeKernel -> GPCCA
# ----------------------------
pk = cr.kernels.PseudotimeKernel(ad, time_key="dpt_pseudotime")
pk.compute_transition_matrix()
pk.write_to_adata()  # optional

g = cr.estimators.GPCCA(pk)

# IMPORTANT: avoid PETSc/Krylov issues
g.compute_schur(n_components=N_COMPONENTS, method="brandts")

# Fit macrostates; don't pass n_components/method here (avoids your earlier TypeError)
g.fit(cluster_key=CLUSTER_KEY, n_states=list(N_STATES_RANGE))

# (1) coarse-grained transition matrix heatmap
g.plot_coarse_T()
savefig(os.path.join(OUTDIR, "01_coarse_T_heatmap.png"))

# (2) terminal states
g.predict_terminal_states(method="top_n", n_states=N_TERMINAL)
g.plot_macrostates(which="terminal", discrete=True, legend_loc="right", s=80)
savefig(os.path.join(OUTDIR, "02_terminal_macrostates_umap.png"))

# (3) initial states: set to cluster 4 mask
g.set_initial_states(states=["4"]) # mask must be length n_obs (you fixed this already)
g.plot_macrostates(which="initial", discrete=True, legend_loc="right", s=80)
savefig(os.path.join(OUTDIR, "03_initial_macrostates_umap.png"))

# (4) fate probabilities (NO PETSc)
g.compute_fate_probabilities(use_petsc=False, solver="gmres")
g.plot_fate_probabilities(same_plot=False)
savefig(os.path.join(OUTDIR, "04_fate_probabilities_umap.png"))

# (5) aggregate fate probabilities by cluster heatmap
cr.pl.aggregate_fate_probabilities(ad, mode="heatmap", cluster_key=CLUSTER_KEY)
savefig(os.path.join(OUTDIR, "05_fate_probabilities_by_cluster_heatmap.png"))

KERNEL_KEY = "cr_pseudotime_kernel_dpt"   # 你自己起名，后面读的时候要一致
OBSP_KEY   = KERNEL_KEY                  # 通常 kernel 会把 transition_matrix 放在 ad.obsp[KEY]

# 1) 把 kernel 写入 adata（transition matrix + params）
pk.write_to_adata(key=KERNEL_KEY)

# 2) 把 GPCCA 结果写入 adata（macrostates/terminal/initial/fate probs/coarse_T 等）
#    GPCCA 在 CellRank2 里用 to_adata() 序列化进 AnnData
g.to_adata()

# 3) 保存“带 CellRank 结果的缓存 h5ad”
CACHE_H5AD = os.path.join(OUTDIR, "adata_CD8_0_1_2_4__DPT+CellRank2_cached.h5ad")
ad.write_h5ad(CACHE_H5AD, compression="gzip")
print(f"[INFO] Cached AnnData saved to: {CACHE_H5AD}")

def lineage_to_df(lineage, obs_names):
    """
    Convert cellrank Lineage object to pandas DataFrame across versions.
    Works when lineage has no .to_df().
    """
    # Newer cellrank
    if hasattr(lineage, "to_df"):
        return lineage.to_df()

    # Most common in older cellrank: lineage.X + lineage.names
    if hasattr(lineage, "X"):
        X = lineage.X
        # sparse -> dense
        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)

        cols = None
        if hasattr(lineage, "names") and lineage.names is not None:
            cols = list(lineage.names)
        elif hasattr(lineage, "lineages") and lineage.lineages is not None:
            cols = list(lineage.lineages)

        if cols is None:
            cols = [f"lineage_{i}" for i in range(X.shape[1])]

        return pd.DataFrame(X, index=obs_names, columns=cols)

    # Fallback: try numpy array protocol
    X = np.asarray(lineage)
    cols = getattr(lineage, "names", None)
    if cols is None:
        cols = [f"lineage_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=obs_names, columns=list(cols))


# ----------------------------
# Fate probabilities table
# ----------------------------
fp = g.fate_probabilities
fp_df = lineage_to_df(fp, ad.obs_names)
fp_df = fp_df.loc[ad.obs_names]  # align
fp_df.to_csv(os.path.join(OUTDIR, "fate_probabilities_per_cell.csv"))

print("[INFO] Fate probability columns:", list(fp_df.columns))

# ----------------------------
# Infer which lineage corresponds to which "branch"
# We infer:
#   A_lineage = lineage most enriched in cluster 1
#   B_lineage = lineage most enriched in (cluster 0 or 2) -> pick the better of 0 and 2
# ----------------------------
A_lineage = infer_lineage_for_cluster(fp_df, ad, "1")
B0 = infer_lineage_for_cluster(fp_df, ad, "0")
B2 = infer_lineage_for_cluster(fp_df, ad, "2")

# choose B as the one with larger mean in cluster0+2 combined
mask0 = (ad.obs[CLUSTER_KEY].astype(str) == "0").values
mask2 = (ad.obs[CLUSTER_KEY].astype(str) == "2").values
mean_B0 = fp_df.loc[ad.obs_names[mask0 | mask2], B0].mean()
mean_B2 = fp_df.loc[ad.obs_names[mask0 | mask2], B2].mean()
B_lineage = B0 if mean_B0 >= mean_B2 else B2

print(f"[INFO] A_lineage (cluster1-enriched): {A_lineage}")
print(f"[INFO] B_lineage (cluster0/2-enriched): {B_lineage}")

# ----------------------------
# Branch analysis (SOFT): mean fate by Respond
# ----------------------------
tmp = pd.DataFrame({
    "Respond": ad.obs[RESP_KEY].astype(str).values,
    "fp_A": fp_df[A_lineage].values,
    "fp_B": fp_df[B_lineage].values,
})
mean_fp = tmp.groupby("Respond")[["fp_A", "fp_B"]].mean()
mean_fp["delta_A_minus_B"] = mean_fp["fp_A"] - mean_fp["fp_B"]
mean_fp.to_csv(os.path.join(OUTDIR, "mean_fate_by_Respond.csv"))

ax = mean_fp[["fp_A", "fp_B"]].plot(kind="bar", rot=0)
ax.set_ylabel("mean fate probability")
ax.set_title(f"Mean fate probability by Respond\nA={A_lineage} (cluster1-like), B={B_lineage} (cluster0/2-like)")
plt.tight_layout()
savefig(os.path.join(OUTDIR, "06_mean_fate_by_Respond_bar.png"))

# ----------------------------
# Branch analysis (HARD): relative-advantage labels (avoids fp>=0.6 sparsity)
# ----------------------------
fpA = fp_df[A_lineage].values
fpB = fp_df[B_lineage].values

branch_rel = np.full(ad.n_obs, "other", dtype=object)
branch_rel[(fpA - fpB >= DELTA) & (fpA >= EPS)] = "A_4to1_like"
branch_rel[(fpB - fpA >= DELTA) & (fpB >= EPS)] = "B_4to2to0_like"
ad.obs["branch_rel"] = branch_rel

# UMAP plot of branch_rel
sc.pl.umap(ad, color="branch_rel", show=False)
savefig(os.path.join(OUTDIR, "07_umap_branch_rel.png"))

# Composition by Respond (within-branch fractions)
ct = pd.crosstab(ad.obs["branch_rel"], ad.obs[RESP_KEY].astype(str), normalize="index")
ct.to_csv(os.path.join(OUTDIR, "branch_rel_composition_by_Respond.csv"))

ax = ct.plot(kind="bar", rot=45)
ax.set_ylabel("fraction within branch")
ax.set_title(f"Branch(rel) composition by Respond\nDELTA={DELTA}, EPS={EPS}")
plt.tight_layout()
savefig(os.path.join(OUTDIR, "08_branch_rel_vs_Respond_bar.png"))

# Also show counts (not normalized)
ct_counts = pd.crosstab(ad.obs["branch_rel"], ad.obs[RESP_KEY].astype(str))
ct_counts.to_csv(os.path.join(OUTDIR, "branch_rel_counts_by_Respond.csv"))

# ----------------------------
# Cluster 1 only: do NR vs R fate bias plot (soft, not thresholded)
# ----------------------------
ad1 = ad[ad.obs[CLUSTER_KEY].astype(str) == "1"].copy()
fp1 = fp_df.loc[ad1.obs_names, [A_lineage, B_lineage]].copy()
tmp1 = pd.DataFrame({
    "Respond": ad1.obs[RESP_KEY].astype(str).values,
    A_lineage: fp1[A_lineage].values,
    B_lineage: fp1[B_lineage].values,
})

# Heatmap: mean fate probs in cluster1 by Respond
mean1 = tmp1.groupby("Respond")[[A_lineage, B_lineage]].mean()
mean1.to_csv(os.path.join(OUTDIR, "cluster1_mean_fate_by_Respond.csv"))

plt.figure(figsize=(6, 3))
plt.imshow(mean1.values, aspect="auto")
plt.xticks(range(mean1.shape[1]), mean1.columns, rotation=45, ha="right")
plt.yticks(range(mean1.shape[0]), mean1.index)
plt.colorbar(label="mean fate probability")
plt.title("Cluster 1: mean fate probabilities by Respond")
plt.tight_layout()
savefig(os.path.join(OUTDIR, "09_cluster1_mean_fate_heatmap.png"))

# Violin plots per lineage in cluster1 (simple)
# Put fate probs into ad1.obs for scanpy plotting
ad1.obs[f"fp_{A_lineage}"] = fp1[A_lineage].values
ad1.obs[f"fp_{B_lineage}"] = fp1[B_lineage].values

safe_scanpy_violin(
    ad1,
    keys=[f"fp_{A_lineage}"],
    groupby=RESP_KEY,
    out_png=os.path.join(OUTDIR, f"10_cluster1_violin_fp_{A_lineage}.png"),
)
safe_scanpy_violin(
    ad1,
    keys=[f"fp_{B_lineage}"],
    groupby=RESP_KEY,
    out_png=os.path.join(OUTDIR, f"11_cluster1_violin_fp_{B_lineage}.png"),
)

# ----------------------------
# Save final AnnData (with dpt + fate + branch labels)
# ----------------------------
#out_h5ad = os.path.join(OUTDIR, "adata_subset_0_1_2_4__with_dpt_cellrank_branch.h5ad")
#ad.write(out_h5ad)
#print(f"[DONE] Wrote: {out_h5ad}")
#print(f"[DONE] Figures/results in: {OUTDIR}")
