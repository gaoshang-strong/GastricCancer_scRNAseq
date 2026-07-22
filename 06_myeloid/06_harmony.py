#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harmony re-processing for macrophages_adata.h5ad

Inputs:
  - /ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/06_myeloid_hypothesis/macrophages_adata.h5ad
    layers['counts'] is raw counts

Outputs (all written under OUTDIR):
  - macrophages_harmony.h5ad  (with X_pca_harmony, UMAP, Leiden clusters)
  - UMAPs colored by patient/Respond/leiden_harmony
  - patient cluster composition tables + stacked barplot
  - marker dotplot by harmony clusters
  - one-vs-rest DE CSVs for each harmony cluster + overview plots
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
H5AD_IN = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "06_myeloid_hypothesis/macrophages_adata.h5ad"
)

OUTDIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "06_myeloid_hypothesis/results_macrophages_harmony"
)
os.makedirs(OUTDIR, exist_ok=True)
sc.settings.figdir = OUTDIR

# obs keys
PATIENT_KEY = "patient"
RESP_KEY = "Respond"  # we will create it from patient id list

# Harmony batch key
BATCH_KEY = PATIENT_KEY

# Harmony/graph/cluster params
N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 30
LEIDEN_RES = 0.6
CLUSTER_KEY = "leiden_harmony"

# Non-responders
NR_PATIENTS = {"PHD001", "PHD002", "PHD008"}

# =========================
# Plot style
# =========================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]

# =========================
# Helpers
# =========================
def keep_present(ad, genes):
    return [g for g in genes if g in ad.var_names]

def ensure_harmonypy():
    """
    Make sure we imported the correct single-cell harmonypy (slowkow/harmonypy),
    not the unrelated harmony-py (NASA) package.
    """
    import harmonypy  # noqa
    if not hasattr(harmonypy, "run_harmony"):
        raise ImportError(
            "Imported 'harmonypy' but it has no attribute 'run_harmony'.\n"
            "This usually means you installed/imported the wrong package.\n"
            "Fix: micromamba install -c conda-forge harmonypy\n"
            "Then verify: python -c \"import harmonypy; print(harmonypy.__file__); print(hasattr(harmonypy,'run_harmony'))\""
        )
    return harmonypy

def stacked_bar_by_patient(ad, outdir, title):
    df = ad.obs[[PATIENT_KEY, CLUSTER_KEY]].copy()
    tab = pd.crosstab(df[PATIENT_KEY], df[CLUSTER_KEY])
    tab_frac = tab.div(tab.sum(axis=1), axis=0)

    tab.to_csv(os.path.join(outdir, "table_cluster_counts_by_patient.csv"))
    tab_frac.to_csv(os.path.join(outdir, "table_cluster_fraction_by_patient.csv"))

    plt.figure(figsize=(10, 4))
    tab_frac.plot(kind="bar", stacked=True, ax=plt.gca(), legend=True)
    plt.ylabel("Fraction within patient (macrophages)")
    plt.xlabel("Patient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bar_cluster_fraction_by_patient.png"), dpi=300)
    plt.close()

# =========================
# Load
# =========================
adata = sc.read_h5ad(H5AD_IN)
print(adata)

assert PATIENT_KEY in adata.obs.columns, f"Missing obs['{PATIENT_KEY}']"
assert "counts" in adata.layers, "Expected raw counts in layers['counts']"

adata.obs[PATIENT_KEY] = adata.obs[PATIENT_KEY].astype(str)

# Create Respond label from patient IDs
adata.obs[RESP_KEY] = np.where(
    adata.obs[PATIENT_KEY].isin(NR_PATIENTS),
    "Non-responder",
    "Responder",
).astype(str)

print("Respond counts:\n", adata.obs[RESP_KEY].value_counts())
print("Patient x Respond:\n", pd.crosstab(adata.obs[PATIENT_KEY], adata.obs[RESP_KEY]))

# =========================
# Preprocess from raw counts
# =========================
# Use raw counts as X
adata.X = adata.layers["counts"].copy()

# Normalize + log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store log1p normalized values for DE/plotting (ALL genes kept)
adata.layers["log1p_norm_harmony"] = adata.X.copy()

# Set .raw to log1p normalized (optional but common)
adata.raw = adata

# HVG selection (batch-aware), DO NOT subset genes (keep full gene set for DE later)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_HVG,
    flavor="seurat_v3",
    batch_key=BATCH_KEY,
    subset=False,
)

# Scale only affects adata.X; keep adata.X as log1p_norm_harmony for now
# We'll scale only for PCA (on HVGs).
adata.X = adata.layers["log1p_norm_harmony"].copy()
sc.pp.scale(adata, max_value=10)

# PCA using HVGs only
sc.tl.pca(adata, n_comps=N_PCS, use_highly_variable=True, svd_solver="arpack")
print("DEBUG X_pca shape:", np.asarray(adata.obsm["X_pca"]).shape)

# =========================
# Harmony (robust, manual write-back to obsm)
# =========================
hm = ensure_harmonypy()

X_pca = np.asarray(adata.obsm["X_pca"])
assert X_pca.ndim == 2, f"X_pca is not 2D: {X_pca.shape}"
assert X_pca.shape[0] == adata.n_obs, f"X_pca rows != n_obs: {X_pca.shape[0]} vs {adata.n_obs}"

meta = adata.obs[[BATCH_KEY]].copy()

ho = hm.run_harmony(X_pca, meta, vars_use=[BATCH_KEY])

Z = np.asarray(ho.Z_corr)
print("DEBUG Z_corr shape:", Z.shape)

assert Z.ndim == 2, f"Z_corr is not 2D: {Z.shape}"

# harmonypy typically returns (n_pcs, n_obs) -> transpose
if Z.shape == (X_pca.shape[1], X_pca.shape[0]):
    X_h = Z.T
elif Z.shape == X_pca.shape:
    X_h = Z
else:
    raise ValueError(
        f"Unexpected Z_corr shape={Z.shape}, expected {X_pca.shape} or {(X_pca.shape[1], X_pca.shape[0])}."
    )

adata.obsm["X_pca_harmony"] = X_h
print("DEBUG X_pca_harmony shape:", adata.obsm["X_pca_harmony"].shape)

# =========================
# Neighbors / UMAP / Leiden on Harmony space
# =========================
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=N_NEIGHBORS)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added=CLUSTER_KEY, resolution=LEIDEN_RES)

# Save integrated object
out_h5ad = os.path.join(OUTDIR, "macrophages_harmony.h5ad")
adata.write_h5ad(out_h5ad, compression="gzip")
print("Saved:", out_h5ad)

# =========================
# UMAP figures
# =========================
sc.pl.umap(
    adata,
    color=[PATIENT_KEY, RESP_KEY, CLUSTER_KEY],
    wspace=0.4,
    save="_umap_patient_respond_cluster.png",
)

# =========================
# Patient composition (cluster fractions)
# =========================
stacked_bar_by_patient(
    adata,
    OUTDIR,
    title="Harmony macrophage clusters: composition per patient",
)

# =========================
# Marker dotplot by Harmony clusters (for annotation)
# =========================
marker_sets = {
    "TAM_C1QC_APOE_TREM2": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "TYROBP", "LST1"],
    "MRC1_CD163": ["MRC1", "CD163", "MSR1", "MARCO", "FOLR2"],
    "Inflammation_NFkB": ["IL1B", "TNF", "CXCL8", "NFKBIA", "TNFAIP3", "NFKB2", "RELB", "PTGS2"],
    "Antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"],
    "Suppress_ligands": ["CD274", "PDCD1LG2", "LGALS9", "VSIR", "IDO1", "IL10", "TGFB1", "CD276"],
}
marker_union = sorted({g for gs in marker_sets.values() for g in gs})
marker_union = keep_present(adata, marker_union)

with open(os.path.join(OUTDIR, "marker_sets_used.txt"), "w") as f:
    for name, gs in marker_sets.items():
        f.write(f"{name}:\n{','.join(keep_present(adata, gs))}\n\n")

sc.pl.dotplot(
    adata,
    var_names=marker_union,
    groupby=CLUSTER_KEY,
    standard_scale="var",
    save="_dotplot_markers_by_harmony_cluster.png",
)

# =========================
# One-vs-rest DE for Harmony clusters (on ALL genes)
# =========================
# IMPORTANT: DE should run on log1p normalized expression (all genes).
adata.X = adata.layers["log1p_norm_harmony"].copy()

sc.tl.rank_genes_groups(
    adata,
    groupby=CLUSTER_KEY,
    method="wilcoxon",
    pts=True,
)

# Overview plots
sc.pl.rank_genes_groups(
    adata,
    n_genes=30,
    sharey=False,
    save="_rank_genes_top30.png",
)
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=10,
    save="_rank_genes_dotplot_top10.png",
)

# Save per-cluster CSV
clusters = sorted(
    adata.obs[CLUSTER_KEY].astype(str).unique(),
    key=lambda x: int(x) if x.isdigit() else x
)

for cl in clusters:
    df_cl = sc.get.rank_genes_groups_df(adata, group=cl)
    df_cl.to_csv(os.path.join(OUTDIR, f"DE_{CLUSTER_KEY}_cluster{cl}_vs_rest_wilcoxon.csv"), index=False)

print("DONE. All outputs in:", OUTDIR)
