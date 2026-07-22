#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
H5AD_PATH = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "01_mapping_raw_scRNA_seq_to_reference/adata_scvi_integrated_all_cells.h5ad"
)

BASE_DIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "06_myeloid_hypothesis"
)
OUT_MONO = os.path.join(BASE_DIR, "results_Monocytes")
OUT_MAC = os.path.join(BASE_DIR, "results_Macrophages")
os.makedirs(OUT_MONO, exist_ok=True)
os.makedirs(OUT_MAC, exist_ok=True)

OUT_H5AD_MONO = os.path.join(BASE_DIR, "monocytes_adata.h5ad")
OUT_H5AD_MAC = os.path.join(BASE_DIR, "macrophages_adata.h5ad")

CELLTYPE_KEY = "majority_voting"
PATIENT_KEY = "patient"
RESP_KEY = "Respond"  # we will create this

NR_PATIENTS = {"PHD001", "PHD002", "PHD008"}  # non-responders

# Clustering
N_NEIGHBORS = 30
LEIDEN_RES = 0.6
CLUSTER_KEY = "leiden_myeloid"

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

def save_patient_cluster_barplot(ad, outdir, title_prefix):
    df = ad.obs[[PATIENT_KEY, RESP_KEY, CLUSTER_KEY]].copy()
    tab = pd.crosstab(df[PATIENT_KEY], df[CLUSTER_KEY])
    tab_frac = tab.div(tab.sum(axis=1), axis=0)

    tab.to_csv(os.path.join(outdir, f"table_{title_prefix}_cluster_counts_by_patient.csv"))
    tab_frac.to_csv(os.path.join(outdir, f"table_{title_prefix}_cluster_fraction_by_patient.csv"))

    plt.figure(figsize=(10, 4))
    tab_frac.plot(kind="bar", stacked=True, ax=plt.gca(), legend=True)
    plt.ylabel("Fraction within patient")
    plt.xlabel("Patient")
    plt.title(f"{title_prefix}: cluster composition per patient")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bar_{title_prefix}_cluster_fraction_by_patient.png"), dpi=300)
    plt.close()

def run_one_vs_rest_DE(ad, outdir, prefix):
    # use log1p_norm if available
    ad_de = ad.copy()
    if "log1p_norm" in ad_de.layers:
        ad_de.X = ad_de.layers["log1p_norm"]
    sc.tl.rank_genes_groups(ad_de, groupby=CLUSTER_KEY, method="wilcoxon", pts=True)

    # save overview plots
    sc.settings.figdir = outdir
    sc.pl.rank_genes_groups(ad_de, n_genes=30, sharey=False, save=f"_{prefix}_rank_genes_top30.png")
    sc.pl.rank_genes_groups_dotplot(ad_de, n_genes=10, save=f"_{prefix}_rank_genes_dotplot_top10.png")

    # save per-cluster CSV
    clusters = sorted(ad_de.obs[CLUSTER_KEY].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    for cl in clusters:
        df_cl = sc.get.rank_genes_groups_df(ad_de, group=cl)
        df_cl.to_csv(os.path.join(outdir, f"DE_{prefix}_cluster{cl}_vs_rest_wilcoxon.csv"), index=False)

def annotate_marker_dotplot(ad, outdir, prefix, marker_sets):
    # marker_sets: dict(name->list genes)
    marker_union = sorted({g for gs in marker_sets.values() for g in gs})
    marker_union = keep_present(ad, marker_union)
    sc.settings.figdir = outdir

    sc.pl.dotplot(
        ad,
        var_names=marker_union,
        groupby=CLUSTER_KEY,
        standard_scale="var",
        save=f"_{prefix}_dotplot_markers_by_cluster.png",
    )

    # write which genes used
    with open(os.path.join(outdir, f"{prefix}_marker_sets_used.txt"), "w") as f:
        for name, gs in marker_sets.items():
            f.write(f"{name}:\n{','.join(keep_present(ad, gs))}\n\n")

def process_one_major_type(adata, major_type, outdir, out_h5ad, prefix):
    # subset
    ad = adata[adata.obs[CELLTYPE_KEY] == major_type].copy()
    print(f"{major_type} subset:", ad.shape)
    assert "X_scVI" in ad.obsm, f"Missing X_scVI in {major_type} subset."

    # neighbors/umap/leiden on scVI
    sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=N_NEIGHBORS)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=LEIDEN_RES, key_added=CLUSTER_KEY)

    # save h5ad
    ad.write_h5ad(out_h5ad, compression="gzip")
    print("Saved:", out_h5ad)

    # basic umaps
    sc.settings.figdir = outdir
    sc.pl.umap(ad, color=[CLUSTER_KEY, PATIENT_KEY, RESP_KEY], wspace=0.4, save=f"_{prefix}_umap_overview.png")

    # patient distribution
    save_patient_cluster_barplot(ad, outdir, title_prefix=prefix)

    # marker dotplot (type-specific)
    if major_type == "Monocytes":
        marker_sets = {
            "Classical_mono": ["S100A8", "S100A9", "FCN1", "LYZ", "LGALS3", "CTSS", "LST1"],
            "Nonclassical_mono": ["FCGR3A", "MS4A7", "LST1", "IFITM3", "CTSS", "LGALS3"],
            "Inflammation_NFkB": ["IL1B", "TNF", "CXCL8", "NFKBIA", "TNFAIP3", "NFKB2", "RELB"],
            "Antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"],
            "Suppress_ligands": ["CD274", "PDCD1LG2", "LGALS9", "VSIR", "IDO1", "IL10", "TGFB1"],
        }
    else:
        marker_sets = {
            "TAM_C1QC_APOE_TREM2": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "TYROBP", "LST1"],
            "MRC1_CD163": ["MRC1", "CD163", "MSR1", "MARCO", "FOLR2"],
            "Inflammation_NFkB": ["IL1B", "TNF", "CXCL8", "NFKBIA", "TNFAIP3", "NFKB2", "RELB", "PTGS2"],
            "Antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"],
            "Suppress_ligands": ["CD274", "PDCD1LG2", "LGALS9", "VSIR", "IDO1", "IL10", "TGFB1", "CD276"],
        }
    annotate_marker_dotplot(ad, outdir, prefix, marker_sets)

    # one-vs-rest DE per myeloid cluster
    run_one_vs_rest_DE(ad, outdir, prefix)

    return ad

# =========================
# Main
# =========================
adata = sc.read_h5ad(H5AD_PATH)
print(adata)

for k in [CELLTYPE_KEY, PATIENT_KEY]:
    assert k in adata.obs.columns, f"Missing adata.obs['{k}']"

adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].astype(str)
adata.obs[PATIENT_KEY] = adata.obs[PATIENT_KEY].astype(str)

# create Respond label
adata.obs[RESP_KEY] = np.where(
    adata.obs[PATIENT_KEY].isin(NR_PATIENTS),
    "Non-responder",
    "Responder",
).astype(str)

print("Respond counts:\n", adata.obs[RESP_KEY].value_counts())
print("Patient x Respond:\n", pd.crosstab(adata.obs[PATIENT_KEY], adata.obs[RESP_KEY]))

# process separately
ad_mono = process_one_major_type(
    adata=adata,
    major_type="Monocytes",
    outdir=OUT_MONO,
    out_h5ad=OUT_H5AD_MONO,
    prefix="Monocytes",
)

ad_mac = process_one_major_type(
    adata=adata,
    major_type="Macrophages",
    outdir=OUT_MAC,
    out_h5ad=OUT_H5AD_MAC,
    prefix="Macrophages",
)

print("DONE.")
print("Monocytes results:", OUT_MONO)
print("Macrophages results:", OUT_MAC)
print("Monocytes h5ad:", OUT_H5AD_MONO)
print("Macrophages h5ad:", OUT_H5AD_MAC)
