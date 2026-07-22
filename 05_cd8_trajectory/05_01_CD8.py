import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# Config (edit if needed)
# =========================
H5AD_PATH = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)

OUTDIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05_trajectory_of_T_cell/results"
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
sc.settings.figdir = OUTDIR

adata = sc.read_h5ad(H5AD_PATH)
print(adata)
sc.pl.umap(adata, color="CD8A", save="_CD8A_umap.png")

sc.pl.violin(
    adata,
    keys="CD8A",
    groupby="leiden_T_0.6",
    stripplot=False,
    rotation=45,
    multi_panel=True,
    save="_CD8A_leiden_T_0.6.png"
)

sc.pl.violin(
    adata,
    keys="CCR7",
    groupby="leiden_T_0.6",
    stripplot=False,
    rotation=45,
    multi_panel=True,
    save="_CCR7_leiden_T_0.6.png"
)

sc.pl.violin(
    adata,
    keys="IL7R",
    groupby="leiden_T_0.6",
    stripplot=False,
    rotation=45,
    multi_panel=True,
    save="_IL7R_leiden_T_0.6.png"
)


