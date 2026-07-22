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
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=10, n_pcs=20)
sc.tl.diffmap(adata)
print(adata.obsm['X_diffmap'])
sc.pl.diffmap(adata, dimensions=(0, 1), color="leiden_T_0.6", save="_DC1_DC2_leiden_T_0.6.png")
sc.pl.diffmap(adata, dimensions=(0, 1), color="patient",      save="_DC1_DC2_patient.png")



