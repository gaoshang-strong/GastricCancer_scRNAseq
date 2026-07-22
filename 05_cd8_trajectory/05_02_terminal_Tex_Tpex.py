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
te_genes = ["NKG7","PRF1","GZMB","GNLY","FGFBP2","IFNG","CX3CR1","KLRD1","KLRC1","ZEB2","TBX21"]
tpex_genes = ["TCF7","LEF1","IL7R","CCR7","LTB","SLAMF6","CXCR5","ICOS"]

# 只用在 adata.var_names 里存在的基因
te_genes = [g for g in te_genes if g in adata.var_names]
tpex_genes = [g for g in tpex_genes if g in adata.var_names]

sc.tl.score_genes(adata, te_genes, score_name="score_Te")
sc.tl.score_genes(adata, tpex_genes, score_name="score_Tpex")

# 画一下检查（diffmap / umap 都行）
sc.pl.umap(adata, color=["score_Te","score_Tpex"], vmin=-0.5, vmax=0.5, save="_Te_Tpex_umap.png")