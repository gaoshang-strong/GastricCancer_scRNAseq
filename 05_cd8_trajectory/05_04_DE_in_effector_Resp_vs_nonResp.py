from scipy import sparse
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import cellrank as cr

# =========================
# Config (edit if needed)
# =========================
H5AD_PATH = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]

adata = sc.read_h5ad(H5AD_PATH)
print(adata)
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_trajectory_of_T_cell/results_DE_in_effector_Resp_vs_nonResp"
os.makedirs(outdir, exist_ok=True)
sc.settings.figdir = outdir

CLUSTER_KEY = "leiden_T_0.6"
EFFECTOR_CLUSTER = "1"          # 你定义的 effector
GROUP_KEY = "Respond"           # 你已有：Responder / Non-responder（或 True/False）

# 1) subset effector
ad_eff = adata[adata.obs[CLUSTER_KEY].astype(str) == EFFECTOR_CLUSTER].copy()
print("Effector cells:", ad_eff.n_obs)

# 2) 用哪个表达矩阵做 DE
# 推荐用 log1p_norm（更常规），如果你有 norm_expr 也可用
ad_eff.X = ad_eff.layers["norm_expr"]

# 3) DE
sc.tl.rank_genes_groups(
    ad_eff,
    groupby=GROUP_KEY,
    method="wilcoxon",
    pts=True,
)

# 4) 导出结果
df = sc.get.rank_genes_groups_df(ad_eff, group="Responder")  # 如果你的标签叫 Responder
# 如果标签不是 "Responder"，先 print(ad_eff.obs[GROUP_KEY].unique())
df.to_csv(os.path.join(outdir, "DE_effector_Responder_vs_NonResponder_wilcoxon.csv"), index=False)

# 5) 可视化：top genes dotplot/heatmap/violin
sc.pl.rank_genes_groups(ad_eff, n_genes=30, sharey=False, save="_effector_R_vs_NR_top30.png")
sc.pl.rank_genes_groups_dotplot(ad_eff, n_genes=15, save="_effector_R_vs_NR_dotplot.png")
sc.pl.rank_genes_groups_heatmap(ad_eff, n_genes=20, show_gene_labels=True, save="_effector_R_vs_NR_heatmap.png")
