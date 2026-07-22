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
# -----------------------------
# Settings
# -----------------------------
CLUSTER_KEY = "leiden_T_0.6"
KEEP = ["0", "1", "2", "4", "7"]

outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_trajectory_of_T_cell/results_subset_CD8_T_cells"
os.makedirs(outdir, exist_ok=True)
sc.settings.figdir = outdir

# -----------------------------
# Subset
# -----------------------------
# make sure it's string
adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype(str)

mask = adata.obs[CLUSTER_KEY].isin(KEEP)
adata_sub = adata[mask].copy()

# optional: store which clusters kept
adata_sub.uns["subset_info"] = {
    "parent_adata": "adata",
    "cluster_key": CLUSTER_KEY,
    "kept_clusters": KEEP,
    "n_obs_parent": int(adata.n_obs),
    "n_obs_subset": int(adata_sub.n_obs),
}

sc.pp.neighbors(adata_sub, use_rep="X_scVI", n_neighbors=30)
sc.tl.umap(adata_sub)

print("Subset done.")
print("Parent:", adata.shape)
print("Subset:", adata_sub.shape)
print(adata_sub.obs[CLUSTER_KEY].value_counts().sort_index())

# -----------------------------
# Save
# -----------------------------
#out_h5ad = os.path.join(outdir, f"subset_{CLUSTER_KEY}__{'-'.join(KEEP)}.h5ad")
#adata_sub.write_h5ad(out_h5ad, compression="gzip")
#print("Saved:", out_h5ad)

cytotoxic_genes = [
    "NKG7", "PRF1", "GZMB", "GNLY", "CTSW", "GZMH", "GZMK"
]

exhaustion_genes = [
    "PDCD1", "TIGIT", "LAG3", "HAVCR2", "TOX", "ENTPD1"
]

# Keep only genes present in var_names
cytotoxic_genes = [g for g in cytotoxic_genes if g in adata_sub.var_names]
exhaustion_genes = [g for g in exhaustion_genes if g in adata_sub.var_names]

sc.tl.score_genes(
    adata_sub,
    gene_list=cytotoxic_genes,
    score_name="score_cytotoxicity",
    use_raw=False,
)
sc.tl.score_genes(
    adata_sub,
    gene_list=exhaustion_genes,
    score_name="score_exhaustion",
    use_raw=False,
)


sc.pl.umap(adata_sub, color=["leiden_T_0.6", "score_cytotoxicity", "score_exhaustion"], wspace=0.4, save="_CD8_subset_scores.png")

sc.tl.diffmap(adata_sub)

sc.pl.diffmap(adata_sub, color=["leiden_T_0.6", "score_cytotoxicity", "score_exhaustion"], wspace=0.4, save="_CD8_subset_scores_diffmap.png")

ad = adata_sub.copy()
ad.X = ad.layers["log1p_norm"]
ad.obs["patient"] = ad.obs["patient"].astype(str)
sc.pp.highly_variable_genes(
    ad,
    n_top_genes=2000,
    flavor="seurat_v3",
    batch_key="patient",
    subset=False,
)

sc.pp.scale(ad, max_value=10)
sc.tl.pca(ad, n_comps=50, use_highly_variable=True)

# -----------------------------
# 4) Neighbors + Diffusion map
# -----------------------------
sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=30, n_pcs=30)
sc.tl.diffmap(ad)

# -----------------------------
# 5) Quick plots (optional)
# -----------------------------
#sc.pl.pca_variance_ratio(ad, n_pcs=50, log=True, save="_hvg_pca_var_ratio.png")

sc.pl.diffmap(
    ad,
    color=["leiden_T_0.6", "score_cytotoxicity", "score_exhaustion"],
    wspace=0.4,
    save="_CD8_subset_scores_diffmap_HVG_PCA.png",
)
sc.pl.diffmap(
    ad,
    color=["leiden_T_0.6", "patient"],
    wspace=0.4,
    save="_CD8_subset_scores_diffmap_patient.png",
)

CLUSTER_KEY = "leiden_T_0.6"
ROOT_CLUSTER = "4"

# cluster 4 的候选细胞
cand = np.where(ad.obs[CLUSTER_KEY].astype(str).values == ROOT_CLUSTER)[0]
assert len(cand) > 0, "No cells found in root cluster 4"

# 在候选里选一个：cytotoxicity + exhaustion 最低的
combo = ad.obs["score_cytotoxicity"].values[cand] + ad.obs["score_exhaustion"].values[cand]
root_idx = cand[np.argmin(combo)]

ad.uns["iroot"] = int(root_idx)
print("iroot set to:", ad.uns["iroot"], "cluster:", ROOT_CLUSTER)

sc.tl.dpt(ad, n_dcs=10)

# 画图看看方向是否合理
sc.pl.diffmap(ad, color=["dpt_pseudotime", CLUSTER_KEY, "score_cytotoxicity", "score_exhaustion"], wspace=0.4, save="_CD8_subset_scores_diffmap_DPT.png")

ck = cr.kernels.ConnectivityKernel(ad)
pk = cr.kernels.PseudotimeKernel(ad, time_key="dpt_pseudotime")

# 计算各自的转移矩阵
ck.compute_transition_matrix()
pk.compute_transition_matrix()

# 融合（权重可调；0.2~0.5 常见）
kernel = 0.3 * pk + 0.7 * ck
kernel.compute_transition_matrix()

g = cr.estimators.GPCCA(kernel)
g.compute_schur(n_components=20)          # 可先用 20；不行再调大/调小
g.compute_macrostates(n_states=2)         # 你这subset里终点不多，6够起步

# 自动打终点（也可以手动）
g.predict_terminal_states()
g.compute_fate_probabilities(
    solver="direct",
    use_petsc=False,
)

# 把 fate prob 写回 ad.obs 方便画图
g.plot_fate_probabilities(same_plot=False, show=False)
plt.savefig(os.path.join(outdir, "cellrank2_fate_probabilities.png"), bbox_inches="tight", dpi=300)
plt.close()

# 2) Terminal states（终点标注）
g.plot_macrostates(which="terminal", basis="umap")  # basis="diffmap" 也行
plt.savefig(os.path.join(outdir, "cellrank2_terminal_macrostates.png"), bbox_inches="tight", dpi=300)
plt.close()

