import scanpy as sc
import os
adata = sc.read_h5ad("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_harmony_cellrank2.h5ad")

import matplotlib.pyplot as plt
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit"

sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15, metric="cosine")
sc.tl.umap(adata, min_dist=0.4, spread=1.0)

# 画图并保存
fig, ax = plt.subplots(figsize=(6, 6))
sc.pl.umap(
    adata,
    color="leiden_T_0.6",
    legend_loc="on data",
    frameon=False,
    ax=ax,
    show=False
)

png_path = os.path.join(outdir, "umap_harmony_leiden_T_0.6.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
