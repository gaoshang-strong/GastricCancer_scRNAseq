import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
all_cells_path = "01_mapping_raw_scRNA_seq_to_reference/adata_scvi_integrated_all_cells.h5ad"
all_cells = sc.read_h5ad(all_cells_path)
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/01_mapping_raw_scRNA_seq_to_reference/results/"

print(all_cells)
non_resp = {"PHD001", "PHD002", "PHD008"}
resp = {"PHD003", "PHD004", "PHD009"}
mapping = {p: "Non-Responder" for p in non_resp}
mapping.update({p: "Responder" for p in resp})

all_cells.obs["Respond"] = all_cells.obs["patient"].astype(str).map(mapping).fillna("Unknown")

# 假设你的对象叫 adata
# sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, fontsize=10)

to_plot = [
    ("majority_voting", "01_UMAP_majority_voting.png"),
    ("predicted_labels", "01_UMAP_predicted_labels.png"),
    ("patient", "01_UMAP_patient.png"),
#    ("Respond", "01_UMAP_Respond.png"),
]

for key, fname in to_plot:
    sc.pl.umap(
        all_cells,
        color=key,
        frameon=False,
        show=False,   # 关键：不弹出图，方便脚本跑完直接存
    )
    plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
    plt.close()

print("Saved to:", outdir)
sc.pl.umap(
        all_cells,
        color="Respond",
        frameon=False,
        show=False,   # 关键：不弹出图，方便脚本跑完直接存
        alpha=0.5,
    )
plt.savefig(os.path.join(outdir, '01_UMAP_Respond.png'), dpi=300, bbox_inches="tight")
plt.close()


import os, scanpy as sc, matplotlib.pyplot as plt

marker = {
    "B cells": ["MS4A1","CD79A","CD74","HLA-DRA","CD37"],
    "Endothelial cells": ["PECAM1","VWF","KDR","ENG","RAMP2"],
    "Epithelial cells": ["EPCAM","KRT19","KRT8","KRT18","MSLN"],
    "Fibroblasts": ["COL1A1","COL1A2","DCN","LUM","COL3A1"],
    "Macrophages": ["CD68","ITGAM","CD163","CSF1R","APOE"],
    "Mast cells": ["TPSAB1","TPSB2","GATA2","CPA3","MS4A2"],
    "Monocytes": ["S100A8","S100A9","FCN1"],
    "Plasma cells": ["XBP1","SDC1","JCHAIN"],
    "T cells": ["CD3D","CD3E","TRAC","LCK","CD247"],
}

use_layer = "log1p_norm"
var_names = [g for genes in marker.values() for g in genes if g in set(all_cells.var_names)]

# 删除 DC / ILC 细胞（按 majority_voting 标签过滤）
plot_ad = all_cells[~all_cells.obs["majority_voting"].isin(["DC", "ILC"])].copy()

sc.pl.dotplot(plot_ad, var_names=var_names, groupby="majority_voting",
              layer=use_layer, standard_scale="var", show=False)
plt.savefig(os.path.join(outdir, "01_marker_dotplot_noDC_noILC_log1p_norm.png"), dpi=300, bbox_inches="tight")
plt.close()

sc.pl.matrixplot(plot_ad, var_names=var_names, groupby="majority_voting",
                 layer=use_layer, standard_scale="var", show=False)
plt.savefig(os.path.join(outdir, "01_marker_heatmap_noDC_noILC_log1p_norm.png"), dpi=300, bbox_inches="tight")
plt.close()

import os, scanpy as sc, matplotlib.pyplot as plt

mp = sc.pl.matrixplot(
    plot_ad,
    var_names=var_names,
    groupby="majority_voting",
    layer="log1p_norm",
    standard_scale="var",
    cmap="viridis",
    figsize=(16, 4),
    show=False,
)

# mp 是 dict：兼容不同 scanpy 版本的 key
ax = mp.get("mainplot_ax", mp.get("heatmap_ax", mp.get("ax", None)))
if ax is None:
    # 兜底：拿第一个 axis
    ax = next(iter(mp.values()))

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

# colorbar（如果存在）
cax = mp.get("colorbar_ax", None)
if cax is not None:
    cax.tick_params(labelsize=14)
    cax.set_title(cax.get_title(), fontsize=14)

plt.savefig(os.path.join(outdir, "marker_heatmap_bigfont.png"), dpi=300, bbox_inches="tight")
plt.close()

