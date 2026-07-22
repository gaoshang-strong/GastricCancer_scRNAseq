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
# ====== 输出目录（按你之前的习惯） ======
all_cells_path = "01_mapping_raw_scRNA_seq_to_reference/adata_scvi_integrated_all_cells.h5ad"
adata = sc.read_h5ad(all_cells_path)
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/01_mapping_raw_scRNA_seq_to_reference/results/"
os.makedirs(outdir, exist_ok=True)

celltype_markers = ["CD3D", "CD3E", "CD247", "TRAC", "LCK"]

# 0) 确保已有 UMAP；没有就算一下（优先 scVI）
if "X_umap" not in adata.obsm_keys():
    if "X_scVI" in adata.obsm_keys():
        sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    else:
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)

# 1) 选择用于表达量上色的 layer（优先 log1p_norm）
layer_to_use = None
for candidate in ["log1p_norm", "norm_expr"]:
    if candidate in adata.layers:
        layer_to_use = candidate
        break

# 2) 决定 gene symbol 映射方式
# 如果 var_names 不是 gene symbol，且有 adata.var["gene_symbol"]，用 gene_symbols 参数
gene_symbols_arg = None
if not all(g in adata.var_names for g in celltype_markers):
    if "gene_symbol" in adata.var.columns:
        gene_symbols_arg = "gene_symbol"
    else:
        raise ValueError("Markers not found in adata.var_names and adata.var['gene_symbol'] is missing.")

# 3) 过滤不存在的基因，避免报错
if gene_symbols_arg is None:
    markers_exist = [g for g in celltype_markers if g in adata.var_names]
else:
    gs = set(adata.var[gene_symbols_arg].astype(str).values)
    markers_exist = [g for g in celltype_markers if g in gs]

missing = sorted(set(celltype_markers) - set(markers_exist))
if missing:
    print("[WARN] Missing markers skipped:", missing)

# 4) 逐个基因画图并保存（每张独立）
for g in markers_exist:
    sc.pl.umap(
        adata,
        color=g,
        gene_symbols=gene_symbols_arg,
        layer=layer_to_use,
        use_raw=False,
        frameon=False,
        vmax="p99",
        show=False,
    )
    fname = f"04_UMAP_marker_{g}_{'layer_'+layer_to_use if layer_to_use else 'X'}.png"
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fpath)