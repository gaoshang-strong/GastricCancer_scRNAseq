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

# ====== 0) 确保已有 UMAP；没有就用 scVI latent 重新算 ======
if "X_umap" not in adata.obsm_keys():
    if "X_scVI" in adata.obsm_keys():
        sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    else:
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)

# ====== 1) 选择用哪个 layer 来上色（优先 log1p_norm） ======
layer_to_use = None
for candidate in ["log1p_norm", "norm_expr"]:
    if candidate in adata.layers:
        layer_to_use = candidate
        break
# 如果 layer_to_use 还是 None：就用 adata.X

# ====== 2) T cell marker genes（你可以按需要增删） ======
tcell_markers = [
    # pan-T
    "CD3D", "CD3E", "CD247", "TRAC", "LCK",
    # CD4 / naïve-memory
    "IL7R", "CCR7", "LTB", "MALAT1",
    # CD8 / cytotoxic
    "CD8A", "CD8B", "NKG7", "GZMB", "PRF1", "GNLY",
    # exhaustion / checkpoint
    "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2",
    # proliferation
    "MKI67", "TOP2A",
]

# ====== 3) 处理“基因名如何匹配”的问题 ======
# 如果 adata.var_names 里就有这些基因，直接用；
# 否则如果有 adata.var["gene_symbol"]，就用 gene_symbols 参数让 Scanpy 自动映射
present_in_varnames = sum(g in adata.var_names for g in tcell_markers)

gene_symbols_arg = None
color_list = tcell_markers

if present_in_varnames < int(0.6 * len(tcell_markers)):
    # 大概率 var_names 不是 gene symbol
    if "gene_symbol" in adata.var.columns:
        gene_symbols_arg = "gene_symbol"
    else:
        # 最后兜底：手动映射 gene_symbol -> var_name
        # (如果你没有 gene_symbol 列，会走到这里并报错更早暴露问题)
        raise ValueError("Neither adata.var_names contains gene symbols nor adata.var['gene_symbol'] exists.")

# 过滤掉不存在的基因（避免画图时报错）
# Scanpy 在 gene_symbols 模式下，会根据 adata.var[gene_symbols_arg] 是否存在来找
if gene_symbols_arg is None:
    genes_exist = [g for g in color_list if g in adata.var_names]
else:
    gs = adata.var[gene_symbols_arg].astype(str).values
    gs_set = set(gs)
    genes_exist = [g for g in color_list if g in gs_set]

missing = sorted(set(color_list) - set(genes_exist))
if len(missing) > 0:
    print(f"[WARN] These markers are missing and will be skipped: {missing}")

# ====== 4) 画 UMAP（连续表达量上色），并保存 ======
# vmax='p99' 能避免极端高表达把色阶拉爆
# use_raw=False 强制用当前 X/layer
sc.pl.umap(
    adata,
    color=genes_exist,
    gene_symbols=gene_symbols_arg,   # None or "gene_symbol"
    layer=layer_to_use,              # None / "log1p_norm" / "norm_expr"
    use_raw=False,
    ncols=4,
    wspace=0.35,
    frameon=False,
    vmax="p99",
    show=False,
)

fname = f"04_UMAP_Tcell_markers_{'layer_'+layer_to_use if layer_to_use else 'X'}.png"
fpath = os.path.join(outdir, fname)
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fpath)
