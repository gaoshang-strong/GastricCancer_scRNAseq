import matplotlib as mpl
import scanpy as sc
import os
import re
import scanpy as sc
import matplotlib.pyplot as plt
# ----------------------------
# Font
# ----------------------------
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

h5ad_path = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/02_adata_T_reclustered_after_drop.h5ad"  # TODO: 改成你的 h5ad
adata = sc.read_h5ad(h5ad_path)

# =========================
# 2) cluster key (auto-detect)
# =========================
preferred_key = "leiden_T_0.6"
if preferred_key in adata.obs.columns:
    cluster_key = preferred_key
else:
    candidates = [c for c in adata.obs.columns if "leiden" in c.lower()]
    if len(candidates) == 0:
        raise ValueError(
            "Cannot find cluster key. Please set cluster_key manually. "
            f"Available obs columns: {adata.obs.columns.tolist()[:50]}..."
        )
    cluster_key = candidates[0]
print(f"[Info] Using cluster_key = {cluster_key}")

# =========================
# 3) Drop cluster 8 (data-level removal)
# =========================
drop_cluster = "8"
before_n = adata.n_obs
mask_keep = adata.obs[cluster_key].astype(str) != drop_cluster
adata = adata[mask_keep].copy()
after_n = adata.n_obs
print(f"[Info] Dropped cluster {drop_cluster}: {before_n} -> {after_n} cells")

# =========================
# 4) Use layer = norm_expr (NOT raw)
# =========================
layer_name = "norm_expr"
if layer_name not in adata.layers.keys():
    raise ValueError(
        f"layers['{layer_name}'] not found. Available layers: {list(adata.layers.keys())}"
    )
use_raw = False  # 强制不用 raw

# =========================
# 5) OUTPUT ROOT
# =========================
out_root = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_T_cell_subtype_signature_genes"
)
os.makedirs(out_root, exist_ok=True)

# =========================
# 6) cluster -> representative genes
# =========================
cluster2genes = {
    "0": ["CCL5", "IFNG", "GZMH", "GZMA", "CD8A", "THEMIS", "RGS1", "CD69", "ID2", "TOX"],
    "1": ["GZMK", "NKG7", "CTSW", "CST7", "CCL4", "KLRG1", "GZMB", "GZMH", "CRTAM", "AOAH"],
    "2": ["ITGAE", "ENTPD1", "CD160", "ABCB1", "ZEB2", "NR3C1", "CAMK4", "CMIP", "DAPK2", "PTPN22"],
    "3": ["FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "TNFRSF4", "ICOS", "BATF", "MAF", "CCR6", "ADAM12"],
    "4": ["IL7R", "ANXA1", "PLCB1", "FOS", "JUN", "FOSB", "ZFP36", "DUSP1", "VIM", "KLF6"],
    "5": ["TCF7", "CCR7", "LTB", "CD28", "IL6ST", "AFF3", "TSHZ2", "FAAH2", "SLC9A9", "TNFSF8"],
    "6": ["MKI67", "TOP2A", "STMN1", "TYMS", "PCLAF", "RRM2", "BIRC5", "TUBA1B", "HMGB2", "NUSAP1"],
    "7": ["ZBTB16", "KLRB1", "SLC4A10", "NCR3", "IL4I1", "IL7R", "RORA", "STAT4", "PHACTR2", "KLRG1"],
    # 8 已删除，不再画
}

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\.-]+", "_", str(s))

def gene_exists(g: str) -> bool:
    return g in adata.var_names

def save_gene_violin(gene: str, out_png: str):
    sc.pl.violin(
        adata,
        keys=gene,
        groupby=cluster_key,
        use_raw=use_raw,
        layer=layer_name,
        stripplot=False,
        show=False,
    )
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# 7) MAIN LOOP
# =========================
missing = []
saved = 0

for clust, genes in cluster2genes.items():
    clust_dir = os.path.join(out_root, str(clust))
    os.makedirs(clust_dir, exist_ok=True)

    for g in genes:
        if not gene_exists(g):
            missing.append((clust, g))
            continue

        out_png = os.path.join(clust_dir, f"VIOLIN_{safe_name(g)}.png")
        save_gene_violin(g, out_png)
        saved += 1

print(f"[Done] saved {saved} violin plots into:\n  {out_root}")

if missing:
    print("\n[Warn] These genes were not found in adata.var_names:")
    for clust, g in missing:
        print(f"  cluster {clust}: {g}")