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
# 1) Settings
# =========================
cluster_key = "leiden_T_0.6"  # 你的cluster列
drop_cluster = "8"            # 删除cluster 8
outdir = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_T_cell_subtype_signature_genes/QC"
)
os.makedirs(outdir, exist_ok=True)

# =========================
# 2) Drop cluster 8
# =========================
if cluster_key not in adata.obs.columns:
    raise ValueError(f"{cluster_key} not found in adata.obs. Available: {list(adata.obs.columns)[:50]}")

before_n = adata.n_obs
adata = adata[adata.obs[cluster_key].astype(str) != str(drop_cluster)].copy()
after_n = adata.n_obs
print(f"[Info] Dropped cluster {drop_cluster}: {before_n} -> {after_n}")

# 固定 cluster 顺序（0..7），避免画图乱序
clusters_sorted = sorted(adata.obs[cluster_key].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
adata.obs[cluster_key] = (
    adata.obs[cluster_key].astype(str).astype("category").cat.set_categories(clusters_sorted, ordered=True)
)

# =========================
# 3) Ensure QC metrics exist (compute if missing)
# =========================
# 常见列名：total_counts, n_genes_by_counts, pct_counts_mt
need_cols = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]

missing = [c for c in need_cols if c not in adata.obs.columns]
if missing:
    print(f"[Warn] Missing QC columns in obs: {missing}. Will compute via sc.pp.calculate_qc_metrics().")

    # 标记线粒体基因（人一般是 MT- 开头）
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    # calculate_qc_metrics 会生成：total_counts, n_genes_by_counts, pct_counts_mt 等
    print("[Info] QC metrics computed. Now available:",
          [c for c in need_cols if c in adata.obs.columns])

# 最终确认
for c in need_cols:
    if c not in adata.obs.columns:
        raise ValueError(f"QC column '{c}' still not found after computation.")

# =========================
# 4) Plot QC violins across clusters
# =========================
qc_to_plot = {
    "total_counts": "UMI (total_counts)",
    "n_genes_by_counts": "Detected genes (n_genes_by_counts)",
    "pct_counts_mt": "Mito percent (pct_counts_mt)",
}

for col, title in qc_to_plot.items():
    sc.pl.violin(
        adata,
        keys=col,
        groupby=cluster_key,
        stripplot=False,
        show=False,
        rotation=0,
    )
    plt.title(title)
    out_png = os.path.join(outdir, f"QC_VIOLIN_{col}_by_{cluster_key}_drop{drop_cluster}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print("[Saved]", out_png)

# =========================
# 5) (Optional) Focus: only show clusters 0 and 2 in a separate plot
# =========================
focus_clusters = {"0", "2"}
mask_focus = adata.obs[cluster_key].astype(str).isin(focus_clusters)
ad_focus = adata[mask_focus].copy()

# 保持顺序：0,2
ad_focus.obs[cluster_key] = (
    ad_focus.obs[cluster_key].astype(str).astype("category").cat.set_categories(["0", "2"], ordered=True)
)

for col, title in qc_to_plot.items():
    sc.pl.violin(
        ad_focus,
        keys=col,
        groupby=cluster_key,
        stripplot=True,   # 只画0和2时可以开散点，更直观
        jitter=0.25,
        show=False,
        rotation=0,
    )
    plt.title(f"{title} (focus clusters 0 vs 2)")
    out_png = os.path.join(outdir, f"QC_VIOLIN_{col}_focus_0_vs_2_drop{drop_cluster}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print("[Saved]", out_png)

print(f"[Done] QC plots saved in:\n  {outdir}")