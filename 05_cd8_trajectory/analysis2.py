import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Config
# ----------------------------
H5AD = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_subset_clusters_0_1_2_4.h5ad"
OUTDIR = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05c_trajectory_rerevisit/DE_cluster4_cluster1_R_vs_NR"
os.makedirs(OUTDIR, exist_ok=True)

CLUSTER_KEY = "leiden_T_0.6"   # 如果你这个 h5ad 里不是这个列名，改这里
RESP_KEY = "Respond"          # 或 "respond"（看你 obs 里实际列名）
TARGET_CLUSTERS = ["4", "1", "2", "0"]  # cluster4 & cluster1

# 画图字体（可删）
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ----------------------------
# Load
# ----------------------------
adata = sc.read_h5ad(H5AD)
print(adata)

# 用 log1p_norm 做 DE（更合适）；如果没有就用当前 X
if "log1p_norm" in adata.layers:
    adata.X = adata.layers["log1p_norm"]
elif "norm_expr" in adata.layers:
    adata.X = adata.layers["norm_expr"]

# ----------------------------
# Helper: unify Respond labels
# ----------------------------
# 统一成两个组名：Responder / Non-Responder
# 兼容你之前出现过的 "Responders" / "Non-responders"
resp = adata.obs[RESP_KEY].astype(str).copy()
resp = resp.replace({
    "Responders": "Responder",
    "Non-responders": "Non-Responder",
    "Non-responders ": "Non-Responder",
    "Non-Responder ": "Non-Responder",
})
adata.obs[RESP_KEY] = pd.Categorical(resp)

print("Respond counts:\n", adata.obs[RESP_KEY].value_counts())

# 只保留这两组（避免别的标签污染）
keep_mask = adata.obs[RESP_KEY].isin(["Responder", "Non-Responder"])
adata = adata[keep_mask].copy()

# ----------------------------
# DE per cluster
# ----------------------------
for cl in TARGET_CLUSTERS:
    ad = adata[adata.obs[CLUSTER_KEY].astype(str) == str(cl)].copy()
    print(f"\n[Cluster {cl}] shape:", ad.shape)
    print(ad.obs[RESP_KEY].value_counts())

    # 必须两组都存在
    if ad.obs[RESP_KEY].nunique() < 2:
        print(f"[WARN] Cluster {cl}: only one group present, skip.")
        continue

    # rank genes
    sc.tl.rank_genes_groups(
        ad,
        groupby=RESP_KEY,
        groups=["Responder"],          # Responder vs reference
        reference="Non-Responder",
        method="wilcoxon",
        pts=True,
        tie_correct=True,
    )

    # save table
    df = sc.get.rank_genes_groups_df(ad, group="Responder")
    out_csv = os.path.join(OUTDIR, f"cluster{cl}_DE_Responder_vs_NonResponder.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    # plots
    # 1) rank plot
    sc.pl.rank_genes_groups(
        ad,
        n_genes=30,
        sharey=False,
        show=False
    )
    plt.savefig(os.path.join(OUTDIR, f"cluster{cl}_rank_genes_groups.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 2) dotplot (top genes)
    top_genes = df["names"].head(20).tolist()
    sc.pl.dotplot(
        ad,
        var_names=top_genes,
        groupby=RESP_KEY,
        standard_scale="var",
        show=False
    )
    plt.savefig(os.path.join(OUTDIR, f"cluster{cl}_dotplot_top20.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 3) heatmap (top genes)
    sc.pl.heatmap(
        ad,
        var_names=top_genes,
        groupby=RESP_KEY,
        swap_axes=True,
        show=False
    )
    plt.savefig(os.path.join(OUTDIR, f"cluster{cl}_heatmap_top20.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

print("\nDone. Outputs in:", OUTDIR)


genes = [
    "NKG7","PRF1","GZMB","CTSW","GZMH",
    "PDCD1","LAG3","HAVCR2","ENTPD1","ITGAE","TOX","CTLA4",
    "HSPA1A","HSPA1B","DNAJB1","HSP90AA1","FOS","JUN",
    "KLF2","BACH2","ZC3H12A",
]

adata = sc.read_h5ad(H5AD)

# 用 log1p_norm 画（更合理）
if "log1p_norm" in adata.layers:
    adata.X = adata.layers["log1p_norm"]
elif "norm_expr" in adata.layers:
    adata.X = adata.layers["norm_expr"]

# 统一 Respond 标签
resp = adata.obs[RESP_KEY].astype(str).replace({
    "Responders": "Responder",
    "Non-responders": "Non-Responder",
    "Non-responders ": "Non-Responder",
})
adata.obs[RESP_KEY] = pd.Categorical(resp)

# 只保留 cluster1/4 且 Respond 两类
m = adata.obs[CLUSTER_KEY].astype(str).isin(TARGET_CLUSTERS) & adata.obs[RESP_KEY].isin(["Responder","Non-Responder"])
ad = adata[m].copy()

# 组合分组：cluster + Respond（四组）
ad.obs["cl_resp"] = (ad.obs[CLUSTER_KEY].astype(str) + "_" + ad.obs[RESP_KEY].astype(str)).astype("category")
order = ["1_Responder","1_Non-Responder","4_Responder","4_Non-Responder"]
ad.obs["cl_resp"] = ad.obs["cl_resp"].cat.set_categories(order, ordered=True)

def resolve_gene(g):
    if g in ad.var_names:
        return g
    if "gene_symbol" in ad.var.columns:
        hit = ad.var.index[ad.var["gene_symbol"].astype(str) == g]
        if len(hit) > 0:
            return hit[0]
    return None

missing = []
for g in genes:
    gg = resolve_gene(g)
    if gg is None:
        missing.append(g)
        continue

    # 注意：不使用 return_fig
    sc.pl.violin(
        ad,
        keys=gg,
        groupby="cl_resp",
        order=order,
        stripplot=False,
        jitter=False,
        rotation=30,
        show=False
    )
    plt.title(f"{g} (cluster1/4 × Respond)")
    plt.savefig(os.path.join(OUTDIR, f"{g}.png"), dpi=300, bbox_inches="tight")
    plt.close()  # 关闭当前 figure，防止叠图/占内存

print("Saved to:", OUTDIR)
if missing:
    print("Missing genes:", missing)