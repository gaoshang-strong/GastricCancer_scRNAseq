# vs_code_de_0_2_1_2.py
# Run this in VS Code debug mode (set breakpoints, run top-down).

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# =========================
# CONFIG (edit these)
# =========================
H5AD_PATH = r"/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_harmony_cellrank2.h5ad"
OUTDIR = r"/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/07_perturb_seq_T_cells"
adata = sc.read_h5ad(H5AD_PATH)
print(adata.obs["Respond"].value_counts())
CLUSTER_COL = "leiden_T_0.6"   # change to your cluster column name (e.g., "leiden", "leiden_T_0.6")
LAYER = 'log1p_norm'                  # e.g. "norm_expr" if you want to use adata.layers["norm_expr"]; None uses adata.X

# If your X/layer is raw counts, set DO_NORM_LOG=True
# If it's already log-normalized, set DO_NORM_LOG=False
DO_NORM_LOG = False
TARGET_SUM = 1e4

METHOD = "wilcoxon"           # "wilcoxon" (fast) or "t-test_overestim_var" etc.
N_GENES = 20000               # how many genes to rank (large => basically all)

# Optional filtering
MIN_CELLS_PER_GROUP = 20      # skip comparison if a group has too few cells
SAVE_TOPK = 300               # also save top-K gene list

# =========================
# Helpers
# =========================
def set_X_from_layer(adata, layer):
    if layer is None:
        return adata
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}")
    ad = adata.copy()
    ad.X = ad.layers[layer]
    return ad

def maybe_norm_log(adata):
    # Only do this if X is counts-like.
    if not DO_NORM_LOG:
        return adata
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=TARGET_SUM)
    sc.pp.log1p(ad)
    return ad

def de_pairwise(adata, cluster_col, a, b, method="wilcoxon", n_genes=20000, min_cells=20):
    """
    Compare cluster a vs b by subsetting only those cells, then rank_genes_groups
    group=a, reference=b.
    """
    cl = adata.obs[cluster_col].astype(str)
    sub = adata[(cl == str(a)) | (cl == str(b))].copy()
    sub.obs[cluster_col] = sub.obs[cluster_col].astype(str)

    vc = sub.obs[cluster_col].value_counts()
    if vc.get(str(a), 0) < min_cells or vc.get(str(b), 0) < min_cells:
        raise ValueError(f"Not enough cells: {a} has {vc.get(str(a),0)}, {b} has {vc.get(str(b),0)}")

    sc.tl.rank_genes_groups(
        sub,
        groupby=cluster_col,
        groups=[str(a)],
        reference=str(b),
        method=method,
        n_genes=n_genes,
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(sub, group=str(a))
    # df columns typically include: names, scores, logfoldchanges, pvals, pvals_adj
    return df

# =========================
# Main
# =========================
os.makedirs(OUTDIR, exist_ok=True)

adata = sc.read_h5ad(H5AD_PATH)
adata = set_X_from_layer(adata, LAYER)
adata = maybe_norm_log(adata)

# Sanity check (set breakpoint here)
# print(adata)
# print(adata.obs.columns)
# print(adata.obs[CLUSTER_COL].value_counts().head())

# 0 vs 2
df_0_vs_2 = de_pairwise(
    adata, cluster_col=CLUSTER_COL, a="0", b="2",
    method=METHOD, n_genes=N_GENES, min_cells=MIN_CELLS_PER_GROUP
)
df_0_vs_2.to_csv(os.path.join(OUTDIR, "DE_cluster0_vs_cluster2.full.csv"), index=False)

top0 = df_0_vs_2.sort_values("pvals_adj").head(SAVE_TOPK)[["names", "logfoldchanges", "pvals_adj"]]
top0.to_csv(os.path.join(OUTDIR, f"DE_cluster0_vs_cluster2.top{SAVE_TOPK}.csv"), index=False)

# 1 vs 2
df_1_vs_2 = de_pairwise(
    adata, cluster_col=CLUSTER_COL, a="1", b="2",
    method=METHOD, n_genes=N_GENES, min_cells=MIN_CELLS_PER_GROUP
)
df_1_vs_2.to_csv(os.path.join(OUTDIR, "DE_cluster1_vs_cluster2.full.csv"), index=False)

top1 = df_1_vs_2.sort_values("pvals_adj").head(SAVE_TOPK)[["names", "logfoldchanges", "pvals_adj"]]
top1.to_csv(os.path.join(OUTDIR, f"DE_cluster1_vs_cluster2.top{SAVE_TOPK}.csv"), index=False)

# Also save just gene lists (ordered by adj-p then abs(logFC))
def save_gene_list(df, out_csv, topk=SAVE_TOPK):
    d = df.copy()
    d["abs_lfc"] = d["logfoldchanges"].abs()
    d = d.sort_values(["pvals_adj", "abs_lfc"], ascending=[True, False])
    genes = d["names"].head(topk).tolist()
    pd.DataFrame({"gene": genes}).to_csv(out_csv, index=False)

save_gene_list(df_0_vs_2, os.path.join(OUTDIR, f"DE_cluster0_vs_cluster2.genes_top{SAVE_TOPK}.csv"))
save_gene_list(df_1_vs_2, os.path.join(OUTDIR, f"DE_cluster1_vs_cluster2.genes_top{SAVE_TOPK}.csv"))

print("DONE. Saved to:", OUTDIR)
print("0 vs 2 top genes:", top0["names"].head(10).tolist())
print("1 vs 2 top genes:", top1["names"].head(10).tolist())

RESPOND_COL = "Respond"   # <<< 如果你列名不是这个，改这里（例如 "response" 或 "resp"）

def to_resp_bin(x: str) -> str:
    s = str(x).strip().lower()
    # 常见写法：Non-responder / NR / 0
    if s in {"nr", "Non-responder", "non-responder", "non_responder", "non responder", "0"}:
        return "NR"
    if s in {"r", "Responder", "1"}:
        return "R"
    # 兜底：只要包含 non 就当 NR
    if "non" in s:
        return "NR"
    return "R"

# 取 cluster 1 子集
cl1 = adata[adata.obs[CLUSTER_COL].astype(str) == "1"].copy()

if RESPOND_COL not in cl1.obs.columns:
    raise ValueError(
        f"RESPOND_COL='{RESPOND_COL}' not in adata.obs. "
        f"Available columns include: {list(adata.obs.columns)[:30]} ..."
    )

cl1.obs["resp_bin"] = pd.Categorical([to_resp_bin(v) for v in cl1.obs[RESPOND_COL]])

# sanity check（建议在 VSCode 里打断点看一下）
print("cluster1 resp_bin counts:", cl1.obs["resp_bin"].value_counts().to_dict())

# NR vs R
df_cl1_nr_vs_r = de_pairwise(
    cl1, cluster_col="resp_bin", a="NR", b="R",
    method=METHOD, n_genes=N_GENES, min_cells=MIN_CELLS_PER_GROUP
)
df_cl1_nr_vs_r.to_csv(os.path.join(OUTDIR, "DE_cluster1_NR_vs_R.full.csv"), index=False)

top_nr = df_cl1_nr_vs_r.sort_values("pvals_adj").head(SAVE_TOPK)[["names", "logfoldchanges", "pvals_adj"]]
top_nr.to_csv(os.path.join(OUTDIR, f"DE_cluster1_NR_vs_R.top{SAVE_TOPK}.csv"), index=False)

save_gene_list(df_cl1_nr_vs_r, os.path.join(OUTDIR, f"DE_cluster1_NR_vs_R.genes_top{SAVE_TOPK}.csv"))

print("cluster1 NR vs R top genes:", top_nr["names"].head(10).tolist())