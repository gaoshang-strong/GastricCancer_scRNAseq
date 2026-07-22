import os
import scanpy as sc
import pandas as pd
import gseapy as gp

# ---- load ----
adata = sc.read_h5ad("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/02_filtered_drop_10-12-5-8-9.h5ad")  # 改成你的h5ad路径
cluster_col = "leiden_T_0.6"
group_col   = "Respond"
clusters = ["0", "1", "2"]
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/03_T_cell_subtype_responder_vs_nonResponder/results"
os.makedirs(outdir, exist_ok=True)

# 你的真实标签名（按需要改：比如 "Non-Responder"）
RESP_NAME   = "Responder"
NONRESP_NAME = "Non-Responder"

GO_SETS = ["GO_Biological_Process_2021", "GO_Molecular_Function_2021", "GO_Cellular_Component_2021"]

# ----------------------------
# Scanpy figure settings
# ----------------------------
sc.settings.figdir = outdir
sc.settings.autoshow = False

# ----------------------------
# Load
# ----------------------------
use_layer = "log1p_norm" if "log1p_norm" in adata.layers.keys() else None
print("Using layer:", use_layer)

def enrich_and_save(gene_list, prefix):
    gene_list = [g for g in gene_list if isinstance(g, str) and g]
    if len(gene_list) < 10:
        return
    enr = gp.enrichr(
        gene_list=gene_list,
        organism="Human",
        gene_sets=GO_SETS,
        outdir=None
    )
    enr.results.to_csv(os.path.join(outdir, f"{prefix}_GO_enrichr.csv"), index=False)

# ----------------------------
# Per-cluster: DE + plots + GO
# ----------------------------
for cl in clusters:
    ad = adata[adata.obs[cluster_col].astype(str) == cl].copy()
    ad = ad[ad.obs[group_col].notna()].copy()

    if ad.n_obs < 20 or ad.obs[group_col].nunique() != 2:
        print(f"skip cluster {cl}: n={ad.n_obs}, groups={ad.obs[group_col].unique().tolist()}")
        continue

    # 固定类别顺序（NonResponder 为 reference）
    ad.obs[group_col] = pd.Categorical(ad.obs[group_col], categories=[NONRESP_NAME, RESP_NAME])

    sc.tl.rank_genes_groups(ad, groupby=group_col, method="wilcoxon", layer=use_layer, use_raw=False)

    # ---- save DE table ----
    df_all = sc.get.rank_genes_groups_df(ad, group=None)
    df_all.insert(0, "cluster", cl)
    df_all.to_csv(os.path.join(outdir, f"DE_cluster_{cl}_{group_col}_wilcoxon.csv"), index=False)

    # ---- plots (use save=...) ----
    sc.pl.rank_genes_groups(ad, n_genes=25, sharey=False, show=False,
                           save=f"_cluster{cl}_rank_genes.png")

    sc.pl.rank_genes_groups_heatmap(ad, n_genes=20, groupby=group_col, show=False,
                                    save=f"_cluster{cl}_heatmap.png")

    sc.pl.rank_genes_groups_dotplot(ad, n_genes=15, groupby=group_col, show=False,
                                    save=f"_cluster{cl}_dotplot.png")

    # violin: Responder top10
    df_r = sc.get.rank_genes_groups_df(ad, group=RESP_NAME)
    top10 = df_r["names"].head(10).tolist()
    sc.pl.violin(ad, keys=top10, groupby=group_col, layer=use_layer, rotation=90, show=False,
                 save=f"_cluster{cl}_violin_top10_{RESP_NAME}.png")

    # ---- GO enrichment ----
    genes_r = df_r.query("pvals_adj < 0.05 and logfoldchanges > 0")["names"].head(300).tolist()
    enrich_and_save(genes_r, f"cluster{cl}_{RESP_NAME}High")

    df_n = sc.get.rank_genes_groups_df(ad, group=NONRESP_NAME)
    genes_n = df_n.query("pvals_adj < 0.05 and logfoldchanges > 0")["names"].head(300).tolist()
    enrich_and_save(genes_n, f"cluster{cl}_{NONRESP_NAME}High")

    print(f"done cluster {cl}")

print("All outputs saved to:", os.path.abspath(outdir))