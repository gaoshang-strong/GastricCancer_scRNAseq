#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze cluster 0/1/2/3 differences (ignore patient/respond).

- Input: h5ad (reclustered T)
- Use: layers['norm_expr'] for DE + plotting (not raw, not .raw)
- Do:
  1) Basic cluster counts
  2) Global DE: each cluster vs rest (wilcoxon)
  3) Pairwise DE among clusters 0/1/2/3
  4) Dotplot + heatmap for marker panels
  5) Violin plots for selected genes

Outputs (png+pdf + csv) to:
  /ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_cluster0123_diff/results
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# Config (edit if needed)
# =========================
H5AD_PATH = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)

OUTDIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "04_cell_level_module_score/results"
)

CLUSTER_KEY = "leiden_T_0.6"
CLUSTERS_FOCUS = ["0", "1", "2", "3"]

LAYER_USE = "norm_expr"         # scVI normalized expression
DE_METHOD = "wilcoxon"
TOP_N_PER_CLUSTER = 20          # for marker panel
DOTPLOT_TOP_N = 10              # shown per cluster in dotplot
VIOLIN_TOP_N = 6                # violin per cluster (auto-picked from DE)

# Extra genes you'd like to always show (feel free to edit)
EXTRA_PANEL_GENES = [
    # cytotoxic
    "NKG7", "PRF1", "GZMB", "GNLY", "CTSW", "CCL5",
    # exhaustion
    "PDCD1", "TOX", "LAG3", "TIGIT", "HAVCR2", "ENTPD1", "CXCL13",
    # TRM-ish
    "ITGAE", "CD69", "CXCR6", "ZNF683", "RGS1",
    # activation/IEG
    "FOS", "JUN", "NR4A1", "NR4A2", "NR4A3", "DUSP1",
    # lineage
    "CD8A", "CD8B", "CD4", "IL7R",
]

# =========================
# Plot style (publication-ish)
# =========================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def sanitize(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def savefig_both(path_base: str, dpi: int = 300) -> None:
    plt.savefig(path_base + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close()


def set_cluster_order(adata: sc.AnnData, key: str) -> None:
    adata.obs[key] = adata.obs[key].astype(str)
    cats = sorted(adata.obs[key].unique(), key=lambda x: int(x) if x.isdigit() else x)
    adata.obs[key] = adata.obs[key].astype("category").cat.set_categories(cats, ordered=True)


def rank_genes_to_df(adata: sc.AnnData, key: str = "rank_genes_groups") -> pd.DataFrame:
    """Convert sc.tl.rank_genes_groups result to a long dataframe."""
    rg = adata.uns[key]
    groups = list(rg["names"].dtype.names)
    rows = []
    for g in groups:
        names = rg["names"][g]
        scores = rg["scores"][g]
        pvals = rg["pvals"][g] if "pvals" in rg else [np.nan] * len(names)
        pvals_adj = rg["pvals_adj"][g] if "pvals_adj" in rg else [np.nan] * len(names)
        lfc = rg["logfoldchanges"][g] if "logfoldchanges" in rg else [np.nan] * len(names)
        for i in range(len(names)):
            rows.append(
                {
                    "group": str(g),
                    "gene": str(names[i]),
                    "score": float(scores[i]) if scores is not None else np.nan,
                    "logfoldchange": float(lfc[i]) if lfc is not None else np.nan,
                    "pval": float(pvals[i]) if pvals is not None else np.nan,
                    "pval_adj": float(pvals_adj[i]) if pvals_adj is not None else np.nan,
                    "rank": i + 1,
                }
            )
    return pd.DataFrame(rows)


def pick_top_markers(df_de: pd.DataFrame, group: str, n: int, exclude: set[str]) -> list[str]:
    sub = df_de[df_de["group"] == group].copy()
    sub = sub.sort_values(["pval_adj", "rank"], ascending=[True, True])
    genes = []
    for g in sub["gene"].tolist():
        if g in exclude:
            continue
        if g.upper().startswith("MT-"):
            continue
        genes.append(g)
        if len(genes) >= n:
            break
    return genes


def main():
    ensure_dir(OUTDIR)
    figdir = os.path.join(OUTDIR, "figures")
    tabdir = os.path.join(OUTDIR, "tables")
    ensure_dir(figdir)
    ensure_dir(tabdir)

    sc.settings.verbosity = 2
    sc.settings.figdir = figdir

    # ---- Load ----
    adata = sc.read_h5ad(H5AD_PATH)

    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{CLUSTER_KEY}'].")

    if LAYER_USE not in adata.layers.keys():
        raise ValueError(f"Missing layers['{LAYER_USE}']. Available: {list(adata.layers.keys())}")

    # ---- Subset clusters 0/1/2/3 ----
    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype(str)
    ad = adata[adata.obs[CLUSTER_KEY].isin(CLUSTERS_FOCUS)].copy()
    set_cluster_order(ad, CLUSTER_KEY)

    # ---- Basic counts ----
    counts = ad.obs[CLUSTER_KEY].value_counts().reindex(CLUSTERS_FOCUS)
    counts.to_csv(os.path.join(tabdir, "table_cluster_counts_0123.csv"), header=["n_cells"])

    # ---- Global DE: each cluster vs rest ----
    sc.tl.rank_genes_groups(
        ad,
        groupby=CLUSTER_KEY,
        method=DE_METHOD,
        use_raw=False,
        layer=LAYER_USE,
        n_genes=TOP_N_PER_CLUSTER,
    )
    df_global = rank_genes_to_df(ad, key="rank_genes_groups")
    df_global.to_csv(os.path.join(tabdir, "DE_global_each_cluster_vs_rest_topN.csv"), index=False)

    # Also export full DE by recomputing with larger n_genes if needed
    # (Scanpy stores only n_genes results per group; keep it topN for now.)

    # ---- Marker panel selection (top markers per cluster) ----
    exclude = set()
    marker_panel = {}
    for c in CLUSTERS_FOCUS:
        marker_panel[c] = pick_top_markers(df_global, group=c, n=DOTPLOT_TOP_N, exclude=exclude)
        exclude.update(marker_panel[c])

    # Flatten unique genes for dotplot/heatmap
    dot_genes = []
    for c in CLUSTERS_FOCUS:
        dot_genes.extend(marker_panel[c])
    dot_genes = [g for g in dict.fromkeys(dot_genes) if g in ad.var_names]

    # ---- Dotplot (marker panel) ----
    if len(dot_genes) > 0:
        dp = sc.pl.dotplot(
            ad,
            var_names=dot_genes,
            groupby=CLUSTER_KEY,
            layer=LAYER_USE,
            standard_scale="var",
            return_fig=True,
            show=False,
        )
        dp.savefig(os.path.join(figdir, "fig_dotplot_top_markers_cluster0123.png"), dpi=300)
        dp.savefig(os.path.join(figdir, "fig_dotplot_top_markers_cluster0123.pdf"))
        plt.close("all")

    # ---- Heatmap of mean expression for dot genes ----
    if len(dot_genes) > 0:
        # compute mean expression per cluster on the layer
        # use sc.get.obs_df for stable extraction
        mean_mat = []
        for c in CLUSTERS_FOCUS:
            sub = ad[ad.obs[CLUSTER_KEY].astype(str) == c]
            df_expr = sc.get.obs_df(sub, keys=dot_genes, layer=LAYER_USE)
            mean_mat.append(df_expr.mean(axis=0))
        mean_df = pd.DataFrame(mean_mat, index=CLUSTERS_FOCUS, columns=dot_genes)
        mean_df.to_csv(os.path.join(tabdir, "table_mean_norm_expr_dot_genes_by_cluster0123.csv"))

        plt.figure(figsize=(max(8, 0.35 * len(dot_genes)), 3.2))
        plt.imshow(mean_df.values, aspect="auto")
        plt.yticks(np.arange(len(CLUSTERS_FOCUS)), CLUSTERS_FOCUS)
        plt.xticks(np.arange(len(dot_genes)), dot_genes, rotation=90, fontsize=7)
        plt.colorbar(label=f"Mean {LAYER_USE}")
        plt.title("Mean expression (norm_expr) of top markers per cluster (0/1/2/3)")
        savefig_both(os.path.join(figdir, "fig_heatmap_mean_norm_expr_top_markers_cluster0123"))

    # ---- Pairwise DE among 0/1/2/3 ----
    pairwise_rows = []
    pairs = [("0","1"), ("0","2"), ("0","3"), ("1","2"), ("1","3"), ("2","3")]
    for a, b in pairs:
        ad_pair = ad[ad.obs[CLUSTER_KEY].isin([a, b])].copy()
        set_cluster_order(ad_pair, CLUSTER_KEY)

        # rank_genes_groups with reference=b => results for group a is (a vs b)
        sc.tl.rank_genes_groups(
            ad_pair,
            groupby=CLUSTER_KEY,
            groups=[a],
            reference=b,
            method=DE_METHOD,
            use_raw=False,
            layer=LAYER_USE,
            n_genes=200,  # get more for tables
        )
        df_ab = rank_genes_to_df(ad_pair, key="rank_genes_groups")
        df_ab["contrast"] = f"{a}_vs_{b}"
        pairwise_rows.append(df_ab)

        df_ab.to_csv(os.path.join(tabdir, f"DE_pairwise_{a}_vs_{b}_top200.csv"), index=False)

    df_pairwise = pd.concat(pairwise_rows, ignore_index=True)
    df_pairwise.to_csv(os.path.join(tabdir, "DE_pairwise_all_contrasts_top200.csv"), index=False)

    # ---- Violin genes: auto-pick + extra panel ----
    auto_violin = []
    for c in CLUSTERS_FOCUS:
        auto_violin.extend(pick_top_markers(df_global, group=c, n=VIOLIN_TOP_N, exclude=set()))
    violin_genes = list(dict.fromkeys(auto_violin + EXTRA_PANEL_GENES))
    violin_genes = [g for g in violin_genes if g in ad.var_names]

    # Save one figure per gene (clean, publication-friendly)
    vdir = os.path.join(figdir, "violin_single_gene")
    ensure_dir(vdir)

    for g in violin_genes:
        ax = sc.pl.violin(
            ad,
            keys=g,
            groupby=CLUSTER_KEY,
            layer=LAYER_USE,
            stripplot=False,
            jitter=False,
            show=False,
        )
        # scanpy may return Axes or list-like; handle both
        if isinstance(ax, list):
            fig = ax[0].figure
        else:
            fig = ax.figure
        fig.set_size_inches(6.5, 3.0)
        fig.suptitle("")
        fig.savefig(os.path.join(vdir, f"VIOLIN_{sanitize(g)}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(vdir, f"VIOLIN_{sanitize(g)}.pdf"), bbox_inches="tight")
        plt.close(fig)

    # ---- Summary marker table (top genes per cluster) ----
    top_tbl = []
    for c in CLUSTERS_FOCUS:
        genes = pick_top_markers(df_global, group=c, n=TOP_N_PER_CLUSTER, exclude=set())
        for rank, gene in enumerate(genes, start=1):
            top_tbl.append({"cluster": c, "rank": rank, "gene": gene})
    pd.DataFrame(top_tbl).to_csv(os.path.join(tabdir, "table_top_markers_per_cluster0123.csv"), index=False)

    # ---- Run info ----
    with open(os.path.join(OUTDIR, "RUN_INFO.txt"), "w") as f:
        f.write(f"H5AD_PATH={H5AD_PATH}\n")
        f.write(f"OUTDIR={OUTDIR}\n")
        f.write(f"CLUSTER_KEY={CLUSTER_KEY}\n")
        f.write(f"CLUSTERS_FOCUS={CLUSTERS_FOCUS}\n")
        f.write(f"LAYER_USE={LAYER_USE}\n")
        f.write(f"DE_METHOD={DE_METHOD}\n")
        f.write(f"TOP_N_PER_CLUSTER={TOP_N_PER_CLUSTER}\n")
        f.write(f"DOTPLOT_TOP_N={DOTPLOT_TOP_N}\n")
        f.write(f"VIOLIN_TOP_N={VIOLIN_TOP_N}\n")

    print("\n[Done] Outputs saved to:")
    print("  ", OUTDIR)
    print("Tables:", tabdir)
    print("Figures:", figdir)
    print("Key files:")
    print("  - tables/DE_global_each_cluster_vs_rest_topN.csv")
    print("  - tables/DE_pairwise_all_contrasts_top200.csv")
    print("  - figures/fig_dotplot_top_markers_cluster0123.*")
    print("  - figures/violin_single_gene/VIOLIN_*.{png,pdf}")


if __name__ == "__main__":
    main()
