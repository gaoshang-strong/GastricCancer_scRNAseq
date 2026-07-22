#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
# ====== 输出目录（按你之前的习惯） ======
all_cells_path = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/adata_scvi_integrated_T_cells.h5ad"
adata = sc.read_h5ad(all_cells_path)
outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results/"
os.makedirs(outdir, exist_ok=True)

LEIDEN_RESOLUTIONS = [0.4, 0.6, 0.8]   # 你也可以只保留一个，比如 [0.6]
N_NEIGHBORS = 15
N_PCS = None  # 不用 PCA，因为用 scVI latent; 保持 None

# Respond 映射（你之前在 R 里就是这么定义的）
NONRESP_PATIENTS = {"PHD001", "PHD002", "PHD008"}  # Non-Responder
RESP_LABELS = ("Responder", "Non-Responder")


# ----------------------------
# Helpers
# ----------------------------
def pick_scvi_rep(adata):
    """Try to find a reasonable scVI latent embedding in adata.obsm."""
    candidates = [
        "X_scVI", "X_scvi", "X_scvi_latent", "X_scVI_latent",
        "X_scANVI", "X_scanvi",
        "X_scvi_umap",  # unlikely
    ]
    # exact match first
    for k in candidates:
        if k in adata.obsm_keys():
            return k

    # fuzzy match fallback
    keys = list(adata.obsm_keys())
    for k in keys:
        lk = k.lower()
        if "scvi" in lk and (lk.startswith("x_") or "latent" in lk):
            return k

    raise RuntimeError(
        f"Cannot find scVI latent embedding in adata.obsm. Available keys: {keys}"
    )


def ensure_respond(adata):
    """Ensure adata.obs has 'Respond' column."""
    if "patient" not in adata.obs.columns:
        raise RuntimeError("adata.obs must contain 'patient' column.")

    if "Respond" not in adata.obs.columns:
        adata.obs["Respond"] = np.where(
            adata.obs["patient"].astype(str).isin(NONRESP_PATIENTS),
            "Non-Responder",
            "Responder",
        )
    # categorical with fixed order
    adata.obs["Respond"] = pd.Categorical(adata.obs["Respond"], categories=list(RESP_LABELS), ordered=True)


def score_modules(adata):
    """Compute simple module scores to help interpret clusters."""
    modules = {
        "score_TCR": ["TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E"],
        "score_Cytotoxic": ["NKG7", "GZMB", "PRF1", "GNLY", "CTSW"],
        "score_Exhaustion": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1"],
        "score_IFNI": ["ISG15", "IFIT1", "IFIT3", "MX1", "STAT1", "IRF7"],
        "score_NFkB": ["NFKB2", "RELB", "TRAF1", "PELI1", "TNFAIP3"],
        "score_Cycling": ["MKI67", "TOP2A", "NUSAP1", "HMGB2", "STMN1"],
    }

    var_names = set(adata.var_names.astype(str))
    for name, genes in modules.items():
        genes_use = [g for g in genes if g in var_names]
        if len(genes_use) < 2:
            print(f"[WARN] {name}: too few genes found in var_names, skip. Found={genes_use}")
            continue
        sc.tl.score_genes(adata, gene_list=genes_use, score_name=name, use_raw=False)
    return list(modules.keys())


def mean_expr_by_group(adata, group_key, genes):
    """Compute mean expression of selected genes per group. Works with sparse or dense X."""
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        return pd.DataFrame()

    X = adata.X
    groups = adata.obs[group_key].astype(str).values
    uniq = pd.unique(groups)

    # map gene -> column index
    gene_idx = {g: int(np.where(adata.var_names == g)[0][0]) for g in genes}

    out = pd.DataFrame(index=uniq, columns=genes, dtype=float)
    for g, j in gene_idx.items():
        col = X[:, j]
        if sparse.issparse(col):
            col = np.asarray(col.todense()).ravel()
        else:
            col = np.asarray(col).ravel()
        s = pd.Series(col, index=groups)
        out[g] = s.groupby(level=0).mean().reindex(uniq).values

    out.index.name = group_key
    return out.sort_index()


def save_crosstabs(adata, cluster_key, outdir):
    """Save cluster composition tables."""
    obs = adata.obs.copy()

    # counts per cluster
    ct_cluster = obs[cluster_key].value_counts().sort_index()
    ct_cluster.to_csv(os.path.join(outdir, f"01_{cluster_key}__counts.csv"), header=["n_cells"])

    # cluster x Respond
    if "Respond" in obs.columns:
        tab = pd.crosstab(obs[cluster_key], obs["Respond"])
        tab.to_csv(os.path.join(outdir, f"01_{cluster_key}__by_Respond.csv"))

    # cluster x patient
    if "patient" in obs.columns:
        tab = pd.crosstab(obs[cluster_key], obs["patient"])
        tab.to_csv(os.path.join(outdir, f"01_{cluster_key}__by_patient.csv"))

    # cluster x t_state (if exists)
    if "t_state" in obs.columns:
        tab = pd.crosstab(obs[cluster_key], obs["t_state"])
        tab.to_csv(os.path.join(outdir, f"01_{cluster_key}__by_t_state.csv"))


def save_module_means(adata, cluster_key, module_cols, outdir):
    obs = adata.obs.copy()
    cols = [c for c in module_cols if c in obs.columns]
    if len(cols) == 0:
        return

    df = obs.groupby(cluster_key)[cols].mean().sort_index()
    df.to_csv(os.path.join(outdir, f"01_{cluster_key}__module_score_means.csv"))


def save_marker_means(adata, cluster_key, outdir):
    # 你关心的“解释用”markers
    markers = [
        # T identity
        "TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E",
        # effector/cytotoxic
        "NKG7", "GZMB", "PRF1", "GNLY", "CTSW",
        # exhaustion
        "PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1",
        # IFN-I
        "ISG15", "IFIT1", "IFIT3", "MX1", "STAT1", "IRF7",
        # NFkB axis (你之前看到的)
        "RELB", "NFKB2", "TRAF1", "PELI1", "OTUD5", "KDM6B",
        # cycling / DDR-ish
        "MKI67", "TOP2A", "NUSAP1",
        # contamination checks (可选看)
        "EPCAM", "KRT8", "KRT18", "KRT19", "TFF1", "GKN1",
        "MS4A1", "CD79A", "MZB1", "JCHAIN",
        "LST1", "TYROBP", "LYZ",
    ]
    df = mean_expr_by_group(adata, cluster_key, markers)
    if df.shape[0] > 0:
        df.to_csv(os.path.join(outdir, f"01_{cluster_key}__marker_mean_expr.csv"))


def rank_markers_per_cluster(adata, cluster_key, outdir, n_top=50):
    # 用 Scanpy 的 wilcoxon 给 cluster 找 marker（可选，快）
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        n_genes=n_top,
        pts=True,
        use_raw=False
    )
    # 导出为长表
    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names
    rows = []
    for g in groups:
        names = rg["names"][g]
        scores = rg["scores"][g]
        pvals = rg["pvals"][g]
        pvals_adj = rg["pvals_adj"][g]
        logfc = rg["logfoldchanges"][g] if "logfoldchanges" in rg else [np.nan]*len(names)
        for i in range(len(names)):
            rows.append({
                "cluster": g,
                "rank": i + 1,
                "gene": names[i],
                "score": scores[i],
                "logFC": logfc[i],
                "pval": pvals[i],
                "padj": pvals_adj[i],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"01_{cluster_key}__rank_genes_groups_wilcoxon_top{n_top}.csv"), index=False)

def save_umaps(adata, cluster_key, outdir):
    """
    Save UMAPs colored by:
      - Respond
      - cluster_key (e.g., leiden_T_0.6)
    """
    # Respond UMAP（如果存在）
    if "Respond" in adata.obs.columns:
        sc.pl.umap(
            adata,
            color="Respond",
            frameon=False,
            show=False,
        )
        plt.savefig(os.path.join(outdir, f"01_umap_Respond.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Leiden UMAP
    if cluster_key in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=cluster_key,
            legend_loc="on data",   # 如果 cluster 多会有点拥挤，可改成 'right margin'
            frameon=False,
            show=False,
        )
        plt.savefig(os.path.join(outdir, f"01_umap_{cluster_key}.png"), dpi=300, bbox_inches="tight")
        plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    print(f"[INFO] Read: {all_cells_path}")
    adata = sc.read_h5ad(all_cells_path)

    # Ensure required obs columns
    ensure_respond(adata)
    if "t_state" not in adata.obs.columns:
        print("[WARN] adata.obs has no 't_state' column. That's OK; will skip related outputs.")

    # Choose scVI embedding
    rep = pick_scvi_rep(adata)
    print(f"[INFO] Use representation: obsm['{rep}'] with shape {adata.obsm[rep].shape}")

    # Neighbors / UMAP
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep=rep, n_pcs=N_PCS, random_state=0)
    sc.tl.umap(adata, random_state=0)
    save_umaps(adata, cluster_key=None, outdir=outdir)  # 只会画 Respond（cluster_key=None 会被跳过）

    # Module scores (for interpretation)
    module_cols = score_modules(adata)

    # Leiden clustering at multiple resolutions
    for res in LEIDEN_RESOLUTIONS:
        key = f"leiden_T_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=0)
        print(f"[INFO] Leiden done: {key} (#clusters={adata.obs[key].nunique()})")
        sc.pl.umap(
            adata,
            color=key,
            legend_loc="on data",   # 或 "right margin"
            frameon=False,
            show=False,
        )
        plt.savefig(os.path.join(outdir, f"01_umap_{key}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Save tables
        save_crosstabs(adata, key, outdir)
        save_module_means(adata, key, module_cols, outdir)
        save_marker_means(adata, key, outdir)

        # Optional: per-cluster markers (top 50)
        rank_markers_per_cluster(adata, key, outdir, n_top=50)

    # Save the reclustered object
    out_h5ad = os.path.join(outdir, "01_adata_T_reclustered.h5ad")
    adata.write(out_h5ad)
    print(f"[INFO] Saved reclustered h5ad: {out_h5ad}")

    # Also print quick summary for the default resolution (0.6 if exists)
    default_key = "leiden_T_0.6" if "leiden_T_0.6" in adata.obs.columns else f"leiden_T_{LEIDEN_RESOLUTIONS[0]}"
    print("\n[SUMMARY] cluster counts:")
    print(adata.obs[default_key].value_counts().sort_index())

    print("\n[SUMMARY] cluster x Respond:")
    print(pd.crosstab(adata.obs[default_key], adata.obs["Respond"]))


if __name__ == "__main__":
    main()
