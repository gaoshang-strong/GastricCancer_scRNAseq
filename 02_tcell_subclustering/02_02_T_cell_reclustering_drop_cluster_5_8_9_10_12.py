#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop selected clusters (5,8,9,10,12) from an existing T-cell clustering,
THEN re-run neighbors+UMAP+Leiden (multi-res) on the remaining cells,
AND export:
  - UMAPs colored by Respond + each Leiden resolution
  - composition tables (cluster counts / by_Respond / by_patient)
  - module score means + marker means
  - rank_genes_groups (wilcoxon) per Leiden key
  - filtered + reclustered h5ad

Adapted to the NEW paths/naming used in our latest pipeline:
  /02_extracting_T_cells_and_clustering/adata_scvi_integrated_T_cells.h5ad
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Matplotlib font (Arial not available; use Liberation Sans)
# =========================
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# =========================
# CONFIG (NEW PATHS)
# =========================
H5AD_PATH = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/adata_scvi_integrated_T_cells.h5ad"

# Which existing key to use for *dropping* clusters (must exist in the input OR will be rerun)
FILTER_CLUSTER_KEY = "leiden_T_0.6"

# Clusters to drop (as strings)
DROP_CLUSTERS = {"5", "8", "9", "10", "12"}

# After dropping, re-run clustering at these resolutions
LEIDEN_RESOLUTIONS = [0.4, 0.6, 0.8]

# Neighbors parameters (scVI latent)
N_NEIGHBORS = 15
N_PCS = None  # keep None when use_rep is scVI latent

# If FILTER_CLUSTER_KEY not present, re-run it at this resolution for filtering
RERUN_FILTER_IF_MISSING = True
FILTER_LEIDEN_RES = 0.6  # used only if FILTER_CLUSTER_KEY missing

# Respond mapping (if Respond not in obs)
NONRESP_PATIENTS = {"PHD001", "PHD002", "PHD008"}
RESP_LABELS = ("Responder", "Non-Responder")

# Output directory (NEW naming)
OUTDIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/"
    f"results_drop_{'-'.join(sorted(DROP_CLUSTERS))}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def pick_scvi_rep(adata):
    """Auto-detect scVI latent embedding in adata.obsm."""
    candidates = [
        "X_scVI", "X_scvi", "X_scvi_latent", "X_scVI_latent",
        "X_scANVI", "X_scanvi",
    ]
    for k in candidates:
        if k in adata.obsm_keys():
            return k
    for k in adata.obsm_keys():
        lk = k.lower()
        if "scvi" in lk and ("latent" in lk or lk.startswith("x_")):
            return k
    raise RuntimeError(f"Cannot find scVI latent in obsm. Keys: {list(adata.obsm_keys())}")


def ensure_respond(adata):
    """Ensure adata.obs has Respond and patient."""
    if "patient" not in adata.obs.columns:
        raise RuntimeError("adata.obs must contain 'patient' column.")
    if "Respond" not in adata.obs.columns:
        adata.obs["Respond"] = np.where(
            adata.obs["patient"].astype(str).isin(NONRESP_PATIENTS),
            "Non-Responder",
            "Responder",
        )
    adata.obs["Respond"] = pd.Categorical(
        adata.obs["Respond"], categories=list(RESP_LABELS), ordered=True
    )


def score_modules(adata):
    """Compute module scores for functional annotation."""
    modules = {
        "score_TCR": ["TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E"],
        "score_Cytotoxic": ["NKG7", "GZMB", "PRF1", "GNLY", "CTSW"],
        "score_Exhaustion": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1"],
        "score_IFNI": ["ISG15", "IFIT1", "IFIT3", "MX1", "STAT1", "IRF7"],
        "score_NFkB": ["NFKB2", "RELB", "TRAF1", "PELI1", "TNFAIP3", "OTUD5", "KDM6B"],
        "score_Cycling": ["MKI67", "TOP2A", "NUSAP1", "HMGB2", "STMN1"],
    }
    var = set(adata.var_names.astype(str))
    computed = []
    for name, genes in modules.items():
        genes_use = [g for g in genes if g in var]
        if len(genes_use) < 2:
            print(f"[WARN] {name}: too few genes found, skip. Found={genes_use}")
            continue
        sc.tl.score_genes(adata, gene_list=genes_use, score_name=name, use_raw=False)
        computed.append(name)
    return computed


def mean_expr_by_group(adata, group_key, genes):
    """Mean expression per group for selected genes."""
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        return pd.DataFrame()

    X = adata.X
    groups = adata.obs[group_key].astype(str).values
    uniq = pd.unique(groups)

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


def save_crosstabs(adata, cluster_key, outdir, prefix=""):
    obs = adata.obs.copy()
    obs[cluster_key] = obs[cluster_key].astype(str)

    # counts per cluster
    obs[cluster_key].value_counts().sort_index().to_csv(
        os.path.join(outdir, f"{prefix}{cluster_key}__counts.csv"),
        header=["n_cells"],
    )

    if "Respond" in obs.columns:
        pd.crosstab(obs[cluster_key], obs["Respond"]).to_csv(
            os.path.join(outdir, f"{prefix}{cluster_key}__by_Respond.csv")
        )

    if "patient" in obs.columns:
        pd.crosstab(obs[cluster_key], obs["patient"]).to_csv(
            os.path.join(outdir, f"{prefix}{cluster_key}__by_patient.csv")
        )

    if "t_state" in obs.columns:
        pd.crosstab(obs[cluster_key], obs["t_state"]).to_csv(
            os.path.join(outdir, f"{prefix}{cluster_key}__by_t_state.csv")
        )


def export_stepB_tables(adata, cluster_key, module_cols, outdir, prefix=""):
    # module means
    if module_cols:
        df_mod = adata.obs.groupby(cluster_key)[module_cols].mean().sort_index()
        df_mod.to_csv(os.path.join(outdir, f"{prefix}{cluster_key}__module_score_means.csv"))

    # marker means
    markers = [
        # identity
        "TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E",
        # cytotoxic
        "NKG7", "GZMB", "PRF1", "GNLY", "CTSW",
        # exhaustion
        "PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1",
        # IFN
        "ISG15", "IFIT1", "IFIT3", "MX1", "STAT1", "IRF7",
        # NFkB
        "RELB", "NFKB2", "TRAF1", "PELI1", "OTUD5", "KDM6B", "TNFAIP3",
        # cycling
        "MKI67", "TOP2A", "NUSAP1", "HMGB2", "STMN1",
        # contamination checks
        "EPCAM", "KRT8", "KRT18", "KRT19", "TFF1", "GKN1",
        "MS4A1", "CD79A", "MZB1", "JCHAIN",
        "LST1", "TYROBP", "LYZ",
    ]
    df_m = mean_expr_by_group(adata, cluster_key, markers)
    if df_m.shape[0] > 0:
        df_m.to_csv(os.path.join(outdir, f"{prefix}{cluster_key}__marker_mean_expr.csv"))


def export_rank_genes_groups(adata, cluster_key, outdir, n_top=50, prefix=""):
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        n_genes=n_top,
        pts=True,
        use_raw=False,
    )
    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names
    rows = []
    for g in groups:
        names = rg["names"][g]
        scores = rg["scores"][g]
        pvals = rg["pvals"][g]
        padj = rg["pvals_adj"][g]
        logfc = rg["logfoldchanges"][g] if "logfoldchanges" in rg else [np.nan] * len(names)
        for i in range(len(names)):
            rows.append({
                "cluster": g,
                "rank": i + 1,
                "gene": names[i],
                "score": scores[i],
                "logFC": logfc[i],
                "pval": pvals[i],
                "padj": padj[i],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"{prefix}{cluster_key}__rank_genes_groups_wilcoxon_top{n_top}.csv"),
        index=False,
    )


def save_umap(adata, color_key, outdir, fname, legend_loc=None):
    sc.pl.umap(
        adata,
        color=color_key,
        legend_loc=legend_loc,
        frameon=False,
        show=False,
    )
    plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Read: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)
    ensure_respond(adata)

    # ---------- Step 0: ensure FILTER_CLUSTER_KEY exists (for dropping) ----------
    if FILTER_CLUSTER_KEY not in adata.obs.columns:
        if not RERUN_FILTER_IF_MISSING:
            raise RuntimeError(
                f"Missing {FILTER_CLUSTER_KEY} in obs. "
                f"Set RERUN_FILTER_IF_MISSING=True or load your reclustered h5ad."
            )
        rep0 = pick_scvi_rep(adata)
        print(f"[INFO] {FILTER_CLUSTER_KEY} missing; rerun neighbors+UMAP+leiden({FILTER_LEIDEN_RES}) using obsm['{rep0}']")
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep=rep0, n_pcs=N_PCS, random_state=0)
        sc.tl.umap(adata, random_state=0)
        sc.tl.leiden(adata, resolution=FILTER_LEIDEN_RES, key_added=FILTER_CLUSTER_KEY, random_state=0)

    adata.obs[FILTER_CLUSTER_KEY] = adata.obs[FILTER_CLUSTER_KEY].astype(str)

    # ---------- Step 1: pre-drop summaries ----------
    pre_dir = os.path.join(OUTDIR, "PRE_drop")
    os.makedirs(pre_dir, exist_ok=True)
    save_crosstabs(adata, FILTER_CLUSTER_KEY, pre_dir, prefix="01_")

    # (optional) visualize pre-drop
    if "X_umap" in adata.obsm_keys():
        save_umap(adata, "Respond", pre_dir, "02_fig_umap_Respond_PRE.png")
        save_umap(adata, FILTER_CLUSTER_KEY, pre_dir, f"02_fig_umap_{FILTER_CLUSTER_KEY}_PRE.png", legend_loc="on data")

    # ---------- Step 2: drop clusters ----------
    before_n = adata.n_obs
    keep = ~adata.obs[FILTER_CLUSTER_KEY].isin(DROP_CLUSTERS)
    adata2 = adata[keep].copy()
    after_n = adata2.n_obs
    print(f"[INFO] Drop clusters {sorted(DROP_CLUSTERS)} from {FILTER_CLUSTER_KEY}: {before_n} -> {after_n} cells")

    # Save filtered object (pre-reclustering)
    out_filtered = os.path.join(OUTDIR, f"02_filtered_drop_{'-'.join(sorted(DROP_CLUSTERS))}.h5ad")
    adata2.write(out_filtered)
    print(f"[INFO] Saved filtered h5ad: {out_filtered}")

    # ---------- Step 3: recompute neighbors + UMAP on filtered cells ----------
    rep = pick_scvi_rep(adata2)
    print(f"[INFO] Recompute neighbors/UMAP using obsm['{rep}'] on filtered cells")
    sc.pp.neighbors(adata2, n_neighbors=N_NEIGHBORS, use_rep=rep, n_pcs=N_PCS, random_state=0)
    sc.tl.umap(adata2, random_state=0)

    # Visualize Respond + patient
    save_umap(adata2, "Respond", OUTDIR, "02_fig_umap_Respond_POST.png")
    if "patient" in adata2.obs.columns:
        save_umap(adata2, "patient", OUTDIR, "02_fig_umap_patient_POST.png")

    # ---------- Step 4: re-cluster at multiple resolutions + export StepB tables ----------
    post_dir = os.path.join(OUTDIR, "POST_drop_recluster")
    os.makedirs(post_dir, exist_ok=True)

    module_cols = score_modules(adata2)

    for res in LEIDEN_RESOLUTIONS:
        key = f"leiden_T_{res}"
        sc.tl.leiden(adata2, resolution=res, key_added=key, random_state=0)
        print(f"[INFO] Leiden done: {key} (#clusters={adata2.obs[key].nunique()})")

        # UMAP
        save_umap(adata2, key, post_dir, f"02_fig_umap_{key}.png", legend_loc="on data")

        # Tables
        save_crosstabs(adata2, key, post_dir, prefix="01_")
        export_stepB_tables(adata2, key, module_cols, post_dir, prefix="02_")
        export_rank_genes_groups(adata2, key, post_dir, n_top=50, prefix="03_")

    # ---------- Step 5: save reclustered object ----------
    out_h5ad = os.path.join(OUTDIR, "02_adata_T_reclustered_after_drop.h5ad")
    adata2.write(out_h5ad)
    print(f"[INFO] Saved reclustered h5ad: {out_h5ad}")

    # ---------- Quick summary for 0.6 if available ----------
    default_key = "leiden_T_0.6" if "leiden_T_0.6" in adata2.obs.columns else f"leiden_T_{LEIDEN_RESOLUTIONS[0]}"
    print("\n[SUMMARY] cluster counts (POST_drop_recluster):")
    print(adata2.obs[default_key].value_counts().sort_index())
    print("\n[SUMMARY] cluster x Respond (POST_drop_recluster):")
    print(pd.crosstab(adata2.obs[default_key], adata2.obs["Respond"]))

    print(f"\n[INFO] Outputs written under: {OUTDIR}")
    print(f" - PRE summaries: {pre_dir}")
    print(f" - POST recluster outputs: {post_dir}")


if __name__ == "__main__":
    main()
