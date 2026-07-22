# ===========================================
# 02b_check_cluster0_and_3/run_cluster_state_analysis.py
# ===========================================
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Publication-ish matplotlib
# -----------------------------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Paths
# -----------------------------
h5ad_path = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "02_adata_T_reclustered_after_drop.h5ad"
)
outdir = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02b_check_cluster0_and_2/results"
)
os.makedirs(outdir, exist_ok=True)

# -----------------------------
# Keys (edit if needed)
# -----------------------------
cluster_key = "leiden_T_0.6"
patient_key = "patient"
respond_key = "Respond"  # e.g., "Responder"/"Non-responder" or 0/1
layer_use = "norm_expr"  # per your request
drop_cluster = "8"       # extra safety, even if already dropped

# Focus clusters for story
# - Primary hypothesis: cluster 0/2 are activated but less cytotoxic than cluster 1
# - Also include cluster 3 as reference because your folder name mentions 0_and_3
focus_clusters = ["0", "1", "2", "3"]

# -----------------------------
# Gene sets (tunable)
# Keep modest size to avoid score instability
# -----------------------------
GENESETS: Dict[str, List[str]] = {
    # TCR activation / immediate early response / antigen stimulation
    "score_activation": [
        "NR4A1", "NR4A2", "NR4A3",
        "FOS", "JUN", "JUNB", "DUSP1", "ZFP36",
        "CD69", "TNFRSF9", "ICOS",
        "IFNG",
    ],
    # Cytolytic machinery / effector killing
    "score_cytotoxic": [
        "NKG7", "PRF1", "GZMB", "GNLY",
        "CTSW", "CST7",
        "KLRD1", "FCER1G",
        "GZMH", "GZMK",  # include one "early cytotoxic" axis
    ],
    # Exhaustion / dysfunction markers
    "score_exhaustion": [
        "PDCD1", "TOX", "LAG3", "HAVCR2", "TIGIT", "CTLA4",
        "ENTPD1", "CXCL13", "LAYN",
    ],
    # TRM / tumor-resident / retention-like program
    "score_trm": [
        "ITGAE", "CD69", "CXCR6", "ZNF683", "RORA",
        "RUNX3", "PRDM1",
    ],
}

# Marker panel for dotplot (identity + activation + cytotoxic + exhaustion + TRM)
DOTPLOT_GENES = [
    # T cell identity / lineage sanity
    "CD3D", "CD3E", "TRAC", "CD8A", "CD8B", "IL7R", "CCR7",
    # Activation
    "NR4A1", "NR4A2", "NR4A3", "CD69", "TNFRSF9", "ICOS", "IFNG",
    # Cytotoxic
    "NKG7", "PRF1", "GZMB", "GNLY", "CTSW", "CST7", "GZMH", "CCL5",
    # Exhaustion
    "PDCD1", "TOX", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "ENTPD1", "CXCL13",
    # TRM
    "ITGAE", "CXCR6", "ZNF683",
]

# ---------------------------------
# Helpers
# ---------------------------------
def _exists_in_var(adata: sc.AnnData, genes: List[str]) -> List[str]:
    vg = set(adata.var_names)
    return [g for g in genes if g in vg]

def _clean_categories(adata: sc.AnnData, key: str) -> None:
    adata.obs[key] = adata.obs[key].astype(str)
    cats = sorted(adata.obs[key].unique(), key=lambda x: int(x) if x.isdigit() else x)
    adata.obs[key] = adata.obs[key].astype("category").cat.set_categories(cats, ordered=True)

def _score_genes_layer_safe(adata: sc.AnnData, gene_list: List[str], score_name: str, layer: str) -> None:
    """
    Compute score on a given layer if scanpy supports layer=...;
    otherwise temporarily swap adata.X.
    """
    gene_list = _exists_in_var(adata, gene_list)
    if len(gene_list) < 3:
        # Avoid nonsense score; still create the column
        adata.obs[score_name] = np.nan
        print(f"[Warn] {score_name}: too few genes found ({len(gene_list)}). Set to NaN.")
        return

    try:
        # scanpy>=1.9 supports layer in score_genes
        sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, layer=layer, use_raw=False)
    except TypeError:
        # Fallback: temporarily use that layer as X
        X_backup = adata.X
        adata.X = adata.layers[layer]
        sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, use_raw=False)
        adata.X = X_backup

def savefig_all(path_base: str, dpi: int = 300) -> None:
    plt.savefig(path_base + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close()

def wilcoxon_signed_rank(x: np.ndarray) -> Tuple[float, float]:
    """Return (stat, p) using scipy if available, else (nan, nan)."""
    try:
        from scipy.stats import wilcoxon
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return np.nan, np.nan
        stat, p = wilcoxon(x)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan

# ---------------------------------
# Load
# ---------------------------------
adata = sc.read_h5ad(h5ad_path)

# sanity checks
for k in [cluster_key, patient_key]:
    if k not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{k}']. Available keys (partial): {list(adata.obs.columns)[:50]}")

if respond_key not in adata.obs.columns:
    print(f"[Warn] adata.obs['{respond_key}'] not found. Will skip responder/non-responder stratified plots.")

if layer_use not in adata.layers.keys():
    raise ValueError(f"layers['{layer_use}'] not found. Available layers: {list(adata.layers.keys())}")

# Drop cluster 8 if present
adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
mask_keep = adata.obs[cluster_key] != str(drop_cluster)
if mask_keep.sum() != adata.n_obs:
    print(f"[Info] Dropping cluster {drop_cluster}: {adata.n_obs} -> {mask_keep.sum()}")
    adata = adata[mask_keep].copy()

_clean_categories(adata, cluster_key)

# Also create a focused subset (0/1/2/3)
focus_present = [c for c in focus_clusters if c in adata.obs[cluster_key].astype(str).unique()]
ad_focus = adata[adata.obs[cluster_key].astype(str).isin(focus_present)].copy()
_clean_categories(ad_focus, cluster_key)

print("[Info] Focus clusters present:", focus_present)
print("[Info] n_cells focus:", ad_focus.n_obs)

# ---------------------------------
# 1) Compute scores on norm_expr layer
# ---------------------------------
for score_name, genes in GENESETS.items():
    _score_genes_layer_safe(ad_focus, genes, score_name=score_name, layer=layer_use)

# Save an h5ad with added scores (useful for later)
ad_focus.write_h5ad(os.path.join(outdir, "adata_focus_with_scores.h5ad"))

# ---------------------------------
# 2) Tables: per-cluster summary of scores
# ---------------------------------
score_cols = list(GENESETS.keys())
cluster_summary = (
    ad_focus.obs
    .groupby(ad_focus.obs[cluster_key].astype(str))[score_cols]
    .agg(["count", "mean", "median", "std"])
)
cluster_summary.to_csv(os.path.join(outdir, "table_cluster_score_summary.csv"))

# ---------------------------------
# 3) Tables: patient-level cluster composition (within all T cells after drop)
#    (Use full adata, not only focus)
# ---------------------------------
# proportions per patient
ct = pd.crosstab(
    adata.obs[patient_key].astype(str),
    adata.obs[cluster_key].astype(str),
    normalize="index"
)
ct.to_csv(os.path.join(outdir, "table_patient_cluster_proportions.csv"))

# also n_cells
n_cells = pd.crosstab(
    adata.obs[patient_key].astype(str),
    adata.obs[cluster_key].astype(str),
    normalize=False
)
n_cells.to_csv(os.path.join(outdir, "table_patient_cluster_counts.csv"))

# ---------------------------------
# 4) Patient-level paired comparison:
#    "cluster 1 more cytotoxic than 0/2/3" within-patient
# ---------------------------------
# Use patient-level mean of cytotoxic score within each (patient, cluster)
df_pc = ad_focus.obs[[patient_key, cluster_key] + score_cols].copy()
df_pc[patient_key] = df_pc[patient_key].astype(str)
df_pc[cluster_key] = df_pc[cluster_key].astype(str)

# patient x cluster means
pc_mean = (
    df_pc.groupby([patient_key, cluster_key])[score_cols]
    .mean()
    .reset_index()
)

# wide for cytotoxic only
cyto_wide = pc_mean.pivot(index=patient_key, columns=cluster_key, values="score_cytotoxic")

paired_rows = []
for target in ["0", "2", "3"]:
    if "1" not in cyto_wide.columns or target not in cyto_wide.columns:
        continue
    both = cyto_wide[["1", target]].dropna()
    diff = both["1"] - both[target]  # positive means cluster1 higher cytotoxic
    stat, p = wilcoxon_signed_rank(diff.values)
    paired_rows.append({
        "compare": f"cluster1_minus_cluster{target}",
        "n_patients_with_both": both.shape[0],
        "median_diff": float(np.median(diff.values)) if both.shape[0] > 0 else np.nan,
        "mean_diff": float(np.mean(diff.values)) if both.shape[0] > 0 else np.nan,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
    })

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(os.path.join(outdir, "table_patient_paired_cytotoxic_cluster1_vs_others.csv"), index=False)

# ---------------------------------
# 5) Figures (publication-friendly): Dotplot + score distributions + paired patient plot
# ---------------------------------

# 5A Dotplot of marker panel (scanpy 1.11.5: get fig from returned axes dict)
genes_dot = _exists_in_var(ad_focus, DOTPLOT_GENES)
if len(genes_dot) < 10:
    print("[Warn] Too few genes found for dotplot. Check var_names format.")
else:
    axes = sc.pl.dotplot(
        ad_focus,
        var_names=genes_dot,
        groupby=cluster_key,
        layer=layer_use,          # norm_expr
        standard_scale="var",
        dendrogram=False,
        show=False,               # <-- 关键：返回 axes dict
    )

    # axes 是 dict: {name: ax}
    any_ax = next(iter(axes.values()))
    fig = any_ax.get_figure()
    fig.set_size_inches(14, 6)
    fig.suptitle("Marker panel (layer=norm_expr), focus clusters", y=1.02)

    fig.savefig(os.path.join(outdir, "fig_dotplot_marker_panel_focus_clusters.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "fig_dotplot_marker_panel_focus_clusters.pdf"),
                bbox_inches="tight")
    plt.close(fig)

# 5B Violin plots for each score by cluster
for s in score_cols:
    sc.pl.violin(
        ad_focus,
        keys=s,
        groupby=cluster_key,
        stripplot=False,
        jitter=False,
        show=False,
        rotation=0,
    )
    plt.title(s)
    savefig_all(os.path.join(outdir, f"fig_violin_{s}_by_{cluster_key}_focus"), dpi=300)

# 5C Boxplot (matplotlib/pandas) for each score by cluster (more publication-like summary)
for s in score_cols:
    dfp = ad_focus.obs[[cluster_key, s]].copy()
    dfp[cluster_key] = dfp[cluster_key].astype(str)

    plt.figure(figsize=(8, 3))
    dfp.boxplot(column=s, by=cluster_key, grid=False)
    plt.suptitle("")
    plt.title(f"{s} by {cluster_key} (focus clusters)")
    plt.xlabel(cluster_key)
    plt.ylabel(s)
    savefig_all(os.path.join(outdir, f"fig_box_{s}_by_{cluster_key}_focus"), dpi=300)

# 5D Patient-level paired line plots: cluster1 vs cluster0/2/3 for cytotoxic
for target in ["0", "2", "3"]:
    if "1" not in cyto_wide.columns or target not in cyto_wide.columns:
        continue
    both = cyto_wide[["1", target]].dropna()
    if both.shape[0] < 3:
        continue

    # optionally annotate respond label
    meta = ad_focus.obs[[patient_key]].drop_duplicates().set_index(patient_key)
    if respond_key in ad_focus.obs.columns:
        meta = meta.join(ad_focus.obs[[patient_key, respond_key]].drop_duplicates().set_index(patient_key))
    meta = meta.loc[both.index]

    plt.figure(figsize=(6, 4))
    # plot paired lines
    for i, pid in enumerate(both.index):
        y0 = both.loc[pid, target]
        y1 = both.loc[pid, "1"]
        plt.plot([0, 1], [y0, y1], alpha=0.7)

    plt.xticks([0, 1], [f"cluster {target}", "cluster 1"])
    plt.ylabel("Patient-mean score_cytotoxic")
    plt.title(f"Paired per-patient cytotoxicity: cluster {target} vs 1 (n={both.shape[0]})")

    # add p-value
    diff = both["1"] - both[target]
    stat, p = wilcoxon_signed_rank(diff.values)
    if not np.isnan(p):
        plt.text(0.02, 0.95, f"Wilcoxon p={p:.3g}", transform=plt.gca().transAxes, va="top")

    savefig_all(os.path.join(outdir, f"fig_paired_patient_cytotoxic_cluster{target}_vs_1"), dpi=300)

# 5E Patient-level cluster proportion barplot (focus on clusters 0/1/2/3)
# (This is a strong figure for "NR enriched states")
ct_focus = ct.copy()
keep_cols = [c for c in focus_present if c in ct_focus.columns]
ct_focus = ct_focus[keep_cols].fillna(0)

# if respond exists, order patients by respond then by cluster0 proportion
patient_order = ct_focus.index.tolist()
if respond_key in adata.obs.columns:
    pat_resp = adata.obs[[patient_key, respond_key]].drop_duplicates().set_index(patient_key)
    pat_resp.index = pat_resp.index.astype(str)
    tmp = ct_focus.join(pat_resp, how="left")
    # sort: Non-responder first (if string), then cluster0 proportion
    # robust sort key:
    def resp_rank(x):
        if pd.isna(x): return 9
        xs = str(x).lower()
        if "non" in xs: return 0
        if "resp" in xs: return 1
        try:
            return int(x)
        except Exception:
            return 9
    tmp["_resp_rank"] = tmp[respond_key].map(resp_rank)
    tmp = tmp.sort_values(["_resp_rank"] + (["0"] if "0" in tmp.columns else []), ascending=[True, False] if "0" in tmp.columns else [True])
    patient_order = tmp.index.tolist()

plt.figure(figsize=(12, 4))
bottom = np.zeros(len(patient_order))
x = np.arange(len(patient_order))

for c in keep_cols:
    vals = ct_focus.loc[patient_order, c].values
    plt.bar(x, vals, bottom=bottom, label=f"cluster {c}")
    bottom += vals

plt.xticks(x, patient_order, rotation=90, fontsize=7)
plt.ylabel("Fraction of T cells (within patient)")
plt.title("Per-patient cluster composition (focus clusters)")
plt.legend(ncol=min(4, len(keep_cols)), frameon=False, fontsize=8)
plt.tight_layout()
savefig_all(os.path.join(outdir, "fig_patient_cluster_composition_focus_stackedbar"), dpi=300)

# ---------------------------------
# 6) Minimal "interpretation-ready" table:
#    medians of scores for clusters 0/1/2/3, plus delta vs cluster1
# ---------------------------------
cluster_medians = (
    ad_focus.obs
    .groupby(ad_focus.obs[cluster_key].astype(str))[score_cols]
    .median()
)
if "1" in cluster_medians.index:
    for t in ["0", "2", "3"]:
        if t in cluster_medians.index:
            cluster_medians.loc[t, "delta_cytotoxic_vs_1"] = cluster_medians.loc[t, "score_cytotoxic"] - cluster_medians.loc["1", "score_cytotoxic"]
            cluster_medians.loc[t, "delta_activation_vs_1"] = cluster_medians.loc[t, "score_activation"] - cluster_medians.loc["1", "score_activation"]
cluster_medians.to_csv(os.path.join(outdir, "table_cluster_median_scores_focus_plus_delta_vs_1.csv"))

print("\n[Done] Outputs saved to:")
print(outdir)
print("\nKey files to review first:")
print(" - fig_dotplot_marker_panel_focus_clusters.(png/pdf)")
print(" - fig_violin_score_*_by_cluster_focus.(png/pdf)")
print(" - fig_paired_patient_cytotoxic_cluster0/2/3_vs_1.(png/pdf)")
print(" - fig_patient_cluster_composition_focus_stackedbar.(png/pdf)")
print(" - table_patient_paired_cytotoxic_cluster1_vs_others.csv")
print(" - table_cluster_score_summary.csv")
