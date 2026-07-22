# 07_02_driver_genes.py
# VS Code debug mode friendly: run top-down, set breakpoints where needed.
#
# Goal:
#   Use your DE lists (from your CD8 AnnData) + Perturb-seq DE_stats (GWCD4i.DE_stats.h5ad)
#   to rank "driver" perturbation targets that best reproduce each DE signature.
#
# Key idea:
#   - Build a signature vector s (gene symbol space) from each DE CSV (0vs2, 1vs2, cluster1 NRvsR).
#   - For each perturbation target p in Perturb-seq, take its intervention effect vector e_p (zscore/log_fc).
#   - Compute cosine similarity score(p) = cos(e_p, s).
#   - Aggregate scores per perturb target (and optionally per culture condition).
#
# Notes:
#   - Your DE CSVs use gene SYMBOLs (scanpy rank_genes_groups_df -> "names").
#   - Perturb-seq var_names are Ensembl IDs; gene symbols are in p.var["gene_name"].
#   - We align via p.var["gene_name"] -> one ENSG per symbol (first occurrence).
#
# Outputs:
#   OUTDIR/driver_ranking_all.csv
#   OUTDIR/top_drivers_for_0vs2.csv
#   OUTDIR/top_drivers_for_1vs2.csv
#   OUTDIR/top_drivers_for_NRvsR.csv
#   OUTDIR/signature_overlap_counts.csv
#   OUTDIR/run_metadata.txt

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# =========================
# CONFIG (edit these)
# =========================
DE_DIR = r"/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/07_perturb_seq_T_cells"
PERT_H5AD = r"/ShangGaoAIProjects/Zang/single_cell_data/scRNA_reference_dataset/T_cell_perturb_seq/GWCD4i.DE_stats.h5ad"   # <<< 改成你下载的 perturb-seq DE_stats h5ad 路径

OUTDIR = os.path.join(DE_DIR, "step2_driver_ranking")
os.makedirs(OUTDIR, exist_ok=True)

# DE outputs you already generated
DE_0VS2 = os.path.join(DE_DIR, "DE_cluster0_vs_cluster2.full.csv")
DE_1VS2 = os.path.join(DE_DIR, "DE_cluster1_vs_cluster2.full.csv")
DE_NRVS_R = os.path.join(DE_DIR, "DE_cluster1_NR_vs_R.full.csv")

# Signature construction thresholds
SIG_FDR = 0.05
SIG_ABS_LFC = 0.25
SIG_TOPK = 500  # maximum genes in signature

# Perturb-seq selection
# If None: score all conditions together and also keep per-condition aggregation.
# If set to one of {"Rest","Stim8hr","Stim48hr"}: filter before scoring.
CONDITION_KEEP = None

# Effect layer used for perturbation vectors (prefer zscore)
EFFECT_LAYER = "zscore"  # or "log_fc"

# Quality filters on perturbations
USE_QC_FILTERS = True
QC_MIN_CELLS_TARGET = 50           # ignore perturbations with too few cells supporting DE
QC_REQUIRE_ONTARGET_SIGNIFICANT = True
QC_EXCLUDE_OFFTARGET_FLAG = True

# Optional: downstream target gene filtering (avoid noisy low-baseMean genes)
FILTER_DOWNSTREAM_BY_BASEMEAN = True
DOWNSTREAM_BASEMEAN_MIN = 0.1      # apply on p.layers["baseMean"]

# Aggregation method when multiple rows per perturb target (e.g., across chunks/replicates)
AGG_FUNC = "mean"  # "mean" or "median"

# =========================
# Helpers
# =========================
def load_signature_from_de(csv_path, name, fdr=0.05, abs_lfc=0.25, topk=500):
    """
    Build a signature vector s in gene-symbol space from scanpy DE csv:
      s_g = sign(logFC) * |logFC|   for genes passing thresholds, else 0.
    Returns:
      sig_df indexed by gene symbol with columns: S, sign, weight
    """
    df = pd.read_csv(csv_path)
    required = {"names", "logfoldchanges"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns {required}. Found: {df.columns.tolist()[:20]}")

    df = df.copy()
    df["gene"] = df["names"].astype(str)

    # filter
    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= fdr]
    df = df[df["logfoldchanges"].abs() >= abs_lfc]

    # prioritize by abs LFC (you can change to pvals_adj first if preferred)
    df["abs_lfc"] = df["logfoldchanges"].abs()
    df = df.sort_values(["abs_lfc"], ascending=False).head(topk)

    sign = np.sign(df["logfoldchanges"].values).astype(float)
    weight = df["abs_lfc"].values.astype(float)

    sig = pd.DataFrame({"gene": df["gene"].tolist(), "sign": sign, "weight": weight})
    sig["S"] = sig["sign"] * sig["weight"]
    sig = sig.drop_duplicates("gene").set_index("gene")
    sig.name = name
    return sig

def cosine_scores(X, s_vec):
    """
    Compute cosine similarity between each row of X and vector s_vec.
    X: (n_obs, n_genes) dense or sparse
    s_vec: (n_genes,)
    Return: (n_obs,)
    """
    s = s_vec.astype(np.float64)
    s_norm = np.sqrt(np.sum(s * s) + 1e-12)

    if sparse.issparse(X):
        X = X.tocsr()
        num = X.dot(s)
        X_sq = X.multiply(X).sum(axis=1)
        X_norm = np.sqrt(np.asarray(X_sq).reshape(-1) + 1e-12)
        return np.asarray(num).reshape(-1) / (X_norm * s_norm + 1e-12)

    X = np.asarray(X, dtype=np.float64)
    num = X @ s
    X_norm = np.sqrt(np.sum(X * X, axis=1) + 1e-12)
    return num / (X_norm * s_norm + 1e-12)

def build_ensg_by_symbol(p):
    """
    Map gene symbol -> one Ensembl ID column in p.var_names.
    p.var["gene_name"] contains symbols; p.var_names contains ENSG IDs.
    Returns:
      ensg_by_symbol: pd.Series mapping symbol -> ENSG
    """
    if "gene_name" not in p.var.columns:
        raise ValueError("p.var missing 'gene_name' column.")
    sym = p.var["gene_name"].astype(str).values
    ensg = p.var_names.astype(str).values
    # first occurrence per symbol
    ensg_by_symbol = pd.Series(ensg, index=sym).drop_duplicates(keep="first")
    # drop empty/NA symbols
    ensg_by_symbol = ensg_by_symbol[ensg_by_symbol.index.notna()]
    ensg_by_symbol = ensg_by_symbol[ensg_by_symbol.index != "nan"]
    ensg_by_symbol = ensg_by_symbol[ensg_by_symbol.index != ""]
    return ensg_by_symbol

def apply_qc_filters(p):
    """
    Filter perturb-seq rows by QC columns in p.obs.
    """
    if not USE_QC_FILTERS:
        return p

    keep = np.ones(p.n_obs, dtype=bool)

    if "n_cells_target" in p.obs.columns and QC_MIN_CELLS_TARGET is not None:
        keep &= (p.obs["n_cells_target"].astype(float).values >= float(QC_MIN_CELLS_TARGET))

    if QC_REQUIRE_ONTARGET_SIGNIFICANT and "ontarget_significant" in p.obs.columns:
        keep &= (p.obs["ontarget_significant"].astype(bool).values)

    if QC_EXCLUDE_OFFTARGET_FLAG and "offtarget_flag" in p.obs.columns:
        keep &= (~p.obs["offtarget_flag"].astype(bool).values)

    return p[keep].copy()

def filter_genes_by_basemean(p_sub):
    """
    Optionally filter columns (genes) by baseMean.
    baseMean is per row x gene in layers. We want to keep genes with decent baseMean
    across most perturbations. Here we filter by global median baseMean across rows.
    """
    if not FILTER_DOWNSTREAM_BY_BASEMEAN:
        return p_sub

    if "baseMean" not in p_sub.layers:
        return p_sub

    bm = p_sub.layers["baseMean"]
    if sparse.issparse(bm):
        bm = bm.tocsr()
        # approximate median with mean for speed; median is costly for sparse big matrices
        col_stat = np.asarray(bm.mean(axis=0)).reshape(-1)
    else:
        col_stat = np.median(np.asarray(bm), axis=0)

    keep_cols = col_stat >= float(DOWNSTREAM_BASEMEAN_MIN)
    if keep_cols.sum() < 50:
        # avoid over-filtering
        return p_sub
    return p_sub[:, keep_cols].copy()

def score_signature(p, sig, ensg_by_symbol, effect_layer):
    """
    Align signature (symbol) with perturb data (ENSG columns), then compute cosine score per row.
    Returns:
      scores (n_obs,), n_overlap_genes
    """
    # overlap in symbol space
    common_symbols = ensg_by_symbol.index.intersection(sig.index)
    if len(common_symbols) < 30:
        raise ValueError(f"Too few overlap genes for signature '{sig.name}': {len(common_symbols)}")

    # map symbols -> ENSG columns
    common_ensg = ensg_by_symbol.loc[common_symbols].values

    # subset perturb data
    p_sub = p[:, common_ensg].copy()
    p_sub = filter_genes_by_basemean(p_sub)

    # IMPORTANT: after filtering genes by baseMean, recompute overlap vectors
    # We need to match the order of p_sub.var["gene_name"] (symbols) to sig.
    sym_sub = p_sub.var["gene_name"].astype(str).values
    # build S vector in the column order
    S = sig.loc[sym_sub, "S"].values.astype(np.float64)

    X = p_sub.layers[effect_layer]
    scores = cosine_scores(X, S)
    return scores, len(sym_sub)

# =========================
# Main
# =========================
# 1) Load signatures
sig_0vs2 = load_signature_from_de(DE_0VS2, "0_vs_2", fdr=SIG_FDR, abs_lfc=SIG_ABS_LFC, topk=SIG_TOPK)
sig_1vs2 = load_signature_from_de(DE_1VS2, "1_vs_2", fdr=SIG_FDR, abs_lfc=SIG_ABS_LFC, topk=SIG_TOPK)
sig_nrvsr = load_signature_from_de(DE_NRVS_R, "NR_vs_R", fdr=SIG_FDR, abs_lfc=SIG_ABS_LFC, topk=SIG_TOPK)

print("Signature sizes:", {"0vs2": sig_0vs2.shape[0], "1vs2": sig_1vs2.shape[0], "NRvsR": sig_nrvsr.shape[0]})

# 2) Load perturb-seq DE_stats
p = sc.read_h5ad(PERT_H5AD)
print("Perturb AnnData:", p)
print("Pert obs columns:", p.obs.columns.tolist())
print("Pert var columns:", p.var.columns.tolist())
print("Pert layers:", list(p.layers.keys()))

# columns we will use (from your inspection)
PERT_COL = "target_contrast_gene_name"   # perturb target gene symbol
COND_COL = "culture_condition"           # Rest / Stim8hr / Stim48hr

if EFFECT_LAYER not in p.layers:
    raise ValueError(f"EFFECT_LAYER='{EFFECT_LAYER}' not found. Available layers: {list(p.layers.keys())}")

# optional condition filtering
if CONDITION_KEEP is not None:
    p = p[p.obs[COND_COL].astype(str) == str(CONDITION_KEEP)].copy()
    print("Filtered to condition:", CONDITION_KEEP, "n_obs =", p.n_obs)

# QC filtering
p_before = p.n_obs
p = apply_qc_filters(p)
print(f"QC filters applied: n_obs {p_before} -> {p.n_obs}")

# build symbol->ENSG map for alignment
ensg_by_symbol = build_ensg_by_symbol(p)
print("Symbol->ENSG map size:", len(ensg_by_symbol))

# 3) Score each signature
scores_0vs2, overlap_0vs2 = score_signature(p, sig_0vs2, ensg_by_symbol, EFFECT_LAYER)
scores_1vs2, overlap_1vs2 = score_signature(p, sig_1vs2, ensg_by_symbol, EFFECT_LAYER)
scores_nrvsr, overlap_nrvsr = score_signature(p, sig_nrvsr, ensg_by_symbol, EFFECT_LAYER)

# 4) Build row-level results
res = pd.DataFrame({
    "perturb": p.obs[PERT_COL].astype(str).values,
    "condition": p.obs[COND_COL].astype(str).values,
    "target_contrast": p.obs["target_contrast"].astype(str).values if "target_contrast" in p.obs.columns else "",
    "n_cells_target": p.obs["n_cells_target"].values if "n_cells_target" in p.obs.columns else np.nan,
    "ontarget_significant": p.obs["ontarget_significant"].values if "ontarget_significant" in p.obs.columns else np.nan,
    "offtarget_flag": p.obs["offtarget_flag"].values if "offtarget_flag" in p.obs.columns else np.nan,
    "score_0vs2": scores_0vs2,
    "score_1vs2": scores_1vs2,
    "score_NRvsR": scores_nrvsr,
})

# 5) Aggregate to driver ranking
group_cols = ["perturb", "condition"]  # keep per-condition; you can later collapse across condition if desired

if AGG_FUNC == "median":
    agg = (
        res.groupby(group_cols)
           .agg(
               n=("score_0vs2", "size"),
               score_0vs2=("score_0vs2", "median"),
               score_1vs2=("score_1vs2", "median"),
               score_NRvsR=("score_NRvsR", "median"),
           )
           .reset_index()
    )
else:
    agg = (
        res.groupby(group_cols)
           .agg(
               n=("score_0vs2", "size"),
               score_0vs2=("score_0vs2", "mean"),
               score_1vs2=("score_1vs2", "mean"),
               score_NRvsR=("score_NRvsR", "mean"),
           )
           .reset_index()
    )

# also provide "across conditions" aggregation (collapse condition)
if AGG_FUNC == "median":
    agg_allcond = (
        res.groupby(["perturb"])
           .agg(
               n=("score_0vs2", "size"),
               score_0vs2=("score_0vs2", "median"),
               score_1vs2=("score_1vs2", "median"),
               score_NRvsR=("score_NRvsR", "median"),
           )
           .reset_index()
    )
else:
    agg_allcond = (
        res.groupby(["perturb"])
           .agg(
               n=("score_0vs2", "size"),
               score_0vs2=("score_0vs2", "mean"),
               score_1vs2=("score_1vs2", "mean"),
               score_NRvsR=("score_NRvsR", "mean"),
           )
           .reset_index()
    )

# 6) Save outputs
res.to_csv(os.path.join(OUTDIR, "row_scores.csv"), index=False)
agg.to_csv(os.path.join(OUTDIR, "driver_ranking_by_condition.csv"), index=False)
agg_allcond.to_csv(os.path.join(OUTDIR, "driver_ranking_all.csv"), index=False)

def save_top(df, score_col, out_name, topn=50):
    d = df.sort_values(score_col, ascending=False).head(topn).copy()
    d.to_csv(os.path.join(OUTDIR, out_name), index=False)

save_top(agg_allcond, "score_0vs2", "top_drivers_for_0vs2.csv", topn=50)
save_top(agg_allcond, "score_1vs2", "top_drivers_for_1vs2.csv", topn=50)
save_top(agg_allcond, "score_NRvsR", "top_drivers_for_NRvsR.csv", topn=50)

pd.DataFrame({
    "signature": ["0vs2", "1vs2", "NRvsR"],
    "n_overlap_genes_used": [overlap_0vs2, overlap_1vs2, overlap_nrvsr],
    "SIG_FDR": [SIG_FDR]*3,
    "SIG_ABS_LFC": [SIG_ABS_LFC]*3,
    "SIG_TOPK": [SIG_TOPK]*3,
    "EFFECT_LAYER": [EFFECT_LAYER]*3,
}).to_csv(os.path.join(OUTDIR, "signature_overlap_counts.csv"), index=False)

# write metadata for reproducibility
with open(os.path.join(OUTDIR, "run_metadata.txt"), "w") as f:
    f.write(f"PERT_H5AD={PERT_H5AD}\n")
    f.write(f"DE_0VS2={DE_0VS2}\n")
    f.write(f"DE_1VS2={DE_1VS2}\n")
    f.write(f"DE_NRVS_R={DE_NRVS_R}\n")
    f.write(f"CONDITION_KEEP={CONDITION_KEEP}\n")
    f.write(f"EFFECT_LAYER={EFFECT_LAYER}\n")
    f.write(f"USE_QC_FILTERS={USE_QC_FILTERS}\n")
    f.write(f"QC_MIN_CELLS_TARGET={QC_MIN_CELLS_TARGET}\n")
    f.write(f"QC_REQUIRE_ONTARGET_SIGNIFICANT={QC_REQUIRE_ONTARGET_SIGNIFICANT}\n")
    f.write(f"QC_EXCLUDE_OFFTARGET_FLAG={QC_EXCLUDE_OFFTARGET_FLAG}\n")
    f.write(f"FILTER_DOWNSTREAM_BY_BASEMEAN={FILTER_DOWNSTREAM_BY_BASEMEAN}\n")
    f.write(f"DOWNSTREAM_BASEMEAN_MIN={DOWNSTREAM_BASEMEAN_MIN}\n")
    f.write(f"AGG_FUNC={AGG_FUNC}\n")
    f.write(f"SIG_FDR={SIG_FDR}\n")
    f.write(f"SIG_ABS_LFC={SIG_ABS_LFC}\n")
    f.write(f"SIG_TOPK={SIG_TOPK}\n")

print("DONE. Outputs saved to:", OUTDIR)
print("Top drivers (all cond) for 0vs2:")
print(agg_allcond.sort_values("score_0vs2", ascending=False).head(10)[["perturb", "score_0vs2", "n"]].to_string(index=False))
print("Top drivers (all cond) for 1vs2:")
print(agg_allcond.sort_values("score_1vs2", ascending=False).head(10)[["perturb", "score_1vs2", "n"]].to_string(index=False))
print("Top drivers (all cond) for NRvsR:")
print(agg_allcond.sort_values("score_NRvsR", ascending=False).head(10)[["perturb", "score_NRvsR", "n"]].to_string(index=False))
