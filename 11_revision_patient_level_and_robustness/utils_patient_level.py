"""
Shared helpers for patient-level (pseudo-replication-safe) re-analysis.

Reviewer #2, point 1: the biological unit of replication is the patient (n=3 vs 3),
not the individual cell. These helpers make it trivial to (a) aggregate any
per-cell quantity to the patient level, (b) build pseudobulk count matrices for
DE, and (c) compare two response groups with a patient-level test while always
plotting the per-patient values so patient-to-patient variability is visible.

Run with the trajectory env that has scanpy:
    /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python

Nothing here computes a cell-level p-value on purpose.
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Cohort definition (single source of truth, kept identical to the pipeline)
# ---------------------------------------------------------------------------
NONRESP_PATIENTS = ("PHD001", "PHD002", "PHD008")
RESP_PATIENTS = ("PHD003", "PHD004", "PHD009")
RESP_ORDER = ["Non-Responder", "Responder"]

# leiden_T_0.6 -> paper cluster label. This is the manual relabelling the authors
# applied downstream; the authoritative source is the dict in
#   02_extracting_T_cells_and_clustering/figure02.ipynb  (cell 5), run on
#   .../02_adata_T_reclustered_after_drop.h5ad :
#       {"4":"1","0":"2","1":"3","2":"4","3":"5","5":"6","6":"7"}
# i.e. paper cluster 1=naive, 2=transitional, 3=effector, 4=exhausted, 5=Treg,
# 6=CD4 progenitor/CM, 7=cycling.
#
# Independently verified by marker genes (Naive IL7R/CCR7/TCF7/SELL; Transit
# CCL5/GZMA/TOX; Effector GZMK/NKG7/CST7/CTSW/GZMH; Exhaust ENTPD1/ITGAE/CD96):
# in BOTH the reclustered and the filtered T-cell objects, leiden 1 = effector,
# 2 = exhausted, 0 = transitional, 4 = naive (marker means match to 3 decimals
# between the two objects, so clusters 0/1/2 are identical across them).
#
# For the CD8 subset (clusters 0/1/2/4) we use the C1..C4 shorthand:
LEIDEN_TO_PAPER = {"4": "C1", "0": "C2", "1": "C3", "2": "C4"}


def add_respond(obs, patient_col="patient", out_col="Respond"):
    """Ensure a categorical Respond column derived from patient id."""
    mapping = {p: "Non-Responder" for p in NONRESP_PATIENTS}
    mapping.update({p: "Responder" for p in RESP_PATIENTS})
    obs = obs.copy()
    obs[out_col] = obs[patient_col].astype(str).map(mapping)
    obs[out_col] = pd.Categorical(obs[out_col], categories=RESP_ORDER, ordered=True)
    return obs


def patient_group_table(obs, patient_col="patient", resp_col="Respond"):
    """One row per patient -> its response group. Drops patients with no group."""
    tab = (
        obs[[patient_col, resp_col]]
        .astype({patient_col: str})
        .dropna(subset=[resp_col])
        .drop_duplicates()
        .set_index(patient_col)[resp_col]
        .astype(str)
    )
    return tab


# ---------------------------------------------------------------------------
# (a) per-patient composition (fractions within each patient)
# ---------------------------------------------------------------------------
def per_patient_fraction(obs, group_key, patient_col="patient", resp_col="Respond"):
    """
    Fraction of each `group_key` category *within each patient*.

    Returns a long DataFrame: patient, Respond, <group_key>, frac, n_cells_patient.
    This is the correct unit for composition comparisons (one value per patient).
    """
    obs = obs.copy()
    obs[patient_col] = obs[patient_col].astype(str)
    obs[group_key] = obs[group_key].astype(str)

    counts = (
        obs.groupby([patient_col, group_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    n_per_patient = counts.sum(axis=1)
    frac = counts.div(n_per_patient, axis=0)

    pg = patient_group_table(obs, patient_col, resp_col)
    long = (
        frac.reset_index()
        .melt(id_vars=patient_col, var_name=group_key, value_name="frac")
    )
    long[resp_col] = long[patient_col].map(pg)
    long["n_cells_patient"] = long[patient_col].map(n_per_patient)
    long = long.dropna(subset=[resp_col])
    return long


# ---------------------------------------------------------------------------
# (b) pseudobulk counts (sum raw counts per patient, optionally within a cluster)
# ---------------------------------------------------------------------------
def pseudobulk_counts(adata, patient_col="patient", counts_layer="counts",
                      cluster_key=None, cluster_value=None, min_cells=10):
    """
    Sum raw UMI counts across cells for each patient (optionally restricted to
    one cluster). Returns (counts_df: genes x samples, coldata: samples x meta).

    Each column is one patient => valid replicate for DESeq2/edgeR.
    """
    ad = adata
    if cluster_key is not None and cluster_value is not None:
        mask = ad.obs[cluster_key].astype(str) == str(cluster_value)
        ad = ad[mask.values]

    X = ad.layers[counts_layer]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    patients = ad.obs[patient_col].astype(str).values
    uniq = pd.unique(patients)

    cols, keep_samples, ncells = [], [], []
    for p in uniq:
        idx = np.where(patients == p)[0]
        if len(idx) < min_cells:
            continue
        summed = np.asarray(X[idx].sum(axis=0)).ravel()
        cols.append(summed)
        keep_samples.append(p)
        ncells.append(len(idx))

    if not cols:
        return pd.DataFrame(), pd.DataFrame()

    counts_df = pd.DataFrame(
        np.vstack(cols).T, index=ad.var_names.astype(str), columns=keep_samples
    ).astype(int)

    pg = patient_group_table(add_respond(ad.obs.copy()))
    coldata = pd.DataFrame(
        {"patient": keep_samples,
         "Respond": [pg.get(p, "NA") for p in keep_samples],
         "n_cells": ncells}
    ).set_index("patient")
    return counts_df, coldata


# ---------------------------------------------------------------------------
# (c) per-patient mean of any per-cell value (module score, regulon AUC, ...)
# ---------------------------------------------------------------------------
def per_patient_mean(values, patient, respond, agg="mean"):
    """
    Aggregate a per-cell vector to one value per patient.

    values, patient, respond: array-likes aligned by cell.
    Returns DataFrame: patient, Respond, value, n_cells.
    """
    df = pd.DataFrame({"patient": np.asarray(patient).astype(str),
                       "Respond": np.asarray(respond).astype(str),
                       "value": np.asarray(values, dtype=float)})
    df = df[np.isfinite(df["value"].values)]
    func = np.median if agg == "median" else np.mean
    out = (df.groupby("patient")
             .agg(value=("value", func),
                  n_cells=("value", "size"),
                  Respond=("Respond", "first"))
             .reset_index())
    return out


# ---------------------------------------------------------------------------
# Patient-level two-group test (Mann-Whitney; n=3 vs 3)
# ---------------------------------------------------------------------------
def patient_level_test(df, value_col="value", resp_col="Respond"):
    """
    Mann-Whitney U between response groups on patient-level values.
    With n=3 vs 3 the smallest attainable two-sided p is 0.1, so we also
    report the group medians / direction, which are what actually matter here.
    """
    g_nr = df.loc[df[resp_col] == "Non-Responder", value_col].astype(float).dropna().values
    g_r = df.loc[df[resp_col] == "Responder", value_col].astype(float).dropna().values
    res = {"n_NR": len(g_nr), "n_R": len(g_r),
           "median_NR": np.median(g_nr) if len(g_nr) else np.nan,
           "median_R": np.median(g_r) if len(g_r) else np.nan,
           "mean_NR": np.mean(g_nr) if len(g_nr) else np.nan,
           "mean_R": np.mean(g_r) if len(g_r) else np.nan}
    if len(g_nr) >= 2 and len(g_r) >= 2:
        try:
            res["p_mwu"] = mannwhitneyu(g_nr, g_r, alternative="two-sided").pvalue
        except ValueError:
            res["p_mwu"] = np.nan
    else:
        res["p_mwu"] = np.nan
    res["direction"] = ("higher_in_R" if res["mean_R"] > res["mean_NR"]
                        else "higher_in_NR")
    return res


# ---------------------------------------------------------------------------
# Standard plot: box + one dot per patient + patient-level p
# ---------------------------------------------------------------------------
def plot_group_compare(df, value_col, title, ylabel, outpath,
                       resp_col="Respond", annotate=None):
    """Box (no fliers) + jittered per-patient points + p annotation."""
    groups = RESP_ORDER
    data = [df.loc[df[resp_col] == g, value_col].astype(float).dropna().values
            for g in groups]

    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.boxplot(data, labels=[f"{g}\n(n={len(d)})" for g, d in zip(groups, data)],
               showfliers=False, widths=0.55)
    rng = np.random.default_rng(0)
    for j, d in enumerate(data, start=1):
        jit = (rng.random(len(d)) - 0.5) * 0.15
        ax.scatter(np.full(len(d), j) + jit, d, s=45, zorder=3,
                   edgecolors="black", linewidths=0.6)
    ax.set_ylabel(ylabel)
    if annotate is None:
        annotate = patient_level_test(df, value_col, resp_col)
    p = annotate.get("p_mwu", np.nan)
    ptxt = f"MWU p={p:.3g}" if np.isfinite(p) else "MWU p=NA"
    ax.set_title(f"{title}\n{ptxt} (patient-level, n={annotate['n_NR']} vs {annotate['n_R']})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
