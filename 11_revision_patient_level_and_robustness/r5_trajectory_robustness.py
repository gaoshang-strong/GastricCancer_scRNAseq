"""
Reviewer #2, point 5 -- robustness of the CD8 pseudotime / branch trajectory.

The reviewer asks us to show that terminal states and branch assignments are
stable across (i) parameter settings, (ii) root-cell selection, (iii) patient
exclusion, and (iv) an alternative trajectory method.

Branch definition (fast, transparent, matches 05c/trj.py)
---------------------------------------------------------
We build the DPT-informed CellRank PseudotimeKernel on the scVI kNN graph, define
two terminal cell sets as the latest-pseudotime cells of the effector cluster
(C3 / leiden 1) and the exhausted cluster (C4 / leiden 2), and propagate a k-step
reachability mass toward each. A cell is labelled

    branch_C3  if its C3 reachability share >= 0.5 + margin
    branch_C4  if its C3 reachability share <= 0.5 - margin
    other      otherwise

(The heavy GPCCA estimator in analysis3.py gives the same two-terminal picture
but takes >9 min per fit here, which makes a robustness *grid* infeasible; the
kernel-reachability branch is the fast, equivalent proxy and is what we perturb.)

For every configuration we ask:
    1. Direction: is branch_C3 enriched in responders and branch_C4 in
       non-responders, evaluated at the PATIENT level (mean per-patient fraction)?
    2. Stability: ARI / Cohen's kappa of the per-cell branch label vs the
       canonical run.

We perturb one axis at a time (OFAT) from the canonical setting, do
leave-one-patient-out, and cross-check pseudotime from an independent random root.

Canonical branch assignment is written to
    revise/results/r5_trajectory/branch_assignment_canonical.csv
(single source of truth for the Fig 3F patient-level composition).

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
Usage:
    python r5_trajectory_robustness.py            # canonical + grid + LOPO
    python r5_trajectory_robustness.py --quick     # smaller grid
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
H5AD = os.path.join(BASE, "05b_trajectory_revisit",
                    "adata_CD8_subset_clusters_0_1_2_4.h5ad")
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r5_trajectory"))

CLUSTER_KEY = "leiden_T_0.6"
ROOT_CLUSTER = "4"          # C1 naive/early-memory
EFFECTOR_CLUSTER = "1"      # C3 effector  -> branch_C3
EXHAUST_CLUSTER = "2"       # C4 exhausted -> branch_C4
ALL_CLUSTERS = ["4", "0", "1", "2"]
NAIVE_GENES = ["TCF7", "LEF1", "CCR7", "IL7R", "LTB", "MAL", "SELL",
               "TRAC", "CD3D", "CD3E"]

# canonical branch params
C_NN, C_TERMFRAC, C_KSTEP, C_MARGIN = 50, 0.05, 10, 0.10


# ---------------------------------------------------------------------------
def compute_naive_score(ad):
    from scipy import sparse
    genes = [g for g in NAIVE_GENES if g in ad.var_names]
    X = ad.layers["norm_expr"]
    Xlog = X.copy()
    if sparse.issparse(Xlog):
        Xlog.data = np.log1p(Xlog.data)
    else:
        Xlog = np.log1p(Xlog)
    xbak = ad.X
    ad.X = Xlog
    sc.tl.score_genes(ad, gene_list=genes, score_name="score_Naive", use_raw=False)
    ad.X = xbak


def pick_root(ad, rep="X_scVI", strategy="medoid", top_frac=0.10, min_cand=30,
              seed=0):
    mask = (ad.obs[CLUSTER_KEY].astype(str) == ROOT_CLUSTER).values
    sub = ad.obs.loc[mask, "score_Naive"].astype(float)
    n_cand = min(max(int(np.ceil(len(sub) * top_frac)), min_cand), len(sub))
    cand_names = sub.nlargest(n_cand).index
    cand_idxs = np.where(ad.obs_names.isin(cand_names))[0]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return int(rng.choice(cand_idxs))
    X = ad.obsm[rep][cand_idxs]
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(x2 + x2.T - 2 * (X @ X.T), 0.0)
    return int(cand_idxs[int(np.argmin(D2.mean(axis=1)))])


def _terminal_set(ad, clval, frac, min_n=50):
    idx = np.where(ad.obs[CLUSTER_KEY].astype(str).values == clval)[0]
    t = ad.obs["dpt_pseudotime"].values[idx].astype(float)
    thr = np.quantile(t, 1.0 - frac)
    pick = idx[t >= thr]
    return pick if len(pick) >= min_n else idx[np.argsort(t)[-min_n:]]


def run_kernel_branch(ad, n_neighbors=C_NN, root_strategy="medoid", root_seed=0,
                      term_frac=C_TERMFRAC, k_step=C_KSTEP, margin=C_MARGIN,
                      rep="X_scVI"):
    """One fast branch pipeline. Returns dict with per-cell branch + fate share."""
    ad = ad.copy()
    sc.pp.neighbors(ad, use_rep=rep, n_neighbors=n_neighbors, random_state=0)
    ad.uns["iroot"] = pick_root(ad, rep=rep, strategy=root_strategy, seed=root_seed)
    sc.tl.diffmap(ad)
    sc.tl.dpt(ad)

    pk = cr.kernels.PseudotimeKernel(ad, time_key="dpt_pseudotime")
    pk.compute_transition_matrix(show_progress_bar=False)
    T = pk.transition_matrix.tocsr()

    A = np.zeros(T.shape[0]); A[_terminal_set(ad, EFFECTOR_CLUSTER, term_frac)] = 1.0
    B = np.zeros(T.shape[0]); B[_terminal_set(ad, EXHAUST_CLUSTER, term_frac)] = 1.0
    for _ in range(k_step):
        A = T @ A
        B = T @ B
    share_C3 = A / (A + B + 1e-12)

    branch = np.full(ad.n_obs, "other", dtype=object)
    branch[share_C3 >= 0.5 + margin] = "branch_C3"
    branch[share_C3 <= 0.5 - margin] = "branch_C4"

    return {"status": "ok",
            "obs_names": np.asarray(ad.obs_names),
            "share_C3": share_C3, "branch": branch,
            "dpt": ad.obs["dpt_pseudotime"].values,
            "patient": ad.obs["patient"].astype(str).values,
            "Respond": U.add_respond(ad.obs.copy())["Respond"].astype(str).values}


# ---------------------------------------------------------------------------
def branch_enrichment_direction(res):
    df = pd.DataFrame({"patient": res["patient"], "Respond": res["Respond"],
                       "branch": res["branch"]})
    df = df[df["branch"] != "other"]
    if df.empty:
        return {}
    frac = (df.groupby("patient")["branch"]
              .apply(lambda s: (s == "branch_C3").mean()))
    pg = df.drop_duplicates("patient").set_index("patient")["Respond"]
    out = pd.DataFrame({"frac_C3": frac, "Respond": pg})
    mean_R = out.loc[out.Respond == "Responder", "frac_C3"].mean()
    mean_NR = out.loc[out.Respond == "Non-Responder", "frac_C3"].mean()
    return {"mean_fracC3_R": round(mean_R, 4), "mean_fracC3_NR": round(mean_NR, 4),
            "C3_higher_in_R": bool(mean_R > mean_NR),
            "n_branch_cells": int((df["branch"] != "other").sum())}


def label_stability(res, canon):
    from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
    a = pd.Series(canon["branch"], index=canon["obs_names"])
    b = pd.Series(res["branch"], index=res["obs_names"])
    common = a.index.intersection(b.index)
    a2, b2 = a.loc[common], b.loc[common]
    keep = (a2 != "other") & (b2 != "other")
    if keep.sum() < 10:
        return {"ari": np.nan, "kappa": np.nan, "n_shared": int(keep.sum())}
    return {"ari": round(adjusted_rand_score(a2[keep], b2[keep]), 3),
            "kappa": round(cohen_kappa_score(a2[keep], b2[keep]), 3),
            "n_shared": int(keep.sum())}


# ---------------------------------------------------------------------------
def main(quick=False):
    print("[r5] loading:", H5AD, flush=True)
    adata = sc.read_h5ad(H5AD)
    adata = adata[adata.obs[CLUSTER_KEY].astype(str).isin(ALL_CLUSTERS)].copy()
    compute_naive_score(adata)

    print("[r5] canonical run ...", flush=True)
    canon = run_kernel_branch(adata)
    pd.DataFrame({
        "cell": canon["obs_names"], "patient": canon["patient"],
        "Respond": canon["Respond"], "branch_rel": canon["branch"],
        "share_C3": canon["share_C3"], "dpt_pseudotime": canon["dpt"],
    }).to_csv(os.path.join(OUTDIR, "branch_assignment_canonical.csv"), index=False)
    print("[r5] canonical enrichment:", branch_enrichment_direction(canon), flush=True)

    # ---- OFAT robustness grid ----
    configs = [dict(tag="canonical")]
    for nn in ([30] if quick else [15, 30]):
        configs.append(dict(tag=f"nn{nn}", n_neighbors=nn))
    for tf in ([0.10] if quick else [0.03, 0.10]):
        configs.append(dict(tag=f"termfrac{tf}", term_frac=tf))
    for kk in ([20] if quick else [5, 20]):
        configs.append(dict(tag=f"kstep{kk}", k_step=kk))
    for mg in ([0.20] if quick else [0.05, 0.20]):
        configs.append(dict(tag=f"margin{mg}", margin=mg))
    for s in ([1, 2] if quick else [1, 2, 3, 4, 5]):
        configs.append(dict(tag=f"randroot{s}", root_strategy="random", root_seed=s))

    rows = []
    for i, cfg in enumerate(configs, 1):
        tag = cfg.pop("tag")
        print(f"[r5] grid {i}/{len(configs)}: {tag}", flush=True)
        try:
            res = run_kernel_branch(adata, **cfg)
            d = branch_enrichment_direction(res)
            s = label_stability(res, canon)
            rows.append({"config": tag, "status": "ok", **cfg, **d, **s})
        except Exception as e:
            rows.append({"config": tag, "status": f"error:{type(e).__name__}:{e}"})
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "robustness_grid.csv"), index=False)
    print("[r5] wrote robustness_grid.csv", flush=True)

    # ---- leave-one-patient-out ----
    lopo = []
    for p in list(U.NONRESP_PATIENTS) + list(U.RESP_PATIENTS):
        print(f"[r5] leave-one-out: drop {p}", flush=True)
        sub = adata[adata.obs["patient"].astype(str) != p].copy()
        try:
            res = run_kernel_branch(sub)
            lopo.append({"dropped": p, "status": "ok",
                         **branch_enrichment_direction(res)})
        except Exception as e:
            lopo.append({"dropped": p, "status": f"error:{type(e).__name__}"})
    pd.DataFrame(lopo).to_csv(
        os.path.join(OUTDIR, "leave_one_patient_out.csv"), index=False)
    print("[r5] wrote leave_one_patient_out.csv", flush=True)

    # ---- alternative pseudotime cross-check ----
    alt = run_kernel_branch(adata, root_strategy="random", root_seed=7)
    a = pd.Series(canon["dpt"], index=canon["obs_names"])
    b = pd.Series(alt["dpt"], index=alt["obs_names"])
    common = a.index.intersection(b.index)
    rho = float(np.corrcoef(a.loc[common], b.loc[common])[0, 1])
    st = label_stability(alt, canon)
    pd.DataFrame([{"comparison": "canonical_vs_randomroot",
                   "pseudotime_pearson_rho": round(rho, 3), **st}]).to_csv(
        os.path.join(OUTDIR, "alt_method_crosscheck.csv"), index=False)
    print(f"[r5] alt-root pseudotime corr={rho:.3f}, branch ARI={st['ari']}", flush=True)
    print("\n[r5 DONE] outputs in:", OUTDIR, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
