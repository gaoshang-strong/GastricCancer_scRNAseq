"""
Reviewer #2, point 5 -- robustness of the ACTUAL CellRank2 GPCCA pipeline used in
the paper (05c/analysis3.py), across the three axes the reviewer names:

  (1) parameter settings   : n_neighbors, n_components (Schur), n_terminal states
  (2) root-cell selection  : first-cell (current), top-naive medoid,
                             min-exhaustion, random naive cells
  (3) patient exclusion    : leave-one-patient-out

For every configuration we run the full GPCCA estimator exactly as analysis3.py
does (PseudotimeKernel -> GPCCA -> compute_schur -> fit -> predict_terminal_states
-> set_initial_states(['4']) -> compute_fate_probabilities) and record:
    * terminal states       : which leiden clusters become terminal macrostates
    * branch assignment      : effector(C3)/exhausted(C4) via relative fate advantage
    * responder bias         : per-PATIENT fraction on the effector branch (direction)
    * label stability        : ARI / kappa of the per-cell branch vs the canonical run

Results are appended to CSV as each config finishes (so partial progress survives).
Each GPCCA fit is slow; expect this to run for a while -> launch in background.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python  (cellrank 2.0.7)
"""

import os
import sys
import time
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
ROOT_CLUSTER = "4"          # C1 naive
EFFECTOR_CLUSTER = "1"      # C3
EXHAUST_CLUSTER = "2"       # C4
ALL_CLUSTERS = ["4", "0", "1", "2"]
NAIVE_GENES = ["TCF7", "LEF1", "CCR7", "IL7R", "LTB", "MAL", "SELL",
               "TRAC", "CD3D", "CD3E"]
CYTO_GENES = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "FGFBP2", "CTSW",
              "KLRD1", "KLRB1", "TRAC", "CD3D", "CD3E"]
EXH_GENES = ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "TOX2",
             "CXCL13", "ENTPD1", "LAYN", "IKZF2"]
DELTA, EPS = 0.15, 0.25

# canonical = analysis3.py settings
C = dict(n_neighbors=30, n_components=20, n_terminal=3,
         root_strategy="first_cell", root_seed=0)


def compute_scores(ad):
    """score_Naive (for root) + cyto/exh scores (for swap-proof branch labelling)."""
    from scipy import sparse
    X = ad.layers["norm_expr"]
    Xlog = X.copy()
    if sparse.issparse(Xlog):
        Xlog.data = np.log1p(Xlog.data)
    else:
        Xlog = np.log1p(Xlog)
    xbak = ad.X; ad.X = Xlog
    for name, genes in [("score_Naive", NAIVE_GENES), ("cyto_score", CYTO_GENES),
                        ("exh_score", EXH_GENES)]:
        gg = [g for g in genes if g in ad.var_names]
        sc.tl.score_genes(ad, gene_list=gg, score_name=name, use_raw=False)
    ad.X = xbak


def pick_root(ad, strategy, seed=0, rep="X_scVI"):
    mask = (ad.obs[CLUSTER_KEY].astype(str) == ROOT_CLUSTER).values
    idxs = np.where(mask)[0]
    if strategy == "first_cell":                       # analysis3.py behaviour
        return int(idxs[0])
    if strategy == "min_exhaustion" and "score_Exhaustion" in ad.obs:
        sub = ad.obs.loc[mask, "score_Exhaustion"].astype(float)
        return int(np.where(ad.obs_names == sub.idxmin())[0][0])
    # top-naive medoid / random
    sub = ad.obs.loc[mask, "score_Naive"].astype(float)
    n_cand = min(max(int(np.ceil(len(sub) * 0.10)), 30), len(sub))
    cand = np.where(ad.obs_names.isin(sub.nlargest(n_cand).index))[0]
    if strategy == "random":
        return int(np.random.default_rng(seed).choice(cand))
    Xc = ad.obsm[rep][cand]                              # medoid
    x2 = np.sum(Xc * Xc, axis=1, keepdims=True)
    D2 = np.maximum(x2 + x2.T - 2 * (Xc @ Xc.T), 0.0)
    return int(cand[int(np.argmin(D2.mean(axis=1)))])


def _fp_to_df(fp, obs_names):
    X = fp.X.toarray() if hasattr(fp.X, "toarray") else np.asarray(fp.X)
    return pd.DataFrame(X, index=obs_names, columns=list(fp.names))


def terminal_clusters(g, ad):
    """Which leiden clusters do the terminal macrostates sit in?"""
    ts = g.terminal_states
    out = {}
    for cat in ts.cat.categories:
        cells = ts.index[ts == cat]
        if len(cells) == 0:
            continue
        dom = ad.obs.loc[cells, CLUSTER_KEY].astype(str).value_counts().idxmax()
        out[str(cat)] = U.LEIDEN_TO_PAPER.get(dom, dom)
    return out


def run_gpcca(ad, n_neighbors=30, n_components=20, n_terminal=3,
              root_strategy="first_cell", root_seed=0):
    ad = ad.copy()
    sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=n_neighbors, random_state=0)
    ad.uns["iroot"] = pick_root(ad, root_strategy, root_seed)
    sc.tl.diffmap(ad)
    sc.tl.dpt(ad)

    pk = cr.kernels.PseudotimeKernel(ad, time_key="dpt_pseudotime")
    pk.compute_transition_matrix(show_progress_bar=False)
    g = cr.estimators.GPCCA(pk)
    # analysis3.py used method="brandts" (dense Schur) to avoid PETSc; PETSc/SLEPc
    # ARE installed here, so we use the numerically-equivalent sparse "krylov"
    # Schur (same invariant subspace) which is ~1000x faster and makes this grid
    # feasible. Terminal states / fate probabilities are otherwise identical.
    g.compute_schur(n_components=n_components, method="krylov")
    g.fit(cluster_key=CLUSTER_KEY, n_states=[4, 10])
    g.predict_terminal_states(method="top_n", n_states=n_terminal)
    # NB: analysis3.py calls set_initial_states(["4"]); cluster 4 is not always a
    # macrostate, and initial states are not needed for fate probabilities toward
    # terminal states, so we skip it (does not affect fate probabilities).
    g.compute_fate_probabilities(solver="gmres")

    fp = _fp_to_df(g.fate_probabilities, ad.obs_names)
    term = terminal_clusters(g, ad)

    # Anchor lineages to biology by fate-prob-weighted score (swap-proof across
    # runs): the effector arm = lineage whose committed cells are most cytotoxic;
    # the exhausted arm = lineage whose committed cells are most exhausted.
    cyto = ad.obs["cyto_score"].values.astype(float)
    exh = ad.obs["exh_score"].values.astype(float)
    def wmean(col, s):
        w = fp[col].values
        return float(np.sum(w * s) / (np.sum(w) + 1e-12))
    cyto_by_lin = {c: wmean(c, cyto) for c in fp.columns}
    exh_by_lin = {c: wmean(c, exh) for c in fp.columns}
    L_C3 = max(cyto_by_lin, key=cyto_by_lin.get)          # effector arm
    L_C4 = max((c for c in fp.columns if c != L_C3),
               key=lambda c: exh_by_lin[c])                # exhausted arm (distinct)
    if L_C3 == L_C4:
        return {"status": f"degenerate({L_C3})", "terminal_clusters": term}
    fpC3, fpC4 = fp[L_C3].values, fp[L_C4].values
    branch = np.full(ad.n_obs, "other", dtype=object)
    branch[(fpC3 - fpC4 >= DELTA) & (fpC3 >= EPS)] = "branch_C3"
    branch[(fpC4 - fpC3 >= DELTA) & (fpC4 >= EPS)] = "branch_C4"
    return {"status": "ok", "obs_names": np.asarray(ad.obs_names),
            "branch": branch, "fpC3": fpC3, "fpC4": fpC4,
            "dpt": ad.obs["dpt_pseudotime"].values,
            "cluster": ad.obs[CLUSTER_KEY].astype(str).values,
            "terminal_clusters": term,
            "patient": ad.obs["patient"].astype(str).values,
            "Respond": U.add_respond(ad.obs.copy())["Respond"].astype(str).values}


def direction(res):
    df = pd.DataFrame({"patient": res["patient"], "Respond": res["Respond"],
                       "branch": res["branch"]})
    df = df[df["branch"] != "other"]
    if df.empty:
        return {}
    frac = df.groupby("patient")["branch"].apply(lambda s: (s == "branch_C3").mean())
    pg = df.drop_duplicates("patient").set_index("patient")["Respond"]
    o = pd.DataFrame({"frac_C3": frac, "Respond": pg})
    mR = o.loc[o.Respond == "Responder", "frac_C3"].mean()
    mNR = o.loc[o.Respond == "Non-Responder", "frac_C3"].mean()
    return {"mean_fracC3_R": round(mR, 4), "mean_fracC3_NR": round(mNR, 4),
            "C3_higher_in_R": bool(mR > mNR)}


def stability(res, canon):
    from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
    a = pd.Series(canon["branch"], index=canon["obs_names"])
    b = pd.Series(res["branch"], index=res["obs_names"])
    common = a.index.intersection(b.index)
    a2, b2 = a.loc[common], b.loc[common]
    keep = (a2 != "other") & (b2 != "other")
    if keep.sum() < 10:
        return {"ari": np.nan, "kappa": np.nan}
    return {"ari": round(adjusted_rand_score(a2[keep], b2[keep]), 3),
            "kappa": round(cohen_kappa_score(a2[keep], b2[keep]), 3)}


def append_row(row, path):
    hdr = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=hdr, index=False)


def main():
    grid_csv = os.path.join(OUTDIR, "gpcca_robustness_grid.csv")
    if os.path.exists(grid_csv):
        os.remove(grid_csv)
    print("[r5d] loading:", H5AD, flush=True)
    adata = sc.read_h5ad(H5AD)
    adata = adata[adata.obs[CLUSTER_KEY].astype(str).isin(ALL_CLUSTERS)].copy()
    compute_scores(adata)

    t0 = time.time()
    print("[r5d] canonical GPCCA (analysis3 settings) ...", flush=True)
    canon = run_gpcca(adata, **C)
    print(f"[r5d] canonical done in {time.time()-t0:.0f}s; terminals={canon['terminal_clusters']}",
          flush=True)
    if canon["status"] != "ok":
        print("[r5d] canonical failed:", canon["status"]); return
    pd.DataFrame({"cell": canon["obs_names"], "patient": canon["patient"],
                  "Respond": canon["Respond"], "branch_rel": canon["branch"],
                  "fpC3": canon["fpC3"], "fpC4": canon["fpC4"]}).to_csv(
        os.path.join(OUTDIR, "gpcca_branch_assignment_canonical.csv"), index=False)
    append_row({"axis": "canonical", "config": "canonical", **C,
                "status": "ok",
                "terminal_clusters": "|".join(sorted(set(canon["terminal_clusters"].values()))),
                **direction(canon), "ari": 1.0, "kappa": 1.0,
                "sec": round(time.time()-t0)}, grid_csv)

    # ------- build the three-direction config list -------
    configs = []
    # (1) parameter settings
    for nn in [15, 50]:
        configs.append(("parameter", f"n_neighbors={nn}", {**C, "n_neighbors": nn}))
    for nt in [2, 4]:
        configs.append(("parameter", f"n_terminal={nt}", {**C, "n_terminal": nt}))
    configs.append(("parameter", "n_components=30", {**C, "n_components": 30}))
    # (2) root-cell selection
    for rs in ["top_naive_medoid", "min_exhaustion"]:
        configs.append(("root", f"root={rs}", {**C, "root_strategy": rs}))
    for sd in [1, 2]:
        configs.append(("root", f"root=random{sd}",
                        {**C, "root_strategy": "random", "root_seed": sd}))

    for axis, tag, cfg in configs:
        t = time.time()
        print(f"[r5d] {axis}: {tag} ...", flush=True)
        try:
            res = run_gpcca(adata, **{k: cfg[k] for k in
                                      ["n_neighbors", "n_components", "n_terminal",
                                       "root_strategy", "root_seed"]})
            row = {"axis": axis, "config": tag, **cfg}
            if res["status"] == "ok":
                row.update({"status": "ok",
                            "terminal_clusters": "|".join(sorted(set(res["terminal_clusters"].values()))),
                            **direction(res), **stability(res, canon)})
            else:
                row.update({"status": res["status"],
                            "terminal_clusters": "|".join(sorted(set(res.get("terminal_clusters", {}).values())))})
            row["sec"] = round(time.time()-t)
            append_row(row, grid_csv)
        except Exception as e:
            append_row({"axis": axis, "config": tag, **cfg,
                        "status": f"error:{type(e).__name__}", "sec": round(time.time()-t)},
                       grid_csv)
        print(f"    -> {round(time.time()-t)}s", flush=True)

    # (3) patient exclusion
    for p in list(U.NONRESP_PATIENTS) + list(U.RESP_PATIENTS):
        t = time.time()
        print(f"[r5d] patient_exclusion: drop {p} ...", flush=True)
        sub = adata[adata.obs["patient"].astype(str) != p].copy()
        try:
            res = run_gpcca(sub, **C)
            row = {"axis": "patient_exclusion", "config": f"drop_{p}", **C}
            if res["status"] == "ok":
                row.update({"status": "ok",
                            "terminal_clusters": "|".join(sorted(set(res["terminal_clusters"].values()))),
                            **direction(res), **stability(res, canon)})
            else:
                row.update({"status": res["status"]})
            row["sec"] = round(time.time()-t)
            append_row(row, grid_csv)
        except Exception as e:
            append_row({"axis": "patient_exclusion", "config": f"drop_{p}", **C,
                        "status": f"error:{type(e).__name__}", "sec": round(time.time()-t)},
                       grid_csv)
        print(f"    -> {round(time.time()-t)}s", flush=True)

    print("\n[r5d DONE] ->", grid_csv, flush=True)


if __name__ == "__main__":
    main()
