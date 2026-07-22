"""
Reviewer #2, point 5 -- two additional robustness analyses that do NOT depend on
calling discrete "terminal states" (which Palantir showed is model-dependent):

(A) PAGA connectivity: a graph-abstraction of cluster-cluster connectivity that
    makes no lineage/terminal claim. It tests whether the branching TOPOLOGY
    (naive C1 connects to effector C3 and to exhausted C4, through transitional C2)
    is present in the data itself. Robust and standard.

(B) Bootstrap late-state recovery: subsample cells many times, recompute DPT, and
    record which clusters occupy the latest-pseudotime tail each time. If the
    effector (C3) and exhausted (C4) clusters are consistently the late states,
    the "two late states" claim is stable even if we don't assert absorbing
    terminals.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5_trajectory_robustness as R5

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r5_trajectory"))
CLUSTER_KEY = R5.CLUSTER_KEY
PAPER = U.LEIDEN_TO_PAPER  # {"4":"C1","0":"C2","1":"C3","2":"C4"}
N_BOOT = 50
LATE_FRAC = 0.10


def load():
    ad = sc.read_h5ad(R5.H5AD)
    ad = ad[ad.obs[CLUSTER_KEY].astype(str).isin(R5.ALL_CLUSTERS)].copy()
    R5.compute_naive_score(ad)
    sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=50, random_state=0)
    return ad


def paga_connectivity(ad):
    sc.tl.paga(ad, groups=CLUSTER_KEY)
    conn = ad.uns["paga"]["connectivities"].toarray()
    cats = list(ad.obs[CLUSTER_KEY].cat.categories) if hasattr(
        ad.obs[CLUSTER_KEY], "cat") else sorted(ad.obs[CLUSTER_KEY].unique())
    df = pd.DataFrame(conn, index=cats, columns=cats)
    df.index = [f"{c}({PAPER.get(str(c), c)})" for c in df.index]
    df.columns = [f"{c}({PAPER.get(str(c), c)})" for c in df.columns]
    df.to_csv(os.path.join(OUTDIR, "paga_connectivity.csv"))
    print("[r5c] PAGA connectivity (leiden(paper)):")
    print(df.round(3).to_string())
    # key edges
    def edge(a, b):
        ia = [i for i, c in enumerate(cats) if str(c) == a][0]
        ib = [i for i, c in enumerate(cats) if str(c) == b][0]
        return round(float(conn[ia, ib]), 3)
    print("\n[r5c] key edges: C1-C2 =", edge("4", "0"),
          "| C2-C3 =", edge("0", "1"), "| C2-C4 =", edge("0", "2"),
          "| C1-C3 =", edge("4", "1"), "| C1-C4 =", edge("4", "2"),
          "| C3-C4 =", edge("1", "2"))


def bootstrap_late_states(ad):
    rng = np.random.default_rng(0)
    n = ad.n_obs
    counts = {c: 0 for c in R5.ALL_CLUSTERS}   # times cluster is the top late-cluster
    frac_in_late = {c: [] for c in R5.ALL_CLUSTERS}
    rows = []
    for b in range(N_BOOT):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        sub = ad[idx].copy()
        sc.pp.neighbors(sub, use_rep="X_scVI", n_neighbors=50, random_state=0)
        sub.uns["iroot"] = R5.pick_root(sub, rep="X_scVI", strategy="medoid")
        sc.tl.diffmap(sub)
        sc.tl.dpt(sub)
        t = sub.obs["dpt_pseudotime"].values
        thr = np.quantile(t, 1 - LATE_FRAC)
        late = sub.obs[CLUSTER_KEY].astype(str).values[t >= thr]
        vc = pd.Series(late).value_counts(normalize=True)
        for c in R5.ALL_CLUSTERS:
            frac_in_late[c].append(float(vc.get(c, 0.0)))
        top = vc.index[0]
        if top in counts:
            counts[top] += 1
        rows.append({"boot": b, **{f"late_frac_{PAPER[c]}": float(vc.get(c, 0.0))
                                   for c in R5.ALL_CLUSTERS}})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTDIR, "bootstrap_late_state_fractions.csv"), index=False)
    summ = pd.DataFrame({
        "cluster": [PAPER[c] for c in R5.ALL_CLUSTERS],
        "mean_frac_in_late_tail": [np.mean(frac_in_late[c]) for c in R5.ALL_CLUSTERS],
        "sd": [np.std(frac_in_late[c]) for c in R5.ALL_CLUSTERS],
        "times_top_late_state": [counts[c] for c in R5.ALL_CLUSTERS],
        "n_boot": N_BOOT,
    })
    summ.to_csv(os.path.join(OUTDIR, "bootstrap_late_state_summary.csv"), index=False)
    print("\n[r5c] bootstrap late-state recovery (top", int(LATE_FRAC*100),
          "% pseudotime,", N_BOOT, "resamples):")
    print(summ.round(3).to_string(index=False))


def main():
    ad = load()
    paga_connectivity(ad)
    bootstrap_late_states(ad)
    print("\n[r5c DONE] outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
