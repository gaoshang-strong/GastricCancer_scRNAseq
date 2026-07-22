"""
Reviewer #2, point 5 (alternative method) -- cross-check the CD8 branch structure
with an INDEPENDENT trajectory method (Palantir), not just a different root of the
same CellRank kernel.

We run Palantir on the same scVI diffusion manifold with the same naive-cluster
root. To make this a fair *method* comparison (isolating the branch-probability
algorithm from terminal-state detection), we give Palantir the SAME two terminal
endpoints the canonical analysis uses -- the latest-pseudotime cell of the
effector cluster (C3) and of the exhausted cluster (C4) -- and then:
  * correlate Palantir pseudotime with the canonical CellRank/DPT pseudotime,
  * derive a Palantir branch label (relative fate advantage) and compare it to the
    canonical branch label (ARI / Cohen's kappa),
  * check the responder-vs-non-responder branch direction at the PATIENT level.
We also report, separately, what Palantir's UNCONSTRAINED terminal detection finds.

Agreement here demonstrates the two-terminal effector/exhaustion split (and its
responder bias) is not an artefact of one algorithm.

Depends on r5_trajectory_robustness.py having written branch_assignment_canonical.csv.
Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python  (palantir 1.4.2)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import palantir
import palantir.core as pal_core

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5_trajectory_robustness as R5

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
H5AD = R5.H5AD
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r5_trajectory"))
CANON_CSV = os.path.join(OUTDIR, "branch_assignment_canonical.csv")

CLUSTER_KEY = R5.CLUSTER_KEY
EFFECTOR_CLUSTER = R5.EFFECTOR_CLUSTER   # "1" -> C3
EXHAUST_CLUSTER = R5.EXHAUST_CLUSTER     # "2" -> C4
DELTA, EPS = 0.15, 0.25


def main():
    if not os.path.exists(CANON_CSV):
        print("[r5b] canonical branch CSV missing; run r5_trajectory_robustness.py first.")
        return
    canon = pd.read_csv(CANON_CSV).set_index("cell")

    print("[r5b] loading + prepping CD8 subset ...")
    ad = sc.read_h5ad(H5AD)
    ad = ad[ad.obs[CLUSTER_KEY].astype(str).isin(R5.ALL_CLUSTERS)].copy()
    R5.compute_naive_score(ad)

    # diffusion manifold on the same scVI space + same naive-cluster medoid root
    sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=50, random_state=0)
    sc.tl.diffmap(ad)
    root = ad.obs_names[R5.pick_root(ad, rep="X_scVI", strategy="medoid")]
    print("[r5b] Palantir early_cell:", root)

    dm = pd.DataFrame(ad.obsm["X_diffmap"], index=ad.obs_names,
                      columns=[f"DC{i+1}" for i in range(ad.obsm["X_diffmap"].shape[1])])

    # ---- (i) record what Palantir's UNCONSTRAINED detection finds ----
    pr_auto = pal_core.run_palantir(dm, early_cell=root)
    bp_auto = pr_auto.branch_probs.reindex(ad.obs_names)
    cl = ad.obs[CLUSTER_KEY].astype(str).values
    auto_term_clusters = [
        pd.Series(bp_auto[c].values, index=cl).groupby(level=0).mean().idxmax()
        for c in bp_auto.columns]
    print("[r5b] unconstrained Palantir terminals map to clusters:",
          auto_term_clusters)

    # ---- (ii) fair comparison: fix the SAME two terminal endpoints ----
    # Use a small SET of latest-pseudotime cells in each of C3 and C4 (single-cell
    # terminals make the absorption sensitive to one dominant patient).
    dpt = canon["dpt_pseudotime"]
    def top_cells(clv, n=5):
        cells = ad.obs_names[ad.obs[CLUSTER_KEY].astype(str) == clv]
        cells = cells.intersection(dpt.index)
        return list(dpt.loc[cells].nlargest(n).index)
    tC3, tC4 = top_cells(EFFECTOR_CLUSTER), top_cells(EXHAUST_CLUSTER)
    terminal_states = pd.Series({**{c: "C3" for c in tC3},
                                 **{c: "C4" for c in tC4}})
    print(f"[r5b] fixed terminals -> {len(tC3)} C3 + {len(tC4)} C4 cells")

    pr = pal_core.run_palantir(dm, early_cell=root,
                               terminal_states=terminal_states)
    pt = pr.pseudotime.reindex(ad.obs_names)
    bp = pr.branch_probs.reindex(ad.obs_names)
    def col_for(label, cells):
        if label in bp.columns:
            return label
        for c in cells:
            if c in bp.columns:
                return c
        clv = EFFECTOR_CLUSTER if label == "C3" else EXHAUST_CLUSTER
        means = {c: bp.loc[ad.obs[CLUSTER_KEY].astype(str) == clv, c].mean()
                 for c in bp.columns}
        return max(means, key=means.get)
    fpC3 = bp[col_for("C3", tC3)].values
    fpC4 = bp[col_for("C4", tC4)].values
    branch = np.full(ad.n_obs, "other", dtype=object)
    branch[(fpC3 - fpC4 >= DELTA) & (fpC3 >= EPS)] = "branch_C3"
    branch[(fpC4 - fpC3 >= DELTA) & (fpC4 >= EPS)] = "branch_C4"

    # argmax-based branch (no abstention) -- fairer method-vs-method comparison
    branch_argmax = np.where(fpC3 >= fpC4, "branch_C3", "branch_C4")

    patient = ad.obs["patient"].astype(str).values
    respond = U.add_respond(ad.obs.copy())["Respond"].astype(str).values
    res = {"obs_names": np.asarray(ad.obs_names), "branch": branch,
           "patient": patient, "Respond": respond}
    res_argmax = {"obs_names": np.asarray(ad.obs_names), "branch": branch_argmax,
                  "patient": patient, "Respond": respond}

    # continuous per-patient mean fate probability toward the effector (C3) terminal
    pp = U.per_patient_mean(fpC3, patient, respond)
    cont = U.patient_level_test(pp, value_col="value")

    # (1) pseudotime correlation vs canonical DPT
    common = canon.index.intersection(ad.obs_names)
    from scipy.stats import spearmanr
    rho = spearmanr(pt.loc[common], canon.loc[common, "dpt_pseudotime"]).correlation

    # (2) branch-label agreement vs canonical (thresholded and argmax)
    canon_res = {"obs_names": canon.index.values, "branch": canon["branch_rel"].values}
    canon_argmax = {"obs_names": canon.index.values,
                    "branch": np.where(canon["share_C3"].values >= 0.5,
                                       "branch_C3", "branch_C4")}
    stab = R5.label_stability(res, canon_res)
    stab_argmax = R5.label_stability(res_argmax, canon_argmax)

    # (3) patient-level R vs NR direction
    direction = R5.branch_enrichment_direction(res)

    # save
    pd.DataFrame({"cell": ad.obs_names, "palantir_pseudotime": pt.values,
                  "palantir_branch": branch, "fp_C3": fpC3, "fp_C4": fpC4}
                 ).to_csv(os.path.join(OUTDIR, "palantir_branch_assignment.csv"),
                          index=False)
    summary = {"method": "Palantir (fixed terminals)",
               "unconstrained_terminal_clusters": "|".join(map(str, auto_term_clusters)),
               "spearman_pseudotime_vs_DPT": round(float(rho), 3),
               "branch_ARI_vs_canonical_thresholded": stab["ari"],
               "branch_ARI_vs_canonical_argmax": stab_argmax["ari"],
               "branch_kappa_vs_canonical_argmax": stab_argmax["kappa"],
               "cont_fpC3_mean_R": round(cont["mean_R"], 4),
               "cont_fpC3_mean_NR": round(cont["mean_NR"], 4),
               "cont_fpC3_higher_in_R": bool(cont["mean_R"] > cont["mean_NR"]),
               "cont_fpC3_p_mwu": cont["p_mwu"],
               **{f"branch_{k}": v for k, v in direction.items()}}
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTDIR, "palantir_crosscheck_summary.csv"), index=False)
    print("[r5b] summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    print("\n[r5b DONE] outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
