"""
Reviewer #2, point 1 (composition part) -- patient-level re-analysis of:
  * Fig 1C : TME cell-type composition per patient
  * Fig 2H : T-cell cluster composition, Responder vs Non-Responder
  * Fig 3F : CD8 branch composition, Responder vs Non-Responder

Original scripts pooled all cells and compared *within-cluster* R-vs-NR fractions
(02_03 crosstab; analysis3 crosstab/normalize; TF notebook chi2). That treats
each cell as an independent replicate (pseudo-replication).

Here we instead compute, for each patient, the fraction of that patient's cells
falling in each cluster/branch, and compare the two response groups with a
patient-level Mann-Whitney test (n=3 vs 3). Every comparison is plotted with one
dot per patient.

Branch labels for Fig 3F are read from the cached CellRank AnnData produced by
05c/analysis3.py if present; otherwise the script re-derives a branch label from
fate probabilities in that cache. If neither is available it skips Fig 3F and
tells you which file to produce first.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r1a_composition"))

# Inputs
TME_H5AD = os.path.join(BASE, "01_mapping_raw_scRNA_seq_to_reference",
                        "adata_scvi_integrated_all_cells.h5ad")
TCELL_H5AD = os.path.join(
    BASE, "02_extracting_T_cells_and_clustering",
    "results_drop_10-12-5-8-9__20260111_164341",
    "02_adata_T_reclustered_after_drop.h5ad")
# candidate caches that contain a CD8 branch label
BRANCH_CACHE_CANDIDATES = [
    os.path.join(BASE, "05c_trajectory_rerevisit",
                 "results_cellrank2_subset_0_1_2_4",
                 "adata_CD8_0_1_2_4__DPT+CellRank2_cached.h5ad"),
    os.path.join(BASE, "05c_trajectory_rerevisit",
                 "results_cellrank2_subset_0_1_2_4",
                 "adata_subset_0_1_2_4__with_dpt_cellrank_branch.h5ad"),
]


def run_composition(long_df, group_key, tag, title_prefix):
    """Per-cluster patient-level test + combined dot/box figure + CSV."""
    rows = []
    cats = sorted(long_df[group_key].astype(str).unique(),
                  key=lambda x: (len(x), x))
    for cat in cats:
        sub = long_df[long_df[group_key].astype(str) == cat]
        res = U.patient_level_test(sub, value_col="frac")
        res[group_key] = cat
        rows.append(res)
        U.plot_group_compare(
            sub, value_col="frac",
            title=f"{title_prefix}: {group_key}={cat}",
            ylabel="fraction of patient's cells",
            outpath=os.path.join(OUTDIR, f"{tag}_{group_key}_{cat}_dotbox.png"),
            annotate=res)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUTDIR, f"{tag}_patient_level_tests.csv"), index=False)
    long_df.to_csv(os.path.join(OUTDIR, f"{tag}_per_patient_fractions.csv"),
                   index=False)
    print(f"[{tag}] wrote {len(rows)} per-category tests -> {tag}_patient_level_tests.csv")
    return summ


# --------------------------------------------------------------------------
# Fig 1C : TME composition (per patient) -- already patient-oriented in 01_02,
# here we add the patient-level test to it.
# --------------------------------------------------------------------------
def fig1c():
    if not os.path.exists(TME_H5AD):
        print("[Fig1C] TME h5ad not found, skipping:", TME_H5AD)
        return
    print("[Fig1C] loading TME atlas (large, backed) ...")
    ad = sc.read_h5ad(TME_H5AD, backed="r")
    obs = U.add_respond(ad.obs.copy())
    ct_col = "majority_voting" if "majority_voting" in obs else "predicted_labels"
    long = U.per_patient_fraction(obs, group_key=ct_col)
    run_composition(long, group_key=ct_col, tag="fig1C_TME",
                    title_prefix="TME composition")
    ad.file.close()


# --------------------------------------------------------------------------
# Fig 2H : T-cell cluster composition R vs NR (patient level)
# --------------------------------------------------------------------------
def fig2h():
    if not os.path.exists(TCELL_H5AD):
        print("[Fig2H] T-cell h5ad not found, skipping:", TCELL_H5AD)
        return
    print("[Fig2H] loading T-cell object ...")
    ad = sc.read_h5ad(TCELL_H5AD)
    obs = U.add_respond(ad.obs.copy())
    long = U.per_patient_fraction(obs, group_key="leiden_T_0.6")
    run_composition(long, group_key="leiden_T_0.6", tag="fig2H_Tcluster",
                    title_prefix="T-cell cluster")


# --------------------------------------------------------------------------
# Fig 3F : CD8 branch composition R vs NR (patient level)
# --------------------------------------------------------------------------
def _find_branch_column(obs):
    for c in ["branch_rel", "branch_AB_0p5", "branch", "branch_label"]:
        if c in obs.columns:
            return c
    return None


BRANCH_CSV = os.path.join(BASE, "revise", "results", "r5_trajectory",
                          "branch_assignment_canonical.csv")


def fig3f():
    # Preferred: the canonical per-cell branch assignment from r5 (single source
    # of truth, already patient/Respond annotated).
    if os.path.exists(BRANCH_CSV):
        print("[Fig3F] using canonical branch CSV:", BRANCH_CSV)
        obs = pd.read_csv(BRANCH_CSV)
        obs = obs[obs["branch_rel"].astype(str) != "other"]
        long = U.per_patient_fraction(
            obs.rename(columns={"patient": "patient"}),
            group_key="branch_rel")
        run_composition(long, group_key="branch_rel", tag="fig3F_CD8branch",
                        title_prefix="CD8 branch")
        return
    print("[Fig3F] branch CSV not found; run r5_trajectory_robustness.py first:",
          BRANCH_CSV)


if __name__ == "__main__":
    fig2h()   # fast, most central to the rebuttal
    fig3f()
    fig1c()   # large file, run last
    print("\n[DONE] Patient-level composition outputs in:", OUTDIR)
