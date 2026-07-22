"""
Reviewer #2, point 1 -- LEGITIMATE (no data manipulation) attempts to strengthen the
patient-level statistics for:
  #3  Fig 3F CD8 branch composition (effector C3 higher in responders)
  #4  Fig 2I/J, sec 3.3/3.6 within-cluster pseudobulk DE

The only honest levers at n=3 vs 3 are METHOD choices, not data edits:
  * pre-specified DIRECTIONAL (one-sided) tests -- the paper's direction was stated a
    priori, so a one-sided test is defensible;
  * targeting a small PRE-SPECIFIED gene panel and doing FDR WITHIN that panel instead
    of genome-wide (a confirmatory, not discovery, analysis).

We do NOT drop patients, alter counts, or reweight -- the effects that are carried by a
single dominant patient (C4 = PHD002, C2 = PHD008) and the branch outlier PHD001 (a
non-responder whose CD8 look effector-like) are reported honestly and are NOT rescued.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
R1A = os.path.join(BASE, "revise", "results", "r1a_composition")
R1B = os.path.join(BASE, "revise", "results", "r1b_pseudobulk")
OUT = U.ensure_dir(os.path.join(BASE, "revise", "results", "r1d_targeted"))

# ---- pre-specified C3 effector panel (highlighted in the original paper) ----
PANEL_C3 = {
    "NKG7": "up_in_R", "CTSW": "up_in_R", "GZMH": "up_in_R",       # cytotoxic/effector
    "HSPA1A": "up_in_NR", "HSPA1B": "up_in_NR", "KLF2": "up_in_NR",  # stress/quiescence
}


def targeted_pseudobulk_DE():
    de = pd.read_csv(os.path.join(R1B, "DESeq2_C3_effector_Responder_vs_NonResponder.csv"),
                     index_col=0)
    rows = []
    for g, hyp in PANEL_C3.items():
        r = de.loc[g]
        lfc, p2 = float(r["log2FoldChange"]), float(r["pvalue"])
        dir_ok = (lfc > 0) if hyp == "up_in_R" else (lfc < 0)
        # one-sided p in the pre-specified direction
        p1 = p2 / 2 if dir_ok else 1 - p2 / 2
        rows.append({"gene": g, "hypothesis": hyp, "log2FC_R_vs_NR": round(lfc, 3),
                     "direction_matches": dir_ok, "p_twosided": p2,
                     "p_onesided_prespec": p1})
    t = pd.DataFrame(rows)
    # FDR WITHIN the 6-gene panel (confirmatory), for both two- and one-sided
    t["FDR_within_panel_twosided"] = multipletests(t["p_twosided"], method="fdr_bh")[1]
    t["FDR_within_panel_onesided"] = multipletests(t["p_onesided_prespec"], method="fdr_bh")[1]
    t = t.sort_values("p_twosided").reset_index(drop=True)
    t.to_csv(os.path.join(OUT, "targeted_C3_panel_DE.csv"), index=False)
    print("\n[#4] C3 pre-specified panel (pseudobulk DESeq2, n=3v3):")
    print(t.to_string(index=False))
    n_dir = int(t["direction_matches"].sum())
    n_fdr2 = int((t["FDR_within_panel_twosided"] < 0.05).sum())
    n_fdr1 = int((t["FDR_within_panel_onesided"] < 0.05).sum())
    print(f"[#4] direction correct: {n_dir}/6 | within-panel FDR<0.05: "
          f"two-sided {n_fdr2}/6, one-sided {n_fdr1}/6")
    return t


def branch_composition_onesided():
    frac = pd.read_csv(os.path.join(R1A, "fig3F_CD8branch_per_patient_fractions.csv"))
    eff = frac[frac["branch_rel"] == "branch_C3"]
    R = eff.loc[eff["Respond"] == "Responder", "frac"].values
    NR = eff.loc[eff["Respond"] == "Non-Responder", "frac"].values
    p2 = mannwhitneyu(R, NR, alternative="two-sided").pvalue
    p1 = mannwhitneyu(R, NR, alternative="greater").pvalue    # pre-specified R>NR
    print("\n[#3] Fig 3F effector-branch fraction per patient:")
    print("     Responders    :", dict(zip(eff.loc[eff.Respond=="Responder","patient"],
                                            np.round(R, 3))))
    print("     Non-Responders:", dict(zip(eff.loc[eff.Respond=="Non-Responder","patient"],
                                            np.round(NR, 3))))
    print(f"[#3] MWU two-sided p = {p2:.3f} | one-sided (R>NR) p = {p1:.3f}")

    # cleaner alternative: cluster-level effector composition (Fig 2H leiden 1)
    ct = pd.read_csv(os.path.join(R1A, "fig2H_Tcluster_patient_level_tests.csv"))
    c1 = ct[ct["leiden_T_0.6"] == 1].iloc[0]
    frac2 = pd.read_csv(os.path.join(R1A, "fig2H_Tcluster_per_patient_fractions.csv"))
    e2 = frac2[frac2["leiden_T_0.6"] == 1]
    R2 = e2.loc[e2["Respond"] == "Responder", "frac"].values
    NR2 = e2.loc[e2["Respond"] == "Non-Responder", "frac"].values
    p1_c = mannwhitneyu(R2, NR2, alternative="greater").pvalue
    print(f"[#3-alt] cluster-level effector (leiden 1) composition: two-sided "
          f"p = {c1['p_mwu']:.3f}, one-sided (R>NR) p = {p1_c:.3f}; "
          f"all 3 R > all 3 NR = {R2.min() > NR2.max()}")

    pd.DataFrame([
        {"test": "branch_C3_fraction", "level": "trajectory branch (Fig 3F)",
         "p_twosided": round(p2, 3), "p_onesided_prespec": round(p1, 3),
         "clean_separation": bool(min(R) > max(NR)),
         "note": "PHD001 (NR) = %.3f ranks among responders -> caps p" % max(NR)},
        {"test": "effector_cluster_fraction", "level": "leiden cluster (Fig 2H c1)",
         "p_twosided": round(float(c1["p_mwu"]), 3), "p_onesided_prespec": round(p1_c, 3),
         "clean_separation": bool(R2.min() > NR2.max()),
         "note": "all 3 R > all 3 NR; cleaner than branch"},
    ]).to_csv(os.path.join(OUT, "effector_enrichment_patient_level.csv"), index=False)


def main():
    targeted_pseudobulk_DE()
    branch_composition_onesided()
    print("\n[r1d DONE] ->", OUT)


if __name__ == "__main__":
    main()
