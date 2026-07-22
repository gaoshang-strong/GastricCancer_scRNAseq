"""
Reviewer #2, point 1 (regulon part) -- patient-level comparison of SCENIC regulon
activity between Responder and Non-Responder within the effector cluster C3.

Original: TF_analysis.ipynb (cells 63/65) and regulon_diff_C3_NR_vs_R.csv compare
per-cell regulon AUC with mannwhitneyu (n_g1=747, n_g2=1434 CELLS, p~1e-140) ->
pseudo-replication.

Here, for each regulon, we average the AUC over each patient's C3 cells (one value
per patient) and compare the response groups with a patient-level Mann-Whitney
test (n=3 vs 3). Every focus regulon gets a box + per-patient-dot figure. We also
export a full patient-level table for all regulons.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
AUC_TSV = os.path.join(BASE, "10_TF_analysis", "pyscenic_run", "cd8", "auc.tsv.gz")
OBS_H5AD = os.path.join(BASE, "10_TF_analysis",
                        "adata_CD8_subset_clusters_0_1_2_4.fixed.h5ad")
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r1c_regulon"))

STATE_COL = "leiden_T_0.6_relabel"   # C3 effector == "3" in this relabel
C3_VALUE = "3"

# regulons the paper highlights for the within-C3 R vs NR contrast (Fig 4F-I)
FOCUS = {
    "responder_up": ["STAT1", "EOMES", "PRDM1", "NFATC2"],
    "nonresponder_up": ["RELB", "NFKB1", "FOXO3", "NFATC1", "NR1D2"],
}


def resolve_regulon(cols, tf, prefer="+"):
    """Map a bare TF name to its regulon column, e.g. 'STAT1' -> 'STAT1(+)'."""
    exact = [c for c in cols if c.upper() in (f"{tf}({prefer})".upper(),)]
    if exact:
        return exact[0]
    hits = [c for c in cols if c.upper().startswith(tf.upper() + "(")]
    plus = [c for c in hits if "(+)" in c]
    return (plus or hits or [None])[0]


def main():
    print("[r1c] loading AUC:", AUC_TSV)
    auc = pd.read_csv(AUC_TSV, sep="\t", index_col=0)   # cells x regulons
    print("[r1c] AUC shape:", auc.shape)

    a = ad.read_h5ad(OBS_H5AD, backed="r")
    obs = U.add_respond(a.obs.copy())
    a.file.close()

    # restrict to C3 cells with a response label, align AUC
    mask = obs[STATE_COL].astype(str) == C3_VALUE
    cells = obs.index[mask]
    cells = cells.intersection(auc.index)
    print(f"[r1c] C3 cells with AUC: {len(cells)}")
    aucC3 = auc.loc[cells]
    patient = obs.loc[cells, "patient"].astype(str).values
    respond = obs.loc[cells, "Respond"].astype(str).values

    # patient-level mean AUC per regulon
    meta = pd.DataFrame({"patient": patient, "Respond": respond}, index=cells)
    pm = aucC3.join(meta)
    per_patient = pm.groupby("patient").agg(
        {**{c: "mean" for c in aucC3.columns}, "Respond": "first"})
    per_patient.to_csv(os.path.join(OUTDIR, "per_patient_mean_AUC_C3.csv"))

    # patient-level test for every regulon
    rows = []
    for reg in aucC3.columns:
        df = per_patient[[reg, "Respond"]].rename(columns={reg: "value"})
        res = U.patient_level_test(df, value_col="value")
        res["regulon"] = reg
        rows.append(res)
    summ = pd.DataFrame(rows).sort_values("p_mwu")
    summ = summ[["regulon", "n_NR", "n_R", "mean_NR", "mean_R",
                 "median_NR", "median_R", "direction", "p_mwu"]]
    summ.to_csv(os.path.join(OUTDIR, "regulon_C3_patient_level_tests.csv"),
                index=False)
    print("[r1c] wrote patient-level tests for", len(summ), "regulons")

    # focus-regulon figures + a compact summary
    focus_rows = []
    for grp, tfs in FOCUS.items():
        for tf in tfs:
            reg = resolve_regulon(list(aucC3.columns), tf)
            if reg is None:
                print(f"[r1c] focus regulon not found: {tf}")
                focus_rows.append({"group": grp, "TF": tf, "regulon": None})
                continue
            df = per_patient[[reg, "Respond"]].rename(columns={reg: "value"})
            res = U.patient_level_test(df, value_col="value")
            U.plot_group_compare(
                df, value_col="value",
                title=f"C3 regulon {reg} ({grp})",
                ylabel="mean regulon AUC (per patient)",
                outpath=os.path.join(OUTDIR, f"C3_regulon_{tf}_dotbox.png"),
                annotate=res)
            focus_rows.append({"group": grp, "TF": tf, "regulon": reg,
                               "mean_NR": res["mean_NR"], "mean_R": res["mean_R"],
                               "direction": res["direction"], "p_mwu": res["p_mwu"]})
    pd.DataFrame(focus_rows).to_csv(
        os.path.join(OUTDIR, "focus_regulon_summary.csv"), index=False)

    print("\n[r1c] focus regulon patient-level summary:")
    print(pd.DataFrame(focus_rows).to_string(index=False))
    print("\n[r1c DONE] outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
