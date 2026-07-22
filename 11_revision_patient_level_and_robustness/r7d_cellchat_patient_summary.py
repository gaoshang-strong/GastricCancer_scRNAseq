"""
Reviewer #2, point 7 -- assemble the patient-level CellChat answer:
  * per-patient myeloid -> CD8 communication strength for each focus LR pair
    (from r7c), as 3 R vs 3 NR box+dots + patient-level MWU  [consistency]
  * per-patient DETECTION counts (in how many of 3 R / 3 NR patients the pair is
    seen at all) -- exposes on/off differences that pooling turns into artefacts
  * merge with the patient-label PERMUTATION p (r7b) and LOPO direction agreement
  * one honest verdict table + a box+dots figure in the paper's Fig-4F-I style

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42

RES = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/revise/results"
CC = os.path.join(RES, "r7_cellchat")
OUT = U.ensure_dir(os.path.join(RES, "r1_final"))   # keep final figures together
COL = {"Non-Responder": "#1f77b4", "Responder": "#ff7f0e"}
BOX = dict(facecolor="#dcdcdc", edgecolor="black", linewidth=1.2)
MED = dict(color="#ff7f0e", linewidth=2.0)
LN = dict(color="black", linewidth=1.2)
rng = np.random.default_rng(0)


def dotbox(ax, dNR, dR, ylabel, annot, title):
    data = [np.asarray(dNR, float), np.asarray(dR, float)]
    ax.boxplot(data, positions=[1, 2], widths=0.58, patch_artist=True,
               showfliers=False, boxprops=BOX, medianprops=MED,
               whiskerprops=LN, capprops=LN, zorder=1)
    for j, (d, g) in enumerate(zip(data, ["Non-Responder", "Responder"]), start=1):
        jit = (rng.random(len(d)) - 0.5) * 0.14
        ax.scatter(np.full(len(d), j) + jit, d, s=60, c=COL[g], edgecolors="black",
                   linewidths=0.6, zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Non-Resp.\n(n=3)", "Resp.\n(n=3)"],
                                              fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.text(0.03, 0.97, annot, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.6, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.margins(x=0.2)


def main():
    pp = pd.read_csv(os.path.join(CC, "cellchat_per_patient_focusLR.csv"))
    # total myeloid -> CD8 strength per patient per LR pair
    strength = (pp.groupby(["lr", "patient", "response"])["prob"].sum()
                .reset_index())
    RESP = {"R": "Responder", "NR": "Non-Responder"}
    strength["group"] = strength["response"].map(RESP)

    # permutation p (r7b): min across this LR pair's source/target combos
    perm = pd.read_csv(os.path.join(CC, "cellchat_patient_permutation_pvals.csv"))
    perm["lr"] = perm["pair"].str.split(" ").str[0]
    perm_min = perm.groupby("lr")["p_perm"].min()

    lrs = sorted(strength["lr"].unique())
    rows = []
    panels = []
    for lr in lrs:
        s = strength[strength["lr"] == lr]
        dR = s.loc[s.group == "Responder", "prob"].values
        dNR = s.loc[s.group == "Non-Responder", "prob"].values
        # pad to 3 with 0 for undetected patients
        dR = np.pad(dR, (0, max(0, 3 - len(dR))))
        dNR = np.pad(dNR, (0, max(0, 3 - len(dNR))))
        nR_det = int((dR > 0).sum()); nNR_det = int((dNR > 0).sum())
        try:
            p = mannwhitneyu(dR, dNR, alternative="two-sided").pvalue
        except ValueError:
            p = np.nan
        direction = "up_in_R" if dR.mean() > dNR.mean() else "up_in_NR"
        rows.append({"lr": lr, "mean_R": dR.mean(), "mean_NR": dNR.mean(),
                     "direction": direction, "detected_R": f"{nR_det}/3",
                     "detected_NR": f"{nNR_det}/3", "MWU_p": round(float(p), 3),
                     "perm_p_min": round(float(perm_min.get(lr, np.nan)), 3)})
        panels.append((lr, dNR, dR, p, nR_det, nNR_det))

    summ = pd.DataFrame(rows).sort_values("MWU_p").reset_index(drop=True)
    summ.to_csv(os.path.join(CC, "cellchat_patient_level_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # individual single-panel PNGs (user assembles the composite themselves)
    pan = U.ensure_dir(os.path.join(OUT, "panels"))
    for lr, dNR, dR, p, nR, nNR in panels:
        f, a = plt.subplots(figsize=(2.9, 3.9))
        dotbox(a, dNR, dR, "comm. prob (sum)",
               f"MWU p={p:.2f}\ndet R {nR}/3, NR {nNR}/3", lr)
        f.tight_layout()
        for ext in ["png", "pdf"]:
            f.savefig(os.path.join(pan, f"cellchat_{lr}.{ext}"), dpi=300,
                      bbox_inches="tight")
        plt.close(f)
    print(f"\n[r7d DONE] {len(panels)} single-panel CellChat figures -> {pan}; "
          "summary CSV written")


if __name__ == "__main__":
    main()
