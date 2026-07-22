"""
Patient-level SCENIC regulon activity for the 3.6 supplement figure -- ALL nine
focus regulons discussed in the text (not just the four in Fig4FI), annotated with
the Welch t-test used in the manuscript (n = 3 vs 3; MWU has no resolution at this
size). Minimal styling: box + per-patient dots, regulon name + p only.

responder-up:      STAT1, EOMES, PRDM1, NFATC2
non-responder-up:  RELB, NFKB1, FOXO3, NFATC1, NR1D2

Output: revise/results/r1_final/FigSx_regulon9_patientlevel.{png,pdf}
Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ttest_ind

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
RES = os.path.join(BASE, "revise", "results")
OUT = os.path.join(RES, "r1_final")

COL = {"Non-Responder": "#1f77b4", "Responder": "#ff7f0e"}
BOX = dict(facecolor="#dcdcdc", edgecolor="black", linewidth=1.2)
MED = dict(color="#ff7f0e", linewidth=2.0)
LINE = dict(color="black", linewidth=1.2)
rng = np.random.default_rng(0)

# text order: responder-up then non-responder-up
REGS = ["STAT1", "EOMES", "PRDM1", "NFATC2",
        "RELB", "NFKB1", "FOXO3", "NFATC1", "NR1D2"]


def panel(ax, nr, r, tf, p):
    ax.boxplot([nr, r], positions=[1, 2], widths=0.58, patch_artist=True,
               showfliers=False, boxprops=BOX, medianprops=MED,
               whiskerprops=LINE, capprops=LINE, zorder=1)
    for j, (d, g) in enumerate([(nr, "Non-Responder"), (r, "Responder")], start=1):
        jit = (rng.random(len(d)) - 0.5) * 0.14
        ax.scatter(np.full(len(d), j) + jit, d, s=60, c=COL[g],
                   edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["NR", "R"], fontsize=10)
    ax.set_title(f"{tf}   p = {p:.3f}", fontsize=11)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.margins(x=0.22)


def main():
    df = pd.read_csv(os.path.join(RES, "r1c_regulon", "per_patient_mean_AUC_C3.csv"))
    nr_mask = df.Respond == "Non-Responder"
    r_mask = df.Respond == "Responder"

    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.4))
    for ax, tf in zip(axes.ravel(), REGS):
        col = f"{tf}(+)"
        nr = df.loc[nr_mask, col].values
        r = df.loc[r_mask, col].values
        p = ttest_ind(r, nr, equal_var=False).pvalue
        panel(ax, nr, r, tf, p)
        print(f"  {tf:8s} Welch t p = {p:.4f}")

    fig.supylabel("Regulon activity (mean AUC per patient)", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"FigSx_regulon9_patientlevel.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[DONE] FigSx_regulon9_patientlevel")


if __name__ == "__main__":
    main()
