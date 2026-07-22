"""
3.7 supplement -- CellChat sensitivity with the PATIENT as the unit of replication.
For each focus ligand-receptor pair, myeloid -> T communication probability is summed
over source (Macrophages/Monocytes) x target (C2/C3/C4) per patient (one value per
patient, matching cellchat_patient_level_summary.csv), then compared R vs NR with the
Mann-Whitney test reported in the text. No focus pair reaches significance.

Pairs undetected in every patient (CXCL9_CXCR3, SPP1_ITGAV_ITGB1) are omitted.
Minimal styling: box + per-patient dots, pair name + MWU p only.

Output: revise/results/r7_cellchat/FigSx_cellchat_sensitivity_patientlevel.{png,pdf}
Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
RES = os.path.join(BASE, "revise", "results", "r7_cellchat")

COL = {"NR": "#1f77b4", "R": "#ff7f0e"}
BOX = dict(facecolor="#dcdcdc", edgecolor="black", linewidth=1.2)
MED = dict(color="#ff7f0e", linewidth=2.0)
LINE = dict(color="black", linewidth=1.2)
rng = np.random.default_rng(0)


def panel(ax, nr, r, lr, p):
    ax.boxplot([nr, r], positions=[1, 2], widths=0.58, patch_artist=True,
               showfliers=False, boxprops=BOX, medianprops=MED,
               whiskerprops=LINE, capprops=LINE, zorder=1)
    for j, (d, g) in enumerate([(nr, "NR"), (r, "R")], start=1):
        jit = (rng.random(len(d)) - 0.5) * 0.14
        ax.scatter(np.full(len(d), j) + jit, d, s=60, c=COL[g],
                   edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["NR", "R"], fontsize=10)
    title = lr.replace("_", "–")
    ax.set_title(f"{title}\nMWU p = {p:.2f}", fontsize=10)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.margins(x=0.22)


def main():
    d = pd.read_csv(os.path.join(RES, "cellchat_per_patient_focusLR.csv"))
    summ = pd.read_csv(os.path.join(RES, "cellchat_patient_level_summary.csv"))
    # per patient x lr: sum over source-target
    g = d.groupby(["patient", "response", "lr"])["prob"].sum().reset_index()

    # keep pairs detected in at least one patient, ordered by MWU p
    keep = summ[(summ.mean_R > 0) | (summ.mean_NR > 0)].sort_values("MWU_p")
    lrs = keep["lr"].tolist()
    pmap = dict(zip(summ.lr, summ.MWU_p))

    ncol = 4
    nrow = int(np.ceil(len(lrs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 3.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ax, lr in zip(axes, lrs):
        sub = g[g.lr == lr]
        nr = sub[sub.response == "NR"]["prob"].values
        r = sub[sub.response == "R"]["prob"].values
        panel(ax, nr, r, lr, float(pmap[lr]))
        print(f"  {lr:20s} MWU p = {pmap[lr]:.2f}")
    for ax in axes[len(lrs):]:
        ax.axis("off")

    fig.supylabel("Myeloid→T communication probability (per patient)",
                  fontsize=12)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RES, f"FigSx_cellchat_sensitivity_patientlevel.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[DONE] FigSx_cellchat_sensitivity_patientlevel  (", len(lrs), "pairs )")


if __name__ == "__main__":
    main()
