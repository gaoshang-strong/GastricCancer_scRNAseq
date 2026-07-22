"""
EXPLORATION (not for submission) -- the best-tuned Palantir run already agrees with
CellRank in RANK (Spearman rho ~ 0.88 on per-cell effector-C3 fate probability), but
the absolute values sit off-diagonal: Palantir uses 2 terminals (fpC3 + fpC4 = 1, so
values span 0-1) while CellRank uses 3 terminals (fpC3 compressed into ~0-0.3).

To make the two land ON the diagonal we apply a MONOTONE recalibration
(quantile-matching Palantir's fpC3 onto CellRank's fpC3 distribution). Because it is
rank-preserving, the Spearman rho is unchanged (0.88) -- this only rescales the axis,
it does not invent agreement. Left panel = raw, right panel = recalibrated.

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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5d_gpcca_robustness as R5D

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42

OUTDIR = R5D.OUTDIR
CLUST_COL = {"4": "#4c78a8", "0": "#f58518", "1": "#54a24b", "2": "#b279a2"}
CLUST_NAME = {"4": "C1 naive", "0": "C2 transit.", "1": "C3 effector", "2": "C4 exhaust."}


def quantile_match(src, ref):
    """Map src onto ref's empirical distribution, preserving src ranks (monotone)."""
    ranks = np.argsort(np.argsort(src))
    ref_sorted = np.sort(ref)
    idx = np.round(ranks / (len(src) - 1) * (len(ref) - 1)).astype(int)
    return ref_sorted[idx]


def panel(ax, x, y, cl, title):
    for c in ["4", "0", "1", "2"]:
        s = cl == c
        ax.scatter(x[s], y[s], s=5, c=CLUST_COL[c], alpha=0.4, linewidths=0,
                   rasterized=True, label=CLUST_NAME[c])
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#555")
    rho = spearmanr(x, y).correlation
    ax.text(0.04, 0.96, f"ρ = {rho:.3f}", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                                   alpha=0.8))
    ax.set_xlabel("CellRank/GPCCA  effector (C3) fate probability", fontsize=9.5)
    ax.set_title(title, fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def main():
    cr = pd.read_csv(os.path.join(OUTDIR, "gpcca_branch_assignment_canonical.csv")
                     ).set_index("cell")
    pal = pd.read_csv(os.path.join(OUTDIR, "palantir_match_best_percell.csv")
                      ).set_index("cell")
    clu = pd.read_csv(os.path.join(OUTDIR, "concordance_percell.csv"),
                      index_col=0)["cluster"].astype(str)

    common = cr.index.intersection(pal.index)
    x = cr.loc[common, "fpC3"].values
    y_raw = pal.loc[common, "palantir_fpC3"].values
    cl = clu.reindex(common).values
    m = np.isfinite(x) & np.isfinite(y_raw)
    x, y_raw, cl = x[m], y_raw[m], cl[m]

    y_cal = quantile_match(y_raw, x)          # monotone recalibration onto CellRank scale

    out = pd.DataFrame({"cell": common[m], "cellrank_fpC3": x,
                        "palantir_fpC3_raw": y_raw,
                        "palantir_fpC3_recalibrated": y_cal})
    out.to_csv(os.path.join(OUTDIR, "palantir_match_recalibrated_percell.csv"),
               index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2))
    panel(axes[0], x, y_raw, cl, "Raw Palantir (2-terminal, fpC3+fpC4=1)")
    axes[0].set_ylabel("Palantir  effector (C3) fate probability", fontsize=9.5)
    panel(axes[1], x, y_cal, cl,
          "Quantile-recalibrated onto CellRank scale (monotone)")
    axes[1].set_ylabel("Palantir fpC3  (recalibrated)", fontsize=9.5)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, markerscale=1.6)
    fig.suptitle("Palantir vs CellRank effector-C3 fate probability: same ranking "
                 "(ρ = 0.88), rescaled to match", fontsize=11.5, y=1.0)
    fig.text(0.5, -0.01, "recalibration is a rank-preserving monotone map "
             "(quantile matching); Spearman ρ is identical in both panels -- only "
             "the scale changes", ha="center", fontsize=7.5, color="#777")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"fig_r5_palantir_match_recalibrated.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[r5m] raw rho = {spearmanr(x, y_raw).correlation:.3f}  "
          f"recalibrated rho = {spearmanr(x, y_cal).correlation:.3f}")
    print("[r5m DONE] ->", OUTDIR)


if __name__ == "__main__":
    main()
