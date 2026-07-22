"""
Combine the S5 effector-(C3)-fate-probability concordance panels into one grid PNG
(companion to r5o pseudotime grid; min-exhaustion root dropped). Reads per-cell
matrices; no GPCCA re-run. Shared axis labels + one cluster legend; per-cell rho.
The Palantir panel uses its own y-range (Palantir fate prob is on a lower scale)
and drops the single y~1 outlier, matching the standalone panel.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
RES = os.path.join(BASE, "revise", "results", "r5_trajectory")
OUTDIR = os.path.join(RES, "S5_concordance_clean")

CLUST_COL = {"4": "#4c78a8", "0": "#f58518", "1": "#54a24b", "2": "#b279a2"}
CLUST_NAME = {"4": "C1 naive", "0": "C2 transit.", "1": "C3 effector", "2": "C4 exhaust."}
CLUST_ORDER = ["4", "0", "1", "2"]

LAYOUT = [
    [("n_neighbors = 15", ("gpcca", "n_neighbors = 15")),
     ("n_neighbors = 50", ("gpcca", "n_neighbors = 50")),
     ("Schur dim = 30",   ("gpcca", "n_components = 30")),
     ("Palantir vs CellRank", ("palantir", None))],
    [("root: top-naive medoid", ("gpcca", "root = top-naive medoid")),
     ("root: random 1", ("gpcca", "root = random #1")),
     ("root: random 2", ("gpcca", "root = random #2")),
     None],
    [("drop PHD001 (NR)", ("gpcca", "drop PHD001 (NR)")),
     ("drop PHD002 (NR)", ("gpcca", "drop PHD002 (NR)")),
     ("drop PHD008 (NR)", ("gpcca", "drop PHD008 (NR)")),
     None],
    [("drop PHD003 (R)", ("gpcca", "drop PHD003 (R)")),
     ("drop PHD004 (R)", ("gpcca", "drop PHD004 (R)")),
     ("drop PHD009 (R)", ("gpcca", "drop PHD009 (R)")),
     None],
]


def draw(ax, x, y, cl, ylim=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[m], y[m], cl[m]
    if ylim is not None:                 # Palantir: drop the single y~1 outlier
        k = y <= 0.3
        x, y, cl = x[k], y[k], cl[k]
    for c in CLUST_ORDER:
        s = cl == c
        ax.scatter(x[s], y[s], s=3, c=CLUST_COL[c], alpha=0.35,
                   linewidths=0, rasterized=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#555555", zorder=3)
    rho = spearmanr(x, y).correlation
    ax.text(0.05, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10)
    ax.tick_params(labelsize=7)
    if ylim is not None:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(*ylim)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def main():
    wide = pd.read_csv(os.path.join(RES, "concordance_percell.csv"), index_col=0)
    wide["cluster"] = wide["cluster"].astype(str)
    pal = pd.read_csv(os.path.join(RES, "palantir_branch_assignment.csv"),
                      index_col="cell")

    nrow, ncol = len(LAYOUT), 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.0 * nrow),
                             squeeze=False)
    for r, row in enumerate(LAYOUT):
        for c in range(ncol):
            ax = axes[r, c]
            cell = row[c]
            if cell is None:
                ax.axis("off"); continue
            title, (kind, tag) = cell
            if kind == "gpcca":
                x = wide["fpC3_canonical"].values
                y = wide[f"fpC3::{tag}"].values
                cl = wide["cluster"].values
                draw(ax, x, y, cl)
            else:  # palantir: own y-scale, drop outlier
                common = wide.index.intersection(pal.index)
                x = wide.loc[common, "fpC3_canonical"].values
                y = pal.loc[common, "fp_C3"].values
                cl = wide.loc[common, "cluster"].values
                draw(ax, x, y, cl, ylim=(-0.01, 0.22))
            ax.set_title(title, fontsize=9.5)

    fig.supxlabel("Canonical C3 fate probability", fontsize=12, y=0.045)
    fig.supylabel("Perturbed C3 fate probability", fontsize=12, x=0.01)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=CLUST_COL[c], markersize=8,
                          label=CLUST_NAME[c]) for c in CLUST_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    out = os.path.join(OUTDIR, "fateprob_grid.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[DONE]", out)


if __name__ == "__main__":
    main()
