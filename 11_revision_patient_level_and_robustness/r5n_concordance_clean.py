"""
Reviewer #2, point 5 -- clean, minimal re-draw of the GPCCA concordance panels
for Supplementary Fig. S5. No GPCCA is re-run: all values are read from the
per-cell matrices already written by r5h / r5l.

Simpler axis labels ("Canonical" vs "Perturbed"), no canonical-baseline footnote,
compact per-cell rho. Only the configurations requested for S5:

    * n_neighbors (15, 50)
    * Schur dimension  (n_components = 30)
    * root-cell strategies (top-naive medoid, random x2, min-exhaustion)
    * leave-one-patient-out (3 NR + 3 R)
    * Palantir vs CellRank cross-check

Output: revise/results/r5_trajectory/S5_concordance_clean/{pseudotime,fateprob}/

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import pandas as pd
import numpy as np
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

# (output-index, csv tag in concordance_percell.csv, clean title)
GPCCA_CONFIGS = [
    ("01", "n_neighbors = 15",        "n_neighbors = 15"),
    ("02", "n_neighbors = 50",        "n_neighbors = 50"),
    ("03", "n_components = 30",       "Schur dim = 30"),
    ("04", "root = top-naive medoid", "root: top-naive medoid"),
    ("05", "root = random #1",        "root: random 1"),
    ("06", "root = random #2",        "root: random 2"),
    ("07", "root = min-exhaustion",   "root: min-exhaustion"),
    ("08", "drop PHD001 (NR)",        "drop PHD001 (NR)"),
    ("09", "drop PHD002 (NR)",        "drop PHD002 (NR)"),
    ("10", "drop PHD008 (NR)",        "drop PHD008 (NR)"),
    ("11", "drop PHD003 (R)",         "drop PHD003 (R)"),
    ("12", "drop PHD004 (R)",         "drop PHD004 (R)"),
    ("13", "drop PHD009 (R)",         "drop PHD009 (R)"),
]

# (metric key, subfolder, x-label, y-label)
METRICS = {
    "dpt":  ("pseudotime", "Canonical pseudotime",   "Perturbed pseudotime"),
    "fpC3": ("fateprob",   "Canonical C3 fate prob.", "Perturbed C3 fate prob."),
}


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def panel(x, y, cl, title, xlab, ylab, path, legend=True):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[m], y[m], cl[m]
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    for c in CLUST_ORDER:
        sel = cl == c
        ax.scatter(x[sel], y[sel], s=4, c=CLUST_COL[c], alpha=0.35,
                   linewidths=0, rasterized=True)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, ls="--", lw=1.1, color="#555555", zorder=3)
    rho = spearmanr(x, y).correlation
    ax.text(0.05, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if legend:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=CLUST_COL[c], markersize=6,
                              label=CLUST_NAME[c]) for c in CLUST_ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.5, 0.0), handletextpad=0.2,
                   columnspacing=0.9)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rho


def main():
    ensure(OUTDIR)
    wide = pd.read_csv(os.path.join(RES, "concordance_percell.csv"), index_col=0)
    wide["cluster"] = wide["cluster"].astype(str)
    cl_all = wide["cluster"].values

    for metric, (sub, xlab, ylab) in METRICS.items():
        d = ensure(os.path.join(OUTDIR, sub))
        for idx, tag, title in GPCCA_CONFIGS:
            col = f"{metric}::{tag}"
            if col not in wide.columns:
                print("[skip] missing", col); continue
            x = wide[f"{metric}_canonical"].values
            y = wide[col].values
            rho = panel(x, y, cl_all, title, xlab, ylab,
                        os.path.join(d, f"{idx}_{sub}.png"))
            print(f"  {sub:10s} {title:24s} rho={rho:.3f}")

    # ---- Palantir cross-check (independent method) ----
    pal = pd.read_csv(os.path.join(RES, "palantir_branch_assignment.csv"),
                      index_col="cell")
    common = wide.index.intersection(pal.index)
    w2 = wide.loc[common]
    p2 = pal.loc[common]
    cl2 = w2["cluster"].values
    # pseudotime: CellRank/DPT vs Palantir
    panel(w2["dpt_canonical"].values, p2["palantir_pseudotime"].values, cl2,
          "Palantir vs CellRank", "CellRank pseudotime", "Palantir pseudotime",
          os.path.join(OUTDIR, "pseudotime", "14_palantir.png"))
    # fate probability: CellRank C3 vs Palantir C3
    panel(w2["fpC3_canonical"].values, p2["fp_C3"].values, cl2,
          "Palantir vs CellRank", "CellRank C3 fate prob.", "Palantir C3 fate prob.",
          os.path.join(OUTDIR, "fateprob", "14_palantir.png"))
    print("[palantir] pseudotime + fateprob panels written")

    print("\n[DONE] clean S5 panels in:", OUTDIR)


if __name__ == "__main__":
    main()
