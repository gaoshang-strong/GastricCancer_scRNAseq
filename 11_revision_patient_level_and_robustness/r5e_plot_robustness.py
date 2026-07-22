"""
Reviewer #2, point 5 -- figure of the GPCCA robustness across the three axes the
reviewer names: parameter settings, root-cell selection, patient exclusion.

Reads gpcca_robustness_grid.csv (from r5d) and draws, per axis, a dumbbell plot of
the per-patient effector(C3)-branch fraction in Non-Responders vs Responders for
each configuration. R dot to the right of NR dot = effector branch enriched in
responders. The canonical run and the single failing config (n_terminal=2, which
structurally forbids the effector terminal) are annotated.

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
from matplotlib.lines import Line2D

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
OUTDIR = os.path.join(BASE, "revise", "results", "r5_trajectory")
GRID = os.path.join(OUTDIR, "gpcca_robustness_grid.csv")

# paper's established response colours (figure02.ipynb): purple=R, blue=NR
COL_R = "#6a3d9a"
COL_NR = "#1f78b4"
INK = "#222222"
MUTED = "#8a8a8a"

AXES = [("parameter", "Parameter settings"),
        ("root", "Root-cell selection"),
        ("patient_exclusion", "Patient exclusion (leave-one-out)")]

PRETTY = {
    "canonical": "canonical (nn 30, nterm 3)",
    "n_neighbors=15": "n_neighbors = 15", "n_neighbors=50": "n_neighbors = 50",
    "n_terminal=2": "n_terminal = 2", "n_terminal=4": "n_terminal = 4",
    "n_components=30": "n_components = 30",
    "root=top_naive_medoid": "root = top-naive medoid",
    "root=min_exhaustion": "root = min-exhaustion",
    "root=random1": "root = random #1", "root=random2": "root = random #2",
    "drop_PHD001": "drop PHD001 (NR)", "drop_PHD002": "drop PHD002 (NR)",
    "drop_PHD008": "drop PHD008 (NR)", "drop_PHD003": "drop PHD003 (R)",
    "drop_PHD004": "drop PHD004 (R)", "drop_PHD009": "drop PHD009 (R)",
}


def main():
    df = pd.read_csv(GRID)
    df = df[df["status"] == "ok"].copy()
    for c in ["mean_fracC3_R", "mean_fracC3_NR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    canon = df[df["axis"] == "canonical"].iloc[0]

    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 8.4),
        gridspec_kw={"height_ratios": [5, 4, 6], "hspace": 0.32})

    for ax, (akey, atitle) in zip(axes, AXES):
        sub = df[df["axis"] == akey].copy()
        # prepend canonical as a reference row in every panel
        rows = pd.concat([pd.DataFrame([canon]), sub], ignore_index=True)
        rows = rows.drop_duplicates(subset=["config"]).reset_index(drop=True)
        y = np.arange(len(rows))[::-1]  # first row at top

        for yi, (_, r) in zip(y, rows.iterrows()):
            nr, rr = r["mean_fracC3_NR"], r["mean_fracC3_R"]
            up = rr > nr
            # connecting line
            ax.plot([nr, rr], [yi, yi], color=(MUTED if up else "#d1495b"),
                    lw=2, zorder=1, solid_capstyle="round")
            # NR (square) and R (circle) markers, 2px surface ring
            ax.scatter(nr, yi, s=70, marker="s", color=COL_NR, zorder=3,
                       edgecolors="white", linewidths=1.4)
            ax.scatter(rr, yi, s=80, marker="o", color=COL_R, zorder=3,
                       edgecolors="white", linewidths=1.4)
            if not up:  # flag the failing config
                ax.annotate("direction fails", (max(nr, rr), yi),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color="#d1495b")

        ax.set_yticks(y)
        labels = [PRETTY.get(c, c) for c in rows["config"]]
        ax.set_yticklabels(labels, fontsize=9)
        # bold the canonical reference row without using mathtext (avoids _ -> subscript)
        for tick, c in zip(ax.get_yticklabels(), rows["config"]):
            if c == "canonical":
                tick.set_fontweight("bold")
        ax.set_title(atitle, fontsize=11, loc="left", color=INK, pad=6)
        ax.set_xlim(-0.02, 1.0)
        ax.grid(axis="x", color="#e8e8e8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)

    axes[-1].set_xlabel("Per-patient fraction on the effector (C3) branch",
                        fontsize=10, color=INK)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_R,
               markersize=10, label="Responder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_NR,
               markersize=9, label="Non-Responder"),
    ]
    axes[0].legend(handles=legend, loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("CD8 effector-branch enrichment is robust across GPCCA "
                 "parameters, root choice, and patient exclusion",
                 fontsize=12, x=0.02, ha="left", y=0.995)
    n_ok = (df["mean_fracC3_R"] > df["mean_fracC3_NR"]).sum()
    fig.text(0.02, 0.008,
             f"Effector branch higher in responders in {n_ok}/{len(df)} configurations "
             f"(the one exception, n_terminal=2, forbids the effector terminal).",
             fontsize=8.5, color=MUTED)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"fig_r5_gpcca_robustness.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[r5e] saved fig_r5_gpcca_robustness.png / .pdf in", OUTDIR)


if __name__ == "__main__":
    main()
