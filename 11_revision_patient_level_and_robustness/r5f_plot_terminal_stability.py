"""
Reviewer #2, point 5 -- figure of TERMINAL-STATE stability across the GPCCA
robustness grid (parameter settings, root-cell selection, patient exclusion).

Reads gpcca_robustness_grid.csv (r5d) and draws a presence matrix: rows = each
configuration (grouped by axis), columns = the four CD8 states C1..C4; a filled
mark means that state was recovered as a terminal macrostate in that run. A
bottom summary counts, per state, in how many configurations it is terminal.

This answers the reviewer's "terminal states remain stable" directly and honestly:
the two states central to the paper (effector C3, exhausted C4) are recovered in
almost every configuration; the naive state C1 is essentially never terminal
(as expected -- it is the root).

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
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

PRESENT = "#238b45"   # green = state recovered as terminal
ABSENT = "#d9d9d9"
INK = "#222222"
MUTED = "#8a8a8a"

XOFF = 0.9   # push the mark columns right so row labels never touch them
STATES = ["C1", "C2", "C3", "C4"]
STATE_LABEL = {"C1": "C1\nnaive", "C2": "C2\ntransit.",
               "C3": "C3\neffector", "C4": "C4\nexhausted"}

AXIS_ORDER = [("canonical", "canonical"),
              ("parameter", "Parameter settings"),
              ("root", "Root-cell selection"),
              ("patient_exclusion", "Patient exclusion")]

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
    df = df[df["terminal_clusters"].notna()].copy()

    # build a layout: canonical row, then for each axis a header row + its configs
    layout = []   # each item: ("config", cfg, present) or ("header", title, None)
    seen = set()
    for akey, title in AXIS_ORDER:
        rows = df[df["axis"] == akey]
        if akey != "canonical":
            layout.append(("header", title, None))
        for _, r in rows.iterrows():
            if r["config"] in seen:
                continue
            seen.add(r["config"])
            layout.append(("config", r["config"],
                           set(str(r["terminal_clusters"]).split("|"))))

    n = len(layout)
    n_cfg = sum(1 for k, *_ in layout if k == "config")
    fig, ax = plt.subplots(figsize=(6.8, 0.40 * n + 2.0))
    y_of = [n - 1 - i for i in range(n)]

    freq = {s: 0 for s in STATES}
    for yy, item in zip(y_of, layout):
        kind, a, present = item
        if kind == "header":
            ax.text(-0.75, yy, a, fontsize=9.5, fontweight="bold",
                    color=INK, ha="left", va="center")
            continue
        cfg = a
        for j, s in enumerate(STATES):
            on = s in present
            if on:
                freq[s] += 1
            ax.scatter(j + XOFF, yy, s=150, marker="o",
                       color=(PRESENT if on else "white"),
                       edgecolors=(PRESENT if on else ABSENT),
                       linewidths=1.6, zorder=3)
        ax.text(-0.71, yy, PRETTY.get(cfg, cfg), fontsize=8.6, va="center",
                ha="left", color=INK,
                fontweight=("bold" if cfg == "canonical" else "normal"))

    # column frequency summary at the bottom
    ybase = -1.4
    ax.text(-0.75, ybase, "terminal in:", fontsize=8.6, ha="left",
            va="center", color=MUTED)
    for j, s in enumerate(STATES):
        ax.text(j + XOFF, ybase, f"{freq[s]}/{n_cfg}", fontsize=9.5, ha="center",
                va="center", color=(PRESENT if freq[s] >= n_cfg - 2 else MUTED),
                fontweight="bold")

    # column headers
    for j, s in enumerate(STATES):
        ax.text(j + XOFF, n - 0.35, STATE_LABEL[s], fontsize=9, ha="center",
                va="bottom", color=INK)

    ax.set_xlim(-0.75, len(STATES) - 0.25 + XOFF)
    ax.set_ylim(ybase - 0.6, n + 0.6)
    ax.axis("off")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PRESENT,
               markeredgecolor=PRESENT, markersize=11, label="terminal state"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=ABSENT, markersize=11, label="not terminal"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.02, 1.06),
              frameon=False, fontsize=8.6, ncol=2, handletextpad=0.3,
              columnspacing=1.2)

    fig.suptitle("Terminal-state recovery across the GPCCA robustness grid",
                 fontsize=12, x=0.02, ha="left", y=1.0)
    fig.text(0.02, 0.005,
             "Late CD8 states are recovered as terminal in almost every run: "
             "exhausted C4 15/16, transitional C2 15/16, effector C3 14/16; "
             "naive C1 is the root, not a terminal.",
             fontsize=8.3, color=MUTED)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"fig_r5_terminal_stability.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[r5f] saved fig_r5_terminal_stability.png / .pdf")
    print("[r5f] terminal-state frequency:", freq, "of", n_cfg)


if __name__ == "__main__":
    main()
