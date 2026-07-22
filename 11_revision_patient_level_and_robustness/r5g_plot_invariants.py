"""
Reviewer #2, point 5 -- the ONE figure that answers "what stays invariant when we
change parameters / root / which patient is excluded?".

Not the (varying) effector-branch magnitude, but the qualitative conclusions the
reviewer asks about: terminal states and branch assignments. For every GPCCA
configuration (grouped by the three axes) we mark whether each invariant holds:

    (1) effector state (C3) is recovered as a terminal
    (2) exhausted state (C4) is recovered as a terminal
    (3) the per-cell branch partition matches the canonical run (ARI >= 0.8)
    (4) the effector arm is enriched in responders (direction preserved)

A green check = holds, a red cross = fails. The bottom row counts, per invariant,
in how many of the 16 configurations it holds. Only n_terminal=2 (which forbids a
3rd terminal) and the extreme min-exhaustion root break any of them.

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

GOOD = "#238b45"    # holds
BAD = "#d1495b"     # fails
INK = "#222222"
MUTED = "#8a8a8a"
ARI_MIN = 0.8

INVARIANTS = [
    ("Effector (C3)\nis terminal", "c3_term"),
    ("Exhausted (C4)\nis terminal", "c4_term"),
    ("Branch partition\n= canonical\n(ARI ≥ 0.8)", "partition"),
    ("Effector arm\nenriched in\nresponders", "direction"),
]

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


def invariants_for(row):
    term = set(str(row["terminal_clusters"]).split("|")) if pd.notna(row["terminal_clusters"]) else set()
    ari = pd.to_numeric(row.get("ari"), errors="coerce")
    ok = row["status"] == "ok"
    return {
        "c3_term": "C3" in term,
        "c4_term": "C4" in term,
        "partition": bool(ok and np.isfinite(ari) and ari >= ARI_MIN),
        "direction": bool(ok and str(row.get("C3_higher_in_R")) == "True"),
    }


def draw_check(ax, x, y, good):
    """green check (holds) or red cross (fails), drawn as vector marks."""
    if good:
        ax.plot([x - 0.12, x - 0.02, x + 0.16], [y, y - 0.14, y + 0.16],
                color=GOOD, lw=2.6, solid_capstyle="round", zorder=3)
    else:
        for dx in (-0.13, 0.13):
            ax.plot([x - dx, x + dx], [y - 0.15, y + 0.15],
                    color=BAD, lw=2.6, solid_capstyle="round", zorder=3)


def main():
    df = pd.read_csv(GRID)

    layout = []
    seen = set()
    for akey, title in AXIS_ORDER:
        rows = df[df["axis"] == akey]
        if akey != "canonical":
            layout.append(("header", title, None))
        for _, r in rows.iterrows():
            if r["config"] in seen:
                continue
            seen.add(r["config"])
            layout.append(("config", r["config"], invariants_for(r)))

    n = len(layout)
    n_cfg = sum(1 for k, *_ in layout if k == "config")
    ncol = len(INVARIANTS)
    xcol = {key: 0.9 + 1.15 * j for j, (_, key) in enumerate(INVARIANTS)}

    fig, ax = plt.subplots(figsize=(8.6, 0.40 * n + 2.2))
    y_of = [n - 1 - i for i in range(n)]

    counts = {key: 0 for _, key in INVARIANTS}
    for yy, item in zip(y_of, layout):
        kind, a, inv = item
        if kind == "header":
            ax.text(-0.75, yy, a, fontsize=9.5, fontweight="bold",
                    color=INK, ha="left", va="center")
            continue
        ax.text(-0.71, yy, PRETTY.get(a, a), fontsize=8.8, va="center",
                ha="left", color=INK,
                fontweight=("bold" if a == "canonical" else "normal"))
        for label, key in INVARIANTS:
            good = inv[key]
            counts[key] += int(good)
            draw_check(ax, xcol[key], yy, good)

    # column headers
    for label, key in INVARIANTS:
        ax.text(xcol[key], n - 0.05, label, fontsize=9, ha="center",
                va="bottom", color=INK, linespacing=1.15)

    # per-invariant tally
    ybase = -1.5
    ax.text(-0.75, ybase, "holds in:", fontsize=8.8, ha="left", va="center",
            color=MUTED)
    for label, key in INVARIANTS:
        c = counts[key]
        ax.text(xcol[key], ybase, f"{c}/{n_cfg}", fontsize=10, ha="center",
                va="center", fontweight="bold",
                color=(GOOD if c >= n_cfg - 2 else MUTED))

    ax.set_xlim(-0.78, 0.9 + 1.15 * (ncol - 1) + 0.7)
    ax.set_ylim(ybase - 0.6, n + 2.6)
    ax.axis("off")

    # legend, on its own row above the column headers
    ly = n + 2.1
    draw_check(ax, 0.9, ly, True)
    ax.text(0.9 + 0.28, ly, "invariant holds", fontsize=9, va="center", color=INK)
    draw_check(ax, 2.85, ly, False)
    ax.text(2.85 + 0.28, ly, "invariant fails", fontsize=9, va="center", color=INK)

    fig.suptitle("What stays invariant across GPCCA parameters, root choice, "
                 "and patient exclusion", fontsize=12.5, x=0.02, ha="left", y=1.02)
    fig.text(0.02, 0.004,
             "The two late terminal states and the responder-biased effector arm "
             "are recovered in almost every configuration; only n_terminal=2 "
             "(no 3rd terminal) and the extreme min-exhaustion root break an invariant.",
             fontsize=8.3, color=MUTED)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"fig_r5_invariants.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[r5g] saved fig_r5_invariants.png / .pdf")
    print("[r5g] holds-in counts:", counts, "of", n_cfg)


if __name__ == "__main__":
    main()
