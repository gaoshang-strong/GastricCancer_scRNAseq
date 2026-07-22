"""
Reviewer #2, point 5 -- split the concordance small-multiples (r5h) into ONE
stand-alone, high-resolution PNG per configuration, so each panel can be dropped
into the supplement individually.

Reuses the already-computed per-cell matrix (concordance_percell.csv) written by
r5h_concordance_scatter.py -- no GPCCA is re-run here. For every configuration in
r5h.CONFIGS we emit two panels (diffusion pseudotime + effector-C3 fate prob),
each a self-contained figure with title, axis labels, cluster legend, Spearman
annotation, and the canonical-baseline footnote.

Output: revise/results/r5_trajectory/concordance_panels/{pseudotime,fateprob}/

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import re
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5h_concordance_scatter as R5H

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTDIR = R5H.OUTDIR
PANEL_DIR = U.ensure_dir(os.path.join(OUTDIR, "concordance_panels"))
DPI = 300

# (metric column, filename tag, subfolder, axis label)
METRICS = [
    ("dpt", "pseudotime", "diffusion pseudotime"),
    ("fpC3", "fateprob", "effector (C3) fate probability"),
]


def slug(s):
    s = s.replace("#", "").replace("(", "").replace(")", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def load():
    wide = pd.read_csv(os.path.join(OUTDIR, "concordance_percell.csv"), index_col=0)
    base = wide[["cluster", "dpt_canonical", "fpC3_canonical"]].copy()
    base["cluster"] = base["cluster"].astype(str)
    per = {}
    for axis, tag, _ in R5H.CONFIGS:
        dc, fc = f"dpt::{tag}", f"fpC3::{tag}"
        if dc not in wide.columns:
            print("[r5i] missing column for:", tag); continue
        d = wide[[dc, fc]].rename(columns={dc: "dpt", fc: "fpC3"}).dropna(how="all")
        per[tag] = (axis, d)
    return base, per


def make_panel(base, d, value, axis, tag, axis_label, path):
    fig, ax = plt.subplots(figsize=(4.4, 4.9))
    R5H._panel(ax, base, d, value)             # scatter + diagonal + rho/n + spines
    ax.set_title(f"{axis}: {tag}", fontsize=11)
    ax.set_xlabel(f"canonical  {axis_label}", fontsize=9.5)
    ax.set_ylabel(f"perturbed  {axis_label}", fontsize=9.5)
    ax.tick_params(labelsize=8)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=R5H.CLUST_COL[c], markersize=7,
                          label=R5H.CLUST_NAME[c]) for c in ["4", "0", "1", "2"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.045), handletextpad=0.3,
               columnspacing=1.1)
    fig.text(0.5, 0.005, R5H.CANON_NOTE, ha="center", va="bottom", fontsize=6.3,
             color="#777777")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    base, per = load()
    n = 0
    for value, mtag, axis_label in METRICS:
        sub = U.ensure_dir(os.path.join(PANEL_DIR, mtag))
        for i, (axis, tag, _) in enumerate(R5H.CONFIGS, 1):
            if tag not in per:
                continue
            fname = f"{i:02d}_{slug(axis)}_{slug(tag)}.png"
            make_panel(base, per[tag][1], value, axis, tag, axis_label,
                       os.path.join(sub, fname))
            n += 1
        print(f"[r5i] {mtag}: {len(per)} panels -> {sub}")
    print(f"[r5i DONE] {n} PNGs ({DPI} dpi) under {PANEL_DIR}")


if __name__ == "__main__":
    main()
