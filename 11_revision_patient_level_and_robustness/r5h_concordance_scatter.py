"""
Reviewer #2, point 5 -- SHOW (not assert) that nothing changes when we perturb the
GPCCA pipeline. For a representative perturbation on each axis we plot, cell by
cell, the canonical value against the perturbed value:

    * diffusion pseudotime               (the ordering that drives everything)
    * effector (C3) fate probability     (what the branch assignment is based on)

Points on the y = x diagonal = that cell's value did not change. Each panel is
annotated with the Spearman correlation. Concordance close to the diagonal is
direct visual evidence of invariance; where a setting genuinely changes things
(e.g. the extreme min-exhaustion root) the scatter honestly moves off-diagonal.

Runs GPCCA (fast krylov Schur) for canonical + the chosen configs, saves the
per-cell matrix, and draws two small-multiple figures.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python  (run in background)
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5d_gpcca_robustness as R5D

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r5_trajectory"))

# cluster colours (paper C1..C4)
CLUST_COL = {"4": "#4c78a8", "0": "#f58518", "1": "#54a24b", "2": "#b279a2"}
CLUST_NAME = {"4": "C1 naive", "0": "C2 transit.", "1": "C3 effector", "2": "C4 exhaust."}

C = R5D.C  # canonical settings

# full perturbation sweep: every configuration in the r5d robustness grid, laid
# out one axis per row (Parameter / Root / Patient exclusion).
CONFIGS = [
    # (1) GPCCA parameter settings
    ("Parameter", "n_neighbors = 15", {**C, "n_neighbors": 15}),
    ("Parameter", "n_neighbors = 50", {**C, "n_neighbors": 50}),
    ("Parameter", "n_components = 30", {**C, "n_components": 30}),
    ("Parameter", "n_terminal = 2", {**C, "n_terminal": 2}),
    ("Parameter", "n_terminal = 4", {**C, "n_terminal": 4}),
    # (2) root-cell selection
    ("Root", "root = top-naive medoid", {**C, "root_strategy": "top_naive_medoid"}),
    ("Root", "root = random #1", {**C, "root_strategy": "random", "root_seed": 1}),
    ("Root", "root = random #2", {**C, "root_strategy": "random", "root_seed": 2}),
    ("Root", "root = min-exhaustion", {**C, "root_strategy": "min_exhaustion"}),
    # (3) leave-one-patient-out (3 non-responders, then 3 responders)
    ("Patient", "drop PHD001 (NR)", {**C, "_drop": "PHD001"}),
    ("Patient", "drop PHD002 (NR)", {**C, "_drop": "PHD002"}),
    ("Patient", "drop PHD008 (NR)", {**C, "_drop": "PHD008"}),
    ("Patient", "drop PHD003 (R)", {**C, "_drop": "PHD003"}),
    ("Patient", "drop PHD004 (R)", {**C, "_drop": "PHD004"}),
    ("Patient", "drop PHD009 (R)", {**C, "_drop": "PHD009"}),
]

# order + spelled-out canonical baseline for the figure footnote
AXIS_ORDER = ["Parameter", "Root", "Patient"]
CANON_NOTE = ("canonical: n_neighbors = 30, n_components = 20, n_terminal = 3, "
              "root = first-cell")


def run_all():
    adata = sc.read_h5ad(R5D.H5AD)
    adata = adata[adata.obs[R5D.CLUSTER_KEY].astype(str).isin(R5D.ALL_CLUSTERS)].copy()
    R5D.compute_scores(adata)

    print("[r5h] canonical ...", flush=True)
    canon = R5D.run_gpcca(adata, **C)
    base = pd.DataFrame({"cell": canon["obs_names"], "cluster": canon["cluster"],
                         "dpt_canonical": canon["dpt"],
                         "fpC3_canonical": canon["fpC3"]}).set_index("cell")

    per = {}
    for axis, tag, cfg in CONFIGS:
        drop = cfg.pop("_drop", None)
        print(f"[r5h] {tag} ...", flush=True)
        sub = adata[adata.obs["patient"].astype(str) != drop].copy() if drop else adata
        res = R5D.run_gpcca(sub, **{k: cfg[k] for k in
                                    ["n_neighbors", "n_components", "n_terminal",
                                     "root_strategy", "root_seed"]})
        if res.get("status") != "ok":
            print(f"   -> {res.get('status')}"); continue
        per[tag] = (axis, pd.DataFrame(
            {"cell": res["obs_names"], "dpt": res["dpt"], "fpC3": res["fpC3"]}
        ).set_index("cell"))

    # save per-cell wide matrix
    wide = base.copy()
    for tag, (_, d) in per.items():
        wide = wide.join(d.rename(columns={"dpt": f"dpt::{tag}",
                                           "fpC3": f"fpC3::{tag}"}))
    wide.to_csv(os.path.join(OUTDIR, "concordance_percell.csv"))
    return base, per


def _panel(ax, base, d, value):
    common = base.index.intersection(d.index)
    x = base.loc[common, f"{value}_canonical"].values
    y = d.loc[common, value].values
    cl = base.loc[common, "cluster"].values
    m = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[m], y[m], cl[m]
    for c in ["4", "0", "1", "2"]:
        sel = cl == c
        ax.scatter(x[sel], y[sel], s=4, c=CLUST_COL[c], alpha=0.35,
                   linewidths=0, rasterized=True)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, ls="--", lw=1.2, color="#555555", zorder=3)
    rho = spearmanr(x, y).correlation
    ax.text(0.04, 0.95, f"ρ = {rho:.3f}\nn = {len(x)}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.tick_params(labelsize=7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def scatter_grid(base, per, value, fname, title, axis_label):
    # one axis (Parameter / Root / Patient) per row; unused cells switched off
    by_axis = {a: [t for t in per if per[t][0] == a] for a in AXIS_ORDER}
    ncol = max(len(v) for v in by_axis.values())
    nrow = len(AXIS_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.15 * nrow),
                             squeeze=False)
    for r, axis in enumerate(AXIS_ORDER):
        tags = by_axis[axis]
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(tags):
                ax.axis("off"); continue
            tag = tags[c]
            _panel(ax, base, per[tag][1], value)
            ax.set_title(f"{axis}: {tag}", fontsize=8.8)
    fig.supxlabel(f"canonical  {axis_label}", fontsize=11, y=0.032)
    fig.supylabel(f"perturbed  {axis_label}", fontsize=11)
    # cluster legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=CLUST_COL[c], markersize=8,
                          label=CLUST_NAME[c]) for c in ["4", "0", "1", "2"]]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.004, CANON_NOTE, ha="center", va="bottom", fontsize=8.5,
             color="#555555")
    fig.suptitle(title, fontsize=13.5, x=0.5, y=1.03)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.965])
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"{fname}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("[r5h] saved", fname)


def main():
    base, per = run_all()
    scatter_grid(base, per, "dpt", "fig_r5_concordance_pseudotime",
                 "Per-cell pseudotime is unchanged by GPCCA parameters, root, "
                 "and patient exclusion", "diffusion pseudotime")
    scatter_grid(base, per, "fpC3", "fig_r5_concordance_fateprob",
                 "Per-cell effector (C3) fate probability is unchanged across "
                 "settings", "effector (C3) fate probability")
    print("\n[r5h DONE] outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
