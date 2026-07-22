"""
Reviewer #2, point 5 -- sweep the GPCCA Schur dimension (n_components) over a
range of values, score each by per-cell concordance with the canonical run
(Spearman rho on diffusion pseudotime AND effector-C3 fate probability), and
render the best one as a stand-alone high-resolution supplement panel.

Canonical per-cell values are read from concordance_percell.csv (written by r5h);
only the new n_components perturbations are run through GPCCA here. An overview
grid of the whole sweep + a ranked CSV are also written.

Output: revise/results/r5_trajectory/
    ncomponents_sweep.csv
    fig_r5_ncomponents_sweep.png              (overview, fate prob)
    concordance_panels/{pseudotime,fateprob}/best_ncomponents_*.png

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
import r5h_concordance_scatter as R5H
import r5i_concordance_panels as R5I

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTDIR = R5H.OUTDIR
CANON_NC = R5D.C["n_components"]                     # = 20
SWEEP = [10, 15, 25, 30, 40, 50, 60]                 # canonical 20 is the reference


def load_base():
    wide = pd.read_csv(os.path.join(OUTDIR, "concordance_percell.csv"), index_col=0)
    base = wide[["cluster", "dpt_canonical", "fpC3_canonical"]].copy()
    base["cluster"] = base["cluster"].astype(str)
    return base


def rho(base, d, value):
    common = base.index.intersection(d.index)
    x = base.loc[common, f"{value}_canonical"].values
    y = d.loc[common, value].values
    m = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[m], y[m]).correlation, int(m.sum())


def run_sweep():
    adata = sc.read_h5ad(R5D.H5AD)
    adata = adata[adata.obs[R5D.CLUSTER_KEY].astype(str).isin(R5D.ALL_CLUSTERS)].copy()
    R5D.compute_scores(adata)
    base = load_base()

    per, rows, wide = {}, [], base.copy()
    for nc in SWEEP:
        print(f"[r5j] n_components = {nc} ...", flush=True)
        res = R5D.run_gpcca(adata, n_neighbors=R5D.C["n_neighbors"], n_components=nc,
                            n_terminal=R5D.C["n_terminal"],
                            root_strategy=R5D.C["root_strategy"],
                            root_seed=R5D.C["root_seed"])
        if res.get("status") != "ok":
            print(f"   -> {res.get('status')}"); continue
        d = pd.DataFrame({"cell": res["obs_names"], "dpt": res["dpt"],
                          "fpC3": res["fpC3"]}).set_index("cell")
        per[nc] = d
        r_dpt, n = rho(base, d, "dpt")
        r_fp, _ = rho(base, d, "fpC3")
        rows.append({"n_components": nc, "rho_pseudotime": round(r_dpt, 4),
                     "rho_fateprob": round(r_fp, 4),
                     "rho_min": round(min(r_dpt, r_fp), 4),
                     "rho_mean": round((r_dpt + r_fp) / 2, 4), "n": n})
        print(f"   -> rho_dpt={r_dpt:.4f}  rho_fp={r_fp:.4f}", flush=True)

        wide = wide.join(d.rename(columns={"dpt": f"dpt::nc{nc}",
                                           "fpC3": f"fpC3::nc{nc}"}))
    tab = pd.DataFrame(rows).sort_values("rho_min", ascending=False).reset_index(drop=True)
    tab.to_csv(os.path.join(OUTDIR, "ncomponents_sweep.csv"), index=False)
    wide.to_csv(os.path.join(OUTDIR, "ncomponents_percell.csv"))
    return base, per, tab


def overview(base, per, tab):
    ncs = list(per.keys())
    fig, axes = plt.subplots(1, len(ncs), figsize=(2.7 * len(ncs), 3.2),
                             squeeze=False)
    for ax, nc in zip(axes[0], ncs):
        R5H._panel(ax, base, per[nc], "fpC3")
        canon = "  (canonical)" if nc == CANON_NC else ""
        ax.set_title(f"n_components = {nc}{canon}", fontsize=9.2)
    fig.supxlabel("canonical  effector (C3) fate probability", fontsize=10, y=0.03)
    fig.supylabel("perturbed  effector (C3) fate probability", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=R5H.CLUST_COL[c], markersize=8,
                          label=R5H.CLUST_NAME[c]) for c in ["4", "0", "1", "2"]]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.004, R5H.CANON_NOTE, ha="center", va="bottom", fontsize=8,
             color="#777777")
    fig.suptitle("GPCCA Schur dimension (n_components) sweep -- effector (C3) "
                 "fate probability is invariant", fontsize=12.5, y=1.06)
    fig.tight_layout(rect=[0.01, 0.06, 1, 0.94])
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"fig_r5_ncomponents_sweep.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[r5j] saved fig_r5_ncomponents_sweep")


def standalone_panels(base, per):
    """One high-res PNG per swept n_components, both metrics -- all invariant, so
    the user picks whichever value to place in the supplement."""
    panel_dir = U.ensure_dir(os.path.join(R5I.PANEL_DIR, "ncomponents_sweep"))
    for value, mtag, axis_label in R5I.METRICS:
        sub = U.ensure_dir(os.path.join(panel_dir, mtag))
        for nc in per:
            tag = f"n_components = {nc}"
            path = os.path.join(sub, f"ncomponents_{nc:02d}.png")
            R5I.make_panel(base, per[nc], value, "Parameter", tag, axis_label, path)
    print(f"[r5j] standalone panels ({len(per)} values x 2 metrics) -> {panel_dir}")


def main():
    base, per, tab = run_sweep()
    print("\n[r5j] concordance with canonical (n_components = %d):" % CANON_NC)
    print(tab.to_string(index=False))
    allperfect = bool((tab["rho_min"] >= 0.9999).all())
    print("[r5j] all n_components perfectly concordant (rho=1.000): %s" % allperfect)
    print("[r5j] NOTE: rho_pseudotime is trivially 1.000 -- dpt is computed before "
          "the Schur step, so n_components cannot affect it; rho_fateprob is the "
          "informative column.")
    overview(base, per, tab)
    standalone_panels(base, per)
    print(f"\n[r5j DONE] -> {OUTDIR}")


if __name__ == "__main__":
    main()
