"""
EXPLORATION (not for submission) -- can we tune Palantir so its per-cell effector
(C3) fate probability roughly matches the canonical CellRank/GPCCA result?

The existing crosscheck (r5b) fed scanpy's diffmap straight into Palantir and got
near-zero branch agreement (ARI ~ 0), largely because Palantir's own terminal
detection never flags the C3 effector cluster. Here we:

  1. build Palantir's OWN multiscale diffusion space from X_scVI (the proper
     pipeline: run_diffusion_maps -> determine_multiscale_space),
  2. anchor the SAME root CellRank uses (first cell of the naive cluster 4),
  3. hand Palantir the SAME terminal cells CellRank's macrostates land on
     (latest-DPT cells of the relevant clusters),
  4. grid over {diffusion n_components} x {2 terminals C3,C4 | 3 terminals C2,C3,C4}

and score each run by per-cell Spearman(Palantir fpC3, CellRank fpC3) + argmax
branch ARI/kappa + pseudotime Spearman. Best run is saved + plotted.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python  (palantir 1.4.2)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

import palantir.utils as pu
import palantir.core as pc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5d_gpcca_robustness as R5D

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42

OUTDIR = R5D.OUTDIR
CLUSTER_KEY = R5D.CLUSTER_KEY
CLUST_COL = {"4": "#4c78a8", "0": "#f58518", "1": "#54a24b", "2": "#b279a2"}
CLUST_NAME = {"4": "C1 naive", "0": "C2 transit.", "1": "C3 effector", "2": "C4 exhaust."}

# clusters -> palantir terminal labels
EFF, EXH, TRANS = "1", "2", "0"      # C3, C4, C2

# grid
DIFF_NCOMP = [10, 20]
TERMSETS = {
    "2term_C3_C4": {EFF: "C3", EXH: "C4"},
    "3term_C2_C3_C4": {TRANS: "C2", EFF: "C3", EXH: "C4"},
}
N_TERM_CELLS = 10
KNN = 30
NUM_WAYPOINTS = 800


def load_canonical():
    cr = pd.read_csv(os.path.join(OUTDIR, "gpcca_branch_assignment_canonical.csv")
                     ).set_index("cell")
    dptc = pd.read_csv(os.path.join(OUTDIR, "concordance_percell.csv"),
                       index_col=0)[["cluster", "dpt_canonical"]]
    dptc["cluster"] = dptc["cluster"].astype(str)
    cr = cr.join(dptc)
    return cr


def terminal_cells(cr, cluster_labels, n=N_TERM_CELLS):
    ts = {}
    for clv, lab in cluster_labels.items():
        cells = cr.index[cr["cluster"] == clv]
        top = cr.loc[cells, "dpt_canonical"].nlargest(n).index
        for c in top:
            ts[c] = lab
    return pd.Series(ts)


def score(fp_pal, cr, obs_names):
    """fp_pal: DataFrame branch_probs indexed by obs_names (has C3, C4 cols)."""
    common = cr.index.intersection(obs_names)
    p3 = fp_pal["C3"].reindex(common).values
    p4 = fp_pal["C4"].reindex(common).values
    c3 = cr.loc[common, "fpC3"].values
    c4 = cr.loc[common, "fpC4"].values
    m = np.isfinite(p3) & np.isfinite(c3)
    rho = spearmanr(p3[m], c3[m]).correlation
    pal_b = np.where(p3[m] >= p4[m], 1, 0)
    cr_b = np.where(c3[m] >= c4[m], 1, 0)
    return {"spearman_fpC3": round(float(rho), 3),
            "argmax_ARI": round(adjusted_rand_score(cr_b, pal_b), 3),
            "argmax_kappa": round(cohen_kappa_score(cr_b, pal_b), 3),
            "n": int(m.sum())}, (common[m], p3[m], c3[m])


def run_one(ms, root, ts, cr):
    pr = pc.run_palantir(ms, early_cell=root, terminal_states=ts, knn=KNN,
                         num_waypoints=NUM_WAYPOINTS, use_early_cell_as_start=True)
    # DataFrame input -> branch_probs columns are terminal CELL barcodes, not the
    # C2/C3/C4 labels. Aggregate columns back to labels via the terminal_states map.
    raw = pr.branch_probs
    col_label = ts.reindex(raw.columns)
    fp = pd.DataFrame(index=raw.index)
    for lab in pd.unique(ts.values):
        cols = col_label.index[col_label.values == lab]
        fp[lab] = raw[list(cols)].sum(axis=1) if len(cols) else 0.0
    for need in ["C3", "C4"]:
        if need not in fp.columns:
            fp[need] = 0.0
    bp = fp
    pt = pr.pseudotime.reindex(cr.index)
    rho_pt = spearmanr(pt.dropna(),
                       cr.loc[pt.dropna().index, "dpt_canonical"]).correlation
    return bp, round(float(rho_pt), 3)


def main():
    cr = load_canonical()
    print("[r5l] loading CD8 subset ...", flush=True)
    ad = sc.read_h5ad(R5D.H5AD)
    ad = ad[ad.obs[CLUSTER_KEY].astype(str).isin(R5D.ALL_CLUSTERS)].copy()
    R5D.compute_scores(ad)

    root = ad.obs_names[R5D.pick_root(ad, "first_cell")]
    print("[r5l] root (CellRank first_cell):", root, flush=True)

    X = pd.DataFrame(ad.obsm["X_scVI"], index=ad.obs_names,
                     columns=[f"scVI{i}" for i in range(ad.obsm["X_scVI"].shape[1])])

    rows, best = [], None
    for nc in DIFF_NCOMP:
        print(f"[r5l] diffusion maps n_components={nc} ...", flush=True)
        dm = pu.run_diffusion_maps(X, n_components=nc, knn=KNN, seed=0)
        ms = pu.determine_multiscale_space(dm)
        for tname, clabels in TERMSETS.items():
            ts = terminal_cells(cr, clabels)
            tag = f"diff{nc}_{tname}"
            print(f"[r5l] palantir: {tag}  ({len(ts)} terminals) ...", flush=True)
            try:
                bp, rho_pt = run_one(ms, root, ts, cr)
                bp = bp.reindex(ad.obs_names)
                sc_, pts = score(bp, cr, ad.obs_names)
                row = {"config": tag, "diff_ncomp": nc, "terminals": tname,
                       "spearman_pseudotime": rho_pt, **sc_}
                rows.append(row)
                print("   ->", row, flush=True)
                if best is None or sc_["spearman_fpC3"] > best[0]:
                    best = (sc_["spearman_fpC3"], tag, bp.copy(), pts, row)
            except Exception as e:
                print(f"   -> error: {type(e).__name__}: {e}", flush=True)
                rows.append({"config": tag, "diff_ncomp": nc, "terminals": tname,
                             "spearman_fpC3": np.nan, "error": str(e)[:80]})

    tab = pd.DataFrame(rows).sort_values("spearman_fpC3", ascending=False,
                                         na_position="last").reset_index(drop=True)
    tab.to_csv(os.path.join(OUTDIR, "palantir_match_grid.csv"), index=False)
    print("\n[r5l] ranked:\n", tab.to_string(index=False), flush=True)

    if best is not None:
        _, tag, bp, (cells, p3, c3), row = best
        pd.DataFrame({"cell": bp.index, "palantir_fpC3": bp["C3"].values,
                      "palantir_fpC4": bp["C4"].values}
                     ).to_csv(os.path.join(OUTDIR, "palantir_match_best_percell.csv"),
                              index=False)
        # concordance scatter for the best
        cl = cr.loc[cells, "cluster"].values
        fig, ax = plt.subplots(figsize=(5.2, 5.4))
        for c in ["4", "0", "1", "2"]:
            s = cl == c
            ax.scatter(c3[s], p3[s], s=5, c=CLUST_COL[c], alpha=0.4, linewidths=0,
                       rasterized=True, label=CLUST_NAME[c])
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#555")
        ax.set_xlabel("CellRank/GPCCA  effector (C3) fate probability", fontsize=10)
        ax.set_ylabel("Palantir  effector (C3) fate probability", fontsize=10)
        ax.set_title(f"Best Palantir match: {tag}\n"
                     f"Spearman ρ = {row['spearman_fpC3']}  |  argmax ARI = "
                     f"{row['argmax_ARI']}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=1.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(OUTDIR, f"fig_r5_palantir_match.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[r5l DONE] best = {tag}  ->  {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
