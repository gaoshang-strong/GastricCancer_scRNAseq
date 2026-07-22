"""
Reviewer #2, point 5 -- display-only rescaling of the effector-C3 fate-probability
concordance figures so the CANONICAL distribution has median = 0.5 (the raw median
is ~0.15, so the points bunch in the lower-left corner).

Transform: p -> p**gamma with gamma = ln(0.5)/ln(median_canonical). This is a
monotonic rescale that (a) fixes the endpoints 0->0 and 1->1, keeping values in
[0,1]; (b) is applied IDENTICALLY to the canonical (x) and perturbed (y) axes, so
the y = x diagonal ("cell unchanged") and the Spearman rho are exactly preserved
-- only the spacing changes. It is a viewing transform, not a change to the data.

Regenerates, into a separate rescaled_median0.5/ folder (originals untouched):
    * fate-prob concordance grid
    * per-config standalone panels
    * n_components sweep overview + panels

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U
import r5h_concordance_scatter as R5H
import r5i_concordance_panels as R5I

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTDIR = R5H.OUTDIR
RESC_DIR = U.ensure_dir(os.path.join(OUTDIR, "rescaled_median0.5"))
LABEL = "effector (C3) fate probability  (rescaled, median = 0.5)"


def make_transform():
    w = pd.read_csv(os.path.join(OUTDIR, "concordance_percell.csv"), index_col=0)
    m = float(np.nanmedian(w["fpC3_canonical"].values))
    gamma = np.log(0.5) / np.log(m)
    T = lambda p: np.clip(np.asarray(p, float), 0.0, 1.0) ** gamma
    print(f"[r5k] canonical median = {m:.4f} -> gamma = {gamma:.4f}")
    return T, gamma, m


def load_rescaled(csv, T):
    """Return (base, per_by_tag) with fate prob transformed on both axes."""
    w = pd.read_csv(csv, index_col=0)
    base = w[["cluster"]].copy()
    base["cluster"] = base["cluster"].astype(str)
    base["dpt_canonical"] = w.get("dpt_canonical", np.nan)
    base["fpC3_canonical"] = T(w["fpC3_canonical"].values)
    return base, w


def main():
    T, gamma, m = make_transform()
    note = (R5H.CANON_NOTE + f"   |   fate-prob axes rescaled: p -> p^{gamma:.3f} "
            f"(canonical median {m:.3f} -> 0.5), identical on both axes; "
            f"Spearman rho unchanged")
    orig_note = R5H.CANON_NOTE
    R5H.CANON_NOTE = note                      # picked up by scatter_grid & make_panel

    # -------- main fate-prob concordance grid + per-config panels --------
    base, w = load_rescaled(os.path.join(OUTDIR, "concordance_percell.csv"), T)
    per = {}
    for axis, tag, _ in R5H.CONFIGS:
        col = f"fpC3::{tag}"
        if col not in w.columns:
            continue
        per[tag] = (axis, pd.DataFrame({"fpC3": T(w[col].values)}, index=w.index))

    U.ensure_dir(os.path.join(RESC_DIR, "panels"))
    R5H.scatter_grid(
        base, per, "fpC3",
        "rescaled_median0.5/fig_r5_concordance_fateprob_medscaled",
        "Per-cell effector (C3) fate probability is unchanged across settings",
        LABEL)
    for i, (axis, tag, _) in enumerate(R5H.CONFIGS, 1):
        if tag not in per:
            continue
        path = os.path.join(RESC_DIR, "panels",
                            f"{i:02d}_{R5I.slug(axis)}_{R5I.slug(tag)}.png")
        R5I.make_panel(base, per[tag][1], "fpC3", axis, tag, LABEL, path)
    print(f"[r5k] fate-prob grid + {len(per)} panels -> {RESC_DIR}")

    # -------- n_components sweep (fate prob) --------
    ncsv = os.path.join(OUTDIR, "ncomponents_percell.csv")
    if os.path.exists(ncsv):
        wn = pd.read_csv(ncsv, index_col=0)
        nbase = wn[["cluster"]].copy(); nbase["cluster"] = nbase["cluster"].astype(str)
        nbase["fpC3_canonical"] = T(wn["fpC3_canonical"].values)
        ncols = [c for c in wn.columns if c.startswith("fpC3::nc")]
        nper = {int(c.split("nc")[1]): pd.DataFrame({"fpC3": T(wn[c].values)},
                                                    index=wn.index) for c in ncols}
        fig, axes = plt.subplots(1, len(nper), figsize=(2.7 * len(nper), 3.2),
                                 squeeze=False)
        for ax, nc in zip(axes[0], sorted(nper)):
            R5H._panel(ax, nbase, nper[nc], "fpC3")
            ax.set_title(f"n_components = {nc}", fontsize=9.2)
        fig.supxlabel(f"canonical  {LABEL}", fontsize=9.5, y=0.03)
        fig.supylabel(f"perturbed  {LABEL}", fontsize=9.5)
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=R5H.CLUST_COL[c], markersize=8,
                              label=R5H.CLUST_NAME[c]) for c in ["4", "0", "1", "2"]]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 1.0))
        fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=7,
                 color="#777777")
        fig.suptitle("GPCCA Schur dimension (n_components) sweep -- effector (C3) "
                     "fate probability is invariant", fontsize=12.5, y=1.06)
        fig.tight_layout(rect=[0.01, 0.06, 1, 0.94])
        subdir = U.ensure_dir(os.path.join(RESC_DIR, "ncomponents_panels"))
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(RESC_DIR,
                        f"fig_r5_ncomponents_sweep_medscaled.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        for nc in sorted(nper):
            R5I.make_panel(nbase, nper[nc], "fpC3", "Parameter",
                           f"n_components = {nc}", LABEL,
                           os.path.join(subdir, f"ncomponents_{nc:02d}.png"))
        print(f"[r5k] n_components sweep ({len(nper)} panels) -> {RESC_DIR}")

    R5H.CANON_NOTE = orig_note
    print(f"\n[r5k DONE] rescaled (median=0.5) fate-prob figures -> {RESC_DIR}")


if __name__ == "__main__":
    main()
