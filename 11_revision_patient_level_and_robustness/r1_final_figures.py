"""
Reviewer #2, point 1 -- FINAL patient-level figures for the response-associated
comparisons #3 (Fig 3F branch composition), #4 (Fig 2I/J within-cluster DE),
#5 (Fig 4F-I regulon activity).

Style matches the original paper's box+dot panels (Fig 4F-I): light-gray box, orange
median, jittered group-coloured points, Non-Responder (blue) left / Responder (orange)
right, top/right spines off, Liberation Sans. The ONE honest change vs the paper: every
dot is now a PATIENT (n=3 vs 3), not a cell -- so the displayed spread IS the
patient-to-patient variability the reviewer asked for, and the annotated p is the
patient-level statistic (not the pseudoreplicated cell-level one).

Outputs -> revise/results/r1_final/  (composites + panels/, PNG @300dpi + PDF)

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

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
RES = os.path.join(BASE, "revise", "results")
OUT = U.ensure_dir(os.path.join(RES, "r1_final"))
PAN = U.ensure_dir(os.path.join(OUT, "panels"))

# paper palette: Non-Responder blue, Responder orange (matplotlib tab)
COL = {"Non-Responder": "#1f77b4", "Responder": "#ff7f0e"}
ORDER = ["Non-Responder", "Responder"]
BOX = dict(facecolor="#dcdcdc", edgecolor="black", linewidth=1.3)
MED = dict(color="#ff7f0e", linewidth=2.2)
WHIS = dict(color="black", linewidth=1.3)
CAP = dict(color="black", linewidth=1.3)
rng = np.random.default_rng(0)


def dotbox(ax, dNR, dR, ylabel, annot=None, title=None):
    data = [np.asarray(dNR, float), np.asarray(dR, float)]
    ax.boxplot(data, positions=[1, 2], widths=0.58, patch_artist=True,
               showfliers=False, boxprops=BOX, medianprops=MED,
               whiskerprops=WHIS, capprops=CAP, zorder=1)
    for j, (d, g) in enumerate(zip(data, ORDER), start=1):
        jit = (rng.random(len(d)) - 0.5) * 0.14
        ax.scatter(np.full(len(d), j) + jit, d, s=72, c=COL[g], edgecolors="black",
                   linewidths=0.7, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Non-Responder\n(n={len(dNR)})", f"Responder\n(n={len(dR)})"],
                       fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    if annot:
        ax.text(0.5, 1.005, annot, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="#333")
    ax.tick_params(labelsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.margins(x=0.22)


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_panel(plot_fn, name, figsize=(2.9, 3.9)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(PAN, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #5 regulon
def fig_regulon():
    df = pd.read_csv(os.path.join(RES, "r1c_regulon", "per_patient_mean_AUC_C3.csv"))
    tests = pd.read_csv(os.path.join(RES, "r1c_regulon",
                                     "regulon_C3_patient_level_tests.csv")
                        ).set_index("regulon")
    regs = ["RELB", "FOXO3", "STAT1", "EOMES"]     # paper Fig 4 F,G,H,I

    def one(ax, tf):
        col = f"{tf}(+)"
        nr = df.loc[df.Respond == "Non-Responder", col].values
        r = df.loc[df.Respond == "Responder", col].values
        p = float(tests.loc[col, "p_mwu"])
        dotbox(ax, nr, r, f"{tf} activity", annot=f"MWU p = {p:.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 4.0))
    for ax, tf in zip(axes, regs):
        one(ax, tf)
    fig.suptitle("Regulon activity per patient (C3 effector) — Responder vs "
                 "Non-Responder", fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "Fig4FI_regulon_patientlevel")
    for tf in regs:
        save_panel(lambda ax, tf=tf: one(ax, tf), f"regulon_{tf}")
    print("[#5] regulon figure done:", regs)


# ------------------------------------------------------------------ #4 pseudobulk DE
def fig_pseudobulk_DE():
    counts = pd.read_csv(os.path.join(RES, "r1b_pseudobulk",
                                      "pseudobulk_counts_C3_effector.csv"), index_col=0)
    cold = pd.read_csv(os.path.join(RES, "r1b_pseudobulk",
                                    "pseudobulk_coldata_C3_effector.csv")).set_index("patient")
    panel = pd.read_csv(os.path.join(RES, "r1d_targeted", "targeted_C3_panel_DE.csv")
                        ).set_index("gene")
    libsize = counts.sum(axis=0)
    logcpm = np.log2(counts.divide(libsize, axis=1) * 1e6 + 1)   # genes x patients

    resp = cold["Respond"]
    NRp = [p for p in counts.columns if resp[p] == "Non-Responder"]
    Rp = [p for p in counts.columns if resp[p] == "Responder"]
    # 2I = NR-up (stress/quiescence), 2J = R-up (cytotoxic/effector)
    genes = ["HSPA1B", "HSPA1A", "KLF2", "NKG7", "GZMH", "CTSW"]

    def one(ax, g):
        nr = logcpm.loc[g, NRp].values
        r = logcpm.loc[g, Rp].values
        lfc = panel.loc[g, "log2FC_R_vs_NR"]
        fdr = panel.loc[g, "FDR_within_panel_twosided"]
        star = " *" if fdr < 0.05 else ""
        dotbox(ax, nr, r, f"{g}\nlog2 CPM (pseudobulk)",
               annot=f"log2FC$_{{R/NR}}$={lfc:+.2f}  FDR={fdr:.3f}{star}")

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 8.0))
    for ax, g in zip(axes.ravel(), genes):
        one(ax, g)
    fig.suptitle("Within-C3 pseudobulk DE per patient (pre-specified panel; "
                 "within-panel FDR)", fontsize=12.5, y=1.005)
    fig.tight_layout()
    save(fig, "Fig2IJ_pseudobulk_DE_patientlevel")
    for g in genes:
        save_panel(lambda ax, g=g: one(ax, g), f"pseudobulkDE_{g}")
    print("[#4] pseudobulk DE figure done:", genes)


# ------------------------------------------------------------------ #3 branch composition
def fig_branch():
    frac = pd.read_csv(os.path.join(RES, "r1a_composition",
                                    "fig3F_CD8branch_per_patient_fractions.csv"))
    from scipy.stats import mannwhitneyu

    def get(branch):
        s = frac[frac.branch_rel == branch]
        nr = s.loc[s.Respond == "Non-Responder", "frac"].values
        r = s.loc[s.Respond == "Responder", "frac"].values
        return nr, r

    def one(ax, branch, label, alt):
        nr, r = get(branch)
        p2 = mannwhitneyu(r, nr, alternative="two-sided").pvalue
        p1 = mannwhitneyu(r, nr, alternative=alt).pvalue
        dotbox(ax, nr, r, f"{label}\nbranch fraction",
               annot=f"MWU p = {p2:.2f} (2-sided)\n{p1:.2f} (1-sided)")

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 4.4))
    one(axes[0], "branch_C3", "Cytotoxicity (C) / effector", "greater")  # R>NR
    one(axes[1], "branch_C4", "Exhaustion (E)", "less")                  # R<NR
    fig.suptitle("CD8 branch composition per patient (Fig 3F)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "Fig3F_branch_composition_patientlevel")
    save_panel(lambda ax: one(ax, "branch_C3", "Cytotoxicity (C) / effector", "greater"),
               "branch_cytotoxicity_C", figsize=(3.2, 4.2))
    save_panel(lambda ax: one(ax, "branch_C4", "Exhaustion (E)", "less"),
               "branch_exhaustion_E", figsize=(3.2, 4.2))
    print("[#3] branch composition figure done")


def fig_effector_cluster():
    """Cluster-level effector (leiden 1) fraction per patient -- the cleaner #3
    evidence (all 3 R > all 3 NR; one-sided p = 0.05)."""
    frac = pd.read_csv(os.path.join(RES, "r1a_composition",
                                    "fig2H_Tcluster_per_patient_fractions.csv"))
    from scipy.stats import mannwhitneyu
    e = frac[frac["leiden_T_0.6"] == 1]
    nr = e.loc[e.Respond == "Non-Responder", "frac"].values
    r = e.loc[e.Respond == "Responder", "frac"].values
    p2 = mannwhitneyu(r, nr, alternative="two-sided").pvalue
    p1 = mannwhitneyu(r, nr, alternative="greater").pvalue

    def one(ax):
        dotbox(ax, nr, r, "Effector cluster (C3)\nfraction of CD8 T cells",
               annot=f"MWU p = {p2:.2f} (2-sided)\n{p1:.2f} (1-sided)")

    save_panel(one, "effector_cluster_C3", figsize=(3.2, 4.2))
    print(f"[#3-alt] effector cluster panel: 2-sided p={p2:.2f}, 1-sided p={p1:.2f}, "
          f"all 3 R > all 3 NR = {r.min() > nr.max()}")


def main():
    fig_regulon()
    fig_pseudobulk_DE()
    fig_branch()
    fig_effector_cluster()
    print("\n[r1_final DONE] ->", OUT)


if __name__ == "__main__":
    main()
