"""
Reviewer #2, point 1 (DE part) -- build patient-level pseudobulk count matrices
for the within-cluster Responder vs Non-Responder comparisons that the paper
reports at the cell level (Fig 2I/J; §3.3 and §3.6, cluster C3 effector).

Original: 03_01_DE.py runs sc.tl.rank_genes_groups (Wilcoxon) on individual
cells within each cluster -> pseudo-replication (each cell treated as a replicate,
n in the thousands, artificially tiny p-values).

Here we sum raw UMI counts across each patient's cells within a cluster, giving
one pseudobulk sample per patient (3 R vs 3 NR). The count matrices + coldata are
written to CSV and then analysed with DESeq2/edgeR in r1b_pseudobulk_DE.R.

We produce pseudobulk for the CD8 clusters compared in the paper:
    leiden 1 = C3 effector   (the main within-state R vs NR comparison)
    leiden 0 = C2 transitional
    leiden 2 = C4 exhausted

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
# use the filtered (pre-recluster) T-cell object that 03_01_DE.py used
H5AD = os.path.join(
    BASE, "02_extracting_T_cells_and_clustering",
    "results_drop_10-12-5-8-9__20260111_164341",
    "02_filtered_drop_10-12-5-8-9.h5ad")
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r1b_pseudobulk"))

CLUSTER_KEY = "leiden_T_0.6"
# leiden -> paper label for filenames
CLUSTERS = {"1": "C3_effector", "0": "C2_transitional", "2": "C4_exhausted"}
MIN_CELLS = 10  # minimum cells for a patient to contribute a pseudobulk sample


def main():
    print("[r1b] loading:", H5AD)
    adata = sc.read_h5ad(H5AD)
    adata.obs = U.add_respond(adata.obs)

    for leiden, label in CLUSTERS.items():
        counts, coldata = U.pseudobulk_counts(
            adata, cluster_key=CLUSTER_KEY, cluster_value=leiden,
            min_cells=MIN_CELLS)
        if counts.empty:
            print(f"[r1b] cluster {leiden} ({label}): no pseudobulk samples, skip")
            continue
        cpath = os.path.join(OUTDIR, f"pseudobulk_counts_{label}.csv")
        mpath = os.path.join(OUTDIR, f"pseudobulk_coldata_{label}.csv")
        counts.to_csv(cpath)
        coldata.to_csv(mpath)
        print(f"[r1b] {label}: {counts.shape[1]} samples "
              f"({(coldata.Respond=='Responder').sum()} R / "
              f"{(coldata.Respond=='Non-Responder').sum()} NR), "
              f"{counts.shape[0]} genes -> {os.path.basename(cpath)}")
        print(coldata.to_string())
    print("\n[r1b DONE] now run:  Rscript revise/code/r1b_pseudobulk_DE.R")


if __name__ == "__main__":
    main()
