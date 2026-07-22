"""
Reviewer #2, point 7 -- build a reproducible CellChat input from the AnnData
objects (the original filtered_cells_cellchat.R relied on an interactively-built
`seu` that is not serialised anywhere).

Senders  : myeloid cells (Macrophages, Monocytes) from the all-cells atlas.
Receivers: CD8 effector/transitional/exhausted states C2/C3/C4
           (= leiden_T_0.6 0/1/2) from the reclustered T-cell object.

We export, for exactly these cells:
    - data matrix (log1p-normalised expression), genes x cells, MatrixMarket .mtx
    - genes.tsv, barcodes.tsv
    - metadata.csv  (cell, ident, patient, Respond, response)

so the R sensitivity script needs no h5ad reader.

Env: /home/sgao30/micromamba/envs/scanpy_micromamba/bin/python
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils_patient_level as U

BASE = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
ALL_H5AD = os.path.join(BASE, "01_mapping_raw_scRNA_seq_to_reference",
                        "adata_scvi_integrated_all_cells.h5ad")
TCELL_H5AD = os.path.join(
    BASE, "02_extracting_T_cells_and_clustering",
    "results_drop_10-12-5-8-9__20260111_164341",
    "02_adata_T_reclustered_after_drop.h5ad")
OUTDIR = U.ensure_dir(os.path.join(BASE, "revise", "results", "r7_cellchat", "input"))

MYELOID = ["Macrophages", "Monocytes"]
CD8_MAP = {"0": "C2", "1": "C3", "2": "C4"}  # leiden_T_0.6 -> paper label
LAYER = "log1p_norm"


def main():
    print("[r7build] CD8 subcluster labels from T-cell object ...")
    tc = sc.read_h5ad(TCELL_H5AD, backed="r")
    tobs = tc.obs
    cd8_label = (tobs["leiden_T_0.6"].astype(str).map(CD8_MAP))
    cd8_cells = cd8_label.dropna()
    tc.file.close()
    print(f"[r7build] CD8 C2/C3/C4 cells: {len(cd8_cells)}")

    print("[r7build] loading all-cells atlas (backed) ...")
    allc = sc.read_h5ad(ALL_H5AD, backed="r")
    obs = allc.obs

    myeloid_mask = obs["majority_voting"].astype(str).isin(MYELOID)
    myeloid_cells = obs.index[myeloid_mask]
    print(f"[r7build] myeloid cells: {len(myeloid_cells)}")

    # ident per cell
    ident = pd.Series(index=allc.obs_names, dtype=object)
    ident.loc[myeloid_cells] = obs.loc[myeloid_cells, "majority_voting"].astype(str)
    ident.loc[cd8_cells.index] = cd8_cells.values
    keep = ident.dropna().index
    # order: keep as in atlas
    keep = [c for c in allc.obs_names if c in set(keep)]
    print(f"[r7build] total cells exported: {len(keep)}")

    # slice expression (load only kept rows into memory)
    idx = allc.obs_names.get_indexer(keep)
    sub = allc[idx]
    X = sub.layers[LAYER]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    # CellChat wants genes x cells
    Xt = X.T.tocsr()
    sio.mmwrite(os.path.join(OUTDIR, "data.mtx"), Xt)
    pd.Series(allc.var_names).to_csv(os.path.join(OUTDIR, "genes.tsv"),
                                     index=False, header=False)
    pd.Series(keep).to_csv(os.path.join(OUTDIR, "barcodes.tsv"),
                           index=False, header=False)

    meta = pd.DataFrame(index=keep)
    meta["ident"] = ident.loc[keep].values
    meta["patient"] = obs.loc[keep, "patient"].astype(str).values
    meta = U.add_respond(meta)
    meta["Respond"] = meta["Respond"].astype(str)
    meta["response"] = meta["Respond"].map({"Responder": "R",
                                            "Non-Responder": "NR"})
    meta.index.name = "cell"
    meta.to_csv(os.path.join(OUTDIR, "metadata.csv"))
    allc.file.close()

    print("[r7build] ident x response:")
    print(pd.crosstab(meta["ident"], meta["response"]))
    print("[r7build] ident x patient:")
    print(pd.crosstab(meta["ident"], meta["patient"]))
    print("\n[r7build DONE] input in:", OUTDIR)


if __name__ == "__main__":
    main()
