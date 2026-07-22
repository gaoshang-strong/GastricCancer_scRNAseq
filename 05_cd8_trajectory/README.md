# Step 5 — CD8⁺ T-cell trajectory

Diffusion pseudotime (DPT) + CellRank fate/branch inference on the CD8⁺ compartment (clusters mapped to C1 naive / C2 transitional / C3 effector / C4 exhausted). Feeds Fig 3 and Fig S4.

The analysis went through **three iterations**; the **canonical branch assignment used in the manuscript comes from the re-revisit (`trj.py`)** and is what the robustness analyses in step 11 perturb.

### Initial trajectory
- `05_01_CD8.py`, `05_01_generate_DC1_DC2.py` — CD8 subset, diffusion components, DPT.
- `05_02_terminal_Tex_Tpex.py` — terminal exhausted / progenitor-exhausted states.
- `05_03_subset_CD8_T_cells.py` — CD8 subsetting.
- `05_04_DE_in_effector_Resp_vs_nonResp.py`, `05_05_top50.py` — effector R-vs-NR DE (cell-level; superseded by patient-level pseudobulk in step 11).

### Revisit
- `05b_01_subset_CD8_Tcell_trajectory_revisit.py` — CD8 subset (clusters 0/1/2/4) → input `.h5ad` for downstream.
- `05b_01_harmony_CD8_cells.py`, `05b_02.py` — alternative integration checks.
- `05b_02_palatinr.py` — Palantir pseudotime.
- `go_enrich_signature_clusters_0_1_2_4.py`, `ckeck_cluster2.py`, `barplot.py` — signature/GO and cluster-2 checks.

### Re-revisit (canonical)
- `trj.py` — **canonical CellRank2 run**: transition kernel, terminal states, and the `branch_rel` (branch C cytotoxic / branch E exhaustion) assignment used throughout the paper.
- `analysis1.py`, `analysis2.py`, `analysis3.py`, `analysis.ipynb` — downstream branch/score analyses and figure panels.

> Cell-level branch-composition comparisons here are descriptive; the patient-level statistic (Fig 3F) and the parameter/root/leave-one-patient-out/Palantir robustness are in step 11 (`r5_*`).
