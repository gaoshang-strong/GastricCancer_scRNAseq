# Step 11 — Patient-level and robustness analyses (revision)

Re-casts every response-associated comparison at the **patient level** (n = 3 vs 3) and adds the trajectory-robustness and communication-sensitivity evidence requested in review. `utils_patient_level.py` holds shared helpers.

## r1 — patient-level composition, DE, and regulons
- `r1a_composition_patient_level.py` — per-patient cluster & branch fractions (incl. Fig 3F CD8 branch composition).
- `r1b_build_pseudobulk.py` → `r1b_pseudobulk_DE.R` — build per-patient pseudobulk, then DESeq2 responder-vs-non-responder DE within each CD8 cluster (**Fig S3**, Fig 2I/J validation).
- `r1c_regulon_patient_level.py` — per-patient mean SCENIC regulon activity, focus regulons.
- `r1d_targeted_patient_level.py` — pre-specified effector/stress gene panel, within-panel FDR.
- `r1e_regulon9_patientlevel.py` — 9-regulon patient-level comparison, Welch t-test (**Fig S9**).
- `r1_final_figures.py` — assembles the patient-level figure panels (Fig 2I/J, 3F, 4F–I).

## r5 — trajectory robustness
- `r5_trajectory_robustness.py`, `r5d_gpcca_robustness.py` — GPCCA re-runs across a parameter grid (n_neighbors, n_components/Schur dim, n_terminal) and root-cell strategies; leave-one-patient-out.
- `r5b_palantir_crosscheck.py`, `r5l_palantir_match.py`, `r5m_palantir_recalibrate.py` — independent Palantir pseudotime/fate cross-check.
- `r5c_paga_bootstrap.py` — PAGA connectivity bootstrap.
- `r5e_plot_robustness.py`, `r5f_plot_terminal_stability.py`, `r5g_plot_invariants.py` — robustness/terminal-stability/invariant summaries.
- `r5h_concordance_scatter.py`, `r5i_concordance_panels.py`, `r5j_ncomponents_sweep.py`, `r5k_rescale_fateprob.py` — per-cell concordance (pseudotime & effector fate prob) vs the canonical run.
- `r5n_concordance_clean.py`, `r5o_pseudotime_grid.py`, `r5q_fateprob_grid.py` — final clean concordance panels/grids (**Fig S5**).

## r7 — CellChat sensitivity
- `r7_build_cellchat_input.py` — assemble CellChat input.
- `r7c_cellchat_per_patient.R` — per-patient CellChat (`population.size = FALSE`).
- `r7b_cellchat_permutation.R` — response-label permutation null.
- `r7_cellchat_sensitivity.R` — equal-size down-sampling.
- `r7d_cellchat_patient_summary.py`, `r7e_cellchat_sensitivity_fig.py` — patient-level summary + figure (**Fig S10**).
