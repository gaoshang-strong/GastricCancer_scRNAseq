# Single-cell transcriptomics of CD8⁺ T-cell heterogeneity and immunotherapy response in gastric cancer

Analysis code for the study *"Single-cell transcriptomics reveals CD8⁺ T-cell heterogeneity linked to immunotherapy response in gastric cancer."*

We profiled tumors from six immune-checkpoint-inhibitor (ICI)-treated gastric cancer patients (three responders, three non-responders) by single-cell RNA sequencing and resolved the CD8⁺ T-cell landscape associated with response, integrating trajectory inference, transcription-factor (regulon) analysis, an external Perturb-seq resource, and cell–cell communication modeling.

> **Biological unit of replication.** The cohort is six patients (n = 3 vs 3). All response-associated comparisons in the final manuscript are performed at the **patient level** (pseudobulk / per-patient aggregation), with patient-to-patient variability shown. The patient-level and robustness analyses live in [`11_revision_patient_level_and_robustness/`](11_revision_patient_level_and_robustness/).

---

## Pipeline overview

| Step | Folder | What it does | Main figures |
|------|--------|--------------|--------------|
| 1 | `01_integration_and_annotation` | scVI/scANVI integration across patients; reference-based cell-type annotation; TME atlas + composition | Fig 1, Fig S1 |
| 2 | `02_tcell_subclustering` | Extract T cells; subcluster in scVI space; annotate 7 T-cell states; QC/marker checks | Fig 2A–G, Fig S2 |
| 3 | `03_response_associated_DE` | Responder vs non-responder differential expression within effector cluster (C3) | Fig 2I–J |
| 4 | `04_module_scores` | Cell-level functional module / gene-set scoring | supporting |
| 5 | `05_cd8_trajectory` | CD8 subset; diffusion pseudotime + CellRank fate/branch inference (three iterations: initial → revisit → re-revisit/canonical) | Fig 3, Fig S4 |
| 6 | `06_myeloid` | Myeloid (macrophage/monocyte) subsetting for the communication hypothesis | supporting Fig 5 |
| 7 | `07_perturb_seq` | External genome-scale Perturb-seq resource; driver-gene prioritization for branch-specific TFs | Fig 4J–K, Fig S8 |
| 8 | `08_cellchat` | Ligand–receptor / cell–cell communication (CellChat) myeloid→T | Fig 5 |
| 9 | `09_other_cell_types` | Epithelial and myeloid compartment exploration | supporting |
| 10 | `10_tf_scenic` | SCENIC transcription-factor / regulon activity | Fig 4A–I, Fig S6, Fig S7 |
| 11 | `11_revision_patient_level_and_robustness` | Patient-level pseudobulk DE, per-patient regulon/composition, trajectory robustness (parameters, roots, leave-one-patient-out, Palantir), CellChat sensitivity | Fig 2I/J, 3F, 4F–I; Fig S3, S5, S9, S10 |

Steps 1–10 follow the exploratory analysis order; step 11 contains the revision analyses that re-cast every response comparison at the patient level and add the robustness/sensitivity evidence.
---