library(SingleCellExperiment)
library(dplyr)
library(ggplot2)

library(zellkonverter)

h5ad_path2 <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/02_adata_T_reclustered_after_drop.h5ad"

sce <- readH5AD(h5ad_path2, use_hdf5 = TRUE, reader="R")

table(sce$Respond)

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(Matrix)
  library(presto)
})

cluster_key <- "leiden_T_0.6"
assay_use <- "norm_expr"          # change to "norm_expr" if no counts

stopifnot(cluster_key %in% colnames(colData(sce)))
stopifnot("Respond" %in% colnames(colData(sce)))
stopifnot(assay_use %in% assayNames(sce))

