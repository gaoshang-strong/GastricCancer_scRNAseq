#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(zellkonverter)   # readH5AD
  library(SingleCellExperiment)
  library(Matrix)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(presto)
})

# =========================
# Paths / Config
# =========================
H5AD_PATH <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/02_filtered_drop_10-12-5-8-9.h5ad"
sce <- readH5AD(H5AD_PATH, use_hdf5 = TRUE, reader="R")
outdir <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/03_T_cell_subtype_responder_vs_nonResponder/results"

# ----------------------------
# user inputs
# ----------------------------
clusters_use <- c("0", "1", "3")
cluster_col  <- "leiden_T_0.6"
group_col    <- "Respond"
assay_use    <- "norm_expr"   # 你SCE里有：X / counts / log1p_norm / norm_expr

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

res_list <- list()

# 你想用的筛选阈值（可按需改）
padj_cut <- 0.05
auc_high <- 0.65   # responder-high
auc_low  <- 0.35   # nonresponder-high

for (cl in c("0","1","3")) {
  
  sce_cl <- sce[, as.character(colData(sce)$leiden_T_0.6) == cl]
  
  X <- as.matrix(assay(sce_cl, "log1p_norm"))
  y <- colData(sce_cl)$Respond
  
  keep <- !is.na(y)
  X <- X[, keep, drop=FALSE]
  y <- droplevels(factor(y[keep]))
  
  res <- as.data.table(presto::wilcoxauc(X, y))
  res[, cluster := cl]
  
  # 保存每个cluster全量结果
  res_list[[cl]] <- res
  fwrite(res, file.path(outdir, paste0("DE_cluster_", cl, "_Respond_presto.csv")))
  
  # --- AUC 筛 signature genes ---
  # 注意：auc 是“当前 res$group 这组更高”的方向性指标
  # 所以 responder 高/NR 高，要用 group 名称来拆
  
  # responder-high
  sig_R <- res[
    group == "Responder" & padj <= padj_cut & auc >= auc_high
  ]
  
  # non-responder-high
  sig_NR <- res[
    group != "Responder" & padj <= padj_cut & auc >= auc_high
  ]
  
  fwrite(sig_R,  file.path(outdir, paste0("SIG_cluster_", cl, "_ResponderHigh_auc", auc_high, "_padj", padj_cut, "_top", topN, ".csv")))
  fwrite(sig_NR, file.path(outdir, paste0("SIG_cluster_", cl, "_NonResponderHigh_auc", auc_high,  "_padj", padj_cut, "_top", topN, ".csv")))
}


