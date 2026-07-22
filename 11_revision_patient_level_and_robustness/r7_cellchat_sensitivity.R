#!/usr/bin/env Rscript
# Reviewer #2, point 7 -- CellChat robustness to patient composition / individual
# samples. The original filtered_cells_cellchat.R pools 3 responders into one
# object and 3 non-responders into another, then compares communication strength.
# With n=3 per group, a difference could be driven by a single patient, and
# CellChat is sensitive to per-group cell abundance.
#
# This script re-runs the myeloid -> CD8 (C2/C3/C4) comparison and adds:
#   (1) permutation-based p-values (CellChat computeCommunProb, nboot)
#   (2) leave-one-patient-out: does each focus LR pair keep its R-vs-NR direction
#       when any single patient is removed?
#   (3) downsampling each cell group to an equal number of cells, repeated, to
#       show differences are not an artefact of unequal abundance.
#
# Input built by r7_build_cellchat_input.py (data.mtx + metadata.csv).
# Run: Rscript revise/code/r7_cellchat_sensitivity.R          # baseline + LOPO + downsample
#      Rscript revise/code/r7_cellchat_sensitivity.R baseline  # baseline only

suppressMessages({
  library(Matrix)
  library(CellChat)
})
set.seed(1)

base   <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
indir  <- file.path(base, "revise", "results", "r7_cellchat", "input")
outdir <- file.path(base, "revise", "results", "r7_cellchat")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

SENDERS   <- c("Macrophages", "Monocytes")
RECEIVERS <- c("C2", "C3", "C4")
NBOOT     <- 100
MIN_CELLS <- 10

# focus ligand-receptor pairs from Fig 5 (CellChatDB interaction_name uses '_')
FOCUS_LR <- c("CXCL9_CXCR3", "CXCL16_CXCR6", "MIF_CD74",
              "ICAM1_SPN", "TNF_TNFRSF1B",
              "GDF15_TGFBR2", "SPP1_ITGAV_ITGB1", "SPP1_CD44",
              "LGALS9_HAVCR2", "ICAM1_ITGAL_ITGB2")

# ---------------------------------------------------------------------------
load_input <- function() {
  data <- as(Matrix::readMM(file.path(indir, "data.mtx")), "CsparseMatrix")
  genes <- readLines(file.path(indir, "genes.tsv"))
  cells <- readLines(file.path(indir, "barcodes.tsv"))
  rownames(data) <- genes; colnames(data) <- cells
  meta <- read.csv(file.path(indir, "metadata.csv"), row.names = 1,
                   stringsAsFactors = FALSE)
  meta <- meta[cells, , drop = FALSE]
  meta$samples <- factor(meta$patient)   # CellChat expects a 'samples' column
  list(data = data, meta = meta)
}

run_cellchat <- function(data, meta) {
  meta$ident <- factor(meta$ident)
  cc <- createCellChat(object = data, meta = meta, group.by = "ident")
  cc@DB <- CellChatDB.human
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  cc <- computeCommunProb(cc, type = "triMean", nboot = NBOOT)
  cc <- filterCommunication(cc, min.cells = MIN_CELLS)
  cc
}

# pull prob (and permutation pval) for focus LR pairs, myeloid -> receivers
extract_focus <- function(cc, tag) {
  prob <- cc@net$prob            # [source, target, LR]
  pval <- cc@net$pval
  lrnames <- dimnames(prob)[[3]]
  src <- intersect(SENDERS, dimnames(prob)[[1]])
  tgt <- intersect(RECEIVERS, dimnames(prob)[[2]])
  rows <- list()
  for (lr in FOCUS_LR) {
    if (!(lr %in% lrnames)) next
    for (s in src) for (t in tgt) {
      rows[[length(rows) + 1]] <- data.frame(
        tag = tag, lr = lr, source = s, target = t,
        prob = prob[s, t, lr], pval = pval[s, t, lr],
        stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0) return(data.frame())
  do.call(rbind, rows)
}

# R vs NR delta on log2 prob for a given data/meta pair
compare_R_NR <- function(data, meta, tag) {
  eps <- 1e-10
  mR  <- meta[meta$response == "R",  , drop = FALSE]
  mNR <- meta[meta$response == "NR", , drop = FALSE]
  ccR  <- run_cellchat(data[, rownames(mR),  drop = FALSE], mR)
  ccNR <- run_cellchat(data[, rownames(mNR), drop = FALSE], mNR)
  fR  <- extract_focus(ccR,  "R")
  fNR <- extract_focus(ccNR, "NR")
  if (nrow(fR) == 0 && nrow(fNR) == 0) return(data.frame())
  m <- merge(fR[, c("lr", "source", "target", "prob", "pval")],
             fNR[, c("lr", "source", "target", "prob", "pval")],
             by = c("lr", "source", "target"), all = TRUE,
             suffixes = c("_R", "_NR"))
  m$prob_R[is.na(m$prob_R)]   <- 0
  m$prob_NR[is.na(m$prob_NR)] <- 0
  m$delta_log2_R_over_NR <- log2((m$prob_R + eps) / (m$prob_NR + eps))
  m$direction <- ifelse(m$prob_R > m$prob_NR, "up_in_R", "up_in_NR")
  m$tag <- tag
  m
}

# ---------------------------------------------------------------------------
main <- function(mode = "full") {
  inp <- load_input()
  data <- inp$data; meta <- inp$meta
  cat("[r7] cells:", ncol(data), " genes:", nrow(data), "\n")

  # (1) baseline R vs NR with permutation p-values
  cat("[r7] baseline R vs NR ...\n")
  base_res <- compare_R_NR(data, meta, "baseline")
  write.csv(base_res, file.path(outdir, "cellchat_baseline_focusLR.csv"),
            row.names = FALSE)
  cat("[r7] baseline focus LR (myeloid -> CD8):\n")
  print(base_res[order(base_res$lr), c("lr", "source", "target",
        "prob_R", "prob_NR", "delta_log2_R_over_NR", "direction")])

  if (mode == "baseline") { cat("[r7] baseline-only done.\n"); return(invisible()) }

  # (2) leave-one-patient-out consistency
  cat("\n[r7] leave-one-patient-out ...\n")
  patients <- sort(unique(meta$patient))
  lopo <- list()
  for (p in patients) {
    cat("  drop", p, "\n")
    keep <- rownames(meta)[meta$patient != p]
    r <- tryCatch(compare_R_NR(data[, keep, drop = FALSE],
                               meta[keep, , drop = FALSE], paste0("drop_", p)),
                  error = function(e) { cat("   error:", conditionMessage(e), "\n");
                                        data.frame() })
    if (nrow(r) > 0) { r$dropped <- p; lopo[[p]] <- r }
  }
  if (length(lopo) > 0) {
    lopo_df <- do.call(rbind, lopo)
    write.csv(lopo_df, file.path(outdir, "cellchat_LOPO_focusLR.csv"),
              row.names = FALSE)
    # per LR/source/target: fraction of LOPO runs agreeing with baseline direction
    key <- paste(base_res$lr, base_res$source, base_res$target)
    base_dir <- setNames(base_res$direction, key)
    lopo_df$key <- paste(lopo_df$lr, lopo_df$source, lopo_df$target)
    lopo_df$agree <- lopo_df$direction == base_dir[lopo_df$key]
    agg <- aggregate(agree ~ key, data = lopo_df, FUN = function(x) mean(x))
    write.csv(agg, file.path(outdir, "cellchat_LOPO_direction_agreement.csv"),
              row.names = FALSE)
    cat("[r7] LOPO direction-agreement fraction (1.0 = robust to dropping any patient):\n")
    print(agg[order(agg$agree), ])
  }

  # (3) downsampling each cell group to an equal size, repeated
  cat("\n[r7] downsampling robustness ...\n")
  grp <- interaction(meta$ident, meta$response, drop = TRUE)
  min_n <- min(table(grp))
  min_n <- max(min_n, 20)
  cat("  downsample each ident x response group to", min_n, "cells,", 3, "reps\n")
  ds <- list()
  for (rep in 1:3) {
    idx <- unlist(lapply(split(rownames(meta), grp), function(cells) {
      if (length(cells) <= min_n) cells else sample(cells, min_n)
    }))
    r <- tryCatch(compare_R_NR(data[, idx, drop = FALSE],
                               meta[idx, , drop = FALSE], paste0("ds_rep", rep)),
                  error = function(e) data.frame())
    if (nrow(r) > 0) { r$rep <- rep; ds[[rep]] <- r }
  }
  if (length(ds) > 0) {
    ds_df <- do.call(rbind, ds)
    write.csv(ds_df, file.path(outdir, "cellchat_downsample_focusLR.csv"),
              row.names = FALSE)
    cat("[r7] downsampling written.\n")
  }

  cat("\n[r7 DONE] outputs in:", outdir, "\n")
}

args <- commandArgs(trailingOnly = TRUE)
main(if (length(args) >= 1) args[1] else "full")
