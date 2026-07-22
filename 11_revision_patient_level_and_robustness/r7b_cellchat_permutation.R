#!/usr/bin/env Rscript
# Reviewer #2, point 7 (permutation testing) -- exact patient-label permutation
# test for the myeloid -> CD8 communication difference.
#
# The observed R-vs-NR delta for each focus LR pair is compared to a null built by
# reassigning the 3-vs-3 response labels across the 6 PATIENTS in every possible
# way (choose(6,3) = 20 assignments; 10 distinct partitions up to label swap).
# This keeps the patient as the unit and asks whether the observed communication
# difference is larger than expected when response is shuffled at the patient level.
# Empirical two-sided p per pair = mean( |delta_perm| >= |delta_obs| ) over the
# null (includes the observed assignment), so the floor at n=3v3 is ~0.1.
#
# nboot=1 for CellChat here: the permutation over patients provides the null; we
# only need CellChat's point-estimate communication probability per run.
#
# Input from r7_build_cellchat_input.py. Run in background (~15-20 min):
#   Rscript revise/code/r7b_cellchat_permutation.R

suppressMessages({ library(Matrix); library(CellChat) })
set.seed(1)

base   <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
indir  <- file.path(base, "revise", "results", "r7_cellchat", "input")
outdir <- file.path(base, "revise", "results", "r7_cellchat")

SENDERS   <- c("Macrophages", "Monocytes")
RECEIVERS <- c("C2", "C3", "C4")
MIN_CELLS <- 10
EPS <- 1e-10
FOCUS_LR <- c("CXCL9_CXCR3", "CXCL16_CXCR6", "MIF_CD74",
              "ICAM1_SPN", "TNF_TNFRSF1B",
              "GDF15_TGFBR2", "SPP1_ITGAV_ITGB1", "SPP1_CD44",
              "LGALS9_HAVCR2", "ICAM1_ITGAL_ITGB2")
TRUE_R <- c("PHD003", "PHD004", "PHD009")   # observed responders

load_input <- function() {
  data <- as(Matrix::readMM(file.path(indir, "data.mtx")), "CsparseMatrix")
  rownames(data) <- readLines(file.path(indir, "genes.tsv"))
  colnames(data) <- readLines(file.path(indir, "barcodes.tsv"))
  meta <- read.csv(file.path(indir, "metadata.csv"), row.names = 1,
                   stringsAsFactors = FALSE)
  meta <- meta[colnames(data), , drop = FALSE]
  list(data = data, meta = meta)
}

run_cc <- function(data, meta) {
  meta$ident <- factor(meta$ident)
  meta$samples <- factor(meta$patient)
  cc <- createCellChat(object = data, meta = meta, group.by = "ident")
  cc@DB <- CellChatDB.human
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  cc <- computeCommunProb(cc, type = "triMean", nboot = 1)
  cc <- filterCommunication(cc, min.cells = MIN_CELLS)
  cc
}

focus_prob <- function(cc) {
  prob <- cc@net$prob
  lr <- dimnames(prob)[[3]]
  src <- intersect(SENDERS, dimnames(prob)[[1]])
  tgt <- intersect(RECEIVERS, dimnames(prob)[[2]])
  out <- list()
  for (l in intersect(FOCUS_LR, lr)) for (s in src) for (t in tgt)
    out[[paste(l, s, t)]] <- prob[s, t, l]
  out
}

# delta_log2(R over NR) for one patient->label assignment
delta_for <- function(data, meta, R_patients) {
  meta$grp <- ifelse(meta$patient %in% R_patients, "R", "NR")
  cR  <- rownames(meta)[meta$grp == "R"]
  cNR <- rownames(meta)[meta$grp == "NR"]
  pR  <- tryCatch(focus_prob(run_cc(data[, cR,  drop = FALSE], meta[cR, ])),
                  error = function(e) list())
  pNR <- tryCatch(focus_prob(run_cc(data[, cNR, drop = FALSE], meta[cNR, ])),
                  error = function(e) list())
  keys <- union(names(pR), names(pNR))
  sapply(keys, function(k) {
    a <- if (!is.null(pR[[k]]))  pR[[k]]  else 0
    b <- if (!is.null(pNR[[k]])) pNR[[k]] else 0
    log2((a + EPS) / (b + EPS))
  })
}

main <- function() {
  inp <- load_input(); data <- inp$data; meta <- inp$meta
  patients <- sort(unique(meta$patient))
  stopifnot(length(patients) == 6)

  # all choose(6,3) assignments of "R"
  combs <- combn(patients, 3, simplify = FALSE)
  cat("[r7b] permutations:", length(combs), "(patient-level 3v3 label shuffles)\n")

  # observed
  obs <- delta_for(data, meta, TRUE_R)

  # null over all assignments (includes observed)
  null_mat <- list()
  for (i in seq_along(combs)) {
    cat(sprintf("  perm %d/%d: R = {%s}\n", i, length(combs),
                paste(combs[[i]], collapse = ",")))
    null_mat[[i]] <- delta_for(data, meta, combs[[i]])
  }
  keys <- Reduce(union, lapply(null_mat, names))
  keys <- union(keys, names(obs))

  # empirical two-sided p per focus pair
  rows <- lapply(keys, function(k) {
    d_obs <- if (!is.null(obs[[k]])) obs[[k]] else NA
    d_null <- sapply(null_mat, function(v) if (!is.null(v[[k]])) v[[k]] else 0)
    p <- if (is.na(d_obs)) NA else mean(abs(d_null) >= abs(d_obs))
    data.frame(pair = k, delta_obs = d_obs,
               p_perm = p, n_perm = length(d_null),
               stringsAsFactors = FALSE)
  })
  res <- do.call(rbind, rows)
  res <- res[order(res$p_perm, -abs(res$delta_obs)), ]
  write.csv(res, file.path(outdir, "cellchat_patient_permutation_pvals.csv"),
            row.names = FALSE)
  cat("\n[r7b] focus-pair patient-level permutation p-values:\n")
  print(res, row.names = FALSE)
  cat("\n[r7b DONE] ->", file.path(outdir, "cellchat_patient_permutation_pvals.csv"), "\n")
}

main()
