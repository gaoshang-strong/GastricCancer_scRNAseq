#!/usr/bin/env Rscript
# Reviewer #2, point 7 (patient-level consistency) -- the most direct answer to
# "differences may be driven by individual samples": run CellChat SEPARATELY for
# each of the 6 patients and record, per patient, the myeloid -> CD8 communication
# probability for every focus ligand-receptor pair. Then (in Python) each focus
# interaction becomes 3 responder values vs 3 non-responder values -> box+dots +
# patient-level MWU, exactly like the other patient-level re-analyses.
#
# population.size = FALSE so the probability does NOT scale with per-patient cell
# abundance (controls for cellular composition, reviewer's 3rd ask). nboot = 1: we
# want the per-patient point estimate; the across-patient spread is the signal.
#
# Input from r7_build_cellchat_input.py. Run in background (~10-15 min):
#   Rscript revise/code/r7c_cellchat_per_patient.R

suppressMessages({ library(Matrix); library(CellChat) })
set.seed(1)

base   <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
indir  <- file.path(base, "revise", "results", "r7_cellchat", "input")
outdir <- file.path(base, "revise", "results", "r7_cellchat")

SENDERS   <- c("Macrophages", "Monocytes")
RECEIVERS <- c("C2", "C3", "C4")
MIN_CELLS <- 10
FOCUS_LR <- c("CXCL9_CXCR3", "CXCL16_CXCR6", "MIF_CD74",
              "ICAM1_SPN", "TNF_TNFRSF1B",
              "GDF15_TGFBR2", "SPP1_ITGAV_ITGB1", "SPP1_CD44",
              "LGALS9_HAVCR2", "ICAM1_ITGAL_ITGB2")

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
  meta$ident   <- factor(meta$ident)
  meta$samples <- factor(meta$patient)
  cc <- createCellChat(object = data, meta = meta, group.by = "ident")
  cc@DB <- CellChatDB.human
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  # population.size = FALSE -> remove the cell-abundance scaling
  cc <- computeCommunProb(cc, type = "triMean", nboot = 1, population.size = FALSE)
  cc <- filterCommunication(cc, min.cells = MIN_CELLS)
  cc
}

focus_rows <- function(cc, patient, response) {
  prob <- cc@net$prob
  if (is.null(prob)) return(data.frame())
  lrnames <- dimnames(prob)[[3]]
  src <- intersect(SENDERS, dimnames(prob)[[1]])
  tgt <- intersect(RECEIVERS, dimnames(prob)[[2]])
  rows <- list()
  for (lr in FOCUS_LR) {
    if (!(lr %in% lrnames)) next
    for (s in src) for (t in tgt) {
      rows[[length(rows) + 1]] <- data.frame(
        patient = patient, response = response, lr = lr,
        source = s, target = t, prob = prob[s, t, lr],
        stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0) return(data.frame())
  do.call(rbind, rows)
}

main <- function() {
  inp <- load_input(); data <- inp$data; meta <- inp$meta
  patients <- sort(unique(meta$patient))
  cat("[r7c] patients:", paste(patients, collapse = ", "), "\n")
  all <- list()
  for (p in patients) {
    resp <- unique(meta$response[meta$patient == p])[1]
    cat("[r7c] CellChat for", p, "(", resp, ") ...\n")
    keep <- rownames(meta)[meta$patient == p]
    r <- tryCatch({
      cc <- run_cc(data[, keep, drop = FALSE], meta[keep, , drop = FALSE])
      focus_rows(cc, p, resp)
    }, error = function(e) { cat("   error:", conditionMessage(e), "\n"); data.frame() })
    if (nrow(r) > 0) all[[p]] <- r
  }
  out <- do.call(rbind, all)
  write.csv(out, file.path(outdir, "cellchat_per_patient_focusLR.csv"),
            row.names = FALSE)
  cat("[r7c DONE] rows:", nrow(out), "->",
      file.path(outdir, "cellchat_per_patient_focusLR.csv"), "\n")
}

main()
