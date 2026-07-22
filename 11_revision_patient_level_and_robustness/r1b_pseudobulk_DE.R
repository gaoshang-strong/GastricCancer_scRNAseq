#!/usr/bin/env Rscript
# Reviewer #2, point 1 (DE part) -- patient-level pseudobulk DE, Responder vs
# Non-Responder, within each CD8 cluster. Replaces the cell-level Wilcoxon in
# 03_01_DE.py. Input CSVs are produced by r1b_build_pseudobulk.py.
#
# Uses DESeq2 (n=3 vs 3 per cluster; low power is expected and reported honestly).
# For each cluster we also print the paper's highlighted genes so the rebuttal can
# state whether they survive at the patient level.
#
# Run with system R (has DESeq2): Rscript revise/code/r1b_pseudobulk_DE.R

suppressMessages({
  library(DESeq2)
})

base <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified"
indir <- file.path(base, "revise", "results", "r1b_pseudobulk")
outdir <- indir

clusters <- c("C3_effector", "C2_transitional", "C4_exhausted")
# genes the paper highlights for the within-C3 R vs NR contrast (Fig 2I/J, 3.3/3.6)
highlight <- list(
  responder_up    = c("NKG7", "CTSW", "GZMH"),
  nonresponder_up = c("HSPA1A", "HSPA1B", "KLF2")
)

run_one <- function(cl) {
  cfile <- file.path(indir, sprintf("pseudobulk_counts_%s.csv", cl))
  mfile <- file.path(indir, sprintf("pseudobulk_coldata_%s.csv", cl))
  if (!file.exists(cfile)) { cat("[skip]", cl, "no counts file\n"); return(invisible()) }

  counts <- read.csv(cfile, row.names = 1, check.names = FALSE)
  coldata <- read.csv(mfile, row.names = 1, check.names = FALSE)
  coldata <- coldata[colnames(counts), , drop = FALSE]
  coldata$Respond <- factor(coldata$Respond,
                            levels = c("Non-Responder", "Responder"))

  nR  <- sum(coldata$Respond == "Responder")
  nNR <- sum(coldata$Respond == "Non-Responder")
  cat(sprintf("\n===== %s : %d Responder vs %d Non-Responder pseudobulk samples =====\n",
              cl, nR, nNR))
  if (nR < 2 || nNR < 2) {
    cat("  too few replicates for DESeq2 (need >=2 per group); skipping test.\n")
    return(invisible())
  }

  # keep genes with a minimal signal across samples
  keep <- rowSums(counts >= 5) >= 2
  counts <- counts[keep, , drop = FALSE]
  cat(sprintf("  genes tested after filtering: %d\n", nrow(counts)))

  dds <- DESeqDataSetFromMatrix(countData = as.matrix(counts),
                                colData = coldata,
                                design = ~ Respond)
  dds <- DESeq(dds, quiet = TRUE)
  # contrast: Responder vs Non-Responder (positive LFC = higher in Responder)
  res <- results(dds, contrast = c("Respond", "Responder", "Non-Responder"))
  res <- res[order(res$padj), ]
  out <- as.data.frame(res)
  out$gene <- rownames(out)
  out <- out[, c("gene", setdiff(colnames(out), "gene"))]
  ofile <- file.path(outdir, sprintf("DESeq2_%s_Responder_vs_NonResponder.csv", cl))
  write.csv(out, ofile, row.names = FALSE)

  nsig <- sum(!is.na(out$padj) & out$padj < 0.05)
  cat(sprintf("  genes with padj<0.05 (patient-level): %d\n", nsig))
  cat(sprintf("  -> %s\n", basename(ofile)))

  # report the paper's highlighted genes
  cat("  paper-highlighted genes (log2FC = higher in Responder if positive):\n")
  for (grp in names(highlight)) {
    for (g in highlight[[grp]]) {
      r <- out[out$gene == g, ]
      if (nrow(r) == 1) {
        cat(sprintf("    [%s] %-7s log2FC=%+6.2f  p=%.3g  padj=%.3g\n",
                    grp, g, r$log2FoldChange, r$pvalue, r$padj))
      } else {
        cat(sprintf("    [%s] %-7s (filtered out / not detected)\n", grp, g))
      }
    }
  }
}

for (cl in clusters) run_one(cl)
cat("\n[r1b_DE DONE] DESeq2 tables in:", outdir, "\n")
