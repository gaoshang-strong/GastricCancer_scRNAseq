suppressPackageStartupMessages({
  library(ggplot2)
  library(data.table)
  library(ggrepel)
  library(pheatmap)
  library(dplyr)
  library(tidyverse)
})

# =========================
# CONFIG
# =========================
ROOT <- "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/07_perturb_seq_T_cells"
OUTDIR <- file.path(ROOT, "R_figs")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# top lists (pass QC)
TOP_2TO3 <- file.path(ROOT, "top_score_2to3.csv")
TOP_4TO3 <- file.path(ROOT, "top_score_4to3.csv")
TOP_2TO4 <- file.path(ROOT, "top_score_2to4_pro_exhaustion.csv")
TOP_C3   <- file.path(ROOT, "top_score_C3_NR_to_R.csv")

GENE_COL <- "target_contrast_gene_name"
COL_2TO3 <- "score_2to3"
COL_4TO3 <- "score_4to3"
COL_2TO4 <- "score_2to4"
COL_C3   <- "score_C3_NR_to_R"

CONSENSUS4 <- c("CD320", "FBXW7", "PARP8", "RANBP6")

# =========================
# Helpers
# =========================
read_top <- function(path) {
  stopifnot(file.exists(path))
  df <- readr::read_csv(path, show_col_types = FALSE)
  if (!(GENE_COL %in% colnames(df))) {
    stop(glue::glue("Missing {GENE_COL} in {path}. Columns: {paste(colnames(df), collapse=', ')}"))
  }
  df <- df %>% mutate(!!GENE_COL := as.character(.data[[GENE_COL]]))
  df
}

# Make a union score table (gene x 4 scores); missing scores filled with 0 for visualization
make_union_table <- function(df2, df4, df24, dfc3) {
  d2  <- df2  %>% select(all_of(GENE_COL), all_of(COL_2TO3)) %>% distinct(.data[[GENE_COL]], .keep_all = TRUE)
  d4  <- df4  %>% select(all_of(GENE_COL), all_of(COL_4TO3)) %>% distinct(.data[[GENE_COL]], .keep_all = TRUE)
  d24 <- df24 %>% select(all_of(GENE_COL), all_of(COL_2TO4)) %>% distinct(.data[[GENE_COL]], .keep_all = TRUE)
  dc3 <- dfc3 %>% select(all_of(GENE_COL), all_of(COL_C3))   %>% distinct(.data[[GENE_COL]], .keep_all = TRUE)
  
  M <- d2 %>%
    full_join(d4,  by = GENE_COL) %>%
    full_join(d24, by = GENE_COL) %>%
    full_join(dc3, by = GENE_COL)
  
  # fill missing with 0 (means "not in that top list")
  for (cc in c(COL_2TO3, COL_4TO3, COL_2TO4, COL_C3)) {
    if (!(cc %in% colnames(M))) M[[cc]] <- 0
    M[[cc]] <- replace_na(M[[cc]], 0)
  }
  M
}

# Order genes for heatmap: emphasize restoration-like pattern (2to3 + 4to3 - 2to4)
order_for_heatmap <- function(M) {
  M %>%
    mutate(
      restoration_index = .data[[COL_2TO3]] + .data[[COL_4TO3]] - .data[[COL_2TO4]],
      magnitude = sqrt((.data[[COL_2TO3]]^2 + .data[[COL_4TO3]]^2 + .data[[COL_2TO4]]^2 + .data[[COL_C3]]^2))
    ) %>%
    arrange(desc(restoration_index), desc(magnitude)) %>%
    select(-restoration_index, -magnitude)
}

# =========================
# Load data
# =========================
df2  <- read_top(TOP_2TO3)
df4  <- read_top(TOP_4TO3)
df24 <- read_top(TOP_2TO4)
dfc3 <- read_top(TOP_C3)

df_p1 <- rbind(df2, df4, dfc3)
df_p1 <- df_p1[!duplicated(df_p1), ]
df_p1$meta_score <- (df_p1$score_2to3+df_p1$score_4to3) / 2 - df_p1$score_2to4
df_p1 <- df_p1[order(df_p1$meta_score, decreasing = T), ]
df_p1_top30 <- df_p1[1:30, c(1,3,4,5,7)]

mat <- df_p1_top30 %>%
  column_to_rownames(GENE_COL) %>%
  as.matrix()

mat <- rbind(df2, df4)[!duplicated(rbind(df2, df4)), c(1,3,4,5,7)] %>%
  column_to_rownames(GENE_COL) %>%
  as.matrix()

mat <- df_p1[, c(1,3,4,6)] %>%
  column_to_rownames(GENE_COL) %>%
  as.matrix()

vmax <- max(abs(mat), na.rm = TRUE)
if (!is.finite(vmax) || vmax == 0) vmax <- 1

# make plot height scale with gene count (avoid unreadable long plots)
ng <- nrow(mat)
png(png_heat, width = 1000, height = max(3000, 18 * ng), res = 300)
pheatmap(
  mat,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  color = colorRampPalette(c("#2c7bb6", "white", "#d7191c"))(101),
  breaks = seq(-vmax, vmax, length.out = 102),
  border_color = NA,
  fontsize_row = ifelse(ng > 120, 6, 8),
  fontsize_col = 10,
  main = "Top perturb-seq drivers"
)
dev.off()

ggplot(df_p1, aes(x=score_2to3, score_2to4)) + geom_point()
ggplot(df_p1, aes(x=score_4to3, score_2to3)) + geom_point()

df_p2 <- rbind(df2, dfc3)
df_p2 <- df_p2[!duplicated(df_p2), ]
ggplot(df_p2, aes(x=score_2to3, score_C3_NR_to_R)) + geom_point()

all_df <- read.csv("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/07_perturb_seq_T_cells/step2_driver_ranking/driver_scores_agg_by_target.csv", header = T)

cor(all_df$score_C3_NR_to_R, all_df$score_2to3, method="spearman")
cor(all_df$score_C3_NR_to_R, all_df$score_4to3, method="spearman")
cor(all_df$score_4to3, all_df$score_2to3, method="spearman")

write.table(df2$target_contrast_gene_name, 'top30_genes_name_2to3.txt', quote = F, row.names = F, col.names = F)

intersect(df2$target_contrast_gene_name, df4$target_contrast_gene_name)




