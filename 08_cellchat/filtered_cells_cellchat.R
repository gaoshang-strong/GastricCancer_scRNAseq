library(Seurat)

# 1) 取髓系（Macrophages + Monocytes + 可选 DC）
myeloid_types <- c("Macrophages", "Monocytes")
my <- subset(seu, subset = majority_voting %in% myeloid_types)

DefaultAssay(my) <- "RNA"

# 用 log1p_norm（你已经放进 RNA 的 data layer 了）
# Seurat v5: AddModuleScore 默认用 data layer（归一化表达）
myeloid_core <- c("LYZ","C1QA","C1QB","C1QC","LST1","TYROBP","FCER1G","CTSS","CSF1R","MS4A7","CD68","APOE")
epi_markers  <- c("EPCAM","KRT8","KRT18","KRT19","TACSTD2","MSLN","CEACAM5")
endo_markers <- c("PECAM1","VWF","KDR","PTPRB","EMCN")

# 2) 算 module scores
my <- AddModuleScore(my, features = list(myeloid_core), name = "score_myeloid_core")
my <- AddModuleScore(my, features = list(epi_markers),  name = "score_epi")
my <- AddModuleScore(my, features = list(endo_markers), name = "score_endo")

# AddModuleScore 会生成 score_myeloid_core1 这类列名
# 3) 设阈值（分位数阈值最稳：不需要你手动猜）
m_thr <- quantile(my$score_myeloid_core1, 0.50)  # 保留髓系核心较高的
e_thr <- quantile(my$score_epi1,          0.85)  # 上皮分数要低
v_thr <- quantile(my$score_endo1,         0.85)  # 内皮分数要低

keep <- (my$score_myeloid_core1 >= m_thr) & (my$score_epi1 <= e_thr) & (my$score_endo1 <= v_thr)
length(keep)
my_clean <- subset(my, cells = colnames(my)[keep])
my_clean
cat("Myeloid before:", ncol(my), " after clean:", ncol(my_clean), "\n")
my_all_cells <- colnames(my)

# 需要删除的髓系 cells = 髓系总集合 - 保留集合
my_all_cells  <- colnames(my)
my_keep_cells <- colnames(my)[keep]          # 用原对象 + keep 得到保留 cell names
my_drop_cells <- setdiff(my_all_cells, my_keep_cells)

length(my_all_cells)
length(my_keep_cells)
length(my_drop_cells)

seu_clean <- subset(seu, cells = setdiff(colnames(seu), my_drop_cells))

cat("All cells before:", ncol(seu), "after:", ncol(seu_clean), "\n")
cat("T cells before:", sum(seu$majority_voting=="T cells"),
    "after:", sum(seu_clean$majority_voting=="T cells"), "\n")
table(seu_clean$cellchat_group)

seu_R  <- subset(seu_clean, subset = response == "R")
seu_NR <- subset(seu_clean, subset = response == "NR")

# --- 创建 Responder 对象 ---
cellchat.R <- createCellChat(object = seu_R, group.by = "ident")
# 注意：因为上面已经 set Idents 了，这里 group.by 直接用 "ident" 最方便

# --- 创建 Non-Responder 对象 ---
cellchat.NR <- createCellChat(object = seu_NR, group.by = "ident")

# --- 设置人类数据库 ---
CellChatDB.use <- CellChatDB.human
cellchat.R@DB <- CellChatDB.use
cellchat.NR@DB <- CellChatDB.use

print("CellChat 对象创建完成！")

# 定义一个函数来处理标准流程，避免写两遍重复代码
run_cellchat_process <- function(seu_obj, db_use, group.by = "ident",
                                 assay = NULL, slot = "data",
                                 min.cells = 10, type = "triMean") {
  # 1) 创建 CellChat
  cellchat <- createCellChat(object = seu_obj, group.by = group.by)
  
  # 2) 指定数据库
  cellchat@DB <- db_use
  
  # 3) 可选：指定 assay（如果你 Seurat 里不是默认 RNA / SCT）
  # assay 不传就用 DefaultAssay(seu_obj)
  if (!is.null(assay)) {
    DefaultAssay(seu_obj) <- assay
  }
  
  # 4) 关键一步：subsetData -> 填充 data.signaling
  # （会从 Seurat 表达矩阵里提取 CellChat 需要的配体/受体相关基因）
  cellchat <- subsetData(cellchat)
  
  # 5) 过表达基因/互作
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  
  # 6) 通讯概率 + 过滤
  cellchat <- computeCommunProb(cellchat, type = type)
  cellchat <- filterCommunication(cellchat, min.cells = min.cells)
  
  # 7) 通路层面 + 聚合网络
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)
  
  return(cellchat)
}

# --- 跑 R / NR ---
cellchat.R  <- run_cellchat_process(seu_R,  CellChatDB.use, group.by = "ident")
cellchat.NR <- run_cellchat_process(seu_NR, CellChatDB.use, group.by = "ident")


object.list <- list(R = cellchat.R, NR = cellchat.NR)
cellchat <- mergeCellChat(object.list, add.names = names(object.list))

print("合并完成，准备画图！")

# 发送者：通常是 Myeloid 或 Epithelial
sources.use <- c("Macrophages", "Monocytes")

# 接收者：你的 CD8 T 细胞亚群
targets.use <- c("2", "3", "4")

# 这里的 comparison = c(1, 2) 表示对比 NR(2) vs R(1)
# 注意：list顺序是 R=1, NR=2

# 增加图形的高度，防止挤在一起
options(repr.plot.width=12, repr.plot.height=10)

p <- netVisual_bubble(cellchat,
                      sources.use = sources.use,
                      targets.use = targets.use,  
                      comparison = c(1, 2), # 1是R，2是NR
                      angle.x = 45,         # x轴标签倾斜45度
                      remove.isolate = TRUE,# 移除不显著的空行
                      title.name = "Myeloid -> CD8 T Interaction Changes")

# 渲染图片
png("../08_other_cell_types/Myeloid_cellchat.png", width = 5, height = 20, units = "in", res= 300)
p
dev.off()

slotNames(cellchat)

names(cellchat@net)
if ("prob" %in% names(cellchat@net)) {
  cat("net$prob class:", class(cellchat@net$prob), "\n")
  cat("net$prob names:", paste(names(cellchat@net$prob), collapse=", "), "\n")
  cat("net$prob dim:", paste(dim(cellchat@net$prob), collapse=" x "), "\n")
}

if (!is.null(cellchat@netP)) {
  cat("netP names:", paste(names(cellchat@netP), collapse=", "), "\n")
  if ("prob" %in% names(cellchat@netP)) {
    cat("netP$prob class:", class(cellchat@netP$prob), "\n")
    cat("netP$prob names:", paste(names(cellchat@netP$prob), collapse=", "), "\n")
  }
}

library(dplyr)
library(tidyr)
library(stringr)

extract_lr_long <- function(net_group) {
  # net_group = cellchat@net$R 或 cellchat@net$NR
  prob <- net_group$prob
  pval <- net_group$pval
  
  stopifnot(length(dim(prob)) == 3)
  dn <- dimnames(prob)
  stopifnot(length(dn) == 3)
  
  df <- as.data.frame(as.table(prob), stringsAsFactors = FALSE) %>%
    setNames(c("source","target","interaction_name","prob")) %>%
    mutate(prob = as.numeric(prob))
  
  if (!is.null(pval)) {
    df_p <- as.data.frame(as.table(pval), stringsAsFactors = FALSE) %>%
      setNames(c("source","target","interaction_name","pval")) %>%
      mutate(pval = as.numeric(pval))
    df <- left_join(df, df_p, by = c("source","target","interaction_name"))
  } else {
    df$pval <- NA_real_
  }
  
  # 解析 ligand/receptor（你的 y 轴显示是 "LIG - REC" 形式）
  df <- df %>%
    mutate(
      ligand  = str_trim(str_split_fixed(interaction_name, "-", 2)[,1]),
      receptor= str_trim(str_split_fixed(interaction_name, "-", 2)[,2])
    )
  
  df
}

dfR  <- extract_lr_long(cellchat@net$R)  %>% mutate(group="R")
dfNR <- extract_lr_long(cellchat@net$NR) %>% mutate(group="NR")

eps <- 1e-6

delta_lr <- bind_rows(dfR, dfNR) %>%
  select(group, source, target, interaction_name, ligand, receptor, prob, pval) %>%
  pivot_wider(
    names_from = group,
    values_from = c(prob, pval),
    values_fill = list(prob = 0, pval = 1)
  ) %>%
  mutate(
    delta_log2 = log2(prob_R + eps) - log2(prob_NR + eps),
    max_prob   = pmax(prob_R, prob_NR, na.rm = TRUE),
    min_p      = pmin(pval_R, pval_NR, na.rm = TRUE),
    direction  = case_when(delta_log2 > 0 ~ "Up_in_R",
                           delta_log2 < 0 ~ "Up_in_NR",
                           TRUE ~ "No_change")
  )

# 看看有哪些 target（找 CD8 的名字）
sort(unique(delta_lr$target))

library(dplyr)
library(tidyr)
library(stringr)
library(pheatmap)
library(tibble)

target_cd8 <- c("2", "3", "4")
send_pat <- "(Macrophage|Monocytes)"

delta_sub <- delta_lr %>%
  filter(target %in% target_cd8) %>%                 # ✅用 %in%
  filter(str_detect(source, send_pat)) %>%
  filter(max_prob > 0) %>%
  mutate(row_id = paste0("T", target, " | ", interaction_name))  # ✅把 target 拼进唯一ID

mat_lr <- delta_sub %>%
  arrange(desc(abs(delta_log2)), desc(max_prob)) %>%
  slice_head(n = 80) %>%
  select(source, row_id, delta_log2) %>%
  pivot_wider(names_from = source, values_from = delta_log2, values_fill = 0) %>%
  column_to_rownames("row_id") %>%
  as.matrix()

pheatmap(
  mat_lr,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  main = paste0("Myeloid → CD8 targets (", paste(target_cd8, collapse = ","), ")  Δlog2(prob) (R - NR)")
)

library(dplyr)
library(tidyr)
library(stringr)
library(pheatmap)
library(tibble)

target_cd8 <- c("2", "3", "4")
send_pat <- "(Macrophage|Monocytes)"

delta_sub_agg <- delta_lr %>%
  filter(target %in% target_cd8) %>%
  filter(str_detect(source, send_pat)) %>%
  filter(max_prob > 0) %>%
  group_by(source, interaction_name) %>%
  summarise(
    delta_log2 = mean(delta_log2, na.rm = TRUE),
    max_prob   = max(max_prob, na.rm = TRUE),
    .groups = "drop"
  )

mat_lr <- delta_sub_agg %>%
  arrange(desc(abs(delta_log2)), desc(max_prob)) %>%
  slice_head(n = 80) %>%
  select(source, interaction_name, delta_log2) %>%
  pivot_wider(names_from = source, values_from = delta_log2, values_fill = 0) %>%
  column_to_rownames("interaction_name") %>%
  as.matrix()

pheatmap(
  mat_lr,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  main = paste0("Myeloid → CD8 (targets ", paste(target_cd8, collapse=","), " aggregated)  Δlog2(prob) (R - NR)")
)

delta_sub <- delta_lr %>%
  filter(target %in% target_cd8) %>%
  filter(str_detect(source, send_pat)) %>%
  filter(max_prob > 0) %>%
  mutate(col_id = paste0(source, "|T", target))  # ✅把 T2/T3/T4 分到列

# 行：interaction_name；列：source|T2、source|T3、source|T4
mat_lr <- delta_sub %>%
  arrange(desc(abs(delta_log2)), desc(max_prob)) %>%
  slice_head(n = 80) %>%
  select(interaction_name, col_id, delta_log2) %>%
  pivot_wider(names_from = col_id, values_from = delta_log2, values_fill = 0) %>%
  column_to_rownames("interaction_name") %>%
  as.matrix()

col_order <- delta_sub %>%
  distinct(source, target, col_id) %>%
  mutate(target = as.integer(target)) %>%
  arrange(source, target) %>%
  pull(col_id)

mat_lr <- mat_lr[, intersect(col_order, colnames(mat_lr)), drop = FALSE]

png('../08_other_cell_types/myeloid/LR_level_difference.png', height = 10, width = 8, units = 'in', res = 200)
pheatmap(
  mat_lr,
  cluster_rows = TRUE,
  cluster_cols = FALSE,  # 你要固定顺序就别聚类列
  main = "Myeloid → CD8 (columns = source|T2,T3,T4): Δlog2(prob) (R - NR)"
)
dev.off()

library(pheatmap)
library(tibble)

assign_axis <- function(x) {
  dplyr::case_when(
    str_detect(x, "^HLA-") ~ "HLA",
    str_detect(x, "^ICAM1|^VCAM1|^PECAM1|^JAM1|^LAMA|^LAMB") ~ "Adhesion",
    str_detect(x, "^NECTIN2") ~ "NECTIN2-TIGIT/CD226",
    str_detect(x, "^LGALS9")  ~ "LGALS9-TIM3",
    str_detect(x, "^CD80|^CD86") ~ "CD80/86-CTLA4/CD28",
    str_detect(x, "^CD274") ~ "PD-L1/PD-1",
    str_detect(x, "^CXCL")  ~ "CXCL chemokines",
    str_detect(x, "^SPP1")  ~ "SPP1",
    str_detect(x, "^MIF")   ~ "MIF",
    TRUE ~ "Other"
  )
}

axis_tab <- delta_sub %>%
  mutate(axis = assign_axis(interaction_name),
         col_id = paste0(source, "|T", target)) %>%
  filter(axis != "Other") %>%
  group_by(axis, col_id) %>%
  summarise(axis_delta = sum(delta_log2, na.rm = TRUE), .groups="drop")

axis_mat <- axis_tab %>%
  pivot_wider(names_from = col_id, values_from = axis_delta, values_fill = 0) %>%
  column_to_rownames("axis") %>%
  as.matrix()

png('../08_other_cell_types/myeloid/axis_level_difference.png', height = 8, width = 8, units = 'in', res = 200)
pheatmap(axis_mat, cluster_rows = TRUE, cluster_cols = TRUE,
         main = "Axis-level Δ (R - NR): columns = source|T2/T3/T4")
dev.off()

library(tidytext)
topN <- 12

rank_df <- delta_sub %>%
  group_by(target) %>%
  # 选绝对差异最大的 2*topN
  arrange(desc(abs(delta_log2)), desc(max_prob), .by_group = TRUE) %>%
  slice_head(n = 2*topN) %>%
  ungroup() %>%
  mutate(label = paste0(source, ": ", interaction_name)) %>%
  group_by(target) %>%
  # 为了好看：在每个target内重新排序
  mutate(label = reorder_within(label, delta_log2, target)) %>%
  ungroup()

png('../08_other_cell_types/myeloid/LR_level_difference_plot.png', height = 8, width = 16, units = 'in', res = 200)
ggplot(rank_df, aes(x = delta_log2, y = label)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_segment(aes(x = 0, xend = delta_log2, yend = label)) +
  geom_point(size = 2) +
  facet_wrap(~target, scales = "free_y") +
  scale_y_reordered() +
  theme_bw() +
  labs(x = "Δlog2(prob) (R - NR)", y = NULL,
       title = "Top differential LR pairs per CD8 target (T2/T3/T4)")
dev.off()

extract_pathway_long <- function(netP_group) {
  prob <- netP_group$prob
  stopifnot(length(dim(prob)) == 3)
  as.data.frame(as.table(prob), stringsAsFactors = FALSE) %>%
    setNames(c("source","target","pathway","prob")) %>%
    mutate(prob = as.numeric(prob))
}

pwR  <- extract_pathway_long(cellchat@netP$R)  %>% mutate(group="R")
pwNR <- extract_pathway_long(cellchat@netP$NR) %>% mutate(group="NR")

eps <- 1e-6
pw_delta <- bind_rows(pwR, pwNR) %>%
  filter(target %in% target_cd8) %>%
  filter(str_detect(source, send_pat)) %>%
  mutate(col_id = paste0(source, "|T", target)) %>%
  pivot_wider(names_from = group, values_from = prob, values_fill = list(prob=0)) %>%
  mutate(delta_log2 = log2(R + eps) - log2(NR + eps))

pw_mat <- pw_delta %>%
  group_by(pathway, col_id) %>%
  summarise(delta_log2 = sum(delta_log2, na.rm=TRUE), .groups="drop") %>%
  pivot_wider(names_from = col_id, values_from = delta_log2, values_fill = 0) %>%
  column_to_rownames("pathway") %>%
  as.matrix()
thr <- 0.5  # 你可以调，比如 0.5/1/2

keep <- apply(pw_mat, 1, function(x) max(abs(x), na.rm = TRUE) >= thr)
pw_mat_f <- pw_mat[keep, , drop = FALSE]

png('../08_other_cell_types/myeloid/pathway_level_difference.png', height = 12, width = 8, units = 'in', res = 200)
pheatmap(pw_mat_f, cluster_rows = TRUE, cluster_cols = TRUE,
         main = "Pathway-level Δ (R - NR): columns = source|T2/T3/T4")
dev.off()

library(dplyr)
library(ggplot2)
library(stringr)

# delta_sub: 你之前已经过滤了 target %in% c("2","3","4") & source 是 myeloid 等
# 这里假设 delta_sub 至少包含：target, source, interaction_name, delta_log2, max_prob

topN <- 12  # 每张图：R更强 topN + NR更强 topN

target_map <- c(
  "2" = "Intermediate T (C2)",
  "3" = "Effector T (C3)",
  "4" = "Exhausted T (C4)"
)

make_barplot_for_target <- function(df, target_id, topN = 12, out_path = NULL) {
  df_t <- df %>%
    filter(target == target_id) %>%
    mutate(label = paste0(source, ": ", interaction_name))
  
  # 两边各取 topN，避免只剩一侧
  top_pos <- df_t %>%
    filter(delta_log2 > 0) %>%
    arrange(desc(delta_log2), desc(max_prob)) %>%
    slice_head(n = topN)
  
  top_neg <- df_t %>%
    filter(delta_log2 < 0) %>%
    arrange(delta_log2, desc(max_prob)) %>%
    slice_head(n = topN)
  
  plot_df <- bind_rows(top_neg, top_pos) %>%
    # 为了画图排序（从负到正）
    arrange(delta_log2) %>%
    mutate(label = factor(label, levels = label))
  
  plot_df <- plot_df %>%
    filter(!(delta_log2 < 0 & delta_log2 > -5))
  
  p <- ggplot(plot_df, aes(x = delta_log2, y = label, fill = delta_log2)) +
    geom_col(width = 0.8) +
    geom_vline(xintercept = 0, linetype = 2) +
    scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c") +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "right",
      axis.title.y = element_blank(),
      axis.text.y = element_text(color = "black")
    ) +
    labs(
      x = "Δlog2(prob)  (R - NR)",
      fill = "Δlog2(prob)",
      title = paste0("Top differential LR pairs: Myeloid → ", target_map[[target_id]])
    ) 
  
  if (!is.null(out_path)) {
    ggsave(out_path, p, width = 8, height = 6, dpi = 300)
  }
  p
}

p_c2 <- make_barplot_for_target(delta_sub, "2", topN = topN, out_path = "../08_other_cell_types/barplot_C2_intermediate.png")
p_c3 <- make_barplot_for_target(delta_sub, "3", topN = topN, out_path = "../08_other_cell_types/barplot_C3_effector.png")
p_c4 <- make_barplot_for_target(delta_sub, "4", topN = topN, out_path = "../08_other_cell_types/barplot_C4_exhausted.png")

p_c2; p_c3; p_c4

library(pheatmap)

# pw_mat: 行=pathway，列=source|T2/T3/T4，值=Δlog2(prob)(R-NR)

# 选择差异最明显的 pathway：按每行 max(|Δ|)
topN_pw <- 30
score_pw <- apply(pw_mat, 1, function(x) max(abs(x), na.rm = TRUE))
pw_keep <- names(sort(score_pw, decreasing = TRUE))[1:min(topN_pw, length(score_pw))]

pw_mat_f <- pw_mat[pw_keep, , drop = FALSE]

# 可选：clamp 防止极端值压扁其它行
clamp <- function(x, lim = 5) pmax(pmin(x, lim), -lim)
pw_mat_f <- clamp(pw_mat_f, lim = 5)

# 列顺序可固定（建议：先 Macrophages 的 T2/T3/T4，再 Monocytes 的 T2/T3/T4）
# 你按自己的 colnames(pw_mat) 实际格式调整下面正则排序
col_order <- colnames(pw_mat_f)[order(colnames(pw_mat_f))]  # 简单字母序
pw_mat_f <- pw_mat_f[, col_order, drop = FALSE]

png("../08_other_cell_types/pathway_heatmap_top_pw.png", width = 2000, height = 1500, res = 300)
pheatmap(
  pw_mat_f,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  fontsize_row = 10,
  fontsize_col = 10,
  main = paste0("Top ", nrow(pw_mat_f), " differential pathways | Δlog2(prob) (R - NR)"),
  border_color = NA
)
dev.off()
