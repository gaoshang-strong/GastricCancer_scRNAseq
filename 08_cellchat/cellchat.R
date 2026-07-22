library(SeuratDisk)
suppressPackageStartupMessages({
  library(CellChat)
  library(Seurat)
  library(future)
  library(ggplot2)
})
library(zellkonverter)
library(SingleCellExperiment)
h5ad_path <- "/tmp/adata_scvi_integrated_all_cells.h5ad"
library(reticulate)

# 1) 指定你已有的 python（换成你自己的环境路径）
use_python("~/micromamba/envs/scanpy_micromamba/bin/python", required = TRUE)

py_config()
Sys.setenv("BASILISK_EXTERNAL_PYTHON" = "1")
Sys.setenv(HDF5_USE_FILE_LOCKING = "FALSE")
sce <- readH5AD(h5ad_path, reader = "R", use_hdf5 = TRUE)  # 直接读
assayNames(sce)
dim(sce)

library(Matrix)

# 取出矩阵
mat_counts <- assay(sce, "counts")
mat_data   <- assay(sce, "log1p_norm")  # 你想用的 log-normalized

# 确保是稀疏矩阵（CellChat/Seurat 更稳）
if (!inherits(mat_counts, "dgCMatrix")) mat_counts <- as(mat_counts, "dgCMatrix")
if (!inherits(mat_data,   "dgCMatrix")) mat_data   <- as(mat_data,   "dgCMatrix")

# 用 counts 建 Seurat
seu <- CreateSeuratObject(counts = mat_counts)

# 把 log1p_norm 写进 data 槽（关键：CellChat 会用这里）
DefaultAssay(seu) <- "RNA"
#seu[["RNA"]]@data <- mat_data

# 加 meta
seu <- AddMetaData(seu, metadata = as.data.frame(colData(sce)))
seu[["RNA"]] <- CreateAssay5Object(counts = mat_counts, data = mat_data)
DefaultAssay(seu) <- "RNA"
# quick check
stopifnot(all(c("patient", "majority_voting") %in% colnames(seu@meta.data)))
table(seu$majority_voting)

non_resp <- c("PHD001", "PHD002", "PHD008")
seu$response <- ifelse(seu$patient %in% non_resp, "NR", "R")
seu$response <- factor(seu$response, levels = c("R", "NR"))

table(seu$response)

CD8_T_cells <- read.csv("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/cell_leiden_T_0.6_relabel.csv", header = T)

seu$cellchat_group <- as.character(seu$majority_voting)
common_cells <- intersect(CD8_T_cells$cell_name, colnames(seu))
seu$cellchat_group[common_cells] <- as.character(CD8_T_cells$leiden_T_0.6_relabel[match(common_cells, CD8_T_cells$cell_name)])
Idents(seu) <- seu$cellchat_group

cell_ids <- colnames(seu)
samples <- sub(".*_", "", cell_ids)
seu$samples <- samples
table(seu$samples)
head(seu$samples)
seu_R  <- subset(seu, subset = response == "R")
seu_NR <- subset(seu, subset = response == "NR")
library(CellChat)

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
                      title.name = "Myeloid/Epithelial -> CD8 T Interaction Changes")

# 渲染图片
p
