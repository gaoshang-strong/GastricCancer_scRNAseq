import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
all_cells_path = "01_mapping_raw_scRNA_seq_to_reference/adata_scvi_integrated_all_cells.h5ad"
adata = sc.read_h5ad(all_cells_path)

outdir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/01_mapping_raw_scRNA_seq_to_reference/results/"
os.makedirs(outdir, exist_ok=True)

celltype_col = "predicted_labels"

# ---------------------------
# 1) 确保 categorical
# ---------------------------
if not pd.api.types.is_categorical_dtype(adata.obs[celltype_col]):
    adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")

cats = list(adata.obs[celltype_col].cat.categories)

# ---------------------------
# 2) 让 Scanpy 自动生成配色（如果还没有）
#    Scanpy 会把颜色写到 adata.uns["majority_voting_colors"]
# ---------------------------
if f"{celltype_col}_colors" not in adata.uns:
    sc.pl.umap(adata, color=celltype_col, show=False)  # 触发 scanpy 生成 colors
    plt.close()

colors = list(adata.uns[f"{celltype_col}_colors"])
if len(colors) != len(cats):
    raise ValueError(
        f"Color length mismatch: {len(colors)} colors vs {len(cats)} categories. "
        "This usually happens if categories changed after colors were created."
    )

ct2color = dict(zip(cats, colors))

# ---------------------------
# 3) 用 Scanpy 配色重画 UMAP（保存）
# ---------------------------
sc.pl.umap(
    adata,
    color=celltype_col,
    frameon=False,
    show=False,
)
umap_path = os.path.join(outdir, f"03_UMAP_{celltype_col}_scanpy_colors.png")
plt.savefig(umap_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", umap_path)

# ---------------------------
# 4) 计算每个患者比例，并用同一套 Scanpy 配色画 stacked bar（保存）
# ---------------------------
non_resp = {"PHD001", "PHD002", "PHD008"}
resp = {"PHD003", "PHD004", "PHD009"}
mapping = {p: "Non-Responder" for p in non_resp}
mapping.update({p: "Responder" for p in resp})

adata.obs["Respond"] = adata.obs["patient"].astype(str).map(mapping).fillna("Unknown")

counts = (
    adata.obs
    .groupby(["patient", celltype_col], observed=True)
    .size()
    .unstack(fill_value=0)
)
props = counts.div(counts.sum(axis=1), axis=0)

# 只画这 6 个病人（如果你想画全部患者，就删掉下面两行）
keep_patients = sorted(list(non_resp | resp))
props = props.loc[keep_patients]

# 排序：Non-Responder 在前，Responder 在后
patient_group = (
    adata.obs[["patient", "Respond"]]
    .drop_duplicates()
    .set_index("patient")
)
props = props.join(patient_group, how="left")
order = (
    props.reset_index()
    .sort_values(["Respond", "patient"], ascending=[True, True])
    ["patient"].tolist()
)
props = props.loc[order]

# 确保列顺序与 cats 一致（否则 legend 颜色会乱）
M = props.reindex(columns=cats).fillna(0)

fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(M)), 5))

bottom = np.zeros(M.shape[0])
x = np.arange(M.shape[0])

for ct in cats:
    ax.bar(x, M[ct].values, bottom=bottom, label=ct, color=ct2color[ct])
    bottom += M[ct].values

ax.set_xticks(x)
ax.set_xticklabels(M.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Cell proportion")
ax.set_title(f"Per-patient cell-type composition ({celltype_col})")
ax.set_ylim(0, 1.02)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)

plt.tight_layout()
bar_path = os.path.join(outdir, f"03_celltype_composition_stackedbar_{celltype_col}_scanpy_colors.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", bar_path)
