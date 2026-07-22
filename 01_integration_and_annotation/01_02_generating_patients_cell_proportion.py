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

# ----------------------------
# 0) 准备 Respond 标签
# ----------------------------
non_resp = {"PHD001", "PHD002", "PHD008"}
resp = {"PHD003", "PHD004", "PHD009"}
mapping = {p: "Non-Responder" for p in non_resp}
mapping.update({p: "Responder" for p in resp})

adata.obs["Respond"] = adata.obs["patient"].astype(str).map(mapping).fillna("Unknown")

#celltype_col = "predicted_labels"  # 或 "_scvi_labels"
celltype_col = "majority_voting"  # 或 "_scvi_labels"

# counts: patient x celltype
counts = (
    adata.obs
    .groupby(["patient", celltype_col], observed=True)
    .size()
    .unstack(fill_value=0)
)

props = counts.div(counts.sum(axis=1), axis=0)

patient_group = (
    adata.obs[["patient", "Respond"]]
    .drop_duplicates()
    .set_index("patient")
)

props = props.join(patient_group, how="left")

# 只画你定义的这6个病人（更聚焦对比；如果你想包含全部患者，把这行注释掉）
keep_patients = sorted(list(non_resp | resp))
props_plot = props.loc[keep_patients].copy()

# 排序：Non-Responder 在前，Responder 在后（突出对比）
order = (
    props_plot.reset_index()
    .sort_values(["Respond", "patient"], ascending=[True, True])  # Non-Responder < Responder
    ["patient"].tolist()
)
props_plot = props_plot.loc[order]

# 仅细胞类型列
celltypes = [c for c in props_plot.columns if c not in ["Respond"]]
M = props_plot[celltypes]

# ----------------------------
# A) 100% stacked bar（最直观）
# ----------------------------
palette = {
    "B cells":            "#1f77b4",  # (31,119,180)
    "DC":                 "#ff7f0e",  # (255,127,14)
    "Endothelial cells":  "#279e68",  # (39,158,104)
    "Epithelial cells":   "#d62728",  # (214,39,40)
    "Fibroblasts":        "#aa40fc",  # (170,64,252)
    "ILC":                "#8c564b",  # (140,86,75)
    "Macrophages":        "#e377c2",  # (227,119,194)
    "Mast cells":         "#b5bd61",  # (181,189,97)
    "Monocytes":          "#17becf",  # (23,190,207)
    "Plasma cells":       "#aec7e8",  # (174,199,232)
    "T cells":            "#ffbb78",  # (255,187,120)
}

fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(M)), 5))

bottom = np.zeros(M.shape[0])
x = np.arange(M.shape[0])

for ct in M.columns:
    ax.bar(
        x,
        M[ct].values,
        bottom=bottom,
        label=ct,
        color=palette.get(ct, "#808080"),  # 若有没在字典里的类型，用灰色兜底
        linewidth=0
    )
    bottom += M[ct].values

ax.set_xticks(x)
ax.set_xticklabels(M.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Cell proportion")
ax.set_title(f"Per-patient cell-type composition ({celltype_col})")
ax.set_ylim(0, 1.02)

ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)

plt.tight_layout()
fpath = os.path.join(outdir, f"02_celltype_composition_stackedbar_{celltype_col}.png")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fpath)

# ----------------------------
# B) Respond vs Non-Responder：每个细胞类型的“患者点 + 组间箱线”
#    （最突出对比 + 后续可加统计检验）
# ----------------------------
long_df = (
    props_plot.reset_index()
    .melt(id_vars=["patient", "Respond"], value_vars=celltypes,
          var_name="celltype", value_name="prop")
)

# 只保留两组（Unknown 不画）
long_df = long_df[long_df["Respond"].isin(["Responder", "Non-Responder"])].copy()

# 画成多面板：每个 celltype 一个小图（便于突出差异）
n_ct = len(celltypes)
ncols = 4 if n_ct >= 4 else n_ct
nrows = int(np.ceil(n_ct / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.2*nrows), squeeze=False)

for i, ct in enumerate(celltypes):
    r, c = divmod(i, ncols)
    ax = axes[r][c]
    sub = long_df[long_df["celltype"] == ct].copy()

    # x 轴分组顺序固定：Non-Responder, Responder
    groups = ["Non-Responder", "Responder"]
    data = [sub.loc[sub["Respond"] == g, "prop"].values for g in groups]

    # box
    ax.boxplot(data, labels=groups, showfliers=False)

    # scatter points（每个患者一个点）
    # 给点一点点水平抖动，避免重叠
    for j, g in enumerate(groups, start=1):
        y = sub.loc[sub["Respond"] == g, "prop"].values
        jitter = (np.random.rand(len(y)) - 0.5) * 0.15
        ax.scatter(np.full_like(y, j, dtype=float) + jitter, y, s=30)

    ax.set_title(ct)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(0.05, sub["prop"].max() * 1.2))

# 多出来的空面板删掉
for k in range(n_ct, nrows*ncols):
    r, c = divmod(k, ncols)
    axes[r][c].axis("off")

plt.tight_layout()
fpath = os.path.join(outdir, f"02_celltype_prop_respond_vs_nonrespond_box_scatter_{celltype_col}.png")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fpath)

# ----------------------------
# C) Heatmap：patient × celltype（带 Respond 注释）
# ----------------------------
try:
    import seaborn as sns

    # 构建注释颜色
    row_colors = props_plot["Respond"].map({"Non-Responder": "tab:orange", "Responder": "tab:blue"}).fillna("gray")

    g = sns.clustermap(
        M,  # 只细胞类型比例
        row_cluster=False, col_cluster=True,
        row_colors=row_colors,
        figsize=(max(8, 0.6*M.shape[1]), max(4, 0.5*M.shape[0] + 2)),
        cbar_kws={"label": "Proportion"}
    )
    g.fig.suptitle(f"Patient × Celltype proportions heatmap ({celltype_col})", y=1.02)
    fpath = os.path.join(outdir, f"02_celltype_prop_heatmap_{celltype_col}.png")
    g.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print("Saved:", fpath)

except ImportError:
    # seaborn 不存在就跳过（不报错）
    print("seaborn not installed, skipped heatmap. If needed: micromamba/conda install seaborn")


# ----------------------------
# A) 100% stacked bar + x-axis gap between groups
# ----------------------------
palette = {
    "B cells":            "#1f77b4",
    "DC":                 "#ff7f0e",
    "Endothelial cells":  "#279e68",
    "Epithelial cells":   "#d62728",
    "Fibroblasts":        "#aa40fc",
    "ILC":                "#8c564b",
    "Macrophages":        "#e377c2",
    "Mast cells":         "#b5bd61",
    "Monocytes":          "#17becf",
    "Plasma cells":       "#aec7e8",
    "T cells":            "#ffbb78",
}

fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(M)), 5))

resp_labels = props_plot.loc[M.index, "Respond"].astype(str)
n_nr = int((resp_labels == "Non-Responder").sum())
n_r  = int((resp_labels == "Responder").sum())

gap = 1.2
x = np.arange(M.shape[0], dtype=float)
if n_nr > 0 and n_r > 0:
    x[n_nr:] += gap

bottom = np.zeros(M.shape[0])
for ct in M.columns:
    ax.bar(
        x,
        M[ct].values,
        bottom=bottom,
        label=ct,
        width=0.9,
        color=palette.get(ct, "#808080"),  # 未匹配到的 celltype 用灰色兜底
        linewidth=0
    )
    bottom += M[ct].values

ax.set_xticks(x)
ax.set_xticklabels(M.index.tolist(), rotation=45, ha="right")
ax.set_ylabel("Cell proportion")
ax.set_title(f"Per-patient cell-type composition ({celltype_col})")
ax.set_ylim(0, 1.02)

if n_nr > 0 and n_r > 0:
    split_pos = (x[n_nr - 1] + x[n_nr]) / 2
    ax.axvline(split_pos, color="k", linewidth=1, alpha=0.6)

trans = ax.get_xaxis_transform()
if n_nr > 0:
    ax.text((x[0] + x[n_nr-1]) / 2, -0.18, "Non-Responder",
            transform=trans, ha="center", va="top")
if n_r > 0:
    ax.text((x[n_nr] + x[-1]) / 2, -0.18, "Responder",
            transform=trans, ha="center", va="top")

ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)

plt.tight_layout()
fpath = os.path.join(outdir, f"02_celltype_composition_stackedbar_{celltype_col}.png")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fpath)
