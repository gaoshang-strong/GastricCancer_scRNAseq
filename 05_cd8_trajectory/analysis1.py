#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Plot style (consistent, publication-friendly)
# ----------------------------
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ----------------------------
# Paths
# ----------------------------
H5AD_PATH = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_subset_clusters_0_1_2_4.h5ad"
)
OUTDIR = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05c_trajectory_rerevisit/results_subset_0_1_2_4__dpt"
)
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Config
# ----------------------------
CLUSTER_KEY = "leiden_T_0.6"
KEEP_CLUSTERS = ["0", "1", "2", "4"]

# Use scVI latent for graph/trajectory (recommended given your pipeline)
USE_REP = "X_scVI"
N_NEIGHBORS = 30  # you can try 15/30/50; 30 is a solid default for ~10k cells
METRIC = "cosine"
N_DCS = 30        # diffusion components
ROOT_STRATEGY = "min_exhaustion"  # automatic root: lowest Exhaustion score
EXHAUSTION_KEY = "score_Exhaustion"

# ----------------------------
# Load & subset
# ----------------------------
adata = sc.read_h5ad(H5AD_PATH)
print(adata)
# Ensure cluster labels are strings
adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype(str)

ad = adata[adata.obs[CLUSTER_KEY].isin(KEEP_CLUSTERS)].copy()
print("[INFO] Subset shape:", ad.shape)
print("[INFO] Subset clusters:\n", ad.obs[CLUSTER_KEY].value_counts())

# ----------------------------
# Recompute neighbors on subset (critical)
# ----------------------------
sc.pp.neighbors(
    ad,
    use_rep=USE_REP,
    n_neighbors=N_NEIGHBORS,
    metric=METRIC,
)

# ----------------------------
# Diffusion map + DPT
# ----------------------------
sc.tl.diffmap(ad, n_comps=N_DCS)

# Pick root cell automatically (robust & reproducible)
if ROOT_STRATEGY == "min_exhaustion":
    if EXHAUSTION_KEY not in ad.obs.columns:
        raise ValueError(f"Missing {EXHAUSTION_KEY} in ad.obs. Available: {list(ad.obs.columns)}")
    # pick the cell with minimum exhaustion (ties resolved by first occurrence)
    root_ix = int(np.argmin(ad.obs[EXHAUSTION_KEY].to_numpy()))
    root_cell = ad.obs_names[root_ix]
    ad.uns["iroot"] = root_ix
    print(f"[INFO] Root cell selected by min {EXHAUSTION_KEY}: {root_cell} (index={root_ix})")
else:
    raise ValueError("Unsupported ROOT_STRATEGY")

sc.tl.dpt(ad)

# DPT results:
# - ad.obs["dpt_pseudotime"]  (0..1)
# - optionally ad.obs["dpt_groups"] if you set groupings

# ----------------------------
# Quick QC plots
# ----------------------------
# 1) UMAP on subset (optional, just for visualization)
sc.tl.umap(ad, min_dist=0.3)

# Color by cluster and pseudotime
fig = sc.pl.umap(
    ad,
    color=[CLUSTER_KEY, "dpt_pseudotime", EXHAUSTION_KEY],
    wspace=0.4,
    size=12,
    show=False,
    return_fig=True,
)
fig.savefig(os.path.join(OUTDIR, "subset_0_1_2_4__umap__cluster_dpt_exhaustion.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 2) Diffmap colored by pseudotime (often clearer than UMAP)
fig = sc.pl.diffmap(
    ad,
    color=["dpt_pseudotime", CLUSTER_KEY],
    components=["1,2", "1,3"],
    wspace=0.4,
    size=12,
    show=False,
    return_fig=True,
)
fig.savefig(os.path.join(OUTDIR, "subset_0_1_2_4__diffmap__dpt.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 3) Simple summary per cluster
summary = (
    ad.obs.groupby(CLUSTER_KEY)["dpt_pseudotime"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_index()
)
summary.to_csv(os.path.join(OUTDIR, "subset_0_1_2_4__dpt_summary_by_cluster.csv"))
print("[INFO] Wrote:", os.path.join(OUTDIR, "subset_0_1_2_4__dpt_summary_by_cluster.csv"))

# ----------------------------
# Save subset object
# ----------------------------
#ad.write_h5ad(os.path.join(OUTDIR, "subset_0_1_2_4__with_dpt.h5ad"))
#print("[DONE] Saved subset with DPT to:", os.path.join(OUTDIR, "subset_0_1_2_4__with_dpt.h5ad"))

import numpy as np
from scipy.sparse.csgraph import connected_components

G = ad.obsp["connectivities"]
n_comp, labels = connected_components(G, directed=False)
print("n_connected_components =", n_comp)
print("component sizes:", np.bincount(labels))

sc.tl.paga(ad, groups="leiden_T_0.6")
sc.pl.paga(ad, show=False)
plt.savefig(os.path.join(OUTDIR, "paga_quickcheck.png"), dpi=300, bbox_inches="tight")
plt.close()

naive_genes = ["TCF7","IL7R","CCR7","LTB","LEF1"]
sc.tl.score_genes(ad, gene_list=[g for g in naive_genes if g in ad.var_names], score_name="score_NaiveLike")

sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=30, metric="cosine")
sc.tl.diffmap(ad, n_comps=30)

mask4 = (ad.obs["leiden_T_0.6"].astype(str) == "4")
score = ad.obs["score_NaiveLike"].to_numpy() - ad.obs["score_Exhaustion"].to_numpy()
root_local = np.argmax(score[mask4])
root_cell = ad.obs_names[mask4][root_local]
root_ix = np.where(ad.obs_names == root_cell)[0][0]
ad.uns["iroot"] = int(root_ix)

print("Root cell chosen in cluster4:", root_cell, "iroot=", root_ix)

sc.tl.dpt(ad, n_dcs=30)
sc.tl.umap(ad, min_dist=0.3)
fig = sc.pl.umap(ad, color=["leiden_T_0.6","dpt_pseudotime","score_NaiveLike","score_Exhaustion"], show=False, return_fig=True)
fig.savefig(f"{OUTDIR}/umap_dpt_root_in4.png", dpi=300, bbox_inches="tight"); plt.close(fig)

fig = sc.pl.diffmap(ad, color=["dpt_pseudotime","leiden_T_0.6"], components=["1,2","1,3"], show=False, return_fig=True)
fig.savefig(f"{OUTDIR}/diffmap_dpt_root_in4.png", dpi=300, bbox_inches="tight"); plt.close(fig)

import cellrank as cr
pk = cr.kernels.PseudotimeKernel(ad, time_key="dpt_pseudotime")
pk.compute_transition_matrix()  # 生成 Markov transition matrix
# 可选：把 kernel 信息写回 adata，方便下次从 h5ad 恢复
pk.write_to_adata()

g = cr.estimators.GPCCA(pk)

# macrostates：建议给一个范围，让它用 minChi 自动挑合适的状态数
g.compute_schur(n_components=20, method="brandts")

# 再 fit（不要再传 n_components / method）
g.fit(cluster_key=CLUSTER_KEY, n_states=[4, 6])
# （热图1）macrostate-level transition：coarse_T（本质就是你要的 macrostate heatmap）
g.plot_coarse_T()
plt.savefig(f"{OUTDIR}/01_coarse_T_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# terminal states：自动挑最稳定的一批（你可以改 n_states）
g.predict_terminal_states(method="top_n", n_states=3)  # 例如先挑 3 个 terminal
g.plot_macrostates(which="terminal", discrete=True, legend_loc="right", s=80)
plt.savefig(f"{OUTDIR}/02_terminal_macrostates_umap.png", dpi=300, bbox_inches="tight")
plt.close()

# 可选：initial states（如果你想显式标注“起点”）
# 你也可以改成 set_initial_states(states=["4"]) 这类“手动指定”
#g.predict_initial_states(n_states=1)
g.set_initial_states(states=["4"])
g.plot_macrostates(which="initial", discrete=True, legend_loc="right", s=80)
plt.savefig(f"{OUTDIR}/03_initial_macrostates_umap.png", dpi=300, bbox_inches="tight")
plt.close()

# fate probabilities：对每个细胞到每个 terminal 的吸收概率（fate prob）
g.compute_fate_probabilities(use_petsc=False, solver="gmres")
g.plot_fate_probabilities(same_plot=False)  # 每条 lineage 单独一张图 :contentReference[oaicite:2]{index=2}
plt.savefig(f"{OUTDIR}/04_fate_probabilities_umap.png", dpi=300, bbox_inches="tight")
plt.close()

# （热图2）按 cluster 聚合 fate probabilities（你要的第二张热图）
# mode="heatmap" 是官方支持的 :contentReference[oaicite:3]{index=3}
cr.pl.aggregate_fate_probabilities(
    ad,
    mode="heatmap",
    cluster_key=CLUSTER_KEY,
)
plt.savefig(f"{OUTDIR}/05_fate_probabilities_by_cluster_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

cr.pl.aggregate_fate_probabilities(
    ad,
    mode="heatmap",
    cluster_key="Respond",   # 或你叫 respond
)

RESPOND_KEY = "Respond"
PT_KEY = "dpt_pseudotime"
SCORES = ["score_Cytotoxic", "score_Exhaustion", "score_TCR"]

cr.pl.aggregate_fate_probabilities(ad, mode="heatmap", cluster_key=RESPOND_KEY)
plt.savefig(f"{OUTDIR}/A_fate_byRespond_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

m1 = ad.obs[CLUSTER_KEY].astype(str) == "1"
df1 = pd.DataFrame({
    "cyt": ad.obs.loc[m1, "score_Cytotoxic"].values,
    "exh": ad.obs.loc[m1, "score_Exhaustion"].values,
    "resp": ad.obs.loc[m1, RESPOND_KEY].astype(str).values,
})

plt.figure(figsize=(6,5))
for grp, dfg in df1.groupby("resp"):
    plt.scatter(dfg["cyt"], dfg["exh"], s=8, alpha=0.6, label=grp)
plt.xlabel("score_Cytotoxic"); plt.ylabel("score_Exhaustion")
plt.title("Cluster 1: Cytotoxic vs Exhaustion (by Respond)")
plt.legend(frameon=False)
plt.savefig(f"{OUTDIR}/B1_cluster1_scatter_cyt_vs_exh.png", dpi=300, bbox_inches="tight")
plt.close()

groups = list(df1["resp"].unique())
data = [df1.loc[df1["resp"]==g, "exh"].values for g in groups]
plt.figure(figsize=(6,4))
plt.violinplot(data, showmedians=True, showextrema=False)
plt.xticks(range(1, len(groups)+1), groups, rotation=15, ha="right")
plt.ylabel("score_Exhaustion")
plt.title("Cluster 1: Exhaustion by Respond")
plt.savefig(f"{OUTDIR}/B2_cluster1_violin_exh.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------
# 3) Lineage-specific pseudotime trends (bin-median), split by Respond
#    Uses CellRank fate probabilities: g.fate_probabilities
# -----------------------
fp = g.fate_probabilities.copy()  # index=cell ids, columns=lineages
# 转成 DataFrame：index=cell, columns=lineages
if hasattr(fp, "to_df"):
    fp = fp.to_df()
else:
    fp = pd.DataFrame(fp.X, index=ad.obs_names, columns=getattr(fp, "names", None))

# 如果 columns 还是 None（极少数版本），用 keys 推断
if fp.columns.isnull().any() or fp.columns.tolist() == [None]*fp.shape[1]:
    fp.columns = list(getattr(g.fate_probabilities, "names", range(fp.shape[1])))

def binned_median(x, y, q=25):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if tmp.shape[0] < 50:
        return None, None
    tmp["bin"] = pd.qcut(tmp["x"], q=q, duplicates="drop")
    y_med = tmp.groupby("bin")["y"].median()
    x_med = tmp.groupby("bin")["x"].median().reindex(y_med.index)
    return x_med.values, y_med.values

TH = 0.60   # fate prob threshold
Q = 25      # bins

pt_all = ad.obs[PT_KEY].values
resp_all = ad.obs[RESPOND_KEY].astype(str).values

for lin in fp.columns:
    mlin = fp[lin].values >= TH
    if mlin.sum() < 200:
        continue
    for score in SCORES:
        y_all = ad.obs[score].values
        plt.figure(figsize=(6,4))
        ok = False
        for grp in sorted(np.unique(resp_all)):
            m = mlin & (resp_all == grp)
            if m.sum() < 80:
                continue
            x, y = binned_median(pt_all[m], y_all[m], q=Q)
            if x is None:
                continue
            plt.plot(x, y, label=f"{grp} (n={m.sum()})")
            ok = True
        if not ok:
            plt.close()
            continue
        plt.xlabel(PT_KEY); plt.ylabel(score)
        plt.title(f"{lin}: {score} vs pseudotime (fp>={TH})")
        plt.legend(frameon=False)
        plt.savefig(f"{OUTDIR}/C_{lin}__{score}.png", dpi=300, bbox_inches="tight")
        plt.close()

print("Saved to:", OUTDIR)


CLUSTER_KEY = "leiden_T_0.6"
RESPOND_KEY = "Respond"

# ---- fate probs -> DataFrame (兼容 cellrank Lineage 对象) ----
fp = g.fate_probabilities
try:
    fp_df = fp.to_df()
except Exception:
    fp_df = pd.DataFrame(fp.X, index=ad.obs_names, columns=getattr(fp, "names", None))
    if fp_df.columns is None or any([c is None for c in fp_df.columns]):
        fp_df.columns = [f"lin{i}" for i in range(fp_df.shape[1])]

# 对齐
fp_df = fp_df.loc[ad.obs_names]

# ---- 只取 cluster1 ----
m1 = ad.obs[CLUSTER_KEY].astype(str) == "1"
ad1_obs = ad.obs.loc[m1, [RESPOND_KEY]].copy()
fp1 = fp_df.loc[m1].copy()
print("cluster1 cells:", fp1.shape[0])
print(ad1_obs[RESPOND_KEY].value_counts())

# =========================
# 图1：cluster1 内按 Respond 聚合 fate prob heatmap
# =========================
mean_by_resp = pd.concat([ad1_obs, fp1], axis=1).groupby(RESPOND_KEY).mean()

plt.figure(figsize=(1.2*mean_by_resp.shape[1] + 2, 2.5))
plt.imshow(mean_by_resp.values, aspect="auto")
plt.yticks(range(mean_by_resp.shape[0]), mean_by_resp.index.astype(str))
plt.xticks(range(mean_by_resp.shape[1]), mean_by_resp.columns.astype(str), rotation=45, ha="right")
plt.colorbar(label="mean fate probability")
plt.title("Cluster 1: mean fate probabilities by Respond")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/cluster1_fate_mean_byRespond_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 图2：cluster1 内每条 lineage fate prob 的 R vs NR violin
# =========================
df_long = fp1.copy()
df_long[RESPOND_KEY] = ad1_obs[RESPOND_KEY].astype(str).values
df_long = df_long.melt(id_vars=[RESPOND_KEY], var_name="lineage", value_name="fate_prob")

# 按 lineage 排序：先按 NR-R 的差（更容易看“NR更偏向0/2终态”）
pivot = df_long.groupby([RESPOND_KEY, "lineage"])["fate_prob"].mean().unstack(0)
# 如果你的组名不是这两个，下面这段会自动跳过排序
if pivot.shape[1] >= 2:
    cols = pivot.columns.tolist()
    # 取前两个组做差：按字母顺序只是为了稳
    c0, c1 = sorted(cols)[:2]
    order = (pivot[c0] - pivot[c1]).sort_values(ascending=False).index.tolist()
else:
    order = sorted(df_long["lineage"].unique())

# 画小提琴：每个 lineage 一个 subplot（最简单稳定，不依赖 seaborn）
groups = sorted(df_long[RESPOND_KEY].unique())
n_lin = len(order)
ncol = 4
nrow = int(np.ceil(n_lin / ncol))

plt.figure(figsize=(4*ncol, 3*nrow))
for i, lin in enumerate(order, start=1):
    ax = plt.subplot(nrow, ncol, i)
    data = [df_long[(df_long["lineage"]==lin) & (df_long[RESPOND_KEY]==g)]["fate_prob"].values for g in groups]
    ax.violinplot(data, showmedians=True, showextrema=False)
    ax.set_title(lin)
    ax.set_xticks(range(1, len(groups)+1))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylim(0, 1)
plt.suptitle("Cluster 1: fate probabilities by lineage (Respond vs Non-Respond)", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/cluster1_fate_violin_byRespond_perLineage.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:\n- cluster1_fate_mean_byRespond_heatmap.png\n- cluster1_fate_violin_byRespond_perLineage.png")
