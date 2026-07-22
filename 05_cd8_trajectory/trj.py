import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Basic matplotlib config
# ----------------------------
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# User inputs
# ----------------------------
H5AD_PATH = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05b_trajectory_revisit/adata_CD8_subset_clusters_0_1_2_4.h5ad"
OUTDIR = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05c_trajectory_rerevisit/results_cellrank2_subset_0_1_2_4"
os.makedirs(OUTDIR, exist_ok=True)

CLUSTER_KEY = "leiden_T_0.6"     # values: 0/1/2/4 in this subset
RESP_KEY = "Respond"            # "Responder"/"Non-Responder"
EMBED_BASIS = "X_umap"          # use existing UMAP in adata
ROOT_CLUSTER = "4"
ROOT_STRATEGY = "top_naive"
NAIVE_KEY = "score_Naive"   # 你已有的 naive 分数列名；如果没有就用下面方法计算
NAIVE_TOP_FRAC = 0.10       # 前10%
NAIVE_MIN_CAND = 30         # 候选太少时的保底（按你数据量可改）
# GPCCA parameters
N_COMPONENTS = 20
N_STATES_RANGE = (4, 10)        # lets minChi pick
N_TERMINAL = 3                  # top_n terminal macrostates

# Branch hard-label parameters (relative advantage)
DELTA = 0.15  # fp_A - fp_B >= DELTA
EPS   = 0.25  # fp_A (or fp_B) >= EPS

# ----------------------------
# Helpers
# ----------------------------
def savefig(path: str):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def pick_root_cell(ad: sc.AnnData, rep: str) -> int:
    """Pick a robust root cell (iroot) using a cluster-defined candidate set."""
    mask_root = (ad.obs[CLUSTER_KEY].astype(str) == str(ROOT_CLUSTER)).values
    idxs = np.where(mask_root)[0]
    if len(idxs) == 0:
        raise ValueError(f"No cells found in ROOT_CLUSTER={ROOT_CLUSTER} for {CLUSTER_KEY}.")

    # ----- candidate selection: top naive within root cluster -----
    if ROOT_STRATEGY == "top_naive":
        if NAIVE_KEY not in ad.obs.columns:
            raise ValueError(f"ROOT_STRATEGY=top_naive but {NAIVE_KEY} not in ad.obs.")
        sub = ad.obs.loc[mask_root, NAIVE_KEY].astype(float)

        # take top 10% (or at least NAIVE_MIN_CAND)
        n_cand = max(int(np.ceil(len(sub) * NAIVE_TOP_FRAC)), NAIVE_MIN_CAND)
        n_cand = min(n_cand, len(sub))
        cand_names = sub.nlargest(n_cand).index
        cand_idxs = np.where(ad.obs_names.isin(cand_names))[0]

        print(f"[INFO] Root candidates: top {n_cand}/{len(sub)} ({NAIVE_TOP_FRAC:.0%}) by {NAIVE_KEY} in cluster {ROOT_CLUSTER}.")
    else:
        # fallback: all cells in root cluster
        cand_idxs = idxs
        print(f"[INFO] Root candidates: all {len(cand_idxs)} cells in cluster {ROOT_CLUSTER}.")

    # ----- pick representation space for medoid -----
    X = ad.obsm[rep][cand_idxs]

    # ----- medoid: most central cell among candidates -----
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D2 = x2 + x2.T - 2 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    medoid_local = int(np.argmin(D2.mean(axis=1)))
    iroot = int(cand_idxs[medoid_local])

    print(f"[INFO] Root cell selected by medoid in {rep}: {ad.obs_names[iroot]} (iroot={iroot})")
    return iroot

def ensure_neighbors_for_dpt(ad: sc.AnnData):
    """
    DPT needs connectivities. If neighbors missing, compute from X_scVI if present,
    otherwise from PCA.
    """
    if "neighbors" in ad.uns and "connectivities" in ad.obsp:
        return

    if "X_scVI" in ad.obsm:
        print("[INFO] Computing neighbors using X_scVI.")
        sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=50)
    else:
        print("[INFO] Computing PCA + neighbors (X_pca).")
        if "X_pca" not in ad.obsm:
            sc.pp.pca(ad, n_comps=50)
        sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=50)

def safe_scanpy_violin(ad, keys, groupby, out_png, rotation=45):
    """
    scanpy violin can differ by version; do show=False then save with matplotlib.
    """
    sc.pl.violin(ad, keys=keys, groupby=groupby, show=False, rotation=rotation)
    savefig(out_png)

def infer_lineage_for_cluster(fp_df: pd.DataFrame, ad: sc.AnnData, cluster_value: str) -> str:
    """
    Find which lineage (column in fp_df) is most associated with a given cluster,
    by mean fate probability within that cluster.
    """
    mask = (ad.obs[CLUSTER_KEY].astype(str) == str(cluster_value)).values
    means = fp_df.loc[ad.obs_names[mask]].mean(axis=0)
    return str(means.idxmax())


ad = sc.read_h5ad(H5AD_PATH)
print(ad)

# If this file is already subset to 0/1/2/4 you can skip. Keep it safe anyway.
keep = ad.obs[CLUSTER_KEY].astype(str).isin(["0", "1", "2", "4"]).values
ad = ad[keep].copy()
print(f"[INFO] Subset shape: {ad.shape}")
print("[INFO] Subset clusters:\n", ad.obs[CLUSTER_KEY].astype(str).value_counts())

from scipy import sparse
NAIVE_GENES = [
    "TCF7", "LEF1", "CCR7", "IL7R", "LTB", "MAL", "SELL",
    "TRAC", "CD3D", "CD3E"
]

genes = [g for g in NAIVE_GENES if g in ad.var_names]

# 1) 构建 log1p(scvi_norm_expr) layer（安全处理 sparse）
X = ad.layers["norm_expr"]
if sparse.issparse(X):
    X_log = X.copy()
    X_log.data = np.log1p(X_log.data)
else:
    X_log = np.log1p(X)

ad.layers["norm_expr_log1p"] = X_log

# 2) 用这个 layer 算 score
Xbak = ad.X
ad.X = ad.layers["norm_expr_log1p"]
sc.tl.score_genes(ad, gene_list=genes, score_name=NAIVE_KEY, use_raw=False)
ad.X = Xbak


out_dir = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/All_results_0123/CD8_Trajectory"
sc.settings.figdir = out_dir
sc.settings.set_figure_params(dpi=200, dpi_save=200)

sc.pl.umap(ad, color=NAIVE_KEY, show=False, save="_naive_score.png")
sc.pl.umap(ad, color='leiden_T_0.6', show=False, save="_CD8_clusters.png")

# ----------------------------
# DPT pseudotime
# ----------------------------
#sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=50)
sc.pp.neighbors(ad, use_rep="X_scVI", n_neighbors=50)
#ensure_neighbors_for_dpt(ad)
iroot = pick_root_cell(ad, rep="X_scVI")
ad.uns["iroot"] = iroot

# Diffmap + DPT
sc.tl.diffmap(ad)
sc.tl.dpt(ad)  # uses ad.uns["iroot"]
if "dpt_pseudotime" not in ad.obs:
    raise RuntimeError("scanpy did not create ad.obs['dpt_pseudotime'].")

# Quick DPT plots
sc.pl.umap(ad, color="dpt_pseudotime", show=False, save="_dpt_pseudotime.png")
#savefig(os.path.join(OUTDIR, "00_umap_cluster_dpt.png"))

CYTO_GENES = ["NKG7","GNLY","GZMB","GZMH","PRF1","FGFBP2","CTSW","KLRD1","KLRB1","TRAC","CD3D","CD3E"]
EXH_GENES  = ["PDCD1","CTLA4","LAG3","HAVCR2","TIGIT","TOX","TOX2","CXCL13","ENTPD1","LAYN","IKZF2"]

# 过滤到数据里存在的基因
ad.X = ad.layers['norm_expr_log1p']
CYTO_GENES = [g for g in CYTO_GENES if g in ad.var_names]
EXH_GENES  = [g for g in EXH_GENES  if g in ad.var_names]
sc.tl.score_genes(ad, gene_list=CYTO_GENES, score_name='cyto_score', use_raw=False)
sc.tl.score_genes(ad, gene_list=EXH_GENES, score_name='exh_score', use_raw=False)

ad.X = Xbak

df = ad.obs[['dpt_pseudotime', 'exh_score']].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

x = df['dpt_pseudotime'].astype(float).values
y = df['exh_score'].astype(float).values

plt.figure(figsize=(5.2, 4.4))
plt.scatter(x, y, s=6, alpha=0.35)
plt.xlabel('dpt_pseudotime')
plt.ylabel('exh_score')

df = ad.obs[['dpt_pseudotime', NAIVE_KEY]].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

x = df['dpt_pseudotime'].astype(float).values
y = df[NAIVE_KEY].astype(float).values

plt.figure(figsize=(5.2, 4.4))
plt.scatter(x, y, s=6, alpha=0.35)
plt.xlabel('dpt_pseudotime')
plt.ylabel(NAIVE_KEY)

def pick_top_frac_in_cluster(ad, cluster_value, frac=0.05, min_n=50):
    mask = (ad.obs[CLUSTER_KEY].astype(str) == str(cluster_value)).values
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No cells in cluster {cluster_value}")
    t = ad.obs['dpt_pseudotime'].values[idx].astype(float)
    thr = np.quantile(t, 1.0 - frac)
    pick = idx[t >= thr]
    if pick.size < min_n:
        # 兜底：取 top min_n（避免太小导致不稳定）
        pick = idx[np.argsort(t)[-min_n:]]
    return np.array(pick, dtype=int)

A_idx = pick_top_frac_in_cluster(ad, cluster_value="2", frac=0.05, min_n=50)  # terminal A from cluster 0
B_idx = pick_top_frac_in_cluster(ad, cluster_value="1", frac=0.05, min_n=50) 

ad.obs["is_terminal_A_c0top5"] = 0
ad.obs["is_terminal_B_c1top5"] = 0
ad.obs.iloc[A_idx, ad.obs.columns.get_loc("is_terminal_A_c0top5")] = 1
ad.obs.iloc[B_idx, ad.obs.columns.get_loc("is_terminal_B_c1top5")] = 1

pk = cr.kernels.PseudotimeKernel(ad, time_key='dpt_pseudotime')
pk.compute_transition_matrix()
T=pk.transition_matrix.tocsr()

n = T.shape[0]

A = np.zeros(n, dtype=float); A[A_idx] = 1.0
B = np.zeros(n, dtype=float); B[B_idx] = 1.0

def kstep_mass(T, indicator, k):
    v = indicator.copy()
    for _ in range(k):
        v = T @ v
    return np.asarray(v).ravel()

for k in [1, 5, 10, 20]:
    uA = kstep_mass(T, A, k)
    uB = kstep_mass(T, B, k)
    ad.obs[f"p_in_A_step{k}"] = uA
    ad.obs[f"p_in_B_step{k}"] = uB
    print(f"k={k}: mean(A)={uA.mean():.4g}, mean(B)={uB.mean():.4g}, frac(A>B)={(uA>uB).mean():.3f}")

# 画一下 k=10 的差值在 UMAP 上（看分叉是否存在）
ad.obs["k10_diff_AminusB"] = ad.obs["p_in_A_step10"] - ad.obs["p_in_B_step10"]
sc.pl.umap(ad, color=["k10_diff_AminusB"], show=False)
plt.show()

# 看分布
plt.figure(figsize=(5.2,3.6))
plt.hist(ad.obs["k10_diff_AminusB"], bins=60, alpha=0.8)
plt.title("Distribution of k=10 reachability diff (A-B)")
plt.xlabel("p_in_A_step10 - p_in_B_step10")
plt.tight_layout()
plt.show()

import numpy as np

x = ad.obs["k10_diff_AminusB"].values.astype(float)

# tau 控制“多快从 0.5 变到两端”，建议用分位数尺度
tau = np.quantile(np.abs(x), 0.7) + 1e-12
s = 1.0 / (1.0 + np.exp(-x / tau))

ad.obs["branch_preference_A_sigmoid"] = s
print("tau:", tau, "saved: branch_preference_A_sigmoid")

sc.pl.umap(ad, color=["branch_preference_A_sigmoid"], show=False, save="_branch_preference_A_sigmoid.png")
plt.show()

# 看分布
plt.figure(figsize=(5.2,3.6))
plt.hist(ad.obs["branch_preference_A_sigmoid"], bins=60, alpha=0.8)
plt.title("Distribution of k=10 reachability diff (A-B)")
plt.xlabel("p_in_A_step10 - p_in_B_step10")
plt.tight_layout()
plt.show()

score_key = "branch_preference_A_sigmoid"
out_key   = "branch_AB_0p5"

ad.obs[out_key] = np.where(ad.obs[score_key].values >= 0.46, "branch_A", "branch_B")
ad.obs[out_key] = pd.Categorical(ad.obs[out_key], categories=["branch_A", "branch_B"])

ad.obs[out_key].value_counts()

import matplotlib.pyplot as plt

X = ad.obsm["X_umap"]
x, y = X[:,0], X[:,1]
lab = ad.obs[out_key].values

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=4, c="lightgrey", alpha=0.4, linewidths=0)
mA = (lab == "branch_A")
plt.scatter(x[mA], y[mA], s=6, c="red", alpha=0.9, linewidths=0)
plt.title("Hard branch assignment (>=0.5 = A)")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hard_branch_A.png"), dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=4, c="lightgrey", alpha=0.4, linewidths=0)
mB = (lab == "branch_B")
plt.scatter(x[mB], y[mB], s=6, c="blue", alpha=0.9, linewidths=0)
plt.title("Hard branch assignment (<0.5 = B)")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hard_branch_B.png"), dpi=200, bbox_inches="tight")
plt.show()

from scipy.stats import rankdata, norm
key_out = "branch_preference_A_sigmoid_z"

x = ad.obs['branch_preference_A_sigmoid'].astype(float).values
m = np.isfinite(x)

z = np.full_like(x, np.nan, dtype=float)
r = rankdata(x[m], method="average")  # 1..n
n = r.size

# Blom / rankit transform: (r - 0.375)/(n + 0.25) 更稳定，避免无穷大
p = (r - 0.375) / (n + 0.25)
z[m] = norm.ppf(p)   # -> approx N(0,1)

ad.obs[key_out] = z

import numpy as np
import matplotlib.pyplot as plt

UMAP = "X_umap"
keyA = "is_terminal_A_c0top5"
keyB = "is_terminal_B_c1top5"

X = ad.obsm[UMAP]
x, y = X[:, 0], X[:, 1]

# 转成bool，兼容0/1、True/False、"True"/"False"
A = ad.obs[keyA].astype(bool).values
B = ad.obs[keyB].astype(bool).values

plt.figure(figsize=(10, 7.5))

# 1) background
plt.scatter(x, y, s=10, c="lightgrey", alpha=0.55, linewidths=0)

# 2) terminal A (red)
plt.scatter(
    x[A], y[A],
    s=80, c="red", edgecolors="black", linewidths=1.2,
    label="A (c0 top5%)"
)

# 3) terminal B (blue)
plt.scatter(
    x[B], y[B],
    s=80, c="dodgerblue", edgecolors="black", linewidths=1.2,
    label="B (c1 top5%)"
)

plt.title("terminal states")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "terminal_states.png"), dpi=200, bbox_inches="tight")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ----- data -----
X = ad.obsm["X_umap"].astype(float)
x = X[:, 0]
y = X[:, 1]
f = ad.obs["branch_preference_A_sigmoid"].values.astype(float)  # 0~1

# ----- grid -----
bins = 26
gx = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), bins)
gy = np.linspace(np.quantile(y, 0.02), np.quantile(y, 0.98), bins)
GX, GY = np.meshgrid(gx, gy, indexing="xy")

def grid_local_mean(x, y, val, GX, GY, radius_frac=0.12, min_points=80):
    xmin, xmax = np.quantile(x, [0.02, 0.98])
    ymin, ymax = np.quantile(y, [0.02, 0.98])
    rx = (xmax - xmin) * radius_frac
    ry = (ymax - ymin) * radius_frac
    r2 = rx*rx + ry*ry

    V = np.full(GX.shape, np.nan, dtype=float)
    ok = np.zeros(GX.shape, dtype=bool)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            x0, y0 = GX[i, j], GY[i, j]
            dx = x - x0
            dy = y - y0
            d2 = dx*dx + dy*dy
            sel = d2 <= r2
            if sel.sum() < min_points:
                continue
            w = np.exp(-d2[sel] / (0.5*r2))
            V[i, j] = np.sum(w * val[sel]) / (np.sum(w) + 1e-12)
            ok[i, j] = True
    return V, ok

# 网格上的平均 f（用于更稳定的梯度）
Fgrid, okF = grid_local_mean(x, y, f, GX, GY, radius_frac=0.12, min_points=80)

# ----- gradient on grid -----
# np.gradient: 对应 axis0=gy, axis1=gx
dFy, dFx = np.gradient(Fgrid, gy, gx)  # dF/dy, dF/dx

ok = okF & np.isfinite(dFx) & np.isfinite(dFy) & np.isfinite(Fgrid)

# ----- make arrows show up also in the middle -----
# 软权重：在 Fgrid 接近 0.5（主干）时也留一点箭头，不要完全没方向
# 如果你希望中间更强，把 tau 调大一点
tau = 0.12
s = np.tanh((Fgrid - 0.5) / tau)  # [-1,1]，中间接近0但不完全“死”

U = s * dFx
V = s * dFy

# 统一缩放，避免极端长箭头
mag = np.sqrt(U**2 + V**2)
cap = np.nanquantile(mag[ok], 0.90)
scale = 0.9 / (cap + 1e-12)   # 0.6~1.2 调整箭头总体长度
U *= scale
V *= scale

# ----- plot -----
plt.figure(figsize=(6.4, 5.4))
plt.scatter(x, y, s=4, c=f, alpha=0.75, linewidths=0)
plt.colorbar(label="branch_preference_A_sigmoid")

plt.quiver(GX[ok], GY[ok], U[ok], V[ok],
           angles="xy", scale_units="xy", scale=1.0,
           width=0.003, alpha=0.9, color="black")

plt.title("UMAP + arrows from branch_preference_A_sigmoid")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "UMAP_arrows_branch_preference_A_sigmoid.png"), dpi=200, bbox_inches="tight")
plt.show()


term = ad.obs[keyA].astype(bool).values | ad.obs[keyB].astype(bool).values

def stay_prob(T, term_mask, k=1):
    Tk = T.copy()
    for _ in range(k-1):
        Tk = Tk @ T
    # p_stay(i) = sum_j Tk[i,j] for j in terminal set
    p = np.asarray(Tk[:, term_mask].sum(axis=1)).ravel()
    return p

p1  = stay_prob(T, term, k=1)
p10 = stay_prob(T, term, k=10)

# 画 UMAP：p10
X = ad.obsm[UMAP].astype(float)
x, y = X[:,0], X[:,1]

plt.figure(figsize=(6.6, 5.6))
plt.scatter(x, y, s=6, c=p10, alpha=0.9, linewidths=0)
plt.colorbar(label="P(stay in terminal set) after 10 steps")
plt.scatter(x[term], y[term], s=45, facecolors="none", edgecolors="black", linewidths=1.2)
plt.title("k=10 terminal-set staying probability (black circles = terminal cells)")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.show()

# 再用直方图对比 terminal vs 非terminal（k=1 和 k=10）
plt.figure(figsize=(7.2, 4.2))
plt.hist(p1[~term], bins=60, alpha=0.5, label="non-terminal (k=1)")
plt.hist(p1[term],  bins=60, alpha=0.7, label="terminal (k=1)")
plt.title("Stay prob to terminal set (k=1)")
plt.legend(frameon=False); plt.tight_layout(); plt.show()

plt.figure(figsize=(7.2, 4.2))
plt.hist(p10[~term], bins=60, alpha=0.5, label="non-terminal (k=10)")
plt.hist(p10[term],  bins=60, alpha=0.7, label="terminal (k=10)")
plt.title("Stay prob to terminal set (k=10)")
plt.legend(frameon=False); plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def binned_mean_curve(x, y, n_bins=60, min_n=15):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    xc, yc = [], []
    for i in range(n_bins):
        sel = (x >= bins[i]) & (x < bins[i+1]) if i < n_bins-1 else (x >= bins[i]) & (x <= bins[i+1])
        if sel.sum() < min_n:
            continue
        xc.append(0.5 * (bins[i] + bins[i+1]))
        yc.append(np.mean(y[sel]))
    return np.array(xc), np.array(yc)

def smooth_rolling(y, win=9):
    # win 必须是奇数更好看；win 越大越平滑（建议 9~21）
    s = pd.Series(y)
    return s.rolling(window=win, center=True, min_periods=max(3, win//3)).mean().to_numpy()

def plot_two_lines(xA, yA, xB, yB, title, ylabel, win=11):
    yA_s = smooth_rolling(yA, win=win)
    yB_s = smooth_rolling(yB, win=win)

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(xA, yA_s, linewidth=2.8, color="red",  label="A")
    plt.plot(xB, yB_s, linewidth=2.8, color="blue", label="B")
    plt.xlabel("within-branch uniform pseudotime (0-1)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title}.png"), dpi=200, bbox_inches="tight")
    plt.show()

# 例子：naive
xA, yA = binned_mean_curve(branch_A_dpt_u, branch_A_naive, n_bins=70, min_n=20)
xB, yB = binned_mean_curve(branch_B_dpt_u, branch_B_naive, n_bins=70, min_n=20)
plot_two_lines(xA, yA, xB, yB, "Naive score trend (smoothed)", "score_Naive", win=13)

# cyto
xA, yA = binned_mean_curve(branch_A_dpt_u, branch_A_cyto, n_bins=70, min_n=20)
xB, yB = binned_mean_curve(branch_B_dpt_u, branch_B_cyto, n_bins=70, min_n=20)
plot_two_lines(xA, yA, xB, yB, "Cytotoxic score trend (smoothed)", "cyto_score", win=13)

# exh
xA, yA = binned_mean_curve(branch_A_dpt_u, branch_A_exh, n_bins=70, min_n=20)
xB, yB = binned_mean_curve(branch_B_dpt_u, branch_B_exh, n_bins=70, min_n=20)
plot_two_lines(xA, yA, xB, yB, "Exhaustion score trend (smoothed)", "exh_score", win=13)

xA, yA = binned_mean_curve(branch_A_dpt_u, branch_A_transit_prob, n_bins=70, min_n=20)
xB, yB = binned_mean_curve(branch_B_dpt_u, branch_B_transit_prob, n_bins=70, min_n=20)
plot_two_lines(xA, yA, xB, yB, "Transit probability trend (smoothed)", "k10_diff_AminusB", win=13)
