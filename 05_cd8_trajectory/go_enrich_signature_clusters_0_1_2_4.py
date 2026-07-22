import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Plot style
# =========================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]

# =========================
# Input / Output
# =========================
h5ad_in = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "05b_trajectory_revisit/adata_CD8_harmony_cellrank2.h5ad"
)
outdir = os.path.dirname(h5ad_in)
os.makedirs(outdir, exist_ok=True)

cluster_col = "leiden_T_0.6"
keep_clusters = ["0", "1", "2", "4"]  # 你要比较的

# Output figs
fig_violin = os.path.join(outdir, "violin_scores_by_cluster_0_1_2_4.png")
fig_dotplot = os.path.join(outdir, "dotplot_markers_by_cluster_0_1_2_4.png")

# =========================
# Load & subset
# =========================
adata = sc.read_h5ad(h5ad_in)
print("[INFO] loaded:", adata)

if cluster_col not in adata.obs.columns:
    raise KeyError(f"Missing adata.obs['{cluster_col}']")

adata.obs[cluster_col] = adata.obs[cluster_col].astype(str)
adata = adata[adata.obs[cluster_col].isin(keep_clusters)].copy()
print("[INFO] subset clusters:", keep_clusters, "->", adata)

# Use log-normalized if present; if you only have counts, we do normalize+log here
# (不会改 layers，只改 adata.X)
if "log1p" not in adata.uns_keys():
    # 如果你之前已经 log1p 过，这里也不会出大问题，但最好避免重复 log1p
    pass

# 如果 adata.X 看起来是 counts（整数且很大），做 normalize+log
X_is_int = False
try:
    X_is_int = np.issubdtype(adata.X.dtype, np.integer)
except Exception:
    X_is_int = False

if X_is_int:
    print("[INFO] adata.X looks like counts -> normalize_total + log1p")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# =========================
# Define marker sets (CD8-focused)
# =========================
naive_markers = [
    "CCR7","IL7R","LTB","MALAT1","TCF7","LEF1","SELL","LST1"  # LST1可能不是T，若出现干扰可删
]
# 细胞毒性（effector/cytotoxic）
cyto_markers = [
    "NKG7","GNLY","PRF1","GZMB","GZMH","GZMK","CTSW","KLRD1","KLRB1","FCGR3A"
]
# Tex / exhaustion
tex_markers = [
    "PDCD1","LAG3","TIGIT","HAVCR2","TOX","TOX2","CTLA4","ENTPD1","CXCL13","LAYN"
]
# Intermediate / activation / memory-like (可理解为过渡态)
inter_markers = [
    "IL32","CD69","HLA-DRA","HLA-DRB1","TNFRSF9","ICOS","IFITM1","IFITM2","JUN","FOS"
]
# Proliferation / cycling（如果 1 是 cycling）
trm_markers   = ["ITGAE","CXCR6","ZNF683","RGS1","CD69","RUNX3","PRDM1","FABP5"]

cycle_markers = [
    "MKI67","TOP2A","TYMS","HMGB2","STMN1","TUBA1B","PCNA"
]

inf_markers = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "OASL", "STAT1", "IRF7", "IFI6", "IFI44L"]
actin_remodel_markers = ["WAS","WIPF1","ARPC1B","ARPC2","ARPC3","ARPC5","CFL1","PFN1","CORO1A","DOCK2","VAV1"]
motility_adhesion_markers = ["ITGB1","ITGA4","ITGAL","ITGB2","VCL","PXN","TLN1","FERMT3","LCP1","MSN","EZR"]
lipid_flippase_markers = ["ATP8A1","ATP8B1","ATP11A","ATP11C","ABCA1","ABCG1","PLA2G6","PLD1","DGKA"]
def present(genes):
    g = [x for x in genes if x in adata.var_names]
    return g

naive_g = present(naive_markers)
cyto_g = present(cyto_markers)
tex_g = present(tex_markers)
inter_g = present(inter_markers)
trm_g = present(trm_markers)
cycle_g = present(cycle_markers)
inf_g = present(inf_markers)
actin_remodel_g = present(actin_remodel_markers)
motility_adhesion_g = present(motility_adhesion_markers)
lipid_flippase_g = present(lipid_flippase_markers)

print("[INFO] markers present:",
      {"naive": len(naive_g), "cyto": len(cyto_g), "tex": len(tex_g), "inter": len(inter_g), "cycle": len(cycle_g)})

# =========================
# Score genes
# =========================
# score_name 会写到 adata.obs
if len(naive_g) > 0:
    sc.tl.score_genes(adata, naive_g, score_name="score_naive", use_raw=False)
if len(cyto_g) > 0:
    sc.tl.score_genes(adata, cyto_g, score_name="score_cytotoxic", use_raw=False)
if len(tex_g) > 0:
    sc.tl.score_genes(adata, tex_g, score_name="score_Tex", use_raw=False)
if len(inter_g) > 0:
    sc.tl.score_genes(adata, inter_g, score_name="score_intermediate", use_raw=False)
if len(cycle_g) > 0:
    sc.tl.score_genes(adata, cycle_g, score_name="score_cycling", use_raw=False)
if len(trm_g) > 0:
    sc.tl.score_genes(adata, trm_g, score_name="score_TRM", use_raw=False)
if len(inf_g) > 0:
    sc.tl.score_genes(adata, inf_g, score_name="score_inf", use_raw=False)
if len(actin_remodel_g) > 0:
    sc.tl.score_genes(adata, actin_remodel_g, score_name="score_actin_remodel", use_raw=False)
if len(motility_adhesion_g) > 0:
    sc.tl.score_genes(adata, motility_adhesion_g, score_name="score_motility_adhesion", use_raw=False)
if len(lipid_flippase_g) > 0:
    sc.tl.score_genes(adata, lipid_flippase_g, score_name="score_lipid_flippase", use_raw=False)
score_cols = [c for c in ["score_naive","score_cytotoxic","score_Tex","score_intermediate","score_cycling","score_TRM","score_inf","score_actin_remodel","score_motility_adhesion","score_lipid_flippase"] if c in adata.obs.columns]
print("[INFO] score cols:", score_cols)

# =========================
# Violin plots (one figure)
# =========================
# Scanpy 会自己分面；为了可控我们用 multi_panel=True
sc.pl.violin(
    adata,
    keys=score_cols,
    groupby=cluster_col,
    rotation=0,
    stripplot=True,
    jitter=0.25,
    multi_panel=True,
    show=False
)
plt.tight_layout()
plt.savefig(fig_violin, dpi=220, bbox_inches="tight")
plt.close()
print("[OK] saved:", fig_violin)

# =========================
# Dotplot markers (更直观：每个cluster均值+表达比例)
# =========================
marker_dict = {
    "Naive": naive_g,
    "Cytotoxic": cyto_g,
    "Tex": tex_g,
    "Intermediate": inter_g,
    "Cycling": cycle_g,
    "TRM": trm_g,
    "Inf": inf_g,
    "Actin Remodel": actin_remodel_g,
    "Motility Adhesion": motility_adhesion_g,
    "Lipid Flippase": lipid_flippase_g,
}
# 去掉空的
marker_dict = {k:v for k,v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata,
    var_names=marker_dict,
    groupby=cluster_col,
    standard_scale="var",
    show=False
)
plt.tight_layout()
plt.savefig(fig_dotplot, dpi=220, bbox_inches="tight")
plt.close()
print("[OK] saved:", fig_dotplot)

print("[DONE]")
