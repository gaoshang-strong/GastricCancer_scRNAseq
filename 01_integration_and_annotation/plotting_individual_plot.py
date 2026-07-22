import os
import numpy as np
#import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
# ====== 输出目录（按你之前的习惯） ======
all_cells_path = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/02_adata_T_reclustered_after_drop.h5ad"
adata = sc.read_h5ad(all_cells_path)
OUTDIR = "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/00_individual_marker_plotting/T_cell"
os.makedirs(OUTDIR, exist_ok=True)


GROUP_COL = "leiden_T_0.6"         # e.g., "cluster" or "celltype" (Cytolytic/Exhaustion/TRM/Treg)
SPLIT_COL = "Respond"        # e.g., "Responder"/"Non-responder" column

# Markers (edit if some genes are missing in your dataset)
tpex_genes = [
    "TCF7", "LEF1", "IL7R", "CCR7", "SELL", "SLAMF6",
    "BCL6", "CXCR5", "IKZF1", "SATB1", "LTB", "MALAT1",
    "CD27", "CD28", "TRAC"
]

# Terminal exhausted CD8 T (checkpoint-high, dysfunction/terminalization)
terminal_tex_genes = [
    "TOX", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "ENTPD1", "CXCL13",
    "LAYN", "BATF", "IRF4", "PRDM1", "TNFRSF9", "CTLA4", "CD160",
    "KLRG1"  # optional; sometimes effector/terminal marker
]
cytolytic_genes = [
    "NKG7", "PRF1", "GZMB", "GNLY", "IFNG",
    "GZMH", "GZMA", "CTSW", "XCL1", "XCL2",
    "CX3CR1", "CCL5", "TRBC1", "TRBC2"
]

genesets = {
    "Tpex_like": tpex_genes,
    "Terminal_Tex": terminal_tex_genes,
    "Cytolytic": cytolytic_genes,
}

# Keep only genes that exist in this adata
varnames = set(adata.var_names.astype(str).tolist())
genesets_present = {k: [g for g in v if g in varnames] for k, v in genesets.items()}

print("Genes present:")
for k, v in genesets_present.items():
    print(f"  {k}: {len(v)} / {len(genesets[k])} -> {v}")

# Ensure grouping columns exist
for col in [GROUP_COL, SPLIT_COL]:
    if col not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{col}']. Available columns: {list(adata.obs.columns)[:30]} ...")

# Make sure expression is log-normalized for plotting (skip if already)
# If your object is raw counts, uncomment the next 3 lines:
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# adata.raw = adata

# Option A: Violin plots per gene set (genes as multiple panels)
for tag, genes in genesets_present.items():
    if len(genes) == 0:
        continue

    tag_dir = os.path.join(OUTDIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    for g in genes:
        sc.pl.violin(
            adata,
            keys=g,               # single gene
            groupby=GROUP_COL,
            layer="log1p_norm",
            jitter=0.25,
            rotation=45,
            show=False,
        )
        #plt.tight_layout()
        out = os.path.join(tag_dir, f"violin_{g}_by_{GROUP_COL}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", out)

TOP_FRAC = 0.005  # top 0.5%

# 1) Build a capped expression matrix ONLY for scoring (do not affect gene violins)
# Use the same layer you intended for scoring (you used layer="log1p_norm" in plots)
LAYER_FOR_SCORING = "norm_expr"
if LAYER_FOR_SCORING not in adata.layers:
    raise ValueError(f"adata.layers['{LAYER_FOR_SCORING}'] not found. Available: {list(adata.layers.keys())}")

X = adata.layers[LAYER_FOR_SCORING]
if hasattr(X, "toarray"):
    X = X.toarray()
else:
    X = np.asarray(X)

caps = np.quantile(X, 1.0 - TOP_FRAC, axis=0)   # per-gene 99.9% quantile across all cells
X_cap = np.minimum(X, caps[None, :])

# Store to a temp layer
CAP_LAYER = f"{LAYER_FOR_SCORING}_cap_top0p1pct_for_score"
adata.layers[CAP_LAYER] = X_cap

# 2) Compute module scores using the capped matrix
# score_genes has no layer argument -> temporarily point adata.X to CAP_LAYER
X_backup = adata.X
adata.X = adata.layers[CAP_LAYER]

for tag, genes in genesets_present.items():
    if len(genes) < 3:
        continue
    sc.tl.score_genes(adata, gene_list=genes, score_name=f"score_{tag}", use_raw=False)

# restore
adata.X = X_backup

# 3) Plot module score violins (scores are already capped via input expression)
sc.pl.violin(
    adata,
    keys=[f"score_{k}" for k in genesets_present.keys() if f"score_{k}" in adata.obs.columns],
    groupby=GROUP_COL,
    # splitby=SPLIT_COL,  # keep commented if your scanpy doesn't support it
    jitter=0.25,
    rotation=45,
    multi_panel=True,
    show=False,
)
#plt.tight_layout()
out = os.path.join(OUTDIR, f"violin_module_scores_by_{GROUP_COL}_cap_top0p1pct.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", out)