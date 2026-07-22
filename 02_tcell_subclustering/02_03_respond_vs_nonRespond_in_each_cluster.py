import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Font (match your pipeline)
mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ----------------------------
# Input
# ----------------------------
in_csv = (
    "/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/"
    "02_extracting_T_cells_and_clustering/results_drop_10-12-5-8-9__20260111_164341/"
    "POST_drop_recluster/01_leiden_T_0.6__by_Respond.csv"
)
outdir = os.path.dirname(in_csv)

# ----------------------------
# Read
# ----------------------------
df = pd.read_csv(in_csv, index_col=0)

# Ensure expected column order (if present)
cols_order = [c for c in ["Responder", "Non-Responder", "Unknown"] if c in df.columns]
df = df[cols_order] if cols_order else df

# Make sure clusters are strings and sorted naturally if possible
df.index = df.index.astype(str)

# ----------------------------
# Plot 1: Stacked barplot (counts)
# ----------------------------
ax = df.plot(kind="bar", stacked=True, figsize=(10, 5))
ax.set_xlabel("Leiden cluster (leiden_T_0.6)")
ax.set_ylabel("Number of cells")
ax.set_title("Cluster composition by Respond (counts)")
ax.legend(title="Respond", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()

out_png_counts = os.path.join(outdir, "fig_barplot_leiden_T_0.6_by_Respond_counts.png")
plt.savefig(out_png_counts, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", out_png_counts)

# ----------------------------
# Plot 2: Stacked barplot (proportions within each cluster)
# ----------------------------
df_prop = df.div(df.sum(axis=1), axis=0).fillna(0)

ax = df_prop.plot(kind="bar", stacked=True, figsize=(10, 5))
ax.set_xlabel("Leiden cluster (leiden_T_0.6)")
ax.set_ylabel("Fraction of cells")
ax.set_ylim(0, 1.0)
ax.set_title("Cluster composition by Respond (within-cluster proportion)")
ax.legend(title="Respond", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()

out_png_prop = os.path.join(outdir, "fig_barplot_leiden_T_0.6_by_Respond_proportion.png")
plt.savefig(out_png_prop, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", out_png_prop)
