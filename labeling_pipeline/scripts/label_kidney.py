import anndata
import pandas as pd

from common import add_labels


adata = anndata.read_h5ad(snakemake.input[0])
adata.X = adata.layers["spliced"]
del adata.layers["spliced"]
del adata.layers["unspliced"]
del adata.layers["spliced_unspliced_sum"]
labels = pd.read_csv(snakemake.input[1], sep="\t")
labels = labels[labels["singlr_clusters_pruned"].notna()]
adata = add_labels(adata, labels, "singlr_clusters_pruned")
adata.obs["tissue"] = "kidney"
adata.write(snakemake.output[0])