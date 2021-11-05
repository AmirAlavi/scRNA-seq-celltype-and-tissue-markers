import anndata
import pandas as pd

from common import add_labels


adata = anndata.read_h5ad(snakemake.input[0])
adata.X = adata.layers["spliced"]
del adata.layers["spliced"]
del adata.layers["unspliced"]
del adata.layers["spliced_unspliced_sum"]
labels = pd.read_csv(snakemake.input[1], sep="\t", index_col=0)
labels = labels[labels["Manuscript_Identity"].notna()]
adata = add_labels(adata, labels, "Manuscript_Identity")
adata.obs["tissue"] = "lung"
adata.write(snakemake.output[0])