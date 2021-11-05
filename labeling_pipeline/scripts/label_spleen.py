import anndata

adata = anndata.read_h5ad(snakemake.input[0])
print(adata.shape)
adata = adata[adata.obs["celltype"].notna() & (adata.obs["celltype"] != "low-quality")]
print(adata.shape)
adata.write(snakemake.output[0])
    