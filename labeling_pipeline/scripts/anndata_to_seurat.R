library(Seurat)

if (!require("SeuratDisk", quietly = TRUE)) {
  remotes::install_github("mojaveazure/seurat-disk")
  library(SeuratDisk)
}

Convert(snakemake@input[[1]], dest = "h5seurat", overwrite = TRUE)
# To test if can successfully load h5seurat file
query.obj <- LoadH5Seurat(snakemake@output[[1]])