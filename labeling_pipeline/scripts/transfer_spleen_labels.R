library(Seurat)

if (!require("SeuratDisk", quietly = TRUE)) {
  remotes::install_github("mojaveazure/seurat-disk")
  library(SeuratDisk)
}

NormAndHVG <- function(data) {
  data <- NormalizeData(data)
  data <- FindVariableFeatures(data)
  return(data)
}

TransferAnnotations <- function(ref, query, add.cell.ids = NULL, project = "SeuratProject") {
  ref <- NormAndHVG(ref)
  query <- NormAndHVG(query)
  transfer.anchors <- FindTransferAnchors(reference = ref, query = query, dims = 1:30)
  annotation.predicted <- TransferData(anchorset = transfer.anchors, refdata = ref$celltype, dims = 1:30)
  return(annotation.predicted)
}

ref.obj <- LoadH5Seurat(snakemake@input[[1]])
Convert(snakemake@input[[2]], dest = "h5seurat", overwrite = TRUE)
query.obj <- LoadH5Seurat(snakemake@output[[1]])

annotation.predicted <- TransferAnnotations(ref.obj, query.obj, c("spleen", snakemake@wildcards$tissue))
query.obj <- AddMetaData(query.obj, metadata = annotation.predicted)
query.obj$celltype <- query.obj$predicted.id

# Necessary because otherwise the Convert function serializes factors as ints
query.obj$tissue <- as.character(query.obj$tissue)

SaveH5Seurat(query.obj, filename = snakemake@output[[1]], overwrite = TRUE)
Convert(snakemake@output[[1]], dest = snakemake@output[[2]])