def add_labels(adata, label_df, label_col):
    # Subset to only the cells for which we have labels
    idx = adata.obs.index.intersection(label_df.index)
    adata = adata[idx, :]
    adata.obs['celltype'] = ""
    adata.obs.loc[idx, 'celltype'] = label_df.loc[idx, label_col]
    return adata