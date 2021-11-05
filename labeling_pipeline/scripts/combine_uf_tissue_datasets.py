from pathlib import Path

import anndata
import pandas as pd
import numpy as np

# Load up Rahul Group's cell type annotations
annotations = pd.read_csv(snakemake.input[0])
annotations.set_index('cell_id', inplace=True)

# Load up HuBMAP metadata
info = pd.read_csv(snakemake.input[1])
info.dropna(axis=0, subset=['localPath'], inplace=True)
info['hexstring'] = info['localPath'].apply(lambda s: Path(str(s)).stem)
info.drop_duplicates(subset=['hexstring'], keep=False, inplace=True)
info.set_index('hexstring', inplace=True)

data_root = Path("data/expression/uf-20210409-094552")
# Load each dataset and add the metadata and annotations
with open(data_root / f"{snakemake.wildcards['tissue']}_datasets.txt", "r") as f:
    hexstrings = f.read().splitlines()
adatas = []
for hexstring in hexstrings:
    expr_path = data_root / hexstring / "expr.h5ad"
    adata = anndata.read_h5ad(expr_path)
    # Only keep the spliced layer, and delete the others to save disk space
    adata.X = adata.layers["spliced"]
    del adata.layers["spliced"]
    del adata.layers["unspliced"]
    del adata.layers["spliced_unspliced_sum"]
    
    # Annotate
    adata.obs.index.name = 'cell_id'
    adata.obs['hexstring'] = hexstring
    adata.obs['Dataset ID'] = info.loc[hexstring]['Dataset ID']
    adata.obs['Donor ID'] = info.loc[hexstring]['Donor ID']
    adata.obs['Dataset Name'] = info.loc[hexstring]['Dataset Name (optional)']
    adata.obs['Assay'] = info.loc[hexstring]['Assay']
    if info.loc[hexstring]['Dataset ID'] in annotations['dataset_id'].values:
        print('Has annotations')
        tmp = pd.merge(adata.obs, annotations, how='left', left_on=['cell_id', 'Dataset ID'], right_on=['cell_id', 'dataset_id'])
        assert(np.alltrue(tmp.index == adata.obs.index))
        adata.obs = tmp
    adatas.append(adata)

adata = anndata.concat(adatas)
adata.obs_names_make_unique()
print(adata.shape)
adata.obs.rename(columns={"annotation": "celltype"}, inplace=True)
adata.obs["tissue"] = snakemake.wildcards["tissue"]

adata.write(snakemake.output[0])
    
    