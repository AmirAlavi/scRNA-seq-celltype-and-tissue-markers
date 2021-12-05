import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
from collections import defaultdict

import csv

import scanpy as sc
import diffxpy.api as de
import pandas as pd
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

from Bio.Cluster import distancematrix

import mygene

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# Data loading
def load_hubmap_data(data_root, verbose=False):
    if not isinstance(data_root, Path):
        data_root = Path(data_root)
    adata = []
    for tissue in ["kidney", "lung", "lymph_node", "spleen", "thymus"]:
        cur_adata = anndata.read_h5ad(data_root / tissue / "labeled.h5ad")
        cur_adata.var["original"] = cur_adata.var_names
        cur_adata.var_names = [ens.split('.')[0] for ens in cur_adata.var_names]
        if verbose:
            print(tissue)
            print(cur_adata.shape)
        sc.pp.filter_cells(cur_adata, min_genes=200)
        sc.pp.filter_genes(cur_adata, min_cells=3)
        if verbose:
            print(cur_adata.shape)
            print()
        adata.append(cur_adata)
    adata = anndata.concat(adata)
    if verbose:
        print(adata.shape)
    return adata

def load_van_der_Wijst_PBMC_data(data_root, verbose=False):
    if not isinstance(data_root, Path):
        data_root = Path(data_root)
    pbmc = []
    for lane in range(1, 9):
        cur_adata = sc.read_10x_mtx(data_root / f"lane_{lane}", var_names='gene_ids', make_unique=False)
        cur_adata.obs["lane"] = lane
        cur_adata.obs_names = [f"{barcode.split('-')[0]}_lane{lane}" for barcode in cur_adata.obs_names]
        if verbose:
            print(lane)
            print(cur_adata.shape)
        sc.pp.filter_cells(cur_adata, min_genes=200)
        sc.pp.filter_genes(cur_adata, min_cells=3)
        if verbose:
            print(cur_adata.shape)
            print()
        pbmc.append(cur_adata)
    pbmc = anndata.concat(pbmc)
    if verbose:
        print(pbmc.shape)
        
    label_df = pd.read_csv(data_root / "../barcodes_to_cell_types/barcodes_to_cell_types.tsv", sep='\t', index_col=0)
    idx = label_df.index.intersection(pbmc.obs_names)
    pbmc = pbmc[idx, :]
    pbmc.obs['celltype'] = ""
    pbmc.obs.loc[idx, 'celltype'] = label_df.loc[idx, 'cell_type']
    pbmc.obs['tissue'] = 'PBMC'
    return pbmc

def unify_celltype_labels(adata):
    # The celltype labels in adata come from independent studies,
    # which used different cell type names and granularity.
    # Here, we unify and merge the common cell type names to increase
    # the number of samples of each immune cell type.
    B_cell_name = 'B Cell'
    T_cell_name = 'T Cell'
    NK_cell_name = 'NK Cell'
    Macrophage_cell_name = "Macrophage"
    Fibroblast_cell_name = "Fibroblast"

    unify_map = {
        'B': B_cell_name,
        'B cell': B_cell_name,
        'B_Plasma': B_cell_name,
        'T': T_cell_name,
        'alpha-beta T cell': T_cell_name,
        'gamma-delta T cell': T_cell_name,
        'CD4+ T': T_cell_name,
        'CD8+ T': T_cell_name,
        'T_Cytotoxic': T_cell_name,
        'T_Regulatory': T_cell_name,
        'NK': NK_cell_name,
        'natural killer cell': NK_cell_name,
        'CD56(bright) NK': NK_cell_name,
        'CD56(dim) NK': NK_cell_name,
        'macrophage': Macrophage_cell_name,
        'Macrophage_Alveolar': Macrophage_cell_name,
        'splenic fibroblast': Fibroblast_cell_name
    }

    adata.obs["celltype"].replace(to_replace=unify_map, inplace=True)

def load_all_human_data(hubmap_data_root="data/label_pipeline_results", pbmc_data_root="data/van_der_Wijst_PBMC/count_matrices_per_lane/"):
    # Get a single adata object that contains all of the scRNA-seq data used
    # in the paper, with labels in .obs['tissue'] and .obs['celltype']
    # Uses the common set of genes across all data.
    data_root = Path(hubmap_data_root)
    adata = load_hubmap_data(data_root)
    data_root = Path(pbmc_data_root)
    adata = anndata.concat([adata, load_van_der_Wijst_PBMC_data(data_root)])
    finalize_adata(adata)
    print(adata.shape)
    add_alt_gene_IDs(adata)
    unify_celltype_labels(adata)
    return adata

def load_tabula_muris_data(data_root, tissues, cell_types=None): # don't filter by cell types just yet, we want other cell types in there for the marker gene finding
    data_root = Path(data_root)
    annotations = pd.read_csv(data_root / "annotations_facs.csv")
    annotations.set_index('cell', inplace=True)
    all_data = []
    for tiss in tissues:
        print(data_root / f"FACS/{tiss}-counts.csv")
        cur_df = pd.read_csv(data_root / f"FACS/{tiss}-counts.csv", index_col=0)
        cur_df = cur_df.T
        common_idx = cur_df.index.intersection(annotations.index)
        cur_meta = annotations.loc[common_idx]
        cur_df = cur_df.loc[common_idx]
        cur_adata = anndata.AnnData(X=cur_df, obs=cur_meta)
        print(cur_adata.shape)
        if cell_types is not None:
            cur_adata = cur_adata[cur_adata.obs['cell_ontology_class'].isin(cell_types)]
        print(cur_adata.shape)
        all_data.append(cur_adata)
    all_data = anndata.concat(all_data)
    all_data.obs.rename(columns={'cell_ontology_class': 'celltype'}, inplace=True)
    nans = all_data.obs['celltype'].isna()
    all_data = all_data[~nans]
    all_data.var['symbol'] = all_data.var.index
    return all_data

def finalize_adata(adata):
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["tissue"] = adata.obs["tissue"].astype("category")
    adata.obs_names_make_unique()

# Data pre-processing
def filter_hvg(adata, verbose=False):
    log_normed = sc.pp.log1p(adata, copy=True)
    if verbose:
        print("log normed")
    sc.pp.highly_variable_genes(log_normed)
    if verbose:
        print("computed hvg")
    adata.var['highly_variable'] = log_normed.var['highly_variable']
    adata.var['dispersions_norm'] = log_normed.var['dispersions_norm']
    highly_variable = adata.var.index[adata.var['highly_variable'] == True]
    return adata[:, highly_variable], highly_variable

# Gene set enrichment analysis
GO_BP_GENE_SET_FILE = 'c5.go.bp.v7.4.symbols.gmt'
go_bp_rs = de.enrich.RefSets(fn=GO_BP_GENE_SET_FILE)

def clean_up_enrich_table_for_printing(table, term_set_name='GO'):
    table['set'] = table.apply(lambda row: ' '.join(row['set'].split('_')[1:]), axis=1)
    table.rename(columns={'qval': 'Corrected p-val', 'set': f'{term_set_name} term'}, inplace=True)
    return table

def do_enrich(gene_list, gene_scores, background, save_file):
    max_score = np.amax(gene_scores)
    print(max_score)
    enr = de.enrich.test(ref=go_bp_rs, scores=gene_scores, gene_ids=gene_list, clean_ref=True, all_ids=background, threshold = max_score + 1.)
    enr_table = enr.summary().loc[enr.summary()['qval'] < 0.05]
    if enr_table.shape[0] > 0:
        print(enr_table.head(n=20))
        enr_table = enr_table.head(n=20)[['set', 'qval']]
        enr_table = clean_up_enrich_table_for_printing(enr_table, 'GO')
        enr_table.to_latex(save_file, index=False)
    else:
        print('NONE SIGNIFICANT')
# Utils
def convert_ensembl_to_entrez_and_symbol(ens):
    mg = mygene.MyGeneInfo()
    print('querying mygene...')
    result = mg.querymany(ens, scopes='ensemblgene', fields='entrezgene,symbol', species='human', verbose=False)
    enz = []
    symbol = []
    for r in result:
        # entrez
        if 'entrezgene' in r:
            enz.append(r['entrezgene'])
        else:
            enz.append(np.nan)
        if 'symbol' in r:
            symbol.append(r['symbol'])
        else:
            symbol.append(np.nan)
    return np.array(enz), np.array(symbol)

def add_alt_gene_IDs(adata):
    # Add alternative gene IDs to the adata object. Useful for downstream functional
    # analyses (e.g. DE analysis)
    enz, symbol = convert_ensembl_to_entrez_and_symbol([ens.split('.')[0] for ens in adata.var_names])
    adata.var['enz'] = enz
    adata.var['symbol'] = symbol
    adata.var['ens'] = adata.var.index
    adata.var['symbol'].loc[adata.var['symbol'] == 'nan'] = adata.var['ens'].loc[adata.var['symbol'] == 'nan']

def get_celltype_and_tissue_frequencies(adata):
    ct_tissue_freq = adata.obs.value_counts(["celltype", "tissue"])
    ct_tissue_freq = ct_tissue_freq.unstack(level=1)
    ct_tissue_freq.fillna(0., inplace=True)
    ct_tissue_freq = ct_tissue_freq.astype(int)
    ct_tissue_freq.columns = ct_tissue_freq.columns.astype(str)
    ct_tissue_freq['num_tissues'] = (ct_tissue_freq > 0).sum(axis=1)
    ct_tissue_freq.sort_values(by="num_tissues", ascending=False, inplace=True)
    ct_tissue_freq.loc['num_celltypes'] = (ct_tissue_freq > 0).sum(axis=0)
    ct_tissue_freq.sort_values(by="num_celltypes", axis=1, ascending=False, inplace=True)
    return ct_tissue_freq

def visualize(adata, tissue_col='tissue', celltype_col='celltype', celltypes_highlight=None, save=None):
    vis_adata = adata.copy()
    sc.pp.normalize_total(vis_adata)
    sc.pp.scale(vis_adata)
    if celltypes_highlight is not None:
        vis_adata.obs.loc[~vis_adata.obs[celltype_col].isin(celltypes_highlight), celltype_col] = 'Other'
    sc.tl.pca(vis_adata)
    sc.pp.neighbors(vis_adata)
    sc.tl.umap(vis_adata)
    sc.pl.umap(vis_adata, color=[tissue_col, celltype_col], save=save)


# DE testing
def add_alt_gene_identifiers_to_df(df, adata):
    symbol_col = []
    enz_col = []
    for gene in df['names']:
        idx = np.where(adata.var_names == gene)[0][0]
        symbol = adata.var['symbol'][idx]
        entrez = adata.var['enz'][idx]
        symbol_col.append(symbol)
        enz_col.append(entrez)
    df['symbol'] = symbol_col
    df['entrez'] = enz_col

def DE_on_counts(count_data, groupby, celltype, gene_clust_id, gene_clust_rank, test='wilcoxon'):
    sc.tl.rank_genes_groups(count_data, groupby=groupby, n_genes=count_data.shape[1], method='wilcoxon')
    de_dfs = {}
    for group in count_data.obs[groupby].unique():
        df = sc.get.rank_genes_groups_df(count_data, group=group)
        df[groupby] = group
        df['celltype'] = celltype
        df['gene_clust_id'] = gene_clust_id
        df['gene_clust_rank'] = gene_clust_rank
        add_alt_gene_identifiers_to_df(df, count_data)
        cols_ordered = ['celltype', groupby, 'gene_clust_rank', 'gene_clust_id', 'scores', 'names', 'entrez', 'symbol', 'logfoldchanges', 'pvals', 'pvals_adj']
        df = df[cols_ordered]
        df.rename(columns={'names': 'ensembl'}, inplace=True)
        de_dfs[group] = df
    return de_dfs

def DE_on_important_features(adata, FI_normalized, selected_clusters, cluster_id_to_feature_ids, out_file, min_label_count=100):
    first_df = True
    for ct in FI_normalized.keys():
        print(ct)
        print()
        FI = FI_normalized[ct]
        sort_idx = np.argsort(FI)[::-1]
        selected_clusters = np.array(selected_clusters)
        sorted_cluster_ids = selected_clusters[sort_idx]
        for i in range(3):
            members = cluster_id_to_feature_ids[sorted_cluster_ids[i]]
            ct_and_feat_adata = adata[adata.obs['celltype'] == ct, members]
            vc = ct_and_feat_adata.obs["tissue"].value_counts()
            tissues_to_keep = vc[vc >= min_label_count].index
            ct_and_feat_adata = ct_and_feat_adata[ct_and_feat_adata.obs["tissue"].isin(tissues_to_keep)]
            print(ct_and_feat_adata.shape)
            de_dfs = DE_on_counts(ct_and_feat_adata, 'tissue', ct, sorted_cluster_ids[i], i + 1)
            for tiss, df in de_dfs.items():
                print(tiss)
                print(df[df['pvals_adj'] < 0.05])
                if first_df:
                    df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=True)
                    first_df = False
                else:
                    df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=False, mode='a')
                print()
            print()
        print()
    
def DE_on_cross_variate_important_features(adata, fi_df_wide, cluster_id_to_feature_ids, out_file, min_label_count=100):
    first_df = True
    for ct in fi_df_wide["Top cell type"].unique():
        # find the top 3 features for this cell type
        fi_df_ct = fi_df_wide[fi_df_wide['Top cell type'] == ct]
        fi_df_ct.sort_values(by='Top cell type score', axis=0, ascending=False, inplace=True)
        for i in range(3):
            feature = fi_df_ct.index[i]
            members = cluster_id_to_feature_ids[feature]
            ct_and_feat_adata = adata[adata.obs['celltype'] == ct, members]
            vc = ct_and_feat_adata.obs["tissue"].value_counts()
            tissues_to_keep = vc[vc >= min_label_count].index
            ct_and_feat_adata = ct_and_feat_adata[ct_and_feat_adata.obs["tissue"].isin(tissues_to_keep)]
            print(ct_and_feat_adata.shape)
            de_dfs = DE_on_counts(ct_and_feat_adata, 'tissue', ct, feature, i + 1)
            for tiss, df in de_dfs.items():
                print(tiss)
                print(df[df['pvals_adj'] < 0.05])
                if first_df:
                    df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=True)
                    first_df = False
                else:
                    df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=False, mode='a')
                print()
            print()
        print()
            
            
def vanilla_DE(adata, celltypes, out_file, min_label_count=100):
    first_df = True

    for ct in celltypes:
        print(ct)
        ct_adata = adata[adata.obs['celltype'] == ct]
        print(ct_adata.shape)
        vc = ct_adata.obs["tissue"].value_counts()
        tissues_to_keep = vc[vc >= min_label_count].index
        ct_adata = ct_adata[ct_adata.obs["tissue"].isin(tissues_to_keep)]
        de_dfs = DE_on_counts(ct_adata, 'tissue', ct, 'NA', 'NA')
        for tiss, df in de_dfs.items():
            print(tiss)
            print(df[df['pvals_adj'] < 0.05])
            if first_df:
                df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=True)
                first_df = False
            else:
                df[df['pvals_adj'] < 0.05].to_csv(out_file, index=False, header=False, mode='a')
            print()
        print()