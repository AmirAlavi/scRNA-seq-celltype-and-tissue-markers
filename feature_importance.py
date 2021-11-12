import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
from collections import defaultdict

import csv

import scanpy as sc
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

def load_tabula_muris_data(tissues, cell_types=None): # don't filter by cell types just yet, we want other cell types in there for the marker gene finding
    annotations = pd.read_csv("data/tabula_muris/annotations_facs.csv")
    annotations.set_index('cell', inplace=True)
    all_data = []
    for tiss in tissues:
        print(f"data/tabula_muris/FACS/{tiss}-counts.csv")
        cur_df = pd.read_csv(f"data/tabula_muris/FACS/{tiss}-counts.csv", index_col=0)
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

# Data processing
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

def subsample_data_by_tissues(adata, n_subsample=10000, bias_celltypes=None, bias_multiple=2.):
    subsets = []
    for tiss in adata.obs["tissue"].unique():
        tiss_idx = np.where(adata.obs["tissue"] == tiss)[0]
        if bias_celltypes is not None:
            probs = np.ones_like(tiss_idx, dtype=np.float32) / bias_multiple
            frequencies = [np.sum(adata.obs["celltype"][tiss_idx] == ct)  for ct in bias_celltypes]
            max_freq = max(frequencies)
            inverse_frequencies = [ (1. / freq * max_freq ) for freq in frequencies]
            for i, ct in enumerate(bias_celltypes):
                preferred_ct_idx = adata.obs["celltype"][tiss_idx] == ct
                probs[preferred_ct_idx] = inverse_frequencies[i]
            probs /= probs.sum()
            tiss_idx = np.random.choice(tiss_idx, size=n_subsample, replace=False, p=probs)
        else:
            tiss_idx = np.random.choice(tiss_idx, size=n_subsample, replace=False)
        tiss_subset = adata[tiss_idx]
        subsets.append(tiss_subset)
    subsets = anndata.concat(subsets)
    return subsets

def compute_feature_correlation(adata, verbose=True):
    X = adata.X.todense()
    
    if verbose:
        print("computing feature correlations...")
    corr = spearmanr(X).correlation
    # Force to be exactly symmetric, it's "close" to being symmetric
    # corr = np.tril(corr) + np.triu(corr.T, 1)
    corr = np.around(corr, 3)
    # dist_matrix = distancematrix(X, transpose=1, dist='s')
    if verbose:
        print("done.")
        print("computing hierarchical clustering...")
    # convert the redundant n*n square matrix form into a condensed nC2 array
    dist_array = ssd.squareform(1. - corr) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
    corr_linkage = hierarchy.ward(dist_array)
    if verbose:
        print("done.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    dendro = hierarchy.dendrogram(
        corr_linkage, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax1.grid('on', axis='y')
    ax1.set_yticks(np.arange(0, 10.1, 0.25))
    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()
    return corr, corr_linkage, (ax1, ax2)

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
    enz, symbol = convert_ensembl_to_entrez_and_symbol([ens.split('.')[0] for ens in adata.var_names])
    adata.var['enz'] = enz
    adata.var['symbol'] = symbol
    adata.var['ens'] = adata.var.index
    adata.var['symbol'].loc[adata.var['symbol'] == 'nan'] = adata.var['ens'].loc[adata.var['symbol'] == 'nan']
    # adata.var.set_index('symbol', inplace=True, verify_integrity=False)

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
    

# Sub-cluster selection
def get_feature_clustered_median_adata(adata, corr_linkage, k):
    cluster_ids = hierarchy.fcluster(corr_linkage, k, criterion='maxclust')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    X_medians = []
    var_names = []
    for cluster_id, members in cluster_id_to_feature_ids.items():
        feature_cluster_subset = adata[:, members].X
        feature_cluster_subset = feature_cluster_subset.todense()
        if feature_cluster_subset.shape[1] > 1:
            feature_cluster_median = np.median(feature_cluster_subset, axis=1, keepdims=True)
        else:
            feature_cluster_median = feature_cluster_subset
        X_medians.append(feature_cluster_median)
        var_names.append(cluster_id)
    X_medians = np.hstack(X_medians)
    adata_de_colineated_medians = anndata.AnnData(X=X_medians, obs=adata.obs, var=pd.DataFrame(index=var_names))
    return adata_de_colineated_medians, cluster_id_to_feature_ids, var_names

def get_clf_score(adata, var1_key='celltype', var1_subset=None, var2_key='tissue', min_label_count=100, verbose=False, random_state=None):
    if var1_subset is None:
        var1_subset = adata.obs[var1_key].unique()
    train_accuracies = []
    valid_accuracies = []
    for v1 in var1_subset:
        adata_sub = adata[adata.obs[var1_key] == v1]
        vc = adata_sub.obs[var2_key].value_counts()
        tissues_to_keep = vc[vc >= min_label_count].index
        adata_sub = adata_sub[adata_sub.obs[var2_key].isin(tissues_to_keep)]
        X = adata_sub.X
        y = adata_sub.obs[var2_key]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        train_accuracies.append(train_acc)
        valid_acc = clf.score(X_test, y_test)
        valid_accuracies.append(valid_acc)
        if verbose:
            print(f"\t\t{v1}")
            print(f"\t\ttissues: {tissues_to_keep}")
            print(adata_sub.shape)
            print(f"\t\ttrain acc: {train_acc}")
            print(f"\t\tvalid acc: {valid_acc}")
    avg_train_acc = np.mean(train_accuracies)
    avg_valid_acc = np.mean(valid_accuracies)
    if verbose:
        print(f"\tavg train acc: {avg_train_acc}")
        print(f"\tavg valid acc: {avg_valid_acc}")
    return avg_train_acc, avg_valid_acc

def degrade_features_wrapper_method(adata, corr_linkage, k_start, k_min, k_step, acc_drop_limit=0.05, var1_key='celltype', var1_subset=None, var2_key='tissue', min_label_count=100, verbose=True, random_state=None):
    k = k_start
    _, base_accuracy = get_clf_score(adata, var1_key=var1_key, var1_subset=var1_subset, var2_key=var2_key, verbose=verbose, random_state=random_state)
    if verbose:
        print(f"base_acc: {base_accuracy}")
    while k > k_min:
        adata_decolineated_medians, _, _ = get_feature_clustered_median_adata(adata, corr_linkage, k)
        train_acc, valid_acc = get_clf_score(adata_decolineated_medians, var1_key=var1_key, var1_subset=var1_subset, var2_key=var2_key, verbose=verbose, random_state=random_state)
        if verbose:
            print(f"{k}: {valid_acc}")
        delta = base_accuracy - valid_acc
        if delta > 0 and (delta / base_accuracy) > acc_drop_limit:
            break
        else:
            k -= k_step
    return k

# Permutation feature importance
def permutation_feature_importance(adata, n_repeats=5, var1_key='celltype', var1_subset=None, var2_key='tissue', verbose=False, random_state=None):
    if var1_subset is None:
        var1_subset = adata.obs[var1_key].unique()
    FI_dict = {}
    for v1 in var1_subset:
        adata_sub = adata[adata.obs[var1_key] == v1]
        X = adata_sub.X
        y = adata_sub.obs[var2_key]
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X, y)
        if verbose:
            print(v1)
            print("Accuracy on train data: {:.2f}".format(clf.score(X, y)))
        FI = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=random_state)
        FI_dict[v1] = FI
    FI_normalized = {}
    FI_orig = {}
    for v1 in var1_subset:
        FI_orig[v1] = FI_dict[v1].importances_mean
        FI = FI_dict[v1].importances_mean
        FI_other_sum = np.zeros_like(FI)
        for v1_other in var1_subset:
            if v1 == v1_other:
                continue
            FI_other = FI_dict[v1_other].importances_mean
            FI_other_sum += FI_other
        FI_other = FI_other_sum / (len(var1_subset) - 1)
        FI_normalized[v1] = FI - FI_other
    return FI_normalized, FI_orig

def build_long_form_FI_dataframe(FI_normalized, FI_orig, selected_clusters):
    col_FI_norm = []
    col_FI_orig = []
    col_celltype = []
    col_feature_id = []
    for celltype in FI_normalized.keys():
        for i in range(len(FI_normalized[celltype])):
            col_feature_id.append(selected_clusters[i])
            col_celltype.append(celltype)
            col_FI_orig.append(FI_orig[celltype][i])
            col_FI_norm.append(FI_normalized[celltype][i])
    df = pd.DataFrame(data = {'Feature ID': col_feature_id, 'Cell type': col_celltype, 'FI Normalized': col_FI_norm, 'FI Original': col_FI_orig})
    return df

def assign_cross_variate_important_features(fi_df, score_col='FI Original'):
    fi_df.drop(columns='FI Normalized', inplace=True)
    fi_df_wide = fi_df.pivot_table(index='Feature ID', columns='Cell type', values=score_col)
    fi_df_wide = fi_df_wide.loc[~(fi_df_wide==0).all(axis=1)]
    fi_df_wide['Top cell type'] = fi_df_wide.idxmax(axis=1)
    fi_df_wide['Top cell type score'] = fi_df_wide.max(axis=1)
    return fi_df_wide
        

def print_feature_importance_results(original_var_names, FI_normalized, FI_orig):
    for ct in FI_normalized.keys():
        print(ct)
        print(FI_normalized[ct].shape)
        sort_idx = np.argsort(FI_normalized[ct])[::-1]
        print(FI_normalized[ct][sort_idx][:10])
        print(FI_orig[ct][sort_idx][:10])
#         for g in original_var_names[sort_idx][:25]:
#             print(g)
        print()

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