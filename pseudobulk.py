import anndata as ad, numpy as np, pandas as pd, scanpy as sc, os, gc, psutil
# Anndata format:
# anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html
# Use CSR rather than CSC for faster pseudobulking:
# e.g. pseudobulking AD took 32 hours with CSC but only 31 minutes with CSR

################################################################################
# Load Alzheimer's data
# Data: portal.brain-map.org/explore/seattle-alzheimers-disease
# Metadata columns: brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.
#                   net/filer_public/b0/a8/b0a81899-7241-4814-8c1f-4a324db51b0b/
#                   sea-ad_donorindex_dataelementbriefdescriptions.pdf
################################################################################

# explictly set working directory 
os.chdir('/nethome/kcni/karbabi/r_projects/reverse_signatures')
# available RAM
print(psutil.virtual_memory().available / (1024 ** 3))

AD_data_file = 'SEA-AD/SEA-AD.h5ad'
if os.path.exists(AD_data_file):
    AD_data = sc.read(AD_data_file)
else:
    print('Preprocessing AD data...')
    from scipy.sparse import vstack
    AD_cell_types = pd.Index(['Astro', 'Chandelier', 'Endo', 'L23_IT', 'L4_IT',
                              'L56_NP', 'L5_ET', 'L5_IT', 'L6_CT', 'L6_IT',
                              'L6_IT_Car3', 'L6b', 'Lamp5', 'Lamp5_Lhx6',
                              'Micro-PVM', 'OPC', 'Oligo', 'Pax6', 'Pvalb',
                              'Sncg', 'Sst', 'Sst_Chodl', 'VLMC', 'Vip'])
    AD_data_per_cell_type = {}
    for cell_type in AD_cell_types:
        print(f'Loading {cell_type}...')
        AD_data_per_cell_type[cell_type] = sc.read(f'SEA-AD/{cell_type}.h5ad')
        # raw counts
        counts = AD_data_per_cell_type[cell_type].raw.X
        del AD_data_per_cell_type[cell_type].raw
        # Convert to int32, but make sure all entries were integers first
        counts_int32 = counts.astype(np.int32)
        assert not (counts != counts_int32).nnz
        AD_data_per_cell_type[cell_type].X = counts_int32
        AD_data_per_cell_type[cell_type].var = \
            AD_data_per_cell_type[cell_type].var\
                .reset_index()\
                .astype({'feature_name': str})\
                .set_index('feature_name')\
                [['gene_ids']]\
                .rename_axis('gene')\
                .rename(columns={'gene_ids': 'Ensembl_ID'})
        del AD_data_per_cell_type[cell_type].uns
        del AD_data_per_cell_type[cell_type].obsm
        del AD_data_per_cell_type[cell_type].obsp
        gc.collect()
    # Alternative to "ad.concat(AD_data_per_cell_type.values(), merge='same')";
    # I suspect my version is more memory-efficient, but not sure
    print('Merging...')
    X = vstack([AD_data_per_cell_type[cell_type].X
                for cell_type in AD_cell_types])
    obs = pd.concat([AD_data_per_cell_type[cell_type].obs
                     for cell_type in AD_cell_types])
    var = AD_data_per_cell_type[AD_cell_types[0]].var
    assert all(var.equals(AD_data_per_cell_type[cell_type].var)
               for cell_type in AD_cell_types[1:])
    AD_data = ad.AnnData(X=X, obs=obs, var=var, dtype=X.dtype)
    # Join with external metadata file; handle two columns with mixed numbers
    # and strings ("Age of Death" and "Fresh Brain Weight"); add broad cell type
    print('Joining with external metadata...')
    AD_metadata = pd.read_excel('SEA-AD/sea-ad_cohort_donor_metadata.xlsx')
    AD_data.obs = AD_data.obs\
        .assign(broad_cell_type=lambda df: np.where(
            df.Class == 'Non-neuronal and Non-neural', df.Subclass, np.where(
            df.Class == 'Neuronal: Glutamatergic', 'Excitatory', np.where(
            df.Class == 'Neuronal: GABAergic', 'Inhibitory', np.nan))))\
        .reset_index()\
        .merge(AD_metadata.rename(columns=lambda col: f'Metadata: {col}'),
               left_on='Donor ID', right_on='Metadata: Donor ID', how='left')\
        .set_index('exp_component_name')\
        .drop('Metadata: Donor ID', axis=1)\
        .assign(**{'Metadata: Age at Death': lambda df:
                       df['Metadata: Age at Death']
                       .replace({'90+': 90})
                       .astype('Int64'),
                   'Metadata: Fresh Brain Weight': lambda df:
                       df['Metadata: Fresh Brain Weight']
                       .replace({'Unavailable': pd.NA})
                       .astype('Int64')})
    print('Saving...')
    # noinspection PyTypeChecker
    AD_data.write(AD_data_file)

################################################################################
# Deal with NAs in covariates
# QC cells (we'll QC genes after pseudobulking)
################################################################################

AD_data.obs = AD_data.obs\
    .assign(**{'APOE4 status': lambda df: df['APOE4 status'].eq('Y'),
               'Metadata: PMI': lambda df: df['Metadata: PMI'].fillna(
                   df['Metadata: PMI'].median()),
               'Study: ACT': lambda df:
                   df['Metadata: Primary Study Name'].eq('ACT'),
               'Study: ADRC Clinical Core': lambda df:
                   df['Metadata: Primary Study Name'].eq('ADRC Clinical Core')})

""" # Filter to cells with >= 200 genes detected
AD_data.obs['Genes detected'].min() # 1001
AD_data = sc.pp.filter_cells(AD_data, min_genes=200)

# Filter to cells with < 5% mitochondrial reads
mitochondrial_genes = AD_data.var.index.str.startswith('MT-')
percent_mito = AD_data[:, mitochondrial_genes].X.sum(axis=1).A1 / \
                AD_data.X.sum(axis=1).A1
AD_data = AD_data[percent_mito < 0.05] # 3 cells  """

################################################################################
# Aggregate to pseudobulk and save
################################################################################

def pseudobulk(dataset, ID_column, cell_type_column):
    # Use observed=True to skip groups where any of the columns is NaN
    groupby = [ID_column, cell_type_column]
    grouped = dataset.obs.groupby(groupby, observed=True)
    # Fill with 0s to avoid auto-conversion to float when filling with NaNs
    pseudobulk = pd.DataFrame(
        0, index=pd.MultiIndex.from_frame(
            grouped.size().rename('num_cells').reset_index()),
        columns=dataset.var_names, dtype=dataset.X.dtype)
    for row_index, group_indices in enumerate(grouped.indices.values()):
        group_counts = dataset[group_indices].X
        pseudobulk.values[row_index] = group_counts.sum(axis=0).A1
    metadata = grouped.first().loc[pseudobulk.index, grouped.nunique().le(1).all()]
    metadata['num_cells'] = dataset.obs.groupby(groupby).size()
    return pseudobulk, metadata

pseudobulk, metadata = pseudobulk(AD_data, ID_column = 'Donor ID', cell_type_column = 'Subclass')
pseudobulk.to_csv('pseudobulk_subclass.csv', index=True)
metadata.to_csv('pseudobulk_subclass_meta.csv', index=True) 
