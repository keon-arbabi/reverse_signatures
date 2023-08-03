import anndata as ad, numpy as np, pandas as pd, scanpy as sc,  os, gc
os.chdir('/nethome/kcni/karbabi/r_projects/reverse_signatures/scripts')
from utils import Timer

os.chdir('/nethome/kcni/karbabi/stlab')

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

AD_data_file = 'SEA-AD/SEA-AD.h5ad'
if os.path.exists(AD_data_file):
    with Timer('Loading AD data'):
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

gc.collect()

################################################################################
# Load Parkinson's data (10 PD/LBD individuals and 8 neurotypical controls)
# Data: singlecell.broadinstitute.org/single_cell/study/SCP1768
# QC procedure: nature.com/articles/s41593-022-01061-1#Sec28
################################################################################

PD_data_file = 'Macosko/Macosko.h5ad'
if os.path.exists(PD_data_file):
    with Timer('Loading PD data'):
        PD_data = sc.read(PD_data_file)
else:
    print('Preprocessing PD data...')
    from scipy.io import mmread  # could also use sc.read_10x_mtx()
    PD_counts = mmread('Macosko/Homo_matrix.mtx.gz').T.tocsr().astype(np.int32)
    PD_genes = pd.read_table('Macosko/Homo_features.tsv.gz', usecols=[0],
                             index_col=0, header=None, names=[None]).index
    PD_cell_types = pd.concat([
        pd.read_table(f'Macosko/{cell_type}_UMAP.tsv', skiprows=[1],
                      usecols=['NAME', 'Cell_Type'], index_col='NAME')\
            .assign(broad_cell_type=lambda df: df.Cell_Type
                    .str.split('_').str[0]
                    .replace({'Astro': 'Astrocyte', 'CALB1': 'Dopaminergic',
                              'Endo': 'Endothelial', 'Ependyma': 'Astrocyte',
                              'Ex': 'Excitatory', 'Inh': 'Inhibitory',
                              'Macro': 'Microglia-PVM', 'MG': 'Microglia-PVM',
                              'Olig': 'Oligodendrocyte', 'OPC': 'OPC',
                              'SOX6': 'Dopaminergic'}))
            for cell_type in ('astro', 'da', 'endo', 'mg', 'nonda',
                              'olig', 'opc')])\
        .astype('category')  # sort cats with pd.CategoricalDtype(ordered=True)
    PD_metadata = pd.read_table('Macosko/METADATA_PD.tsv.gz', skiprows=[1],
                                index_col='NAME', low_memory=False)\
        .loc[pd.read_table('Macosko/Homo_bcd.tsv.gz', usecols=[0], index_col=0,
                           header=None, names=[None]).index]\
        .astype('category')\
        .astype({'Donor_Age': int, 'Donor_PMI': float})\
        .join(PD_cell_types)
    assert len(PD_genes) == 41625, len(PD_genes)
    assert len(PD_metadata) == 434354, len(PD_metadata)
    assert (PD_num_typed_cells := PD_metadata.Cell_Type.notna().sum()) == \
           387483, PD_num_typed_cells
    assert PD_counts.shape == (len(PD_metadata), len(PD_genes)), PD_counts.shape
    PD_data = ad.AnnData(PD_counts, obs=PD_metadata, var=pd.DataFrame(
        index=PD_genes), dtype=PD_counts.dtype)
    # noinspection PyTypeChecker
    PD_data.write(PD_data_file)

################################################################################
# Load schizophrenia data
# SZBDMulti-Seq:
# - medrxiv.org/content/10.1101/2020.11.06.20225342v1.full
# - synapse.org/#!Synapse:syn22963646
# - psychencode.synapse.org/Explore/Studies/DetailsPage?study=syn22755055
# SZBDMulticohort:
# - medrxiv.org/content/10.1101/2022.08.31.22279406v1
# - synapse.org/#!Synapse:syn25922167
# - psychencode.synapse.org/Explore/Studies/DetailsPage?study=syn25946131
# Download: mamba install synapseclient,
#           synapse -u shreejoy -p danfelsky get syn30877938
################################################################################

SCZ_data_file = 'SZBDMulticohort/SZBDMulticohort.h5ad'
if os.path.exists(SCZ_data_file):
    with Timer('Loading SCZ data'):
        SCZ_data = sc.read(SCZ_data_file)
else:
    print('Preprocessing SCZ data...')
    from rpy2.robjects import r
    from utils import r2py
    SCZ_data = sc.read('SZBDMulticohort/combinedCells_ACTIONet.h5ad')
    # Convert counts to int32, but make sure all entries were integers first
    counts = SCZ_data.layers['counts']
    counts_int32 = counts.astype(np.int32)
    assert not (counts != counts_int32).nnz
    SCZ_data.X = counts_int32.tocsr()
    del SCZ_data.layers
    # same as r2py(r.readRDS('SZBDMulticohort/combinedCells_ACTIONet.rds'))\
    #     ['rowMaps']['listData']['ACTION_V']['NAMES']
    SCZ_gene_names = r2py(r.readRDS('SZBDMulticohort/ACTIONet_summary.rds')\
        .rx2('unified_feature_specificity').rownames)
    SCZ_data.var.index = SCZ_gene_names
    # Neither ID nor Internal_ID are unique for each person, but the combo is.
    # Fix a number of columns that are incorrectly categorical/string/float
    # instead of float/int/bool.
    SCZ_data.obs = SCZ_data.obs\
        .assign(PRS=lambda df: df.PRS.replace({'NA': np.nan}).astype(float))\
        .astype({'Age': int, 'PMI': float, 'EUR_Ancestry': float,
                 'EAS_Ancestry': float, 'AMR_Ancestry': float,
                 'SAS_Ancestry': float, 'AFR_Ancestry': float,
                 'Benzodiazepines': int, 'Anticonvulsants': int,
                 'AntipsychTyp': int, 'AntipsychAtyp': int, 'Antidepress': int,
                 'Lithium': int, 'umis': int, 'genes': int,
                 'assigned_archetype': int})\
        .astype({'Benzodiazepines': bool, 'Anticonvulsants': bool,
                 'AntipsychTyp': bool, 'AntipsychAtyp': bool,
                 'Antidepress': bool, 'Lithium': bool,
                 'assigned_archetype': 'category'})\
        .assign(broad_cell_type=lambda df: df.Celltype.str.split('-').str[0]
                .replace({'Ast': 'Astrocyte', 'Endo': 'Endothelial',
                          'Ex': 'Excitatory', 'In': 'Inhibitory',
                          'Mic': 'Microglia-PVM', 'Oli': 'Oligodendrocyte',
                          'OPC': 'OPC', 'Pericytes': 'Pericyte'}),
                unique_donor_ID=lambda df:
                    (df.ID.astype(str) + '_' + df.Internal_ID.astype(str))
                    .astype('category'))
    # noinspection PyTypeChecker
    SCZ_data.write(SCZ_data_file)

################################################################################
# Load major depressive disorder data
# - https://www.nature.com/articles/s41467-023-38530-5
# Download: 
# https://cells.ucsc.edu/dlpfc-mdd/matrix.mtx.gz 
# https://cells.ucsc.edu/dlpfc-mdd/features.tsv.gz 
# https://cells.ucsc.edu/dlpfc-mdd/barcodes.tsv.gz 
# https://cells.ucsc.edu/dlpfc-mdd/meta.tsv 
################################################################################

MDD_data_file = 'Maitra/Maitra.h5ad'
if os.path.exists(MDD_data_file):
    with Timer('Loading MDD data'):
        MDD_data = sc.read(MDD_data_file)
else:
    print('Preprocessing MDD data...')
    from scipy.io import mmread  # could also use sc.read_10x_mtx()
    MDD_counts = mmread('Maitra/matrix.mtx.gz').T.tocsr().astype(np.int32)
    MDD_genes = pd.read_table('Maitra/features.tsv.gz', usecols=[0], index_col=0, header=None, names=[None]).index
    MDD_cell_types = pd.read_table('Maitra/meta.tsv', usecols=['Cell','Broad'], index_col='Cell')\
        .assign(broad_cell_type=lambda df: df.Broad
        .replace({'Ast':'Astrocyte',
                  'End':'Endothelial',
                  'ExN':'Excitatory',
                  'InN':'Inhibitory',
                  'Mic':'Microglia-PVM',
                  'Oli':'Oligodendrocyte',
                  'OPC':'OPC'})).astype('category')\
        .query("Broad != 'Mix'")\
        .astype('category')
    MDD_metadata = pd.read_table('Maitra/meta.tsv')
    MDD_metadata = MDD_metadata\
        .assign(ID_column= MDD_metadata["Cell"].str.split(".", n=1).str[0])\
        .set_index("Cell")\
        .astype('category')\
        .query("Broad != 'Mix'")\
        .combine_first(MDD_cell_types)
    index_map = {index: i for i, index in enumerate(MDD_metadata.index)}
    rows_to_keep = [index_map[index] for index in MDD_metadata.index]
    MDD_counts = MDD_counts[rows_to_keep, :]
    assert len(MDD_genes) == 36588, len(MDD_genes)
    assert len(MDD_metadata) == 156911, len(MDD_metadata)
    assert (MDD_num_typed_cells := MDD_metadata.Cluster.notna().sum()) == \
           156911, MDD_num_typed_cells
    assert MDD_counts.shape == (len(MDD_metadata), len(MDD_genes)), MDD_counts.shape
    MDD_data = ad.AnnData(MDD_counts, obs=MDD_metadata, var=pd.DataFrame(
        index=MDD_genes), dtype=MDD_counts.dtype)
    # noinspection PyTypeChecker
    MDD_data.write(MDD_data_file)

################################################################################
# Deal with NAs in covariates
################################################################################

AD_data.obs = AD_data.obs\
    .assign(**{'APOE4 status': lambda df: df['APOE4 status'].eq('Y'),
               'Metadata: PMI': lambda df: df['Metadata: PMI'].fillna(
                   df['Metadata: PMI'].median()),
               'Study: ACT': lambda df:
                   df['Metadata: Primary Study Name'].eq('ACT'),
               'Study: ADRC Clinical Core': lambda df:
                   df['Metadata: Primary Study Name'].eq('ADRC Clinical Core')})

################################################################################
# Define data fields we're using
################################################################################

data_files = {'AD': AD_data_file, 'PD': PD_data_file, 'SCZ': SCZ_data_file, 'MDD': MDD_data_file}
datasets = {'AD': AD_data, 'PD': PD_data, 'SCZ': SCZ_data, 'MDD': MDD_data}
original_datasets = datasets  # for reference
covariate_columns = {
    'AD': ['Age at death', 'sex', 'APOE4 status', 'Metadata: PMI'], 
    'PD': ['organ__ontology_label', 'sex', 'Donor_Age', 'Donor_PMI'],
    'SCZ': ['Batch', 'Gender', 'Age', 'PMI'],
    'MDD': ['Batch', 'Chemistry', 'Condition', 'Sex']
    }
cell_type_column = {'AD': 'broad_cell_type', 'PD': 'broad_cell_type',
                    'SCZ': 'broad_cell_type', 'MDD': 'broad_cell_type'}
fine_cell_type_column = {'AD': 'Supertype', 'PD': 'Cell_Type',
                         'SCZ': 'Celltype', 'MDD': 'fine_cell_type'}
ID_column = {'AD': 'Donor ID', 'PD': 'donor_id', 'SCZ': 'unique_donor_ID', 'MDD': 'ID_column'}
phenotype_column = {'AD': 'disease', 'PD': 'disease__ontology_label', 'SCZ': 'Phenotype', 'MDD': 'Condition'}
control_name = {'AD': 'normal', 'PD': 'normal', 'SCZ': 'CON', 'MDD': 'Control'}
assert (dataset_sizes := {dataset_name: dataset.shape
                          for dataset_name, dataset in datasets.items()}) == \
       {'AD': (1378211, 36517), 'PD': (434354, 41625), 'SCZ': (468727, 17658), 'MDD': (156911, 36588)
        }, dataset_sizes

# Make sure covariates are the same for all cells of a given person + cell-type

for dataset_name, dataset in datasets.items():
    assert (nunique := dataset.obs.groupby([
        ID_column[dataset_name], cell_type_column[dataset_name]], observed=True)
        [covariate_columns[dataset_name]].nunique()).eq(1).all().all(), \
        (dataset_name, nunique[nunique.ne(1).any(axis=1)])

################################################################################
# QC cells (we'll QC genes after pseudobulking)
################################################################################

def print_dataset_sizes(label, datasets):
    dataset_sizes = {dataset_name: dataset.shape
                     for dataset_name, dataset in datasets.items()}
    print(f'{label}: {dataset_sizes}')

# Filter to cells with >= 200 genes detected

for dataset_name, dataset in datasets.items():
    if dataset_name != 'AD_data':
        with Timer(f'[{dataset_name}] Filtering to cells with >= 200 genes'):
            sc.pp.filter_cells(dataset, min_genes=200)
            gc.collect()

print_dataset_sizes('After filtering to cells with >= 200 genes', datasets)
# {'AD': (1378211, 36517), 'PD': (434354, 41625), 'SCZ': (468727, 17658)}

# Filter to cells with < 5% mitochondrial reads

for dataset_name, dataset in datasets.items():
    if dataset_name != 'AD_data':
        with Timer(f'[{dataset_name}] Filtering to cells <5% mitochondrial reads'):
            mitochondrial_genes = dataset.var.index.str.startswith('MT-')
            percent_mito = dataset[:, mitochondrial_genes].X.sum(axis=1).A1 / \
                        dataset.X.sum(axis=1).A1
            datasets[dataset_name] = dataset[percent_mito < 0.05]

print_dataset_sizes('After filtering to cells with <5% mitochondrial reads',
                    datasets)
# {'AD': (1378207, 36517), 'PD': (376235, 41625), 'SCZ': (444796, 17658)}

################################################################################
# Aggregate to pseudobulk and save
################################################################################

# def sparse_variance(sparse_matrix, axis=None):
#     return ((sparse_matrix - type(sparse_matrix)(np.full(
#         sparse_matrix.shape, sparse_matrix.mean(axis)))).power(2)).mean(axis)

def pseudobulk(dataset, ID_column, cell_type_column):
    # Fill with 0s to avoid auto-conversion to float when filling with NaNs;
    # skip groups where any of the columns is NaN
    groupby = [ID_column, cell_type_column]
    grouped = dataset.obs.groupby(groupby, observed=True, sort=False)
    pseudobulk = pd.DataFrame(
        0, index=pd.MultiIndex.from_tuples(grouped.indices),
        columns=dataset.var_names, dtype=dataset.X.dtype)
    # variance = pd.DataFrame(
    #     0, index=pd.MultiIndex.from_tuples(grouped.indices),
    #     columns=dataset.var_names, dtype=float)
    for group, group_indices in grouped.indices.items():
        group_counts = dataset[group_indices].X
        pseudobulk.loc[group] = group_counts.sum(axis=0).A1
        # variance.loc[group] = sparse_variance(group_counts, axis=0).A1
    # Take all columns of obs that have a unique value for each group;
    # reorder obs to match counts
    obs = grouped.first().loc[pseudobulk.index, grouped.nunique().le(1).all()]
    # Add number of cells as a covariate
    obs['num_cells'] = dataset.obs.groupby(groupby).size()
    # Construct AnnData object
    pseudobulk = ad.AnnData(pseudobulk, obs=obs, var=dataset.var,
                            # layers={'variance': variance},
                            dtype=dataset.X.dtype)
    # Reset index because h5ad can't store multi-indexes, but add all columns in
    # groupby joined with _ as an index
    pseudobulk.obs = pseudobulk.obs\
        .reset_index()\
        .pipe(lambda df: df.set_axis(df[groupby].astype(str)
                                     .apply('_'.join, axis=1)))
    return pseudobulk

pseudobulks = {}
for dataset_name, dataset in datasets.items():
    data_directory = data_files[dataset_name].split("/")[0]
    pseudobulk_file = f'{data_directory}/pseudobulk.h5ad'
    cell_type_counts_file = f'{data_directory}/cell_type_counts.tsv'
    # if os.path.exists(pseudobulk_file): continue
    with Timer(f'[{dataset_name}] Pseudobulking'):
         pseudobulks[dataset_name] = pseudobulk(
             dataset, ID_column[dataset_name], cell_type_column[dataset_name])
    with Timer(f'[{dataset_name}] Saving pseudobulk'):
        # noinspection PyTypeChecker
        pseudobulks[dataset_name].write(pseudobulk_file)
    with Timer(f'[{dataset_name}] Saving cell-type counts'):
        cell_type_counts = dataset.obs\
            .groupby([cell_type_column[dataset_name],
                      fine_cell_type_column[dataset_name]],
                     observed=True, sort=False)\
            [ID_column[dataset_name]]\
            .value_counts()\
            .unstack()\
            .pipe(lambda df: df.loc[:, df.sum() > 0])
        cell_type_counts.to_csv(cell_type_counts_file, sep='\t')

print_dataset_sizes('After pseudobulking', pseudobulks)
# {'AD': (709, 36517), 'PD': (144, 41625), 'SCZ': (1052, 17658)}





















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

os.chdir('/nethome/kcni/karbabi/r_projects/reverse_signatures/scripts')

pseudobulk, metadata = pseudobulk(AD_data, ID_column = 'Donor ID', cell_type_column = 'Subclass')
pseudobulk.to_csv('data/pseudobulks/SEA-AD_pseudobulk_subclass.csv', index=True)
metadata.to_csv('data/pseudobulks/SEA-AD_pseudobulk_subclass_meta.csv', index=True) 