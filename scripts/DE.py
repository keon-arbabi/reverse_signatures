import numpy as np, os, pandas as pd, scanpy as sc, warnings
from functools import reduce
from itertools import combinations, product
from rpy2.robjects import Formula, r
from scipy.stats import pearsonr
from utils import array_to_rvector, bonferroni, df_to_rdf, df_to_rmatrix, fdr, \
    percentile_transform, rmatrix_to_df, rdf_to_df, unalias, Timer
r.library('edgeR', quietly=True)

################################################################################
# Define data fields we're using
################################################################################

covariate_columns = {
    'AD': ['Age at death', 'sex', 'APOE4 status', 'Metadata: PMI', 'Study: ACT', 'num_cells'],
    # 'Study: ADRC Clinical Core' is multicollinear
    'PD': ['organ__ontology_label', 'sex', 'Donor_Age', 'Donor_PMI', 'num_cells'],
    'SCZ': ['Batch', 'Gender', 'Age', 'PMI', 'num_cells']}  # TODO should use log(# cells)
    # 'Cohort' is collinear with Batch; 'HTO' is also collinear with something
cell_type_column = {'AD': 'broad_cell_type', 'PD': 'broad_cell_type',
                    'SCZ': 'broad_cell_type'}
fine_cell_type_column = {'AD': 'Supertype', 'PD': 'Cell_Type',
                         'SCZ': 'Celltype'}
ID_column = {'AD': 'Donor ID', 'PD': 'donor_id', 'SCZ': 'unique_donor_ID'}
phenotype_column = {'AD': 'disease', 'PD': 'disease__ontology_label',
                    'SCZ': 'Phenotype'}
control_name = {'AD': 'normal', 'PD': 'normal', 'SCZ': 'CON'}

################################################################################
# Load pseudobulks
################################################################################

def get_sample_sizes(datasets):
    return {trait: dataset.obs.groupby(phenotype_column[trait])
            [ID_column[trait]].nunique().to_dict()
            for trait, dataset in datasets.items()}

def get_number_of_genes(datasets):
    return {trait: len(dataset.var)
            for trait, dataset in datasets.items()}

# Load pseudobulks
# Note: already removed 3 PD folks (3839, 4340, 5730) w/ all-NA cell-type labels

pseudobulk_files = {'AD': 'SEA-AD/pseudobulk.h5ad',
                    'PD': 'Macosko/pseudobulk.h5ad',
                    'SCZ': 'SZBDMulticohort/pseudobulk.h5ad'}
pseudobulks = {trait: sc.read(pseudobulk_file)
               for trait, pseudobulk_file in pseudobulk_files.items()}
assert (sample_sizes := get_sample_sizes(pseudobulks)) == {
    'AD': {'dementia': 42, 'normal': 47},
    'PD': {'Lewy body dementia': 4, 'Parkinson disease': 6, 'normal': 8},
    'SCZ': {'CON': 75, 'SZ': 65}}, sample_sizes
assert (number_of_genes := get_number_of_genes(pseudobulks)) == \
       {'AD': 36517, 'PD': 41625, 'SCZ': 17658}, number_of_genes

# Split PD into PD and LBD

pseudobulks = {
    'AD': pseudobulks['AD'],
    'PD': pseudobulks['PD'][pseudobulks['PD'].obs.disease__ontology_label !=
                            'Lewy body dementia'],
    'LBD': pseudobulks['PD'][pseudobulks['PD'].obs.disease__ontology_label !=
                             'Parkinson disease'],
    'SCZ': pseudobulks['SCZ']}
covariate_columns['LBD'] = covariate_columns['PD']
cell_type_column['LBD'] = cell_type_column['PD']
fine_cell_type_column['LBD'] = fine_cell_type_column['PD']
ID_column['LBD'] = ID_column['PD']
phenotype_column['LBD'] = phenotype_column['PD']
control_name['LBD'] = control_name['PD']
assert (sample_sizes := get_sample_sizes(pseudobulks)) == {
    'AD': {'dementia': 42, 'normal': 47},
    'PD': {'Parkinson disease': 6, 'normal': 8},
    'LBD': {'Lewy body dementia': 4, 'normal': 8},
    'SCZ': {'CON': 75, 'SZ': 65}}, sample_sizes

# For AD, remove 5 people (all controls) not in the two main studies (ACT and
# ADRC Clinical Core) since 2 of 5 have suspiciously low numbers of certain
# types of non-neurons

pseudobulks['AD'] = pseudobulks['AD'][
    pseudobulks['AD'].obs['Study: ACT'] |
    pseudobulks['AD'].obs['Study: ADRC Clinical Core']]
assert (sample_sizes := get_sample_sizes(pseudobulks)) == {
    'AD': {'dementia': 42, 'normal': 42},
    'PD': {'Parkinson disease': 6, 'normal': 8},
    'LBD': {'Lewy body dementia': 4, 'normal': 8},
    'SCZ': {'CON': 75, 'SZ': 65}}, sample_sizes

# Filter to autosomal/chrX coding genes present in all three datasets,
# remapping gene aliases; sort genes alphabetically
# TODO should ideally convert to log(CPMs) before gene filtering

coding_genes = pd.read_table(
    'gene_annotations/coding_genes_hg38_gencode_v41.bed',
    header=None, usecols=[0, 3], names=['chrom', 'gene'], index_col='gene')\
    .query('chrom != "chrY"').index  # TODO and chrom != "chrM", and non-readthrough
assert len(coding_genes) == 19924, len(coding_genes)
for trait, dataset in pseudobulks.items():
    dataset.var.index = unalias(gene_list=dataset.var.index,
                                target_gene_list=coding_genes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        # so later gene indexing works
        pseudobulks[trait] = dataset.copy()

genes_in_common = reduce(pd.Index.intersection, (
    dataset.var.index.dropna() for dataset in pseudobulks.values()))\
    .sort_values()
for trait, dataset in pseudobulks.items():
    pseudobulks[trait] = dataset[:, dataset.var.index.isin(
        genes_in_common)][:, genes_in_common]

assert (number_of_genes := get_number_of_genes(pseudobulks)) == \
       {'AD': 15433, 'PD': 15433, 'LBD': 15433, 'SCZ': 15433}, number_of_genes
for trait, dataset in pseudobulks.items():
    assert dataset.var.index.equals(genes_in_common)

# Stratify by cell type; only keep cells in the 7 major cell types

major_cell_types = pd.Index(['Astrocyte', 'Endothelial', 'Excitatory',
                             'Inhibitory', 'Microglia-PVM', 'OPC',
                             'Oligodendrocyte'])
cell_type_pseudobulks = {
    (trait, cell_type):
        dataset[dataset.obs[cell_type_column[trait]] == cell_type]
    for trait, dataset in pseudobulks.items()
    for cell_type in major_cell_types}

# For each cell type, filter to:
# - genes with at least 1 count in 80% of controls in all studies
#   (Note: splitting into PD and LBD doesn't affect this filter, since we only
#   use controls, which are shared between PD and LBD, to decide the threshold)
# - people with at least 10 cells of that type

sufficiently_expressed_genes = {
    cell_type: reduce(pd.Index.intersection, (
        (dataset := cell_type_pseudobulks[trait, cell_type]).var.index[
            np.percentile(dataset.X[dataset.obs[phenotype_column[
                trait]] == control_name[trait]], 20, axis=0) >= 1]
        for trait in pseudobulks)) for cell_type in major_cell_types}
assert (num_genes := {cell_type: len(gene_set) for cell_type, gene_set in
                      sufficiently_expressed_genes.items()}) == {
    'Astrocyte': 10203, 'Endothelial': 674, 'Excitatory': 13686,
    'Inhibitory': 12382, 'Microglia-PVM': 766, 'OPC': 7983,
    'Oligodendrocyte': 10033}, num_genes
filtered_pseudobulks = {
    (trait, cell_type):
        dataset[dataset.obs.num_cells >= 10,
                sufficiently_expressed_genes[cell_type]]
    for (trait, cell_type), dataset in cell_type_pseudobulks.items()}