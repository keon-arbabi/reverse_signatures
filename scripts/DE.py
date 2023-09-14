import numpy as np, pandas as pd, scanpy as sc, warnings, os, sys
from functools import reduce
from itertools import combinations, product
from rpy2.robjects import Formula, r
from scipy.stats import pearsonr
sys.path.append('projects/reverse_signatures/scripts')
from utils import array_to_rvector, bonferroni, df_to_rdf, df_to_rmatrix, fdr, \
    rmatrix_to_df, rdf_to_df, Timer
r.library('edgeR', quietly=True)
warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures/')

################################################################################
# Load pseudobulks
################################################################################

pseudobulk_files = {'SEAAD-MTG': 'data/pseudobulk/SEAAD-MTG-broad.h5ad',
                    'SEAAD-DLPFC': 'data/pseudobulk/SEAAD-DLPFC-broad.h5ad',
                    'p400': 'data/pseudobulk/p400-broad.h5ad'}
pseudobulks = {trait: sc.read(pseudobulk_file)
               for trait, pseudobulk_file in pseudobulk_files.items()}

for trait, pseudobulk in pseudobulks.items():
    if 'SEAAD' in trait:
        pseudobulk.obs['Dx'] = \
            np.where(pseudobulk.obs['Consensus Clinical Dx (choice=Alzheimers disease)'] == 'Checked', 1,
                     np.where(pseudobulk.obs['Consensus Clinical Dx (choice=Control)'] == 'Checked', 0, np.nan))
    if trait == 'p400':
        pseudobulk.obs['Dx'] = np.where(pseudobulk.obs['cogdx'].isin([4, 5]), 1, 0) 

covariate_columns = {'SEAAD-MTG': ['Age at death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'num_cells'],
                     'SEAAD-DLPFC': ['Age at death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'num_cells'],
                     'p400': ['age_death', 'sex', 'apoe_genotype', 'pmi', 'num_cells']} #TODO p400 batch or study?

cell_type_column = {'SEAAD-MTG': 'broad_cell_type', 'SEAAD-DLPFC': 'broad_cell_type', 'p400': 'broad_cell_type'}
fine_cell_type_column = {'SEAAD-MTG': 'Subclass', 'SEAAD-DLPFC': 'Subclass', 'p400': 'cell_type'}
ID_column = {'SEAAD-MTG': 'donor_id', 'SEAAD-DLPFC': 'donor_id', 'p400': 'projid'}
phenotype_column = {'SEAAD-MTG': 'Dx', 'SEAAD-DLPFC': 'Dx', 'p400': 'Dx'}
control_name = {'SEAAD-MTG': 0, 'SEAAD-DLPFC': 0, 'p400': 0}

################################################################################
# Preprocess 
################################################################################

def get_sample_sizes(datasets):
    return {trait: dataset.obs.groupby(phenotype_column[trait])
            [ID_column[trait]].nunique().to_dict()
            for trait, dataset in datasets.items()}

def get_number_of_genes(datasets):
    return {trait: len(dataset.var)
            for trait, dataset in datasets.items()}

# filter to coding genes present in all three datasets

coding_genes = pd.read_table('data/differential-expression/coding_genes_hg38_gencode_v44.bed',
                             header=None, usecols=[0, 3], names=['chrom', 'gene'], index_col='gene')
assert len(coding_genes) == 20765, len(coding_genes)
for trait, pseudobulk in pseudobulks.items():
    pseudobulk = pseudobulk[:, coding_genes.index.intersection(pseudobulk.var_names)]
    pseudobulks[trait] = pseudobulk
    
genes_in_common = reduce(pd.Index.intersection, (
    dataset.var.index.dropna() for dataset in pseudobulks.values()))\
    .sort_values()
for trait, dataset in pseudobulks.items():
    pseudobulks[trait] = dataset[:, dataset.var.index.isin(
        genes_in_common)][:, genes_in_common]
    
assert (number_of_genes := get_number_of_genes(pseudobulks)) == \
       {'SEAAD-MTG': 16421, 'SEAAD-DLPFC': 16421, 'p400': 16421}, number_of_genes
for trait, dataset in pseudobulks.items():
    assert dataset.var.index.equals(genes_in_common)

# stratify by cell type: 7 major cell types

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte', 'Microglia-PVM', 'OPC', 'Endothelial'
cell_type_pseudobulks = {(trait, cell_type):
    dataset[dataset.obs[cell_type_column[trait]] == cell_type]
        for trait, dataset in pseudobulks.items()
        for cell_type in broad_cell_types}
    
# for each cell type, filter to:
# - genes with at least 1 count in 80% of controls in all studies
# - people with at least 10 cells of that type

sufficiently_expressed_genes = {
    cell_type: reduce(pd.Index.intersection, (
        (dataset := cell_type_pseudobulks[trait, cell_type]).var.index[
            np.percentile(dataset.X[dataset.obs[phenotype_column[trait]] ==\
                control_name[trait]], 20, axis=0) >= 1]
        for trait in pseudobulks)) for cell_type in broad_cell_types}
{cell_type: len(gene_set) for cell_type, gene_set in sufficiently_expressed_genes.items()}

filtered_pseudobulks = {(trait, cell_type):
    dataset[dataset.obs.index[dataset.obs['num_cells'] >= 10], :]
    [:, dataset.var.index.isin(sufficiently_expressed_genes[cell_type])]
    for (trait, cell_type), dataset in cell_type_pseudobulks.items()}

    
################################################################################
# Create normalized count matrices and design matrices per dataset and cell type
# - Some covariates (like which brain region the samples came from) may be the
#   same for all samples from a given cell type; remove them for that cell type
################################################################################

normalized_count_matrices, design_matrices = {}, {}
for (trait, cell_type), dataset in filtered_pseudobulks.items():
    with Timer(f'[{trait}, {cell_type}] '
               f'Creating normalized count + design matrices'):
        count_matrix = df_to_rmatrix(pd.DataFrame(
            dataset.X.T, index=dataset.var.index, columns=dataset.obs.index))
        diagnosis_and_covariate_matrix = dataset.obs.assign(
            diagnosis=lambda df: df[phenotype_column[trait]]
                .eq(control_name[trait])
                .replace({False: 'Case', True: 'Control'})
                .astype(pd.CategoricalDtype(['Control', 'Case'])))\
            [['diagnosis'] + covariate_columns[trait]]\
            .pipe(lambda df: df.loc[:, df.nunique() > 1])\
            .pipe(df_to_rdf)
        design_matrices[trait, cell_type] = r['model.matrix'](
            Formula('~.'), data=diagnosis_and_covariate_matrix)
        normalized_count_matrices[trait, cell_type] = \
            r.calcNormFactors(r.DGEList(
                count_matrix,
                group=diagnosis_and_covariate_matrix.rx2('diagnosis')))
    
    

