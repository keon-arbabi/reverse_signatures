import numpy as np, pandas as pd, scanpy as sc, warnings, os, sys
from functools import reduce
from rpy2.robjects import Formula, r
from scipy.stats import combine_pvalues
sys.path.append('projects/reverse_signatures/scripts')
from projects.reverse_signatures.old.utils import series_to_rvector, df_to_rdf, df_to_rmatrix, array_to_rvector, \
    rmatrix_to_df, rdf_to_df, bonferroni, fdr, percentile_transform, Timer
    
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', 10)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures/')

r.library('edgeR', quietly=True)
r.library('disgenet2r', quietly=True)
r.source('scripts/R_functions.R')

################################################################################
# Load pseudobulks
################################################################################

# 'SEAAD-MTG': 'data/pseudobulk/SEAAD-MTG-broad.h5ad',
# 'SEAAD-DLPFC': 'data/pseudobulk/SEAAD-DLPFC-broad.h5ad',
pseudobulk_files = {'p400': 'data/pseudobulk/p400-broad.h5ad'}
pseudobulks = {trait: sc.read(pseudobulk_file)
               for trait, pseudobulk_file in pseudobulk_files.items()}

pseudobulk = sc.read('data/pseudobulk/p400-broad.h5ad')
ids = pd.read_csv('tmp.csv')['ID']

pseudobulk = pseudobulk[pseudobulk.obs['broad_cell_type'] == "Astrocyte"].copy()
pseudobulk = pseudobulk[pseudobulk.obs['projid'].isin(ids).astype(bool)].copy()
pseudobulk.X.shape
pseudobulk.X[(pseudobulk.obs['projid'] == 10514454).astype(bool), pseudobulk.var.index == 'ABCA7']


# Note: num_cells is the number of cells for that broad_cell_type per subject 
for trait, pseudobulk in pseudobulks.items():
    library_sizes = pseudobulk.X.sum(axis=1)
    assert not np.any(library_sizes == 0), f'{trait} has libraries with no counts'
    pseudobulk.obs['library_size'] = library_sizes
    pseudobulk.obs['log_library_size'] = np.log2(pseudobulk.obs['library_size'])
    pseudobulk.obs['log_num_cells'] = np.log2(pseudobulk.obs['num_cells'] + 1)

for trait, pseudobulk in pseudobulks.items():
    if 'SEAAD' in trait:
        pseudobulk.obs['Dx'] = \
            np.where(pseudobulk.obs['Consensus Clinical Dx (choice=Alzheimers disease)']
                     .eq('Checked'), 1,
            np.where( pseudobulk.obs['Consensus Clinical Dx (choice=Control)']
                     .eq('Checked'), 0, np.nan)).astype(float)
        pseudobulk = pseudobulk[pseudobulk.obs['Dx'].notna() & \
            (pseudobulk.obs['ACT'] | pseudobulk.obs['ADRC Clinical Core'])]
        pseudobulks[trait] = pseudobulk
    if trait == 'p400':
        pseudobulk.obs['Dx'] = pseudobulk.obs['pmAD'].fillna(0).astype(float)
        pseudobulk.obs['apoe4'] = pseudobulk.obs['apoe_genotype'].astype(str).str.count('4')
        pseudobulk.obs['pmi'] = pseudobulk.obs['pmi'].fillna(pseudobulk.obs['pmi'].median())
        pseudobulk = pseudobulk[pseudobulk.obs['Dx'].notna()]
        pseudobulks[trait] = pseudobulk

covariate_columns = {'SEAAD-MTG': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'log_library_size'],
                     'SEAAD-DLPFC': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'log_library_size'],
                     'p400': ['age_death', 'sex', 'apoe4', 'pmi', 'log_library_size']}

cell_type_column = {'SEAAD-MTG': 'broad_cell_type', 'SEAAD-DLPFC': 'broad_cell_type', 'p400': 'broad_cell_type'}
fine_cell_type_column = {'SEAAD-MTG': 'Subclass', 'SEAAD-DLPFC': 'Subclass', 'p400': 'cell_type'}

ID_column = {'SEAAD-MTG': 'donor_id', 'SEAAD-DLPFC': 'donor_id', 'p400': 'projid'}
phenotype_column = {'SEAAD-MTG': 'Dx', 'SEAAD-DLPFC': 'Dx', 'p400': 'Dx'}
control_name = {'SEAAD-MTG': 0, 'SEAAD-DLPFC': 0, 'p400': 0}

for trait, pseudobulk in pseudobulks.items():
    for col in ['Dx'] + covariate_columns[trait]:
        assert not pseudobulk.obs[col].isna().any(), f"'{col}' in {trait} has NaNs"
            
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

assert (sample_sizes := get_sample_sizes(pseudobulks)) ==\
   {'SEAAD-MTG': {0: 32, 1: 23}, 'SEAAD-DLPFC': {0: 32, 1: 21}, 'p400': {0: 275, 1: 161}}, sample_sizes
assert (number_of_genes := get_number_of_genes(pseudobulks)) == \
    {'SEAAD-MTG': 36517, 'SEAAD-DLPFC': 36517, 'p400': 18552}, number_of_genes
    
# Filter to coding genes present in all three datasets

coding_genes = pd.read_table('data/differential-expression/coding_genes_hg38_gencode_v44.bed',
                             header=None, usecols=[0, 3], names=['chrom', 'gene'],
                             index_col='gene')
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

# Stratify by cell type

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte',\
    'Microglia-PVM', 'OPC', 'Endothelial'
cell_type_pseudobulks = {(trait, cell_type):
    dataset[dataset.obs[cell_type_column[trait]] == cell_type]
        for trait, dataset in pseudobulks.items()
        for cell_type in broad_cell_types}
    
# For each cell type, filter to:
# - Genes with at least 1 count in 80% of controls OR cases in all studies
# - Subjects with at least 10 cells of that type

sufficiently_expressed_genes = {
    cell_type: reduce(pd.Index.intersection, (
        (dataset := cell_type_pseudobulks[trait, cell_type]).var.index[
            np.percentile(dataset.X[np.logical_or(
                (dataset.obs[phenotype_column[trait]] ==
                    control_name[trait]).to_numpy(dtype=bool),
                (dataset.obs[phenotype_column[trait]] !=
                    control_name[trait]).to_numpy(dtype=bool))], 20, axis=0) >= 1]
        for trait in pseudobulks)) for cell_type in broad_cell_types}

assert (num_genes := {cell_type: len(gene_set) for cell_type, gene_set in
                      sufficiently_expressed_genes.items()}) == {
                          'Excitatory': 15976, 'Inhibitory': 14787, 
                          'Oligodendrocyte': 13640, 'Astrocyte': 14042, 
                          'Microglia-PVM': 12233, 'OPC': 12674, 'Endothelial': 6498}, num_genes
                      
filtered_pseudobulks = {(trait, cell_type):
    dataset[dataset.obs.index[dataset.obs.num_cells >= 10].tolist(),
            sufficiently_expressed_genes[cell_type]]
    for (trait, cell_type), dataset in cell_type_pseudobulks.items()}

################################################################################
# Create normalized count matrices and design matrices per dataset and cell type
# - Some covariates (like which brain region the samples came from) may be the
#   same for all samples from a given cell type; remove them for that cell type
################################################################################
            
normalized_count_matrices, design_matrices, group_vectors = {}, {}, {}
for (trait, cell_type), dataset in filtered_pseudobulks.items():
    with Timer(f'[{trait}, {cell_type}] '
               f'Creating normalized count + design matrices'):
        diagnosis_and_covariate_matrix = dataset.obs.assign(
            diagnosis=dataset.obs[phenotype_column[trait]]
                .eq(control_name[trait])
                .replace({False: 'Case', True: 'Control'})
                .astype(pd.CategoricalDtype(['Control', 'Case'])))\
            [['diagnosis'] + covariate_columns[trait]]
        assert not diagnosis_and_covariate_matrix.isna().any().any(),\
            "NAs detected in diagnosis and covariate matrix"
        count_matrix = df_to_rmatrix(pd.DataFrame(
            dataset.X.T, index=dataset.var.index, columns=dataset.obs.index))
        group_vectors[trait, cell_type] = diagnosis_and_covariate_matrix['diagnosis']\
            .pipe(series_to_rvector)
        diagnosis_and_covariate_matrix = diagnosis_and_covariate_matrix\
            .pipe(lambda df: df.loc[:, df.nunique() > 1])\
            .pipe(df_to_rdf)
        design_matrices[trait, cell_type] = r['model.matrix'](
                    Formula('~.'), data=diagnosis_and_covariate_matrix)
        normalized_count_matrices[trait, cell_type] = \
            r.calcNormFactors(r.DGEList(
                count_matrix,
                method = "TMM",
                group=diagnosis_and_covariate_matrix.rx2('diagnosis')))

################################################################################
# Differential expression with limma-voom
# - voomByGroup: https://github.com/YOU-k/voomByGroup 
################################################################################

def aggregate(DEs):
    return pd.concat([DEs[trait, cell_type]
                      .assign(trait=trait, 
                              cell_type=cell_type,
                              bonferroni=lambda df: bonferroni(df.p),
                              fdr=lambda df: fdr(df.p))
                      for trait, cell_type in DEs])\
        .reset_index()\
        .set_index(['trait', 'cell_type', 'gene'])

def print_calibration(DE, quantile=0.5):
    print(DE.groupby(['trait', 'cell_type']).p.quantile(quantile).unstack().T
          .rename_axis(None).rename_axis(None, axis=1).applymap('{:.2g}'.format)
          .replace({'nan': ''}))

def print_num_hits(DE, method='fdr', threshold=0.05):
    print(DE[method].lt(threshold).groupby(['trait', 'cell_type']).sum()
          .unstack().T.rename_axis(None).rename_axis(None, axis=1)
          .applymap('{:.0f}'.format).replace({'nan': ''}))

def print_hits(DE, method='fdr', threshold=0.05):
    print(DE.query(f'{method} < {threshold}').to_string())
    
def print_hits(DE, method='fdr', threshold=0.05, num_hits=None):
    filtered = DE.query(f'{method} < {threshold}')
    sorted_and_limited = (filtered.sort_values(['trait', 'cell_type', 'fdr'])
                          .groupby(['trait', 'cell_type'])
                          .head(num_hits))
    print(sorted_and_limited.to_string())

make_voom_plots = True
if make_voom_plots:
    os.makedirs('results/voom', exist_ok=True)

DE_limma_vooms, log_CPMs = {}, {}
for trait, cell_type in normalized_count_matrices:
    with Timer(f'[{trait}, {cell_type}] Running limma-voom'):
        normalized_count_matrix = normalized_count_matrices[trait, cell_type]
        design_matrix = design_matrices[trait, cell_type]
        group = group_vectors[trait, cell_type]
        
        assert list(r.rownames(design_matrix)) == list(r.colnames(normalized_count_matrix))

        non_estimable = r('limma:::nonEstimable')(design_matrix)
        if not r['is.null'](non_estimable)[0]:
            to_drop = [name for name in r.colnames(design_matrix) if name in non_estimable]
            design_matrix = df_to_rdf(rmatrix_to_df(design_matrix).drop(
                columns=to_drop, errors='ignore'))
            print(" ".join(non_estimable), "dropped due to colinearity")
        
        if make_voom_plots:
            r.png(f'results/voom/{trait}_{cell_type}.png')
            voom_output = r.voomByGroup(
                normalized_count_matrix, group,
                design_matrix, plot='combine')
            r['dev.off']()
        else:
            voom_output = r.voomByGroup(
                normalized_count_matrix, group,
                design_matrix, plot='none')
        # copy() avoids array corruption (when R frees the underlying memory?)
        log_CPMs[trait, cell_type] = rmatrix_to_df(voom_output.rx2('E')).copy()
        lmFit_output = r.lmFit(voom_output, design_matrix)
        eBayes_output = r.eBayes(lmFit_output)
        SE = np.array(eBayes_output.rx2('stdev.unscaled').rx(True, 'diagnosisCase')) * \
            np.array(r.sqrt(eBayes_output.rx2('s2.post')))
        res = r.topTable(eBayes_output, coef='diagnosisCase', number=np.inf, confint=True)
        res_df = rdf_to_df(res)
        DE_limma_vooms[trait, cell_type] = rdf_to_df(res)\
            .assign(lfcse=SE)\
            .rename_axis('gene')\
            [['AveExpr', 'logFC', 'CI.L', 'CI.R', 'P.Value', 'lfcse']]\
            .rename(columns={'P.Value': 'p'})
            
DE_limma_voom = aggregate(DE_limma_vooms)
# DE_limma_voom.to_csv('results/voom/limma_voom.tsv.gz', sep='\t', compression='gzip')

# https://github.com/unistbig/metapro/blob/master/R/metapro.R

def wFisher(p, weight=None, is_onetail=True, eff_sign=None):
    from scipy.stats import gamma
    p = np.array(p)
    weight = np.ones(len(p)) if weight is None else np.array(weight)
    mask = ~np.isnan(p)
    p, weight = p[mask], weight[mask]
    
    if not is_onetail:
        eff_sign = np.array(eff_sign)[mask]
    if len(p) != len(weight):
        raise ValueError("Length mismatch between 'p' and 'weight'")
    
    N, ratio = len(p), weight / weight.sum()
    Gsum = lambda p_values: np.sum(gamma.ppf(p_values, a=N * ratio, scale=2))
    
    if is_onetail:
        resultP = gamma.sf(Gsum(p), a=N, scale=2)
        return pd.Series({"p_combined": min(1, resultP)})
    
    p1, p2 = p.copy(), p.copy()
    pos, neg = eff_sign > 0, eff_sign < 0
    p1[pos], p1[neg] = p[pos] / 2, 1 - p[neg] / 2
    p2[pos], p2[neg] = 1 - p[pos] / 2, p[neg] / 2
    resultP1, resultP2 = gamma.sf(Gsum(p1), a=N, scale=2), gamma.sf(Gsum(p2), a=N, scale=2)
    resultP = 2 * min(resultP1, resultP2)
    overall_eff_direction = "+" if resultP1 <= resultP2 else "-"
    
    return pd.Series({"p_combined": min(1, resultP), "overall_eff_direction": overall_eff_direction})

DE_limma_voom_combined = DE_limma_voom\
    .groupby(['cell_type', 'gene'])\
    .apply(lambda df: wFisher(df['p'], 
                              weight=1 / (df['lfcse'] ** 2),
                              eff_sign=df['logFC'], is_onetail=False))\
    .rename(columns={"p_combined": "p"})\
    .assign(fdr=lambda df: fdr(df['p']))\
    [['p', 'fdr', 'overall_eff_direction']]
# DE_limma_voom_combined.to_csv('results/voom/limma_voom_combined.tsv.gz', sep='\t', compression='gzip')

print(DE_limma_voom_combined.loc[DE_limma_voom_combined['fdr'] < 0.01]
    .reset_index()
    .groupby('cell_type')['gene'].nunique())

################################################################################

# load results 
DE_limma_voom = pd.read_csv('results/voom/limma_voom.tsv.gz', sep='\t')\
    .set_index(['trait', 'cell_type', 'gene'])  

print_num_hits(DE_limma_voom)    

'''print_num_hits(DE_limma_voom)
                SEAAD-DLPFC SEAAD-MTG p400
Astrocyte                 2       191  323
Endothelial               2         0    6
Excitatory                0        31  799
Inhibitory                0         0  274
Microglia-PVM             0         0   11
OPC                       1         0    4
Oligodendrocyte           7        54  138
'''

DE_limma_voom_combined = pd.read_csv('results/voom/limma_voom_combined.tsv.gz', sep='\t')\
    .set_index(['cell_type', 'gene'])  

DE_tain = pd.read_csv('data/differential-expression/de_aspan_voombygroup_p400.tsv', sep='\t')\
    .rename(columns={'study': 'trait', 'gene_id': 'gene', 'log_fc': 'logFC', 'p_value': 'p'})\
    .assign(cell_type = lambda df: df.cell_type
            .replace({'Astro': 'Astrocyte',
                        'Endo': 'Endothelial',
                        'Glut': 'Excitatory',
                        'GABA': 'Inhibitory',
                        'Micro': 'Microglia-PVM',
                        'Oligo': 'Oligodendrocyte',
                        'OPC': 'OPC'}))\
    .set_index(['trait', 'cell_type', 'gene'])\
    .query('cell_type in @broad_cell_types & ids == "allids"')
            
print_num_hits(DE_tain, method='fdr', threshold=0.05)
print_num_hits(DE_limma_voom, method='fdr', threshold=0.05)

n_DE_limma = DE_limma_voom.query('fdr < 0.05 & trait == "p400"').groupby('cell_type').size()
n_DE_tain = DE_tain.query('fdr < 0.05').groupby('cell_type').size()

overlap_stats = (
    pd.merge(
        DE_limma_voom.query('fdr < 0.05 & trait == "p400"'),
        DE_tain.query('fdr < 0.05'),
        left_index=True, right_index=True, how='left', suffixes=('_limma', '_tain')
    )
    .groupby(level='cell_type')
    .apply(lambda x: pd.Series({
        'n_overlaps': (x['logFC_limma'] * x['logFC_tain'] > 0).sum(),
        'percent_limma': ((x['logFC_limma'] * x['logFC_tain'] > 0).sum() / n_DE_limma.get(x.name, 1)),
        'percent_tain': ((x['logFC_limma'] * x['logFC_tain'] > 0).sum() / n_DE_tain.get(x.name, 1))
    }))
    .reset_index()
)
print(overlap_stats)

'''
         cell_type  n_overlaps  percent_limma  percent_tain
0        Astrocyte       126.0       0.372781      0.807692
1      Endothelial         0.0       0.000000      0.000000
2       Excitatory       494.0       0.574419      0.809836
3       Inhibitory       174.0       0.519403      0.494318
4    Microglia-PVM         1.0       0.090909      1.000000
5              OPC         2.0       0.666667      1.000000
6  Oligodendrocyte        42.0       0.315789      0.954545
'''