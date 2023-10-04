import numpy as np, pandas as pd, scanpy as sc, warnings, os, sys
from functools import reduce
from itertools import combinations, product
from rpy2.robjects import Formula, r
from scipy.stats import pearsonr, combine_pvalues
sys.path.append('projects/reverse_signatures/scripts')
from utils import array_to_rvector, series_to_rvector, df_to_rdf, df_to_rmatrix,\
    rmatrix_to_df, rdf_to_df, bonferroni, fdr, percentile_transform, Timer
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', 10)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures/')

r.library('edgeR', quietly=True)
r.library('disgenet2r', queitly=True)
r.source('scripts/voomByGroup.R')

################################################################################
# Load pseudobulks
################################################################################

pseudobulk_files = {'SEAAD-MTG': 'data/pseudobulk/SEAAD-MTG-broad.h5ad',
                    'SEAAD-DLPFC': 'data/pseudobulk/SEAAD-DLPFC-broad.h5ad',
                    'p400': 'data/pseudobulk/p400-broad.h5ad'}
pseudobulks = {trait: sc.read(pseudobulk_file)
               for trait, pseudobulk_file in pseudobulk_files.items()}

for trait, pseudobulk in pseudobulks.items():
    library_sizes = pseudobulk.X.sum(axis=1)
    assert not np.any(library_sizes == 0), f'{trait} has libraries with no counts'
    pseudobulk.obs['library_size'] = library_sizes
    pseudobulk.obs['log_library_size'] = np.log10(pseudobulk.obs['library_size'])
    pseudobulk.obs['log_num_cells'] = np.log10(pseudobulk.obs['num_cells'])

for trait, pseudobulk in pseudobulks.items():
    if 'SEAAD' in trait:
        pseudobulks[trait] = \
            pseudobulk[pseudobulk.obs['Consensus Clinical Dx (choice=Alzheimers disease)'].eq('Checked') |
                       pseudobulk.obs['Consensus Clinical Dx (choice=Control)'].eq('Checked')]
        pseudobulk.obs['Dx'] = \
            pseudobulk.obs['Consensus Clinical Dx (choice=Alzheimers disease)'].eq('Checked')
        pseudobulks[trait] = \
            pseudobulk[pseudobulk.obs['ACT'] | pseudobulk.obs['ADRC Clinical Core']]
    if trait == 'p400':
        pseudobulk.obs['Dx'] = pseudobulk.obs['cogdx'].isin([4, 5])

covariate_columns = {'SEAAD-MTG': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 
                                   'log_num_cells', 'log_library_size'],
                     'SEAAD-DLPFC': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT',
                                     'log_num_cells', 'log_library_size'],
                     'p400': ['age_death', 'sex', 'apoe_genotype', 'pmi',
                              'log_num_cells', 'log_library_size']}
cell_type_column = {'SEAAD-MTG': 'broad_cell_type', 'SEAAD-DLPFC': 'broad_cell_type', 
                    'p400': 'broad_cell_type'}
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

assert (sample_sizes := get_sample_sizes(pseudobulks)) ==\
   {'SEAAD-MTG': {0: 61, 1: 23}, 'SEAAD-DLPFC': {0: 59, 1: 21}, 'p400': {0: 275, 1: 162}}, sample_sizes
assert (number_of_genes := get_number_of_genes(pseudobulks)) == \
    {'SEAAD-MTG': 36517, 'SEAAD-DLPFC': 36517, 'p400': 18552}, number_of_genes
    
# filter to coding genes present in all three datasets

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

# stratify by cell type: 7 major cell types

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte',\
    'Microglia-PVM', 'OPC', 'Endothelial'
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
            np.percentile(dataset.X[dataset.obs[phenotype_column[
                trait]] == control_name[trait]], 20, axis=0) >= 1]
        for trait in pseudobulks)) for cell_type in broad_cell_types}

assert (num_genes := {cell_type: len(gene_set) for cell_type, gene_set in
                      sufficiently_expressed_genes.items()}) == {
                          'Excitatory': 15989, 'Inhibitory': 14820, 
                          'Oligodendrocyte': 13602, 'Astrocyte': 14054, 
                          'Microglia-PVM': 12179, 'OPC': 12718, 'Endothelial': 7130}, num_genes
                      
filtered_pseudobulks = {
    (trait, cell_type): 
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
        valid_indices = diagnosis_and_covariate_matrix.dropna().index
        if len(dataset.obs) - len(valid_indices) > 3:
            raise ValueError(">3 rows with NAs detected")
        count_matrix = df_to_rmatrix(pd.DataFrame(
            dataset.X.T[:, dataset.obs.index.get_indexer(valid_indices)],
            index=dataset.var.index, columns=valid_indices))
        group_vectors[trait, cell_type] = diagnosis_and_covariate_matrix['diagnosis']\
            .loc[valid_indices]\
            .pipe(series_to_rvector)
        diagnosis_and_covariate_matrix = diagnosis_and_covariate_matrix\
            .loc[valid_indices]\
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
# - voomByGroup: https://github.com/YOU-k/voomByGroup/blob/main/voomByGroup.R
################################################################################

def aggregate(DEs):
    return pd.concat([DEs[trait, cell_type]
                      .assign(trait=trait, cell_type=cell_type,
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

DE_limma_vooms = {}
log_CPMs = {}

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
        res = r.topTable(eBayes_output, coef='diagnosisCase', number=np.inf,
                         adjust_method='none', sort_by='p', confint=True)
        DE_limma_vooms[trait, cell_type] = rdf_to_df(res)\
            .rename_axis('gene')\
            [['AveExpr', 'logFC', 'CI.L', 'CI.R', 'P.Value']]\
            .rename(columns={'P.Value': 'p'})
            
# save results
DE_limma_voom = aggregate(DE_limma_vooms)
#DE_limma_voom.to_csv('results/voom/limma_voom_logcells.tsv.gz', sep='\t', compression='gzip')

DE_limma_voom_combined = DE_limma_voom\
    .groupby(['cell_type', 'gene'])['p']\
    .apply(lambda df: combine_pvalues(df, method='fisher')[1])
#DE_limma_voom_combined.to_csv('results/voom/limma_voom_combined.tsv', sep='\t')
    
################################################################################

# load results 
DE_limma_voom = pd.read_csv('results/voom/limma_voom_logcells.tsv.gz', sep='\t')\
    .set_index(['trait', 'cell_type', 'gene'])  

print_num_hits(DE_limma_voom, method='fdr', threshold=0.05)

# log_library_size
'''                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte                 0         1   411
Endothelial               1         0     7
Excitatory                1         3  1050
Inhibitory                0         0   415
Microglia-PVM             2         0    13
OPC                       1         0     5
Oligodendrocyte         561        24   201
'''
# log_library_size + log_num_cells
'''                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte                 0         2   406
Endothelial               1         0     2
Excitatory                3         3  1141
Inhibitory              430         0   410
Microglia-PVM             0         0    13
OPC                       1         0     3
Oligodendrocyte           0        22   182
'''
 
print_calibration(DE_limma_voom
                .assign(percentile=lambda df: df
                        .groupby(['trait', 'cell_type'])
                        .AveExpr
                        .transform(percentile_transform))
                .query('percentile < 10'))
 
# log_library_size
'''                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte              0.42      0.48  0.38
Endothelial            0.47      0.54  0.45
Excitatory             0.23      0.24  0.34
Inhibitory             0.21      0.26  0.44
Microglia-PVM          0.37      0.48  0.44
OPC                    0.43       0.5  0.46
Oligodendrocyte        0.24       0.4  0.37
''' 

# log_library_size + log_num_cells
'''                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte              0.41      0.47  0.39
Endothelial            0.47      0.53  0.45
Excitatory             0.23      0.24  0.33
Inhibitory              0.2      0.27  0.44
Microglia-PVM          0.38      0.47  0.45
OPC                    0.45      0.44  0.46
Oligodendrocyte        0.37      0.45  0.37
'''

print_hits(DE_limma_voom, method='fdr', threshold=0.05, num_hits=10)

'''
trait       cell_type       gene                                                                                       
SEAAD-DLPFC Endothelial     CHSY1      9.541515  0.519139  0.315709  0.722569  2.710333e-06  1.932468e-02  1.932468e-02
            Excitatory      FAM171B    7.413468  0.174580  0.106485  0.242675  2.405058e-06  3.845447e-02  3.066335e-02
                            SLC7A14    6.367587  0.318102  0.191052  0.445152  3.835556e-06  6.132671e-02  3.066335e-02
                            FZD3       7.499330  0.247341  0.144583  0.350099  8.053310e-06  1.287644e-01  4.292146e-02
            Inhibitory      PBX3       5.150162  0.503064  0.298350  0.707778  5.463352e-06  8.096687e-02  4.110475e-02
                            ALS2CL     2.492332 -0.339811 -0.489441 -0.190180  2.227407e-05  3.301018e-01  4.110475e-02
                            CES4A      4.812628 -0.398634 -0.574966 -0.222302  2.403422e-05  3.561872e-01  4.110475e-02
                            USP12      5.444890  0.188213  0.104242  0.272183  2.770381e-05  4.105704e-01  4.110475e-02
                            GRIA3      8.945683  0.178306  0.098559  0.258052  2.884359e-05  4.274620e-01  4.110475e-02
                            TRIM17     1.150403 -0.400264 -0.582951 -0.217577  4.004363e-05  5.934466e-01  4.110475e-02
                            LAS1L      4.430782 -0.250100 -0.364276 -0.135924  4.019256e-05  5.956537e-01  4.110475e-02
                            NGDN       3.509946 -0.171217 -0.249619 -0.092815  4.218005e-05  6.251083e-01  4.110475e-02
                            SNX16      4.211400  0.228068  0.123448  0.332689  4.337849e-05  6.428691e-01  4.110475e-02
                            SMAD2      6.366159  0.182662  0.098783  0.266540  4.409679e-05  6.535144e-01  4.110475e-02
            OPC             FGL1       3.039806  1.350743  0.836245  1.865241  1.416512e-06  1.801520e-02  1.801520e-02
SEAAD-MTG   Astrocyte       COL8A1     1.311568  1.574584  1.037370  2.111799  1.119544e-07  1.573408e-03  1.573408e-03
                            CERCAM     3.735362  0.411870  0.245397  0.578344  4.513291e-06  6.342979e-02  3.171490e-02
            Excitatory      ARHGAP24   4.472153  0.677050  0.452656  0.901443  5.490717e-08  8.779107e-04  8.779107e-04
                            MRPL48     4.636607  0.219148  0.132623  0.305672  2.867455e-06  4.584774e-02  2.292387e-02
                            ADRA1D    -1.331570 -0.783426 -1.105007 -0.461846  6.090882e-06  9.738711e-02  3.246237e-02
            Oligodendrocyte PYGL       3.421172  0.976195  0.603073  1.349317  1.445244e-06  1.965821e-02  1.965821e-02
                            IFIH1      2.747243  0.700352  0.414573  0.986130  5.345015e-06  7.270289e-02  2.317434e-02
                            OGFRL1     4.229555  0.919827  0.539799  1.299855  6.758617e-06  9.193070e-02  2.317434e-02
                            FAP        3.114715  0.903317  0.528038  1.278596  7.493558e-06  1.019274e-01  2.317434e-02
                            VSIR       2.423761  1.198227  0.696960  1.699495  8.518724e-06  1.158717e-01  2.317434e-02
                            BIRC3      2.580700  0.975974  0.562455  1.389493  1.073856e-05  1.460658e-01  2.434431e-02
                            LHFPL2     4.622467  0.823906  0.464190  1.183622  1.821802e-05  2.478014e-01  3.400654e-02
                            ANGPT1     2.559527  0.751076  0.419668  1.082484  2.183063e-05  2.969402e-01  3.400654e-02
                            BEND7      2.123067  0.600786  0.333171  0.868402  2.560220e-05  3.482411e-01  3.400654e-02
                            P3H2       4.273301  0.788007  0.436540  1.139475  2.616363e-05  3.558777e-01  3.400654e-02
p400        Astrocyte       CREB5      4.985435  0.435004  0.299693  0.570316  6.594967e-10  9.268566e-06  9.268566e-06
                            HMGN2      5.477695 -0.286752 -0.377908 -0.195597  1.463382e-09  2.056637e-05  1.028318e-05
                            KLF4       1.708971  0.572011  0.379764  0.764259  9.847756e-09  1.384004e-04  4.613345e-05
                            ALDH1A1    7.884295 -0.453921 -0.609597 -0.298245  1.880198e-08  2.642430e-04  6.606076e-05
                            AK4        6.704768  0.185163  0.121020  0.249307  2.568668e-08  3.610006e-04  7.220012e-05
                            NOL3       4.439560 -0.222186 -0.299994 -0.144378  3.576574e-08  5.026517e-04  8.377529e-05
                            ADAM33     3.822175  0.493732  0.314624  0.672840  1.004656e-07  1.411944e-03  1.833809e-04
                            ALK        6.731758  0.250498  0.159503  0.341492  1.043865e-07  1.467047e-03  1.833809e-04
                            INPP5D     2.682237  0.476252  0.299385  0.653120  1.926917e-07  2.708089e-03  3.008987e-04
                            AP3B2      5.401773 -0.247790 -0.340899 -0.154681  2.641525e-07  3.712399e-03  3.712399e-04
            Endothelial     SLC38A2    8.708283  0.306976  0.191263  0.422689  2.861213e-07  2.040045e-03  2.040045e-03
                            DLL4       6.016983  0.483983  0.283974  0.683992  2.690430e-06  1.918277e-02  9.591384e-03
            Excitatory      NRIP2      2.273171  0.350403  0.251536  0.449269  1.225678e-11  1.959737e-07  1.959737e-07
                            HES4       3.853992 -0.441433 -0.568167 -0.314699  2.624009e-11  4.195528e-07  2.097764e-07
                            ADAMTS2    4.361591  0.450830  0.313570  0.588090  2.908792e-10  4.650867e-06  1.550289e-06
                            SLC16A6    2.811462 -0.309632 -0.405290 -0.213975  5.093580e-10  8.144125e-06  2.036031e-06
                            GTF2H3     5.012298 -0.117691 -0.155903 -0.079479  3.084475e-09  4.931767e-05  9.863535e-06
                            NSMCE1     3.060265 -0.137044 -0.184554 -0.089534  2.626869e-08  4.200100e-04  7.000167e-05
                            PXN        2.333824  0.216647  0.141105  0.292190  3.136818e-08  5.015458e-04  7.164940e-05
                            WDR64      2.860880  0.613933  0.395984  0.831882  5.372907e-08  8.590741e-04  1.073843e-04
                            AGBL1      3.772189 -0.504876 -0.685917 -0.323834  7.204624e-08  1.151947e-03  1.279942e-04
                            TMEM107    3.415203 -0.157574 -0.214849 -0.100300  1.061507e-07  1.697243e-03  1.697243e-04
            Inhibitory      KCNMB4     6.066520  0.141121  0.098561  0.183681  2.002561e-10  2.967796e-06  2.967796e-06
                            GTF2H3     5.094409 -0.107004 -0.144692 -0.069317  4.243315e-08  6.288593e-04  3.144297e-04
                            ACVR1C     3.547316  0.192773  0.121214  0.264332  1.901219e-07  2.817607e-03  6.831124e-04
                            HMGN2      5.191573 -0.197587 -0.271543 -0.123631  2.377790e-07  3.523885e-03  6.831124e-04
                            ODF2L      7.314679  0.100348  0.062629  0.138066  2.661022e-07  3.943634e-03  6.831124e-04
                            EPB41      6.621838  0.194116  0.121046  0.267185  2.765637e-07  4.098674e-03  6.831124e-04
                            ATRNL1    10.726753  0.138231  0.085758  0.190704  3.451537e-07  5.115177e-03  7.307396e-04
                            SCN1A      9.265393  0.101581  0.062168  0.140993  6.039789e-07  8.950967e-03  1.028679e-03
                            MKLN1      8.399612  0.087267  0.053362  0.121171  6.247038e-07  9.258110e-03  1.028679e-03
                            FBN1       7.445120  0.108958  0.066150  0.151765  8.242201e-07  1.221494e-02  1.221494e-03
            Microglia-PVM   PTPRG      7.234962  0.935920  0.698637  1.173203  6.533596e-14  7.957267e-10  7.957267e-10
                            FLT1       5.147248  0.673443  0.403901  0.942984  1.289855e-06  1.570915e-02  5.964511e-03
                            RASGRP3    5.720994  0.372904  0.221869  0.523939  1.704078e-06  2.075396e-02  5.964511e-03
                            EMID1      5.345206 -0.352326 -0.496177 -0.208475  2.050536e-06  2.497348e-02  5.964511e-03
                            LPL        3.858617  0.551977  0.322067  0.781888  3.210405e-06  3.909952e-02  5.964511e-03
                            CHI3L1     3.882210  0.594699  0.346633  0.842765  3.315105e-06  4.037467e-02  5.964511e-03
                            SLC6A6     5.783228  0.248531  0.144703  0.352360  3.428161e-06  4.175158e-02  5.964511e-03
                            PSTPIP1    3.721334  0.429621  0.245999  0.613242  5.595074e-06  6.814241e-02  8.517801e-03
                            NOD1       5.499973 -0.210230 -0.302805 -0.117654  1.030895e-05  1.255528e-01  1.395031e-02
                            PARD3B     7.042697 -0.247962 -0.362543 -0.133381  2.582953e-05  3.145779e-01  3.145779e-02
            OPC             SLC38A2    6.653175  0.194823  0.122709  0.266936  1.757402e-07  2.235064e-03  2.235064e-03
                            GPR158     7.422594 -0.287326 -0.396795 -0.177858  3.792317e-07  4.823069e-03  2.411534e-03
                            AGMO       6.349418  0.282226  0.157518  0.406934  1.103279e-05  1.403151e-01  4.677169e-02
            Oligodendrocyte PLD5       7.984329  0.286385  0.192032  0.380739  5.091079e-09  6.924885e-05  3.428025e-05
                            FANCB      5.223941  0.233905  0.156046  0.311765  7.177529e-09  9.762875e-05  3.428025e-05
                            SLC25A37   5.186938 -0.198906 -0.265219 -0.132593  7.560708e-09  1.028408e-04  3.428025e-05
                            FFAR1      1.663535 -0.473748 -0.642194 -0.305302  5.630617e-08  7.658765e-04  1.575972e-04
                            GIPR       3.609557  0.576544  0.371350  0.781739  5.793161e-08  7.879858e-04  1.575972e-04
                            FANCC      6.105682  0.306102  0.194344  0.417860  1.204450e-07  1.638293e-03  2.730488e-04
                            KCNIP3     4.254985  0.365943  0.227512  0.504374  3.154965e-07  4.291384e-03  6.130548e-04
                            ZNF488     5.028610  0.290161  0.179540  0.400781  3.864819e-07  5.256927e-03  6.571159e-04
                            P3H2       3.754451  0.347828  0.210701  0.484955  8.976414e-07  1.220972e-02  1.237660e-03
                            C8orf82    3.562671  0.216019  0.130570  0.301468  9.738298e-07  1.324603e-02  1.237660e-03
                            '''

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
'''             p400
Astrocyte        156
Endothelial        0
Excitatory       610
Inhibitory       352
Microglia-PVM      1
OPC                2
Oligodendrocyte   44
'''
overlaps = pd.merge(
    DE_limma_voom.query('fdr < 0.10'),
    DE_tain.query('fdr < 0.10'),
    left_index=True, right_index=True, how='left', suffixes=('_limma', '_tain'))\
        .groupby(level='cell_type').apply(lambda x: pd.Series({
            'overlap_count': (x['logFC_limma'] * x['logFC_tain'] > 0).sum(),
            'total_genes_fdr': len(set(x.index.get_level_values('gene'))),
            'percent': (x['logFC_limma'] * x['logFC_tain'] > 0).sum() / len(set(x.index.get_level_values('gene')))
    })
).reset_index()
print(overlaps)
'''         cell_type  overlap_count  total_genes_fdr   percent
0        Astrocyte          304.0            795.0  0.382390
1      Endothelial            2.0             13.0  0.153846
2       Excitatory          981.0           2515.0  0.390060
3       Inhibitory          523.0           3005.0  0.174043
4    Microglia-PVM            8.0             24.0  0.333333
5              OPC            2.0             11.0  0.181818
6  Oligodendrocyte          137.0           2133.0  0.064229
'''
# differences here are possibly due to Tain additionally using braak and cerad as filters for case-controls status 