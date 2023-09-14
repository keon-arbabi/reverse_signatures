import numpy as np, pandas as pd, scanpy as sc, warnings, os, sys
from functools import reduce
from itertools import combinations, product
from rpy2.robjects import Formula, r
from scipy.stats import pearsonr
sys.path.append('projects/reverse_signatures/scripts')
from utils import array_to_rvector, series_to_rvector, df_to_rdf, df_to_rmatrix,  \
    rmatrix_to_df, rdf_to_df, bonferroni, fdr, percentile_transform, Timer
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures/')

r.library('edgeR', quietly=True)
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
    if 'SEAAD' in trait:
        pseudobulk.obs['Dx'] = np.where(
            pseudobulk.obs['Consensus Clinical Dx (choice=Alzheimers disease)'] == 'Checked', 1,
            np.where(pseudobulk.obs['Consensus Clinical Dx (choice=Control)'] == 'Checked', 0, 0)
        )
        pseudobulks[trait] = pseudobulk[pseudobulk.obs['ACT'] | pseudobulk.obs['ADRC Clinical Core']]
    if trait == 'p400':
        pseudobulk.obs['Dx'] = np.where(pseudobulk.obs['cogdx'].isin([4, 5]), 1, 0)

covariate_columns = {'SEAAD-MTG': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'num_cells'],
                     'SEAAD-DLPFC': ['Age at Death', 'Sex', 'APOE4 status', 'PMI', 'ACT', 'num_cells'],
                     'p400': ['age_death', 'sex', 'apoe_genotype', 'pmi', 'num_cells']} #TODO p400 batch or study?
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
    {'SEAAD-MTG': 16421, 'SEAAD-DLPFC': 16421, 'p400': 16421}, number_of_genes
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

sufficiently_expressed_genes = {cell_type: 
    reduce(pd.Index.intersection,
           ((dataset := cell_type_pseudobulks[trait, cell_type]).var.index[
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
                group=diagnosis_and_covariate_matrix.rx2('diagnosis')))

################################################################################
# Differential expression with limma-voom
# - voomByGroup: https://github.com/YOU-k/voomByGroup/blob/main/voomByGroup.R
################################

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

def print_num_hits(DE, method='bonferroni', threshold=0.05):
    print(DE[method].lt(threshold).groupby(['trait', 'cell_type']).sum()
          .unstack().T.rename_axis(None).rename_axis(None, axis=1)
          .applymap('{:.0f}'.format).replace({'nan': ''}))

def print_hits(DE, method='bonferroni', threshold=0.05):
    print(DE.query(f'{method} < {threshold}').to_string())

make_voom_plots = True
if make_voom_plots:
    os.makedirs('results/voom', exist_ok=True)

DE_limma_vooms = {}
log_CPMs = {}
for trait, cell_type in normalized_count_matrices:
    with Timer(f'[{trait}, {cell_type}] Running limma-voom'):
        normalized_count_matrix = \
            normalized_count_matrices[trait, cell_type]
        design_matrix = design_matrices[trait, cell_type]
        group = group_vectors[trait, cell_type]
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

DE_limma_voom = aggregate(DE_limma_vooms)
# save results
DE_limma_voom.to_csv('results/voom/limma_voom.tsv.gz', sep='\t',
                     compression='gzip')

print_calibration(DE_limma_voom)
'''
                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte              0.35      0.43   0.3
Endothelial            0.43      0.49  0.44
Excitatory             0.22      0.23  0.24
Inhibitory             0.16      0.21  0.33
Microglia-PVM           0.3      0.47  0.36
OPC                    0.34      0.51  0.41
Oligodendrocyte         0.2      0.38  0.29
'''
print_calibration(DE_limma_voom
                  .assign(percentile=lambda df: df
                          .groupby(['trait', 'cell_type'])
                          .AveExpr
                          .transform(percentile_transform))
                  .query('percentile < 10'))
'''                SEAAD-DLPFC SEAAD-MTG  p400
Astrocyte              0.42      0.51   0.4
Endothelial            0.45      0.54  0.46
Excitatory             0.24      0.27  0.35
Inhibitory             0.21      0.27  0.45
Microglia-PVM          0.31      0.49  0.45
OPC                    0.41      0.51  0.47
Oligodendrocyte        0.26      0.45  0.37
'''
print_num_hits(DE_limma_voom)
'''                SEAAD-DLPFC SEAAD-MTG p400
Astrocyte                 0         1   27
Endothelial               1         0    2
Excitatory                0         1   45
Inhibitory                0         0   18
Microglia-PVM             2         0    5
OPC                       1         0    2
Oligodendrocyte           4         0   16
'''
print_num_hits(DE_limma_voom, method='fdr', threshold=0.01)
'''                SEAAD-DLPFC SEAAD-MTG p400
Astrocyte                 0         0   89
Endothelial               0         0    1
Excitatory                0         1  316
Inhibitory                0         0   65
Microglia-PVM             0         0    6
OPC                       0         0    2
Oligodendrocyte           0         0   50
'''
print_hits(DE_limma_voom)
'''                                    AveExpr     logFC      CI.L      CI.R             p    bonferroni           fdr
trait       cell_type       gene                                                                                       
SEAAD-MTG   Excitatory      ARHGAP24   4.472153  0.703682  0.458751  0.948613  1.783623e-07  2.851835e-03  2.851835e-03
            Astrocyte       COL8A1     1.311568  1.427536  0.861414  1.993657  3.058992e-06  4.299108e-02  4.299108e-02
SEAAD-DLPFC Oligodendrocyte EPS8       6.671009  0.421575  0.260397  0.582754  1.553504e-06  2.113077e-02  1.022017e-02
                            SALL1      5.283115  0.456733  0.281185  0.632281  1.733134e-06  2.357409e-02  1.022017e-02
                            ZDHHC2     6.051999  0.385488  0.234092  0.536885  2.682728e-06  3.649047e-02  1.022017e-02
                            SOCS6      5.462246  0.340261  0.205860  0.474662  3.005491e-06  4.088069e-02  1.022017e-02
            Microglia-PVM   PPARG      6.935466  0.709132  0.435152  0.983112  1.921651e-06  2.340379e-02  2.340379e-02
                            LPL        3.866582  1.189717  0.713058  1.666376  3.958602e-06  4.821182e-02  2.410591e-02
            OPC             FGL1       3.039806  1.305035  0.809457  1.800612  1.310536e-06  1.666739e-02  1.666739e-02
            Endothelial     CHSY1      9.541515  0.510808  0.307402  0.714215  3.649317e-06  2.601963e-02  2.601963e-02
p400        Excitatory      NRIP2      2.273171  0.350461  0.251581  0.449341  1.221322e-11  1.952772e-07  1.820365e-07
                            HES4       3.853992 -0.445365 -0.572815 -0.317914  2.277021e-11  3.640729e-07  1.820365e-07
                            ADAMTS2    4.361591  0.451514  0.313815  0.589212  3.098764e-10  4.954615e-06  1.651538e-06
                            SLC16A6    2.811462 -0.310628 -0.406931 -0.214324  5.808111e-10  9.286589e-06  2.321647e-06
                            GTF2H3     5.012298 -0.118690 -0.157136 -0.080245  2.837199e-09  4.536398e-05  9.072796e-06
                            PXN        2.333824  0.224075  0.146968  0.301182  2.085328e-08  3.334231e-04  5.557052e-05
                            NSMCE1     3.060265 -0.137391 -0.184933 -0.089848  2.479424e-08  3.964351e-04  5.663359e-05
                            WDR64      2.860880  0.619980  0.401806  0.838154  4.135710e-08  6.612587e-04  8.265733e-05
                            AGBL1      3.772189 -0.505036 -0.686897 -0.323174  8.126429e-08  1.299335e-03  1.443705e-04
                            TMEM107    3.415203 -0.158131 -0.215535 -0.100726  1.023439e-07  1.636377e-03  1.521933e-04
                            PHKA1      2.421737  0.269364  0.171501  0.367226  1.047049e-07  1.674126e-03  1.521933e-04
                            LAP3       2.913163 -0.151146 -0.206904 -0.095388  1.603314e-07  2.563539e-03  2.031866e-04
                            MAP3K21    5.563942  0.104855  0.066051  0.143659  1.749253e-07  2.796881e-03  2.031866e-04
                            CUBN       3.175294  0.271233  0.170794  0.371672  1.779106e-07  2.844613e-03  2.031866e-04
                            ZDHHC23    4.633151 -0.165890 -0.227531 -0.104250  1.952784e-07  3.122307e-03  2.081538e-04
                            LAMC1      6.023220  0.134074  0.083629  0.184519  2.732084e-07  4.368330e-03  2.730206e-04
                            RIIAD1     1.901533 -0.221604 -0.305271 -0.137936  2.993963e-07  4.787048e-03  2.815910e-04
                            SYTL1      0.632933  0.250189  0.154989  0.345389  3.673190e-07  5.873063e-03  3.262813e-04
                            HMGN2      4.790745 -0.175264 -0.242499 -0.108028  4.532755e-07  7.247423e-03  3.643546e-04
                            DNAH11     2.077512  0.337256  0.207848  0.466663  4.557566e-07  7.287093e-03  3.643546e-04
                            PHYH       3.645417 -0.154216 -0.213807 -0.094626  5.449043e-07  8.712476e-03  4.148798e-04
                            TNS3       2.515089  0.379703  0.232516  0.526890  5.902480e-07  9.437476e-03  4.289762e-04
                            MGAT4A     6.561639  0.087859  0.053532  0.122186  7.188083e-07  1.149303e-02  4.996967e-04
                            SLA        1.317052  0.310719  0.188356  0.433082  8.729628e-07  1.395780e-02  5.403573e-04
                            SFT2D2     2.695843  0.274050  0.166125  0.381975  8.734847e-07  1.396615e-02  5.403573e-04
                            NBEAL2     2.678562  0.172651  0.104642  0.240661  8.786847e-07  1.404929e-02  5.403573e-04
                            TMEM43     4.746778  0.086516  0.052302  0.120729  9.668095e-07  1.545832e-02  5.725303e-04
                            HSD11B2    2.049518  0.408628  0.246664  0.570592  1.021329e-06  1.633003e-02  5.832154e-04
                            STK4       6.229272  0.082329  0.049529  0.115129  1.155227e-06  1.847092e-02  6.158380e-04
                            PPEF1      2.959489 -0.270879 -0.378798 -0.162961  1.155491e-06  1.847514e-02  6.158380e-04
                            HNRNPLL    6.584434  0.136971  0.082195  0.191748  1.264494e-06  2.021799e-02  6.521932e-04
                            PRDM6      2.768874  0.219181  0.131142  0.307219  1.402935e-06  2.243152e-02  7.008813e-04
                            TERT      -1.301100 -0.451983 -0.633767 -0.270198  1.446562e-06  2.312908e-02  7.008813e-04
                            ADAMTSL3   5.141912  0.173024  0.103206  0.242841  1.562320e-06  2.497993e-02  7.347039e-04
                            EDA        3.945990  0.208388  0.123681  0.293095  1.852141e-06  2.961388e-02  8.417938e-04
                            DNAJC18    5.555817 -0.108049 -0.152017 -0.064080  1.899534e-06  3.037166e-02  8.417938e-04
                            TRIM54     4.062666 -0.242342 -0.341112 -0.143572  1.968600e-06  3.147595e-02  8.417938e-04
                            PDYN       1.079854  0.414822  0.245636  0.584008  2.000636e-06  3.198816e-02  8.417938e-04
                            TGFBR3     3.290432  0.195718  0.115668  0.275768  2.134038e-06  3.412113e-02  8.569404e-04
                            SCGN       0.995607  0.513189  0.303250  0.723129  2.143825e-06  3.427762e-02  8.569404e-04
                            TPM3       6.244188 -0.100744 -0.142047 -0.059440  2.253501e-06  3.603123e-02  8.788104e-04
                            NR3C1      8.281277  0.120965  0.071148  0.170782  2.494523e-06  3.988493e-02  9.336052e-04
                            NF2        5.197970  0.094544  0.055571  0.133517  2.547773e-06  4.073634e-02  9.336052e-04
                            CAPN7      6.393924  0.086590  0.050883  0.122298  2.569180e-06  4.107863e-02  9.336052e-04
                            LSAMP     11.657400  0.153227  0.089746  0.216708  2.850640e-06  4.557888e-02  1.012864e-03
            Inhibitory      KCNMB4     6.066520  0.142780  0.099976  0.185583  1.575862e-10  2.335427e-06  2.335427e-06
                            GTF2H3     5.094409 -0.108844 -0.146925 -0.070763  3.467793e-08  5.139269e-04  2.569635e-04
                            ACVR1C     3.547316  0.195810  0.124103  0.267517  1.307555e-07  1.937796e-03  6.459320e-04
                            ATRNL1    10.726753  0.139734  0.087569  0.191899  2.214851e-07  3.282410e-03  7.061879e-04
                            HMGN2      5.191573 -0.199575 -0.274282 -0.124867  2.382550e-07  3.530940e-03  7.061879e-04
                            EPB41      6.621838  0.193555  0.120380  0.266729  3.098456e-07  4.591912e-03  7.312081e-04
                            ODF2L      7.314679  0.100199  0.062085  0.138313  3.639304e-07  5.393449e-03  7.312081e-04
                            SCN1A      9.265393  0.102839  0.063598  0.142080  3.947143e-07  5.849665e-03  7.312081e-04
                            MKLN1      8.399612  0.087602  0.053697  0.121507  5.671658e-07  8.405398e-03  8.719263e-04
                            FBN1       7.445120  0.110442  0.067635  0.153249  5.883443e-07  8.719263e-03  8.719263e-04
                            SESN1      7.379379  0.117464  0.070497  0.164430  1.258374e-06  1.864910e-02  1.644367e-03
                            REEP3      5.271958  0.110541  0.066173  0.154908  1.377635e-06  2.041655e-02  1.644367e-03
                            HES4       4.414816 -0.267291 -0.374784 -0.159799  1.442427e-06  2.137677e-02  1.644367e-03
                            FSD1L      6.402078  0.084425  0.050327  0.118523  1.594663e-06  2.363291e-02  1.688065e-03
                            SCCPDH     4.974599 -0.087742 -0.123367 -0.052117  1.803165e-06  2.672290e-02  1.731919e-03
                            MPV17      4.696606 -0.112718 -0.158556 -0.066880  1.869818e-06  2.771070e-02  1.731919e-03
                            ST8SIA1    7.709185  0.112471  0.065759  0.159183  3.011127e-06  4.462490e-02  2.558222e-03
                            DSCC1      1.895044 -0.270379 -0.382833 -0.157924  3.107152e-06  4.604799e-02  2.558222e-03
            Oligodendrocyte PLD5       7.984329  0.286052  0.191864  0.380240  4.988709e-09  6.785642e-05  3.147115e-05
                            SLC25A37   5.186938 -0.200768 -0.267272 -0.134263  6.103255e-09  8.301647e-05  3.147115e-05
                            FANCB      5.223941  0.234060  0.156227  0.311892  6.941144e-09  9.441344e-05  3.147115e-05
                            FFAR1      1.663535 -0.474918 -0.643384 -0.306451  5.253569e-08  7.145905e-04  1.538524e-04
                            GIPR       3.609557  0.577363  0.372043  0.782683  5.655507e-08  7.692621e-04  1.538524e-04
                            FANCC      6.105682  0.303990  0.192051  0.415929  1.527012e-07  2.077042e-03  3.461737e-04
                            ZNF488     5.028610  0.288204  0.177576  0.398832  4.607554e-07  6.267195e-03  8.953136e-04
                            CEMIP2     4.905084  0.228044  0.138445  0.317643  8.259931e-07  1.123516e-02  1.354127e-03
                            KCNIP3     4.254985  0.365786  0.221589  0.509982  8.959817e-07  1.218714e-02  1.354127e-03
                            P3H2       3.754451  0.347330  0.208925  0.485735  1.162272e-06  1.580923e-02  1.530704e-03
                            SLC6A9     5.179841  0.242246  0.145460  0.339033  1.237887e-06  1.683775e-02  1.530704e-03
                            ZNF33B     4.016982  0.234877  0.140389  0.329365  1.455750e-06  1.980112e-02  1.635341e-03
                            NRIP2      3.878943  0.248830  0.148425  0.349236  1.562964e-06  2.125944e-02  1.635341e-03
                            PTGDS     10.183929 -0.185332 -0.260424 -0.110240  1.720403e-06  2.340092e-02  1.671494e-03
                            C8orf82    3.562671  0.212242  0.125948  0.298536  1.863869e-06  2.535235e-02  1.690157e-03
                            KEL        4.811180 -0.244192 -0.344493 -0.143890  2.352568e-06  3.199963e-02  1.999977e-03
            Astrocyte       CREB5      4.985435  0.435448  0.300458  0.570437  5.796519e-10  8.146428e-06  8.146428e-06
                            HMGN2      5.477695 -0.290762 -0.382705 -0.198818  1.207689e-09  1.697285e-05  8.486427e-06
                            KLF4       1.708971  0.586753  0.391934  0.781573  6.593467e-09  9.266458e-05  3.088819e-05
                            AK4        6.704768  0.186984  0.122535  0.251434  2.196174e-08  3.086503e-04  7.496156e-05
                            NOL3       4.439560 -0.224439 -0.302336 -0.146543  2.720430e-08  3.823292e-04  7.496156e-05
                            ALDH1A1    7.884295 -0.449609 -0.606487 -0.292731  3.200294e-08  4.497694e-04  7.496156e-05
                            ADAM33     3.822175  0.501726  0.322672  0.680780  6.268241e-08  8.809385e-04  1.258484e-04
                            ALK        6.731758  0.249999  0.158836  0.341162  1.162490e-07  1.633763e-03  2.042204e-04
                            HSPA2      4.739849  0.302821  0.190162  0.415481  2.020750e-07  2.839962e-03  3.155514e-04
                            INPP5D     2.682237  0.482344  0.302052  0.662636  2.293337e-07  3.223056e-03  3.223056e-04
                            ZNF441     4.129412 -0.336611 -0.463890 -0.209332  3.116005e-07  4.379233e-03  3.981121e-04
                            AP3B2      5.401773 -0.246805 -0.341281 -0.152329  4.290696e-07  6.030144e-03  5.025120e-04
                            NUP210     4.657570 -0.171596 -0.238698 -0.104494  7.347294e-07  1.032589e-02  7.942990e-04
                            SLC17A5    5.652339  0.242445  0.147348  0.337542  7.922316e-07  1.113402e-02  7.952874e-04
                            ARHGAP29   6.240597  0.310888  0.188048  0.433727  9.474649e-07  1.331567e-02  8.877115e-04
                            SDSL       2.202813 -0.405235 -0.565829 -0.244642  1.017886e-06  1.430537e-02  8.940857e-04
                            PFKP       7.614982  0.352586  0.211576  0.493596  1.266604e-06  1.780085e-02  1.047109e-03
                            PKD1L1     3.397099 -0.392585 -0.550055 -0.235115  1.358081e-06  1.908647e-02  1.055268e-03
                            ODC1       3.282332 -0.260115 -0.364999 -0.155232  1.536189e-06  2.158960e-02  1.055268e-03
                            GPCPD1     5.698213 -0.376717 -0.528654 -0.224780  1.545117e-06  2.171507e-02  1.055268e-03
                            PCDHB16    3.319616  0.263788  0.157176  0.370399  1.621978e-06  2.279527e-02  1.055268e-03
                            MRGPRF     1.315723  0.524507  0.312357  0.736656  1.651907e-06  2.321590e-02  1.055268e-03
                            SLC2A4RG   4.580885 -0.239840 -0.337086 -0.142595  1.747837e-06  2.456411e-02  1.068005e-03
                            FBXO2      6.167882  0.262857  0.155497  0.370217  2.068123e-06  2.906540e-02  1.211058e-03
                            OXR1       7.937094  0.170574  0.100179  0.240968  2.615110e-06  3.675275e-02  1.470110e-03
                            HMGN1      6.293019 -0.228871 -0.323544 -0.134198  2.754998e-06  3.871875e-02  1.489183e-03
                            CFAP221    5.319244 -0.250693 -0.354715 -0.146671  2.951867e-06  4.148554e-02  1.536501e-03
            Microglia-PVM   PTPRG      7.234962  0.936465  0.696277  1.176652  1.207153e-13  1.470191e-09  1.470191e-09
                            FLT1       5.147248  0.673180  0.403469  0.942891  1.321150e-06  1.609028e-02  5.755415e-03
                            EMID1      5.345206 -0.357279 -0.501485 -0.213073  1.571862e-06  1.914370e-02  5.755415e-03
                            RASGRP3    5.720994  0.371472  0.220340  0.522605  1.890275e-06  2.302166e-02  5.755415e-03
                            LPL        3.858617  0.553236  0.323698  0.782775  2.944660e-06  3.586302e-02  7.172604e-03
            OPC             SLC38A2    6.653175  0.197886  0.125767  0.270006  1.142121e-07  1.452549e-03  1.452549e-03
                            GPR158     7.422594 -0.288278 -0.397776 -0.178780  3.502798e-07  4.454859e-03  2.227430e-03
            Endothelial     SLC38A2    8.708283  0.303331  0.185816  0.420846  5.801024e-07  4.136130e-03  4.136130e-03
                            DLL4       6.016983  0.479121  0.277565  0.680677  3.979610e-06  2.837462e-02  1.418731e-02
                            '''