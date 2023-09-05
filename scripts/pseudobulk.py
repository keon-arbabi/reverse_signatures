import anndata as ad, numpy as np, pandas as pd, scanpy as sc, sys, os, gc, warnings
sys.path.append(os.path.expanduser('~wainberg'))
from utils import Timer, get_pseudobulk
warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures/data')

# Anndata format:
# anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html
# Use CSR rather than CSC for faster pseudobulking:
# e.g. pseudobulking AD took 32 hours with CSC but only 31 minutes with CSR

################################################################################
# Load SEAAD Alzheimer's data
# Documentation: https://portal.brain-map.org/explore/seattle-alzheimers-disease/seattle-alzheimers-disease-brain-cell-atlas-download
# Download: https://cellxgene.cziscience.com/collections/1ca90a2d-2943-483d-b678-b809bf464c30
# Metadata columns: https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/b0/a8/b0a81899-7241-4814-8c1f-4a324db51b0b/sea-ad_donorindex_dataelementbriefdescriptions.pdf
################################################################################

# select either DLPFC or MTG
region = 'DLPFC'

data_file = f'single-cell/SEAAD/{region}/SEAAD-{region}.h5ad'
if os.path.exists(data_file):
    with Timer(f'[SEAAD {region}] Loading AD data'):
        data = sc.read(data_file)
else:
    with Timer(f'[SEAAD {region}] Preprocessing AD data'):
        from scipy.sparse import vstack
        cell_types = pd.Index(['Astrocyte', 'Chandelier', 'Endothelial', 'L23_IT', 'L4_IT',
                                'L56_NP', 'L5_ET', 'L5_IT', 'L6_CT', 'L6_IT',
                                'L6_IT_Car3', 'L6b', 'Lamp5', 'Lamp5_Lhx6',
                                'Microglia-PVM', 'OPC', 'Oligodendrocyte', 'Pax6', 'Pvalb',
                                'Sncg', 'Sst', 'Sst_Chodl', 'VLMC', 'Vip'])
        data_per_cell_type = {}
        for cell_type in cell_types:
            print(f'[SEAAD {region}] Loading {cell_type}...')
            data_per_cell_type[cell_type] = sc.read(f'single-cell/SEAAD/{region}/{cell_type}.h5ad')
            # check that there is no file name mismatch
            assert all(data_per_cell_type[cell_type].obs['Subclass'].str.replace('/', '')\
                       .str.replace(' ', '_') == cell_type),\
                        f"Mismatch detected in {cell_type}"
            counts = data_per_cell_type[cell_type].raw.X
            del data_per_cell_type[cell_type].raw
            # convert to int32, but make sure all entries were integers first
            counts_int32 = counts.astype(np.int32)
            assert not (counts != counts_int32).nnz
            data_per_cell_type[cell_type].X = counts_int32
            data_per_cell_type[cell_type].var = \
                data_per_cell_type[cell_type].var\
                    .reset_index()\
                    .astype({'feature_name': str})\
                    .set_index('feature_name')\
                    [['gene_ids']]\
                    .rename_axis('gene')\
                    .rename(columns={'gene_ids': 'Ensembl_ID'})
            # QC per cell type; there are memory issues doing this for the whole dataset
            # should already be performed by original authors 
            num_cells_i = data_per_cell_type[cell_type].shape[0]
            sc.pp.filter_cells(data_per_cell_type[cell_type], min_genes=200)
            mitochondrial_genes = data_per_cell_type[cell_type].var.index.str.startswith('MT-')
            percent_mito = data_per_cell_type[cell_type][:, mitochondrial_genes].X.sum(axis=1).A1 / \
                data_per_cell_type[cell_type].X.sum(axis=1).A1
            data_per_cell_type[cell_type] = data_per_cell_type[cell_type][percent_mito < 0.05]
            num_cells_f = data_per_cell_type[cell_type].shape[0]
            print(f"[SEAAD {region}] Dropped {num_cells_i - num_cells_f} {cell_type} cells after QC.")
            # clean-up
            del data_per_cell_type[cell_type].uns
            del data_per_cell_type[cell_type].obsm
            del data_per_cell_type[cell_type].obsp
            gc.collect()
        # combine across cell types 
        print(f'[SEAAD {region}] Merging...')
        X = vstack([data_per_cell_type[cell_type].X
                    for cell_type in cell_types])
        obs = pd.concat([data_per_cell_type[cell_type].obs
                        for cell_type in cell_types])
        var = data_per_cell_type[cell_types[0]].var
        assert all(var.equals(data_per_cell_type[cell_type].var)
                for cell_type in cell_types[1:])
        data = ad.AnnData(X=X, obs=obs, var=var, dtype=X.dtype)
        
        print(f'[SEAAD {region}] Joining with external metadata...')
        pseudo_progression_scores = pd.read_csv('single-cell/SEAAD/pseudoprogression_scores_meta.csv')
        metadata = pd.read_excel('single-cell/SEAAD/sea-ad_cohort_donor_metadata.xlsx')\
            .drop(['CERAD score','LATE','APOE4 Status','Cognitive Status',
                'Overall AD neuropathological Change','Highest Lewy Body Disease',
                'Thal','Braak'], axis=1)
        data.obs = data.obs\
            .drop(['PMI', 'Years of education', 'sex', 'Age at death'], axis=1)\
            .assign(broad_cell_type=lambda df: np.where(
                df.Class == 'Non-neuronal and Non-neural', df.Subclass, np.where(
                df.Class == 'Neuronal: Glutamatergic', 'Excitatory', np.where(
                df.Class == 'Neuronal: GABAergic', 'Inhibitory', np.nan))))\
            .reset_index()\
            .merge(pseudo_progression_scores, on = 'donor_id', how='left')\
            .merge(metadata, left_on='donor_id', right_on='Donor ID', how='left')\
            .drop('Donor ID', axis=1)\
            .loc[:, lambda df: ~df.columns.str.contains('choice')]\
            .set_index('exp_component_name')\
            .astype({col: pd.CategoricalDtype(categories=cat, ordered=True) 
                     for col, cat in {
                        'Cognitive status': ['Reference', 'No dementia', 'Dementia'],
                        'ADNC': ['Reference', 'Not AD', 'Low', 'Intermediate', 'High'],
                        'Braak stage': ['Reference', 'Braak 0', 'Braak II', 'Braak III', 'Braak IV', 'Braak V', 'Braak VI'],
                        'Thal phase': ['Reference', 'Thal 0', 'Thal 1', 'Thal 2', 'Thal 3', 'Thal 4', 'Thal 5'],
                        'CERAD score': ['Reference', 'Absent', 'Sparse', 'Moderate', 'Frequent'],
                        'LATE-NC stage': ['Staging Precluded by FTLD with TDP43 or ALS/MND or TDP-43 pathology is unclassifiable',
                                          'Reference', 'Not Identified','LATE Stage 1', 'LATE Stage 2', 'LATE Stage 3'],
                        'Atherosclerosis': ['Mild', 'Moderate', 'Severe'],
                        'Arteriolosclerosis': ['Mild', 'Moderate', 'Severe']}.items()})\
            .assign(**{
                'PMI': lambda df: df['PMI'].fillna(df['PMI'].median())\
                    .astype(float),
                'Fresh Brain Weight': lambda df: df['Fresh Brain Weight']
                    .replace({'Unavailable': pd.NA})
                    .astype('Int64'),
                'APOE4 status': lambda df: df['APOE4 status']\
                    .eq('Y'),
                'Neurotypical reference': lambda df: df['Neurotypical reference']\
                    .eq('True'),
                'ACT': lambda df: df['Primary Study Name']\
                    .eq('ACT'),
                'ADRC Clinical Core': lambda df: df['Primary Study Name'].eq('ADRC Clinical Core')})\
            .assign(**{key: lambda df, key=key: pd.to_numeric(df[key].replace('90+', '90')).astype('Int64')
                    for key in ('Age at Death','Age of onset cognitive symptoms','Age of Dementia diagnosis')})\
            .astype({'assay': 'category', 'Brain pH': float, 'Fraction mitochrondrial UMIs': float, 
                     'Genes detected': 'Int64', 'Last CASI Score': 'Int32', 'Lewy body disease pathology': 'category', 
                     'Microinfarct pathology': 'category', 'Number of UMIs': 'Int64', 'RIN': float, 'Sex': 'category', 
                     'Specimen ID': 'category', 'Subclass': 'category', 'Supertype': 'category', 
                     'Total Microinfarcts (not observed grossly)': 'Int32', 'Total microinfarcts in screening sections': 'Int32',
                     'Years of education': 'Int32', 'assay': 'category', 'broad_cell_type': 'category', 'cell_type': 'category', 
                     'disease': 'category', 'self_reported_ethnicity': 'category', 'tissue': 'category'})\
            .pipe(lambda df: pd.concat([df, pd.get_dummies(df['Subclass']).groupby(df['donor_id']).transform('sum').add_suffix('_num')], axis=1))

        print(f'[SEAAD {region}] Saving...')
        data.write(data_file)
        
if not os.path.exists(f'pseudobulk/SEAAD-{region}-broad.h5ad'):
    with Timer(f'[SEAAD {region}] Pseudobulking and saving'):
            pseudobulk = get_pseudobulk(data, 'donor_id', 'broad_cell_type', return_anndata=True)
            pseudobulk.write(f'pseudobulk/SEAAD-{region}-broad.h5ad')

################################################################################
# Load p400 Alzheimer's data
# Documentation and data: https://www.synapse.org/#!Synapse:syn23650894
# AnnData object provided by Vilas Menon
################################################################################

with Timer('[p400] Loading AD data'):
    data = sc.read('single-cell/p400/p400_qced_shareable.h5ad')
    tmp1 = data.obs\
        .assign(broad_cell_type=lambda df: df.subset
            .replace({
                'CUX2+': 'Excitatory',
                'Astorcytes': 'Astrocyte',
                'Microglia': 'Microglia-PVM',
                'Oligodendrocytes': 'Oligodendrocyte'})).astype('category')

    print('[p400] Joining with external metadata...')
    # check that all duplicate projids have the same metadata
    assert not any(pd.read_csv('single-cell/p400/dataset_978_basic_04-21-2023_ordered.csv')\
                    [lambda df: df.duplicated('projid', keep=False)]\
                    .groupby('projid').apply(lambda x: x.nunique() > 1).sum(axis=1) > 0)
    metadata = pd.read_csv('single-cell/p400/dataset_978_basic_04-21-2023_ordered.csv')\
        .drop_duplicates(subset='projid')\
        .drop(['braaksc','ceradsc','pmi','niareagansc','msex'], axis=1)
    tmp = tmp1\
        .reset_index()\
        .merge(metadata, on='projid', how='left')\
        .apply(lambda col: col.replace({' ': pd.NA, 'NA': pd.NA, '<NA>': pd.NA, 'nan': pd.NA}))\
        .dropna(axis=1, how='all')\
        .set_index('index')\
        .astype({'braaksc': 'Int32', 'ceradsc': 'Int32', 'niareagansc': 'Int32',
                'cogdx': 'Int32', 'dlbdx': 'Int32', 'arteriol_scler': 'Int32',
                'caa_4gp': 'Int32', 'cvda_4gp2': 'Int32', 'tomm40_hap': 'Int32',
                'tdp_st4': 'Int32'})\
        .astype({'study': 'category', 'apoe_genotype': 'category', 'amyloid': float, 
                'plaq_d': float, 'plaq_n': float, 'braaksc': 'category', 'ceradsc': 'category', 
                'gpath': float, 'niareagansc': 'category', 'tangles': float, 'nft': float,
                'cogdx': 'category', 'age_bl': float, 'age_death': float, 'pmi': float, 
                'race7': 'category', 'educ': 'Int32', 'spanish': 'category', 'ldai_bl': float,
                'smoking': 'category', 'cancer_bl': 'category', 'headinjrloc_bl': 'category', 
                'thyroid_bl': 'category', 'agreeableness': 'Int32', 'conscientiousness': 'Int32', 
                'extraversion_6': 'Int32', 'neuroticism_12': 'Int32', 'openness': 'Int32', 
                'chd_cogact_freq': float, 'lifetime_cogact_freq_bl': float, 'ma_adult_cogact_freq': float, 
                'ya_adult_cogact_freq': float, 'hspath_typ': 'category', 'dlbdx': 'category', 
                'arteriol_scler': 'category', 'caa_4gp': 'category', 'cvda_4gp2': 'category', 
                'ci_num2_gct': 'category', 'ci_num2_mct': 'category', 'emotional_neglect': 'Int32',
                'family_pro_sep': 'Int32', 'financial_need': 'Int32', 'parental_intimidation': 'Int32',
                'parental_violence': 'Int32', 'tot_adverse_exp': 'Int32', 'angerin': 'Int32', 
                'angerout': 'Int32', 'angertrait': 'Int32', 'disord_regiment': 'Int32', 
                'explor_rigid': 'Int32', 'extrav_reserv': 'Int32', 'haanticipatoryworry': 'Int32', 
                'hafatigability': 'Int32', 'hafearuncetainty': 'Int32', 'harmavoidance': 'Int32', 
                'hashyness': 'Int32', 'impul_reflect': 'Int32', 'nov_seek': 'Int32', 'tomm40_hap': 'category', 
                'age_first_ad_dx': float, 'marital_now_bl': 'category', 'agefirst': 'Int32', 
                'agelast': 'Int32', 'menoage': 'Int32', 'mensage': 'Int32', 'natura': 'category', 
                'othspe00': 'category', 'whatwas': 'category', 'med_con_sum_bl': 'category', 
                'ad_reagan': 'category', 'mglia123_caud_vm': float, 'mglia123_it': float, 'mglia123_mf': float,
                'mglia123_put_p': float, 'mglia23_caud_vm': float, 'mglia23_it': float, 'mglia23_mf': float,
                'mglia23_put_p': float, 'mglia3_caud_vm': float, 'mglia3_it': float, 'mglia3_mf': float,
                'mglia3_put_p': float, 'tdp_st4': 'category', 'cog_res_age12': float, 'cog_res_age40': float, 
                'tot_cog_res': float, 'early_hh_ses': float, 'income_bl': 'Int32', 'ladder_composite': float, 
                'mateduc': 'Int32', 'pareduc': 'Int32', 'pateduc': 'Int32', 
                'q40inc': 'Int32'})\
        .pipe(lambda df: pd.concat([df, pd.get_dummies(df['Subclass']).groupby(df['donor_id']).transform('sum').add_suffix('_num')], axis=1))


################################################################################
# Load Parkinson's data (10 PD/LBD individuals and 8 neurotypical controls)
# Data: singlecell.broadinstitute.org/single_cell/study/SCP1768
# QC procedure: nature.com/articles/s41593-022-01061-1#Sec28
################################################################################

PD_data_file = 'single-cell/Macosko/Macosko.h5ad'
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

SCZ_data_file = 'single-cell/SZBDMulticohort/SZBDMulticohort.h5ad'
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
# - https://cells.ucsc.edu/dlpfc-mdd/matrix.mtx.gz 
# - https://cells.ucsc.edu/dlpfc-mdd/features.tsv.gz 
# - https://cells.ucsc.edu/dlpfc-mdd/barcodes.tsv.gz 
# - https://cells.ucsc.edu/dlpfc-mdd/meta.tsv 
################################################################################

MDD_data_file = 'single-cell/Maitra/Maitra.h5ad'
if os.path.exists(MDD_data_file):
    with Timer('Loading MDD data'):
        MDD_data = sc.read(MDD_data_file)
else:
    print('Preprocessing MDD data...')
    from scipy.io import mmread  # could also use sc.read_10x_mtx()
    MDD_counts = mmread('single-cell/Maitra/matrix.mtx.gz').T.tocsr().astype(np.int32)
    MDD_genes = pd.read_table('single-cell/Maitra/features.tsv.gz', usecols=[0], index_col=0,
                              header=None, names=[None]).index
    MDD_metadata = pd.read_table('single-cell/Maitra/meta.tsv')\
        .assign(ID_column=lambda df: df.Cell.str.split(".", n=1).str[0])\
        .merge(pd.read_csv('single-cell/Maitra/Male_female_metadata_combined_Sharing_20230605.csv',
                           usecols=['ID_column','Age','PMI']), how='left')\
        .assign(broad_cell_type=lambda df: df.Broad
                .replace({'Ast':'Astrocyte',
                        'End':'Endothelial',
                        'ExN':'Excitatory',
                        'InN':'Inhibitory',
                        'Mic':'Microglia-PVM',
                        'Oli':'Oligodendrocyte',
                        'OPC':'OPC'}))\
        .astype('category')\
        .astype({'Age': int, 'PMI': float})\
        .set_index('Cell')

    assert len(MDD_genes) == 36588, len(MDD_genes)
    assert len(MDD_metadata) == 160711, len(MDD_metadata)
    assert (MDD_num_typed_cells := MDD_metadata.Cluster.notna().sum()) ==\
        160711, MDD_num_typed_cells
    assert MDD_counts.shape == (len(MDD_metadata), len(MDD_genes)), MDD_counts.shape
    MDD_data = ad.AnnData(MDD_counts, obs=MDD_metadata, 
                          var=pd.DataFrame(index=MDD_genes), dtype=MDD_counts.dtype)
    # noinspection PyTypeChecker
    MDD_data.write(MDD_data_file)

################################################################################
# Define data fields we're using
################################################################################

data_files = {'AD': AD_data_file, 'PD': PD_data_file, 'SCZ': SCZ_data_file, 'MDD': MDD_data_file}
datasets = {'AD': data, 'PD': PD_data, 'SCZ': SCZ_data, 'MDD': MDD_data}
covariate_columns = {
    'AD': ['Age at death', 'sex', 'APOE4 status', 'Metadata: PMI'], 
    'PD': ['organ__ontology_label', 'sex', 'Donor_Age', 'Donor_PMI'],
    'SCZ': ['Batch', 'Gender', 'Age', 'PMI'],
    'MDD': ['Batch', 'Sex', 'Chemistry', 'Condition', 'Age', 'PMI']
    }
cell_type_column = {'AD': 'broad_cell_type', 'PD': 'broad_cell_type', 
                    'SCZ': 'broad_cell_type', 'MDD': 'broad_cell_type'}
fine_cell_type_column = {'AD': 'Supertype', 'PD': 'Cell_Type', 
                         'SCZ': 'Celltype', 'MDD': 'fine_cell_type'}
ID_column = {'AD': 'Donor ID', 'PD': 'donor_id', 
             'SCZ': 'unique_donor_ID', 'MDD': 'ID_column'}
phenotype_column = {'AD': 'disease', 'PD': 'disease__ontology_label', 
                    'SCZ': 'Phenotype', 'MDD': 'Condition'}
control_name = {'AD': 'normal', 'PD': 'normal', 
                'SCZ': 'CON', 'MDD': 'Control'}
assert (dataset_sizes := {dataset_name: dataset.shape
                          for dataset_name, dataset in datasets.items()}) == \
       {'AD': (1378211, 36517), 'PD': (434354, 41625), 
        'SCZ': (468727, 17658), 'MDD': (156911, 36588)}, dataset_sizes

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
# {'AD': (1378211, 36517), 'PD': (434354, 41625), 'SCZ': (468727, 17658), 'MDD': (156911, 36588)}

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
# {'AD': (1378207, 36517), 'PD': (376235, 41625), 'SCZ': (444796, 17658), 'MDD': (156911, 36588)}





pseudobulks = {}
for dataset_name, dataset in datasets.items():
    data_directory = data_files[dataset_name].split("/")[0]
    pseudobulk_file = f'pseudobulk/{data_directory}-pseudobulk.h5ad'
    # if os.path.exists(pseudobulk_file): continue
    with Timer(f'[{dataset_name}] Pseudobulking'):
         pseudobulks[dataset_name] = pseudobulk(
             dataset, ID_column[dataset_name], cell_type_column[dataset_name])
    with Timer(f'[{dataset_name}] Saving pseudobulk'):
        # noinspection PyTypeChecker
        pseudobulks[dataset_name].write(pseudobulk_file)

print_dataset_sizes('After pseudobulking', pseudobulks)
# {'AD': (709, 36517), 'PD': (144, 41625), 'SCZ': (1052, 17658)}

