import matplotlib.pyplot as plt, numpy as np, os, sys, warnings,\
    pandas as pd, scanpy as sc, seaborn as sns, optuna
from rpy2.robjects import r
import matplotlib.gridspec as gridspec
from scipy.stats import sem, zscore
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.append('projects/reverse_signatures/scripts')
from utils import Timer, rmatrix_to_df, array_to_rmatrix, rdf_to_df

warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures')
r.library('RcppML', quietly=True)

################################################################################
# Run NMF
################################################################################

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte',\
    'Microglia-PVM', 'OPC', 'Endothelial'
study_names = 'SEAAD-MTG', 'SEAAD-DLPFC', 'p400'

k_max = 20
n_trials = 20
save_name = 'DEG_L1_fdr01'

def preprocess_data(study_name, cell_type, gene_selection='deg'):
    data = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
    data = data[data.obs['broad_cell_type'] == cell_type, :]
    if gene_selection == 'hvg':
        hvg = np.argpartition(-np.var(data.X, axis=0), 2000)[:2000]
        data = data[:, hvg].copy()
    elif gene_selection == 'deg':
        degs = pd.read_csv('results/voom/limma_voom_combined.tsv', sep='\t')\
            .query('cell_type == @cell_type & fdr < 0.01')
        degs = degs['gene'].astype(str).tolist()
        data = data[:, data.var_names.isin(degs)].copy()
    if 'SEAAD' in study_name:
        case_samples = (data.obs['Consensus Clinical Dx (choice=Alzheimers disease)'].eq('Checked')) & \
                    (data.obs['ACT'] | data.obs['ADRC Clinical Core'])
        data = data[case_samples, :].copy()
    elif study_name == 'p400':
        case_samples = data.obs['cogdx'].isin([4, 5])
        data = data[case_samples, :].copy()
    data.X = np.log1p(data.X * (1000000 / data.X.sum(axis=1))[:, None])
    data.X *= 1 / np.log(2)
    assert not np.any(data.X < 0), "Array contains negative numbers"
    return data

def nmf_objective(trial, log_CPMs_R, MSE_trial, k_1se_trial, study_name, cell_type, k_max, L1=True, L2=False):
    L1_w = trial.suggest_float('L1_w', 0.001, 0.999, log=True) if L1 else 0
    L1_h = trial.suggest_float('L1_h', 0.001, 0.999, log=True) if L1 else 0
    L2_w = trial.suggest_float('L2_w', 0.001, 0.999, log=True) if L2 else 0
    L2_h = trial.suggest_float('L2_h', 0.001, 0.999, log=True) if L2 else 0

    r.options(**{'RcppML.verbose': True})
    MSE = r.crossValidate(log_CPMs_R,
                          k=r.c(*range(1, k_max + 1)), L1=r.c(L1_w, L1_h), L2=r.c(L2_w, L2_h),
                          seed=0, reps=3, tol=1e-2, maxit=np.iinfo('int32').max)
    MSE = rdf_to_df(MSE)\
        .astype({'k': int, 'rep': int})\
        .set_index(['k', 'rep'])\
        .squeeze()\
        .rename('MSE')
    mean_MSE = MSE.groupby('k').mean()
    k_best = int(mean_MSE.idxmin())
    k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + sem(MSE[k_best])][0])
    MSE_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h] = MSE
    k_1se_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h] = k_1se
    print(f'[{study_name} {cell_type}]: {k_1se=}')
    error = mean_MSE[k_1se]
    return error

def plot_MSE(axes, idx, study_name, cell_type, MSE_trial, k_1se_trial, MSE_final, best_params):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    for (current_study, current_cell_type, L1_w, L1_h, L2_w, L2_h), MSE in MSE_trial.items():
        if current_study == study_name and current_cell_type == cell_type:
            mean_MSE = MSE.groupby('k').mean()
            k_1se = k_1se_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h]
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=0.08)
            ax.scatter(k_1se, mean_MSE[k_1se], color='black', s=16, alpha=0.08)
    mean_MSE = MSE_final.groupby('k').mean()
    k_final = k_1se_trial[study_name, cell_type, *best_params.values()]
    ax.plot(mean_MSE.index, mean_MSE.values, color='red')
    ax.scatter(k_final, mean_MSE[k_final], color='red', s=50)
    ax.set_xticks(ticks=mean_MSE.index)
    ax.set_yscale('log')
    ax.set_title(r"$\bf{" + f"{study_name}\;{cell_type}" + r"}$"
                + "\nMSE across Optuna trials\n"
                + f"Selected L1_w: {best_params['L1_w']:.3f}, L1_h: {best_params['L1_h']:.3f}, "
                + f"L2_w: {best_params['L2_w']:.3f}, L2_h: {best_params['L2_h']:.3f}")
    ax.set_xlabel('k')
    ax.set_ylabel('Mean MSE')
    ax.axis('on')

MSE_trial, k_1se_trial = {}, {}
for study_name in study_names:

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    [ax.axis('off') for ax in axes.flatten()]
    
    for idx, cell_type in enumerate(broad_cell_types):
        
        pseudobulk = preprocess_data(study_name, cell_type)
        log_CPMs_R = array_to_rmatrix(pseudobulk.X.T)
        gene_names = pseudobulk.var_names
        samp_names = pseudobulk.obs_names

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0, multivariate=True), 
                                    direction='minimize')
        study.optimize(lambda trial: nmf_objective(trial, log_CPMs_R, MSE_trial, k_1se_trial,\
            study_name, cell_type, k_max), n_trials=n_trials)

        best_params = {param: study.best_trial.params.get(param, 0)
                       for param in ['L1_w', 'L1_h', 'L2_w', 'L2_h']}
        L1_w, L1_h, L2_w, L2_h = best_params.values()
        MSE_final = MSE_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h]

        NMF_results = r.nmf(log_CPMs_R, 
                            k=k_1se_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h], 
                            L1=r.c(L1_w, L1_h), L2=r.c(L2_w, L2_h),
                            seed=0, tol=1e-5, maxit=np.iinfo('int32').max)
        
        W = rmatrix_to_df(NMF_results.slots['w']).set_axis(gene_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))
        H = rmatrix_to_df(NMF_results.slots['h']).T.set_axis(samp_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))
        #save
        os.makedirs('results/MSE', exist_ok=True)
        os.makedirs('results/NMF', exist_ok=True)
        MSE_final.to_csv(f'results/MSE/{study_name}-{cell_type}_MSE_{save_name}.tsv', sep='\t')
        W.to_csv(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', sep='\t')
        H.to_csv(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', sep='\t')

        plot_MSE(axes, idx, study_name, cell_type, MSE_trial, k_1se_trial, MSE_final, best_params)

    plt.tight_layout()
    plt.savefig(f"results/MSE/{study_name}_plots_{save_name}.png", dpi=300)

################################################################################
# Examine signatures 
################################################################################

def plot_heatmap(ax, data, color='viridis', yticks=1, xticks=1, square=False, col_order=None):
    data = data.iloc[:, col_order] if col_order is not None else data
    sns.heatmap(data, ax=ax, cmap=color, cbar_kws={'shrink': 0.5}, 
                yticklabels=yticks, xticklabels=xticks, square=square, rasterized=True)
    ytick_fontsize = 12 - 0.04 * len(ax.get_yticklabels()) if len(ax.get_yticklabels()) > 20 else 12
    xtick_fontsize = 12 - 0.04 * len(ax.get_xticklabels()) if len(ax.get_xticklabels()) > 20 else 12
    ax.tick_params(axis='y', labelsize=ytick_fontsize)
    ax.tick_params(axis='x', labelsize=xtick_fontsize)

def get_optimal_col_ordering(data_list):
    aggregated_data = pd.concat(data_list, axis=0)
    col_order = leaves_list(linkage(pdist(aggregated_data.T), 'average')) if aggregated_data.shape[1] > 1 else slice(None)
    return col_order

for study_name in study_names:
    H_data_list = []
    row_ratios = []
    for cell_type in broad_cell_types:
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0).T
        H.columns = H.columns.str.replace(f'{cell_type}_', '', regex=False)
        H_data_list.append(H)
        row_ratios.append(H.shape[0])

    col_order = get_optimal_col_ordering(H_data_list)

    fig = plt.figure(figsize=(14, sum(row_ratios)*0.7))
    gs = gridspec.GridSpec(len(broad_cell_types), 1, hspace=0.1, height_ratios=row_ratios)

    for idx, (cell_type, nrows) in enumerate(zip(broad_cell_types, row_ratios)):
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)
        ax = fig.add_subplot(gs[idx])
        plot_heatmap(ax, H.T, color='rocket', xticks=(idx == len(broad_cell_types) - 1), col_order=col_order)
        ax.set_ylabel(f'{cell_type}', rotation=90, labelpad=15, fontsize=14, fontweight='bold')
        ax.yaxis.set_label_coords(-0.15, 0.5)

    plt.savefig(f'results/{study_name}_tmp.png', dpi=400)
    plt.clf()





def minmax_scale(data):
    return (data - data.min()) / (data.max() - data.min())

def plot_heatmap(ax, data, color='viridis', yticks=1, xticks=1, square=False, col_order=None, norm=False):
    data = data.iloc[:, col_order] if col_order is not None else data
    if norm:
        data = data.apply(minmax_scale, axis=0, result_type='broadcast')
    sns.heatmap(data, ax=ax, cmap=color, cbar_kws={'shrink': 0.5}, 
                yticklabels=yticks, xticklabels=xticks, square=square, rasterized=True)
    ytick_fontsize = 12 - 0.04 * len(ax.get_yticklabels()) if len(ax.get_yticklabels()) > 20 else 12
    xtick_fontsize = 12 - 0.04 * len(ax.get_xticklabels()) if len(ax.get_xticklabels()) > 20 else 12
    ax.tick_params(axis='y', labelsize=ytick_fontsize)
    ax.tick_params(axis='x', labelsize=xtick_fontsize)

def get_optimal_col_ordering(data_list):
    aggregated_data = pd.concat(data_list, axis=0)
    col_order = leaves_list(linkage(pdist(aggregated_data.T), 'average')) if aggregated_data.shape[1] > 1 else slice(None)
    return col_order

def get_metadata(study_name, H_index):
    adata = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
    adata = adata[adata.obs['broad_cell_type'] == cell_type, :]
    if 'SEAAD' in study_name:
        cols = ['ADNC', 'Braak stage', 'Thal phase', 'Last CASI Score',
                'CERAD score', 'LATE-NC stage', 'Atherosclerosis', 'Arteriolosclerosis',
                'Lewy body disease pathology', 'Microinfarct pathology',
                'Continuous Pseudo-progression Score', 'APOE4 status', 'ACT', 
                'ADRC Clinical Core', 'Age at Death', 'Age of onset cognitive symptoms',
                'Age of Dementia diagnosis', 'Sex', 'PMI', 'Brain pH', 'RIN',
                'Fresh Brain Weight', 'Years of education', 'self_reported_ethnicity']
    if study_name == 'p400':
        cols = ['ad_reagan', 'age_death', 'age_first_ad_dx', 'amyloid', 'apoe_genotype', 
                'arteriol_scler', 'braaksc', 'caa_4gp', 'cancer_bl', 'ceradsc', 
                'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12', 'cog_res_age40', 
                'cogdx', 'cogdx_stroke', 'cogn_global_random_slope', 'cvda_4gp2', 
                'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl', 
                'ldai_bl', 'mglia123_caud_vm', 'mglia123_it', 'mglia123_mf', 'mglia123_put_p', 
                'mglia23_caud_vm', 'mglia23_it', 'mglia23_mf', 'mglia23_put_p', 'mglia3_caud_vm', 
                'mglia3_it', 'mglia3_mf', 'mglia3_put_p', 'med_con_sum_bl', 'msex', 
                'nft', 'niareagansc', 'plaq_d', 'plaq_n', 'pmi', 
                'race7', 'smoking', 'tangles', 'tdp_st4', 'thyroid_bl', 
                'tomm40_hap', 'tot_cog_res']
        cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                    'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
        cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                     'Sst_num', 'Sst Chodl_num', 'Vip_num'])
    meta = adata.obs[cols].loc[H_index]
    meta_transformed = meta.apply(lambda col: col.astype('category').cat.codes if col.dtype.name == 'category' \
                                  else col.astype(int) if col.dtype == bool \
                                  else col).astype(float).apply(lambda col: col.fillna(col.median()))
    return meta_transformed

for study_name in study_names:
    H_data_list = []
    row_ratios = []
    for cell_type in broad_cell_types:
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0).T
        H_index = H.columns
        H.columns = H.columns.str.replace(f'{cell_type}_', '', regex=False)
        H_data_list.append(H)
        row_ratios.append(H.shape[0])

    meta = get_metadata(study_name, H_index)
    col_order = get_optimal_col_ordering(H_data_list)

    fig = plt.figure(figsize=(14, (len(broad_cell_types) + 1) * 3))
    gs = gridspec.GridSpec(len(broad_cell_types) + 1, 1, hspace=0.1, height_ratios=[15] + row_ratios)  # Fixing meta's ratio to 1

    meta_ax = fig.add_subplot(gs[0])
    plot_heatmap(meta_ax, meta.T, color='YlGnBu', xticks=False, col_order=col_order, norm=True)

    for idx, (cell_type, H) in enumerate(zip(broad_cell_types, H_data_list)):
        ax = fig.add_subplot(gs[idx + 1])
        plot_heatmap(ax, H, color='rocket', xticks=(idx == len(broad_cell_types) - 1), col_order=col_order)
        ax.set_ylabel(f'{cell_type}', rotation=90, labelpad=15, fontsize=14, fontweight='bold')
        ax.yaxis.set_label_coords(-0.15, 0.5)

    plt.savefig(f'results/{study_name}_tmp1.png', dpi=400)
    plt.clf()















save_name = 'DEG_L1_fdr01'
    
def plot_heatmap(data, color='viridis', title='', yticks=1, xticks=1, square=False, optimal_ordering=True, figsize=None):
    if figsize: plt.figure(figsize=figsize)
    row_order = leaves_list(linkage(pdist(data), 'average', optimal_ordering=optimal_ordering)) if data.shape[0] > 1 else slice(None)
    col_order = leaves_list(linkage(pdist(data.T), 'average', optimal_ordering=optimal_ordering)) if data.shape[1] > 1 else slice(None)
    
    ax = sns.heatmap(data.iloc[row_order, col_order],
                     cmap=color, cbar_kws={'shrink': 0.5},
                     yticklabels=yticks, xticklabels=xticks, 
                     square=square, rasterized=True)
    
    ytick_fontsize = 12 - 0.07 * len(ax.get_yticklabels()) if len(ax.get_yticklabels()) > 20 else 12
    xtick_fontsize = 12 - 0.07 * len(ax.get_xticklabels()) if len(ax.get_xticklabels()) > 20 else 12
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=ytick_fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=xtick_fontsize)
    ax.set_title(title)
    plt.tight_layout()

def get_correlations(H, meta):
    # drop columns with constant values and convert unordered categories and bools
    non_ordered_cols = meta.select_dtypes('category').columns[~meta.select_dtypes('category').apply(lambda df: df.cat.ordered)]
    meta = meta\
        .assign(**{col: meta[col].astype(int) for col in meta.select_dtypes('bool').columns})\
        .assign(**{col_name: col.cat.codes for col_name, col in meta.select_dtypes('category').items() if col.cat.ordered})\
        .pipe(lambda df: df.join(pd.get_dummies(meta[non_ordered_cols], prefix_sep='_').astype(int)) if non_ordered_cols.any() else df)\
        .drop(columns=meta.select_dtypes('category').columns)
    meta = meta[meta.columns[meta.nunique() > 1]]
    
    cor_matrix = H.apply(lambda col_H: meta.apply(lambda col_meta: col_H.corr(col_meta, method='pearson')))
    return cor_matrix

for study_name in study_names:
    for cell_type in broad_cell_types:
        W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', index_col=0)
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)
        
        os.makedirs('results/basis', exist_ok=True)
        os.makedirs('results/coefficient', exist_ok=True)
        
        # plot_heatmap(W, color='rocket', yticks=False, optimal_ordering=True, figsize=(6,7),
        #              title=f'Basis Vectors W (Metagenes), {study_name}-{cell_type}')
        # plt.savefig(f'results/basis/{study_name}-{cell_type}_heatmap.png', dpi=400)
        # plt.clf()

        plot_heatmap(H.T, color='rocket', xticks=False, optimal_ordering=True, figsize=(8,4),
                     title=f'Coefficient Matrix H (Metasamples), {study_name}-{cell_type}')
        plt.savefig(f'results/coefficient/{study_name}-{cell_type}_heatmap.png', dpi=400)
        plt.clf()

for study_name in study_names:
    for cell_type in broad_cell_types:
        
        data = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
        data = data[data.obs['broad_cell_type'] == cell_type, :]

        if 'SEAAD' in study_name:
            cols = ['ADNC', 'Braak stage', 'Thal phase', 'Last CASI Score',
                    'CERAD score', 'LATE-NC stage', 'Atherosclerosis', 'Arteriolosclerosis',
                    'Lewy body disease pathology', 'Microinfarct pathology',
                    'Continuous Pseudo-progression Score', 'APOE4 status', 'ACT', 
                    'ADRC Clinical Core', 'Age at Death', 'Age of onset cognitive symptoms',
                    'Age of Dementia diagnosis', 'Sex', 'PMI', 'Brain pH', 'RIN',
                    'Fresh Brain Weight', 'Years of education', 'self_reported_ethnicity']
        if study_name == 'p400':
            cols = ['ad_reagan', 'age_death', 'age_first_ad_dx', 'amyloid', 'apoe_genotype', 
                    'arteriol_scler', 'braaksc', 'caa_4gp', 'cancer_bl', 'ceradsc', 
                    'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12', 'cog_res_age40', 
                    'cogdx', 'cogdx_stroke', 'cogn_global_random_slope', 'cvda_4gp2', 
                    'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl', 
                    'ldai_bl', 'mglia123_caud_vm', 'mglia123_it', 'mglia123_mf', 'mglia123_put_p', 
                    'mglia23_caud_vm', 'mglia23_it', 'mglia23_mf', 'mglia23_put_p', 'mglia3_caud_vm', 
                    'mglia3_it', 'mglia3_mf', 'mglia3_put_p', 'med_con_sum_bl', 'msex', 
                    'nft', 'niareagansc', 'plaq_d', 'plaq_n', 'pmi', 
                    'race7', 'smoking', 'tangles', 'tdp_st4', 'thyroid_bl', 
                    'tomm40_hap', 'tot_cog_res']
        if cell_type == 'Excitatory':
                cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                             'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
        if cell_type == 'Inhibitory':
            cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                            'Sst_num', 'Sst Chodl_num', 'Vip_num'])
    
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)
        meta = data.obs[cols].loc[H.index]

        assoc = get_correlations(H, meta)
        plot_heatmap(assoc, color='RdBu_r', optimal_ordering=True, figsize=(11,6), title=f'{study_name}-{cell_type}')
        os.makedirs('results/assoc', exist_ok=True)
        plt.savefig(f"results/assoc/{study_name}-{cell_type}_plot.png", dpi=300)
        plt.clf()


def peek_metagenes(study_name, cell_type):
    W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', index_col=0)
    print(pd.DataFrame({col: W.nlargest(20, col).index.tolist() for col in W.columns}))

for cell_type in broad_cell_types:
    print(cell_type)
    peek_metagenes('p400', cell_type)

'''
Excitatory
   Metagene 1 Metagene 2 Metagene 3 Metagene 4 Metagene 5 Metagene 6 Metagene 7
0        SYT1      WDR64   PDCD1LG2       GLUL       GLUL       SPP1     MS4A13
1       LSAMP    HSD11B2       PSG8       SPP1       SPP1       GLUL      WDR64
2       RALYL       GLUL      PAPPA       TPM2       SCGN       KRT5      SPON2
3       NEGR1     ITGA10     PCDH12      SPON2   SLC25A18     PHLDA2      MS4A8
4       FGF14       SPP1        TYR        MAF        MAF       PKP1      ITGA4
5       KCNQ5   SLC25A18      CHST9   SLC25A18      GRIP2     CARTPT     CARTPT
6       SNTG1       OIT3     SCUBE3       TGFA    SLC14A1       PDYN      BATF3
7        RYR2     SFT2D2       PSG9      ITGA4       TPM2      PRELP     TRIM73
8       RIMS2      MAML2     CELA2B      BATF3      WDR64       CHGA      ARAP3
9      ADGRL3       TPM2    ANKRD53     CELA2B    HSD11B2      MAP1A      PRTN3
10    KHDRBS2      CASP6     CYP3A4      MS4A8        SLA     NT5DC2      IFT27
11      LRRC7    ADAMTS2      TRPA1       ETV4       PKP1     CLDND2       RRM2
12     PPP3CA       CUBN    SLC28A2     SFT2D2  MAP1LC3B2      GPR26       MYH7
13     SLC8A1       MYH7      MEOX1      WDR64      CASP6       CCNO      RERGL
14      CNTN1    CCDC178    SLC22A6     SCUBE3        RGR       SCGN        PGC
15      PTPRK      ITGA4    SLC22A8     PCDH12       PRLR      ATOX1      NPHS1
16      NELL2       PDYN    DGAT2L6      RERGL       TGFA       MYH7     KLHL35
17      NCAM2     CELA2B      PSG11     TRIM73       SYT6     PLCXD2    HSD11B2
18      PCDH7     PCDH12        PLG       TNS3      ARAP3     ATP5PD       CPT2
19       PCLO      OVCH1      HIPK4    SLC14A1       CUBN       SYN1      TNNC2
Inhibitory
   Metagene 1 Metagene 2 Metagene 3 Metagene 4 Metagene 5
0     CNTNAP2      HIPK4       SPP1       SPP1       SPP1
1       ERBB4      FUCA2       POLQ       TPBG       SRPX
2       CADM2       PSG8      FUCA2      PDZD4      LAMB4
3       CSMD1     CELA2B       SRPX      CCND1       SMTN
4        NRG3       POLQ     PCDH12     BOLA2B      FUCA2
5       LSAMP      ARAP3      HIPK4     SYCE1L       OIT3
6      ADGRB3      RGPD1     CELA2B       TPM2       VASP
7        DLG2     MS4A13      RGPD1     HAPLN2       TCN2
8    IL1RAPL1    SLC28A2    RANBP3L      ATOX1      WDR86
9        SYT1    C1QTNF7    C1QTNF7     CHCHD7        CCS
10      PCDH9     PCDH12      PDE5A       CD83    RANBP3L
11      GRIK2    CEACAM1      CCND1       ZPR1       TPM2
12      CNTN5    SLC22A6      GFRAL     FN3KRP     HAPLN2
13      PLCB1       MXD3     LRRIQ3     EIF1AD      REP15
14      MAGI2       ETV4     ALOX12       NEFH       MXD3
15      NLGN1       MYOT      LAMB4      APOA1       TPBG
16      SNTG1        CCS       MYOT      CXXC5       DOK3
17     CCSER1      WDR86     GALNT3     FAHD2B      HARS2
18       TCF4        REN     ACVR1C      FGF22       POLQ
19      NEGR1     TRIM73       TPBG        CCS       NPFF
Oligodendrocyte
   Metagene 1 Metagene 2 Metagene 3 Metagene 4 Metagene 5
0       ERBIN    PPP2R2B  C14orf132      GRIA4     COL5A2
1       PEX5L      NCAM2    FAM186A      PGBD5      SIDT1
2     SLC44A1      EDIL3      AP3B2     SLC6A9      STAT4
3       ELMO1     MAN2A1      TRPV5       P3H2     TESPA1
4      NCKAP5    SLC44A1     IZUMO4     BAHCC1       MLIP
5       NCAM2     SPOCK3      SCYL1    RAPGEF3     MICAL2
6       EDIL3       WWOX      USH1C     ZFYVE9       CRYM
7      SPOCK3      GRIA4       GNAZ    SHROOM1       FHL1
8        QDPR      ELMO1      CLVS1      SH2D6    TSPAN13
9      MAN2A1      PEX5L      ACAP1      HHATL       ZBBX
10      CDH20     NCKAP5       LDB3    RAPGEF1      VWA3A
11    CNTNAP2    CNTNAP2      FOXP4      GDPD1      LAMB1
12     MAP4K5       LDB3      TTLL3      CES4A       STRC
13    SLC38A2      CDH20       PTK6      FANCB     CAMK2A
14    PPP2R2B      CADM1      BLCAP     POU3F2       IRS1
15     PIEZO2      ERBIN        MVP      SCYL1      CRHR1
16       ANLN      CLCA4    COLEC12       GNAZ     CYP2E1
17     ZSWIM6      FKBP5      CES4A      CLCA4       FZD1
18       WWOX      LRRC1     IFNGR2      RESF1     OGFRL1
19     GPRC5B     ZNF652     COL9A3      KLHL8       SDC2
Astrocyte
   Metagene 1 Metagene 2 Metagene 3
0        RORA      RPL34      LRRC7
1        NEBL      RPS12       CD38
2     RANBP3L       TLE5      RPS12
3        NFIB     RPL35A      RPL15
4       SASH1      RPL15      SSBP3
5       DPP10       RPSA  MAP1LC3B2
6      SAMD4A       RPS7       TLE5
7         LPP      RPL35      RPL34
8    SLC39A11    ATP5IF1      RPL35
9       DOCK7       RPL6       RPSA
10      ARAP2      SSBP3     RPL35A
11     PDZRN3     NDUFA4       RPL6
12      SYNE1       BEX3     ZNF441
13     AHCYL1       BEX2    SLC28A2
14      TTC28     TMSB4X       RPS7
15    SEPTIN7      RPL32      CHMP3
16     TBC1D5      RPS14     PRSS23
17       CHD9      RPL26       NOL3
18       DGKG      CHMP3       CRY2
19      ASTN2      LZTS3      IFT27
Microglia-PVM
   Metagene 1 Metagene 2 Metagene 3 Metagene 4
0      ATP8B4       DPYD     S100A4    TMEM14A
1       PRKCA      FOXP1      ANXA2      CORO6
2        TLN2     ATP8B4     IL27RA     RNF187
3       MKNK1       SNCA   ARHGAP10      RPH3A
4         ADK      PRKCA        FGR     KCNMB2
5      FRMD4B      ALCAM      P2RY8       PTMS
6       FOXP1       DTNA       EYA2    STARD10
7        DTNA     FRMD4B       FLT1     CHI3L1
8      RHBDF2      TNPO1     IFNLR1      LRFN4
9     NCKAP1L        ADK      PTPRG       NPM3
10     GPCPD1       CD81    SLC43A3     FAM89B
11      SPIN1     RHBDF2      KCNN4     LGALS1
12   PPP1R12A   PPP1R12A     ADARB1       ELK1
13       SNCA       CTSD     CLEC5A     LRRC39
14    ZNF518A       TLN2     ACOT11  TNFAIP8L3
15     MCF2L2    NCKAP1L      SYNE3     PDE10A
16     YTHDC2        CPM      APOBR      GRWD1
17      WDPCP      TFCP2     CALHM2       RIC3
18     PARD3B        GSN        ZYX  TNFRSF11B
19      TFCP2    ZNF518A        VIM    PSTPIP1
OPC
   Metagene 1 Metagene 2
0       PCDH9      VSNL1
1      ADGRB3    COX7A2L
2       KCND2    CDK2AP1
3        VCAN      PAPPA
4       LRRC7    SLC16A9
5         DMD      CORO6
6     GALNT13      STAT4
7      BRINP3     CHCHD7
8       PCDH7       ODC1
9       EDIL3     VPS33B
10    KHDRBS3  C20orf203
11      GPM6B      KLKB1
12     SLAIN1       TUT1
13        FER    ZC2HC1B
14      TAFA1     ZNF433
15     TBC1D5       ARSG
16      ARAP2      CRTAP
17      GDAP1    CWF19L1
18      FLRT2       CDK7
19    SEPTIN7      KCNB2
Endothelial
   Metagene 1
0     GALNT18
1       WWTR1
2       DUSP1
3       USP34
4     SLC38A2
5       GPM6A
6       STAU2
7       SDCBP
8      NFKBIA
9        RHOJ
10      RBM17
11     LRRC32
12      PINK1
13      CIRBP
14      ACER3
15      SMIM3
16     CTDSP1
17    CDK2AP1
18      CABP1
19      SORL1
>>> 
'''









for study_name in study_names:
    for cell_type in broad_cell_types:

        adata = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        if 'SEAAD' in study_name:
            meta = adata.obs[['Age at Death','Cognitive status','ADNC','Braak stage','Thal phase',
                              'CERAD score','APOE4 status','Lewy body disease pathology','LATE-NC stage',
                              'Microinfarct pathology','PMI','disease','Sex','self_reported_ethnicity','num_cells']]
        if study_name == 'p400':
            meta = adata.obs[['apoe_genotype','amyloid','braaksc','ceradsc','gpath','tangles',
                              'cogdx','age_death','age_first_ad_dx','educ','msex','race7','pmi','num_cells']]
        if cell_type == 'Excitatory':
            cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                            'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
        if cell_type == 'Inhibitory':
            cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                            'Sst_num', 'Sst Chodl_num', 'Vip_num'])

        W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_DEG_L1.tsv', index_col=0)
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_DEG_L1.tsv', index_col=0)

        meta = meta.loc[H.index]
        colors_dict = meta.apply(lambda col: col.astype('category').cat.codes / (len(col.cat.categories) - 1) if col.dtype.name == 'category' \
                                 else col.astype(int) if col.dtype == bool \
                                      else (col - col.min()) / (col.max() - col.min()))
        col_colors = colors_dict.applymap(plt.cm.plasma)

        # heatmap for coefficient matrix H (metasamples)
        cluster_grid = sns.clustermap(H.T, method='average', cmap='viridis', standard_scale=1, 
                                      xticklabels=False, col_colors=col_colors, figsize=(10, 7))
        cluster_grid.ax_heatmap.set_xlabel('Samples')
        cluster_grid.cax.yaxis.set_label_position('left')
        cluster_grid.cax.set_ylabel('Weight', rotation=90, labelpad=10, verticalalignment='center')
        plt.suptitle(f'Coefficient Matrix H (Metasamples), {study_name}-{cell_type}', y=1.02)
        # save
        os.makedirs('results/coefficient', exist_ok=True)
        plt.savefig(f'results/coefficient/{study_name}-{cell_type}_basis_vectors_heatmap.png')

        # heatmap for basis vectors W (metagenes)
        cluster_grid = sns.clustermap(W, method='average', cmap='viridis', standard_scale=1, 
                                      yticklabels=False, figsize=(7, 10))
        plt.suptitle(f'Basis Vectors W (Metagenes), {study_name}-{cell_type}', y=1.02)
        cluster_grid.ax_heatmap.set_ylabel('Genes')
        cluster_grid.cax.yaxis.set_label_position('left')
        cluster_grid.cax.set_ylabel('Weight', rotation=90, labelpad=10, verticalalignment='center')
        # save
        os.makedirs('results/basis', exist_ok=True)
        plt.savefig(f'results/basis/{study_name}-{cell_type}_basis_vectors_heatmap.png')

# consensus matrices 
for study_name in study_names:
    for cell_type in broad_cell_types:

        # load pseudobulks
        adata = sc.read(f'data/pseudobulks/{study_name}-pseudobulk.h5ad')
        adata.obs = adata.obs.assign(study_name=study_name)
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]
        # subset to the 2000 most highly variable genes 
        hvg = highly_variable_genes(adata).highly_variable
        adata = adata[:, hvg].copy()
        # convert counts to CPMs
        adata.X = np.log2((adata.X * 1000000 / adata.X.sum(axis=1)[:, None]) + 1)

        # convert to R
        # RcppML internally coerces to a dgcMatrix, so transpose the counts
        assert not np.any(adata.X < 0), "Array contains negative numbers"
        log_CPMs_R = array_to_rmatrix(adata.X.T)
        gene_names = adata.var_names
        samp_names = adata.obs_names

        # get k_1se
        MSE = pd.read_table(f'results/MSE/{study_name}-{cell_type}_MSE.tsv')\
            .astype({'k': int, 'rep': int})\
            .set_index(['k', 'rep'])\
            .squeeze()\
            .rename('MSE')

        mean_MSE = MSE.groupby('k').mean()
        k_best = int(mean_MSE.idxmin())
        k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + sem(MSE[k_best])][0])

        n_runs = 100
        consensus_matrix = np.zeros((samp_names.size, samp_names.size))

        for run in range(n_runs):
            NMF_run = r.nmf(log_CPMs_R, k=k_1se, seed=run, tol=1e-5, maxit=np.iinfo('int32').max, L1=r.c(0.01, 0.01))
            H_run = rmatrix_to_df(NMF_run.slots['h']).T.set_axis(samp_names)

            # Cluster membership based on max row of each column of H
            cluster_membership = np.argmax(H_run.values, axis=1)

            # Create connectivity matrix using broadcasting
            connectivity_matrix = (cluster_membership[:, None] == cluster_membership[None, :]).astype(int)
            consensus_matrix += connectivity_matrix

        consensus_matrix /= n_runs

        sns.clustermap(consensus_matrix, method='average', cmap="YlGnBu", xticklabels=False, yticklabels=False, figsize=(10, 10))
        plt.suptitle(f'Consensus Matrix {study_name}-{cell_type}, k={k_1se} nruns={n_runs}', y = 0.99)
        # save
        os.makedirs('results/consensus', exist_ok=True)
        plt.savefig(f'results/consensus/{study_name}-{cell_type}_consensus_heatmap.png')
        