import matplotlib.pyplot as plt, numpy as np, os, sys, warnings,\
    pandas as pd, scanpy as sc, seaborn as sns, optuna
from rpy2.robjects import r
import matplotlib.gridspec as gridspec
from scipy.stats import sem, zscore
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.append('projects/reverse_signatures/scripts')
from projects.reverse_signatures.old.utils import Timer, rmatrix_to_df, array_to_rmatrix, df_to_rdf, rdf_to_df, index_to_rvector

warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures')
r.library('RcppML', quietly=True)

################################################################################
# Run NMF
################################################################################

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte',\
    'Microglia-PVM', 'OPC', 'Endothelial'
study_names = ['p400']
#'SEAAD-MTG', 'SEAAD-DLPFC'

k_max = 30
n_trials = 20
save_name = ''

def preprocess_data(study_name, cell_type, gene_selection='deg', threshold=0.05, filt_cases=True):
    data = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
    data = data[data.obs['broad_cell_type'] == cell_type, :]
    if 'SEAAD' in study_name and filt_cases:
        case_samples = \
            np.where(data.obs['Consensus Clinical Dx (choice=Alzheimers disease)'].eq('Checked'), True,
            np.where(data.obs['Consensus Clinical Dx (choice=Control)'].eq('Checked'), False, False))
        data = data[case_samples, :].copy()
    elif study_name == 'p400' and filt_cases:
        case_samples = data.obs['pmAD'].eq(1).fillna(False).to_numpy(dtype=bool)
        data = data[case_samples, :].copy()
    if gene_selection == 'hvg':
        hvg = np.argpartition(-np.var(data.X, axis=0), 2000)[:2000]
        data = data[:, hvg].copy()
    elif gene_selection == 'deg':
        # degs = pd.read_csv('results/voom/limma_voom_combined.tsv.gz', sep='\t')\
        #     .query('cell_type == @cell_type & fdr < @threshold')
        degs = pd.read_csv('results/voom/limma_voom.tsv.gz', sep='\t')\
             .query('trait == @study_name & cell_type == @cell_type & fdr < @threshold')
        degs = degs['gene'].astype(str).tolist()
        print(f'[{study_name} {cell_type}]: {len(degs)}')
        data = data[:, data.var_names.isin(degs)].copy()
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

broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte'
study_name = 'p400'
 
for cell_type in broad_cell_types:
    pseudobulk = preprocess_data(study_name, cell_type, gene_selection='deg', threshold=0.05, filt_cases=True)
    genes = index_to_rvector(pseudobulk.var_names)
    samps = index_to_rvector(pseudobulk.obs_names)
    mat = pseudobulk.X.T
    mat = mat / np.median(mat, axis=1)[:, None]
    mat = array_to_rmatrix(mat)
    W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', index_col=0)\
        .pipe(df_to_rdf)
    H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)\
        .pipe(df_to_rdf)
        
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
                'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12',
                'cog_res_age40', 'cogdx', 'cogdx_stroke', 'cogn_global_random_slope',
                'cvda_4gp2', 'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl',
                'ldai_bl', 'med_con_sum_bl', 'msex', 'nft', 'niareagansc', 'plaq_d',
                'plaq_n', 'pmi', 'race7', 'smoking', 'tangles', 'tdp_st4', 'thyroid_bl',
                'tomm40_hap', 'tot_cog_res']
    cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
    cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                    'Sst_num', 'Sst Chodl_num', 'Vip_num'])
    meta = pseudobulk.obs[cols].pipe(df_to_rdf)
    os.makedirs('results/explore/p400_cases', exist_ok=True)

    r('''
        function(study_name, cell_type, genes, samps, mat, W, H, meta) {
        suppressPackageStartupMessages({
            library(ComplexHeatmap)
            library(circlize)
            library(seriation)
        })
        rownames(mat) = genes
        colnames(mat) = samps
        o1 = seriation::seriate(dist(mat), method = "OLO")
        o2 = seriation::seriate(dist(t(mat)), method = "OLO")
        ht = HeatmapAnnotation(df = meta,
                                simple_anno_size = unit(0.15, "cm"),
                                annotation_name_gp = gpar(fontsize = 5),
                                show_legend = FALSE)     
        hb = HeatmapAnnotation(df = H,
                    simple_anno_size = unit(0.3, "cm"),
                    annotation_name_gp = gpar(fontsize = 8),
                    show_legend = FALSE)         
        hr = rowAnnotation(df = W,
                            simple_anno_size = unit(0.3, "cm"),
                            annotation_name_gp = gpar(fontsize = 8),
                            show_legend = FALSE)

        #col_fun = colorRamp2(range(mat), hcl_palette = "Batlow", reverse = TRUE)
        col_fun = colorRamp2(quantile(mat, probs = c(0.01, 0.99)), hcl_palette = "Batlow", reverse = TRUE)

        file_name = paste0("results/explore/p400_cases/", cell_type, "trimc_scaled.png")
        png(file = file_name, width=7, height=10, units="in", res=1200)
        h = Heatmap(
            mat,
            row_order = seriation::get_order(o1),
            column_order = seriation::get_order(o2),
            show_row_names = FALSE,
            show_column_names = FALSE,
            bottom_annotation = hb,
            top_annotation = ht,
            left_annotation = hr,
            col = col_fun,
            name = paste0(study_name, "\n", cell_type),
            show_heatmap_legend = TRUE
        )
        draw(h)
        dev.off()
        }
        ''')(study_name, cell_type, genes, samps, mat, W, H, meta)












for cell_type in broad_cell_types:
    pseudobulk = preprocess_data(study_name, cell_type, gene_selection='deg', threshold=0.05, filt_cases=False)
    genes = index_to_rvector(pseudobulk.var_names)
    samps = index_to_rvector(pseudobulk.obs_names)
    mat = pseudobulk.X.T
    
    W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', index_col=0)\
        .pipe(df_to_rdf)
    H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)
    
    case_samples = H.index.tolist()
    is_case = pseudobulk.obs_names.isin(case_samples)
    case_mat = mat[:, is_case]
    control_mat = mat[:, ~is_case]
    
    median_norm = np.median(case_mat, axis=1)[:, None]
    case_mat_normalized = case_mat / median_norm
    control_mat_normalized = control_mat / median_norm
    
    case_indices = np.where(is_case)[0]
    control_indices = np.where(~is_case)[0]
    
    from rpy2.robjects.vectors import IntVector
    case_indices_r = IntVector(case_indices+1)
    control_indices_r = IntVector(control_indices+1)
        
    case_mat_r = array_to_rmatrix(case_mat_normalized)
    control_mat_r = array_to_rmatrix(control_mat_normalized)
    
    cols = ['ad_reagan', 'age_death', 'age_first_ad_dx', 'amyloid', 'apoe_genotype',
        'arteriol_scler', 'braaksc', 'caa_4gp', 'cancer_bl', 'ceradsc',
        'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12',
        'cog_res_age40', 'cogdx', 'cogdx_stroke', 'cogn_global_random_slope',
        'cvda_4gp2', 'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl',
        'ldai_bl', 'med_con_sum_bl', 'msex', 'nft', 'niareagansc', 'plaq_d',
        'plaq_n', 'pmi', 'race7', 'smoking', 'tangles', 'tdp_st4', 'thyroid_bl',
        'tomm40_hap', 'tot_cog_res', 'pmAD']
    cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
    cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                    'Sst_num', 'Sst Chodl_num', 'Vip_num'])
    meta = pseudobulk.obs[cols].pipe(df_to_rdf)
    os.makedirs('results/explore/p400_cases_controls', exist_ok=True)

    r('''
        function(study_name, cell_type, genes, samps, case_mat_r, control_mat_r, case_indices_r, control_indices_r, W, meta) {
            suppressPackageStartupMessages({
                library(ComplexHeatmap)
                library(circlize)
                library(seriation)
            })
            rownames(case_mat_r) = genes
            rownames(control_mat_r) = genes
            
            colnames(case_mat_r) = samps[case_indices_r] 
            colnames(control_mat_r) = samps[control_indices_r]

            o2_cases = seriation::seriate(dist(t(case_mat_r)), method = "OLO")
            o2_controls = seriation::seriate(dist(t(control_mat_r)), method = "OLO")

            combined_mat = cbind(case_mat_r, control_mat_r) 
            o1 = seriation::seriate(dist(combined_mat), method = "OLO")

            hb = HeatmapAnnotation(df = meta$pmAD,
                simple_anno_size = unit(0.3, "cm"),
                annotation_name_gp = gpar(fontsize = 8),
                show_legend = FALSE)  
            ht = HeatmapAnnotation(df = meta,
                simple_anno_size = unit(0.15, "cm"),
                annotation_name_gp = gpar(fontsize = 5),
                show_legend = FALSE)       
            hr = rowAnnotation(df = W,
                simple_anno_size = unit(0.3, "cm"),
                annotation_name_gp = gpar(fontsize = 8),
                show_legend = FALSE)

            col_fun = colorRamp2(range(combined_mat), hcl_palette = "Batlow", reverse = TRUE)
            #col_fun = colorRamp2(quantile(combined_mat, probs = c(0.01, 0.99)), hcl_palette = "Batlow", reverse = TRUE)
            file_name = paste0("results/explore/p400_cases_controls/", cell_type, "_heatmap.png")
            
            print(length(c(seriation::get_order(o2_cases), seriation::get_order(o2_controls))))
            print(ncol(combined_mat))
            
            png(file = file_name, width=7, height=10, units="in", res=1200)
            h = Heatmap(
                combined_mat,
                cluster_rows = F,
                cluster_columns = F,
                row_order = seriation::get_order(o1),
                column_order = c(seriation::get_order(o2_cases), ncol(case_mat_r) + seriation::get_order(o2_controls)),
                show_row_names = FALSE,
                show_column_names = FALSE,
                bottom_annotation = hb,
                top_annotation = ht,
                left_annotation = hr,
                col = col_fun,
                name = paste0(study_name, "\n", cell_type),
                show_heatmap_legend = TRUE
            )
            draw(h)
            dev.off()
        }
        ''')(study_name, cell_type, genes, samps, case_mat_r, control_mat_r, case_indices_r, control_indices_r, W, meta)








case_indices = [i for i, x in enumerate(is_case) if x]
control_indices = [i for i, x in enumerate(is_case) if not x]

# Ensure they are of correct length
print("Total number of samples:", len(samps))
print("Number of case indices:", len(case_indices))
print("Number of control indices:", len(control_indices))

# Convert indices to R vectors
case_indices_r = index_to_rvector(pd.Index(case_indices))
control_indices_r = index_to_rvector(pd.Index(control_indices))









import pandas as pd
import os
from pyComplexHeatmap import Heatmap, HeatmapAnnotation, rowAnnotation, colorRamp2
import numpy as np
import seriation

for study_name in study_names:
    for cell_type in broad_cell_types:
        pseudobulk = preprocess_data(study_name, cell_type, gene_selection='deg', threshold=0.01, filt_cases=True)
        genes = pseudobulk.var_names
        samps = pseudobulk.obs_names
        mat = pseudobulk.X.T

        W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W_{save_name}.tsv', index_col=0)
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{save_name}.tsv', index_col=0)

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
                    'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12', 
                    'cog_res_age40', 'cogdx', 'cogdx_stroke', 'cogn_global_random_slope', 
                    'cvda_4gp2', 'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl', 
                    'ldai_bl',  'med_con_sum_bl', 'msex', 'nft', 'niareagansc', 'plaq_d',
                    'plaq_n', 'pmi', 'race7', 'smoking', 'tangles',  'tdp_st4', 'thyroid_bl',
                    'tomm40_hap', 'tot_cog_res']
        cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                    'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
        cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                     'Sst_num', 'Sst Chodl_num', 'Vip_num'])

        meta = pseudobulk.obs[cols]
        os.makedirs('results/explore', exist_ok=True)

        o1 = seriation.seriate(np.array(mat), method="OLO")
        o2 = seriation.seriate(np.array(mat.T), method="OLO")

        ht = HeatmapAnnotation(df=meta)
        hb = HeatmapAnnotation(df=H)
        hr = rowAnnotation(df=W)

        col_fun = colorRamp2(np.linspace(np.min(mat), np.max(mat), num=256), 'Batlow', reverse=True)

        file_name = f"results/explore/{study_name}_{cell_type}_fdr01.png"
        hm = Heatmap(
            mat,
            row_order=o1,
            column_order=o2,
            bottom_annotation=hb,
            top_annotation=ht,
            left_annotation=hr,
            col=col_fun,
            name=f"{study_name}\n{cell_type}"
        )
        hm.save(file_name, dpi=1200, width=7, height=10)









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
                'chd_cogact_freq', 'ci_num2_gct', 'ci_num2_mct', 'cog_res_age12', 
                'cog_res_age40', 'cogdx', 'cogdx_stroke', 'cogn_global_random_slope', 
                'cvda_4gp2', 'dlbdx', 'dxpark', 'educ', 'gpath', 'headinjrloc_bl', 
                'ldai_bl', 'mglia123_caud_vm', 'mglia123_it', 'mglia123_mf', 'mglia123_put_p', 
                'mglia23_caud_vm', 'mglia23_it', 'mglia23_mf', 'mglia23_put_p', 'mglia3_caud_vm', 
                'mglia3_it', 'mglia3_mf', 'mglia3_put_p', 'med_con_sum_bl', 'msex', 
                'nft', 'niareagansc', 'plaq_d', 'plaq_n', 'pmi', 'race7', 'smoking', 'tangles', 
                'tdp_st4', 'thyroid_bl', 'tomm40_hap', 'tot_cog_res']
        
        cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                    'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
        cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                     'Sst_num', 'Sst Chodl_num', 'Vip_num'])
    meta = adata.obs[cols].loc[H_index]
    meta_transformed = meta.apply(lambda col: col.astype('category').cat.codes if col.dtype.name == 'category'
                                  else col.astype(int) if col.dtype == bool
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















save_name = ''
    
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
        
        plot_heatmap(W, color='rocket', yticks=False, optimal_ordering=True, figsize=(6,7),
                     title=f'Basis Vectors W (Metagenes), {study_name}-{cell_type}')
        plt.savefig(f'results/basis/{study_name}-{cell_type}_heatmap.png', dpi=400)
        plt.clf()

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
        