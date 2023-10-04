import matplotlib.pyplot as plt, numpy as np, os, sys, warnings,\
    pandas as pd, scanpy as sc, seaborn as sns, optuna
from rpy2.robjects import r
from scipy.stats import sem
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
sys.path.append('projects/reverse_signatures/scripts')
from utils import Timer, rmatrix_to_df, array_to_rmatrix, rdf_to_df
warnings.filterwarnings("ignore", category=FutureWarning)
r.library('RcppML', quietly=True)
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures')

# for each cell broad cell type and study
broad_cell_types = 'Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte',\
    'Microglia-PVM', 'OPC', 'Endothelial'
study_names = 'SEAAD-DLPFC'

cell_type = 'Inhibitory'
study_name = 'SEAAD-DLPFC'

MSE_trial = {}
k_1se_trial = {}

def process_data(study_name, cell_type):
        adata = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        # # subset to the 2000 most highly variable genes
        # hvg = np.argpartition(-np.var(adata.X, axis=0), 2000)[:2000]
        # adata = adata[:, hvg].copy()
        
        # subset to case-control differentially expressed genes
        degs = pd.read_csv('data/differential-expression/de_aspan_voombygroup_p400.tsv', sep='\t')\
            .assign(broad_cell_type=lambda df: df.cell_type
                    .replace({'Astro': 'Astrocyte',
                              'Endo': 'Endothelial',
                              'Glut': 'Excitatory',
                              'GABA': 'Inhibitory',
                              'Micro': 'Microglia-PVM',
                              'Oligo': 'Oligodendrocyte',
                              'OPC': 'OPC'}))\
            .query(f'broad_cell_type == "{cell_type}" & ids == "allids" & study == "p400"')\
            .query('p_value < 0.05')
        degs = degs['gene'].astype(str).tolist()
        adata = adata[:, adata.var_names.isin(degs)].copy()
        
        # convert to log CPMs
        adata.X = np.log1p(adata.X * (1000000 / adata.X.sum(axis=1))[:, None])
        adata.X *= 1 / np.log(2)

        assert not np.any(adata.X < 0), "Array contains negative numbers"
        log_CPMs_R = array_to_rmatrix(adata.X.T)
        gene_names = adata.var_names
        samp_names = adata.obs_names










flavour = 'DEG_L1'
for study_name in [study_names]:

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    [ax.axis('off') for ax in axes.flatten()]
    
    for idx, cell_type in enumerate(broad_cell_types):
        
        adata = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        # # subset to the 2000 most highly variable genes
        # hvg = np.argpartition(-np.var(adata.X, axis=0), 2000)[:2000]
        # adata = adata[:, hvg].copy()
        
        # subset to case-control differentially expressed genes
        degs = pd.read_csv('data/differential-expression/de_aspan_voombygroup_p400.tsv', sep='\t')\
            .assign(broad_cell_type=lambda df: df.cell_type
                    .replace({'Astro': 'Astrocyte',
                              'Endo': 'Endothelial',
                              'Glut': 'Excitatory',
                              'GABA': 'Inhibitory',
                              'Micro': 'Microglia-PVM',
                              'Oligo': 'Oligodendrocyte',
                              'OPC': 'OPC'}))\
            .query(f'broad_cell_type == "{cell_type}" & ids == "allids" & study == "p400"')\
            .query('p_value < 0.05')
        degs = degs['gene'].astype(str).tolist()
        adata = adata[:, adata.var_names.isin(degs)].copy()
        
        # convert to log CPMs
        adata.X = np.log1p(adata.X * (1000000 / adata.X.sum(axis=1))[:, None])
        adata.X *= 1 / np.log(2)

        assert not np.any(adata.X < 0), "Array contains negative numbers"
        log_CPMs_R = array_to_rmatrix(adata.X.T)
        gene_names = adata.var_names
        samp_names = adata.obs_names

        # use optuna to select best k, and L1 and L2 for W and H:
        # https://github.com/optuna/optunadl.acm.org/doi/10.1145/3292500.3330701
        def objective(trial):
        
            L1_w = trial.suggest_float('L1_w', 0.001, 0.999, log=True)
            L1_h = trial.suggest_float('L1_h', 0.001, 0.999, log=True)

            # run NMF with RcppML, selecting k via 3-fold cross-validation (3 is default):
            # https://github.com/zdebruine/RcppML/blob/main/R/nmf.R
            kmin, kmax = 1, 20
            r.options(**{'RcppML.verbose': True})
            MSE = r.crossValidate(log_CPMs_R,
                                  k=r.c(*range(kmin, kmax + 1)), L1=r.c(L1_w, L1_h),
                                  seed=0, reps=3, tol=1e-2, maxit=np.iinfo('int32').max)
            MSE = rdf_to_df(MSE)\
                .astype({'k': int, 'rep': int})\
                .set_index(['k', 'rep'])\
                .squeeze()\
                .rename('MSE')
            
            # choose the smallest k that has a mean MSE (across the three folds) within 1
            # standard error of the k with the lowest mean MSE
            mean_MSE = MSE.groupby('k').mean()
            k_best = int(mean_MSE.idxmin())
            k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + sem(MSE[k_best])][0])
            
            MSE_trial[study_name, cell_type, L1_w, L1_h] = MSE
            k_1se_trial[study_name, cell_type, L1_w, L1_h] = k_1se
            print(f'[{study_name} {cell_type}]: {k_1se=}')
            
            error = mean_MSE[k_1se]
            return error

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
                                    direction='minimize')
        study.optimize(objective, n_trials=30)

        L1_w, L1_h = study.best_trial.params.values()
        
        MSE_final = MSE_trial[study_name, cell_type, L1_w, L1_h]
        k_final = k_1se_trial[study_name, cell_type, L1_w, L1_h]
        print(f'[{study_name} {cell_type}]: {k_final=}')
        
        # run NMF with best k, L1_w, L1_h
        NMF_results = r.nmf(log_CPMs_R,
                            k=k_final, L1=r.c(L1_w, L1_h),
                            seed=0, tol=1e-5, maxit=np.iinfo('int32').max)

        # get W and H matrices
        W = rmatrix_to_df(NMF_results.slots['w'])\
            .set_axis(gene_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))
        H = rmatrix_to_df(NMF_results.slots['h'])\
            .T\
            .set_axis(samp_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))
            
        # save MSE results
        os.makedirs('results/MSE', exist_ok=True)
        MSE_final.to_csv(f'results/MSE/{study_name}-{cell_type}_MSE_{flavour}.tsv', sep='\t')
        # save W and H matrices
        os.makedirs('results/NMF', exist_ok=True)
        W.to_csv(f'results/NMF/{study_name}-{cell_type}_W_{flavour}.tsv', sep='\t')
        H.to_csv(f'results/NMF/{study_name}-{cell_type}_H_{flavour}.tsv', sep='\t')
        
        # plot MSE and k_1se across all and best trial(s)
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        for trial_n, ((current_study, current_cell_type, L1_w, L1_h), MSE) in enumerate(MSE_trial.items()):
            if current_study != study_name or current_cell_type != cell_type:
                continue
            mean_MSE = MSE.groupby('k').mean()
            k_1se = k_1se_trial[study_name, cell_type, L1_w, L1_h]
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=0.1)
            ax.scatter(k_1se, mean_MSE[k_1se], color='black', s=16, alpha=0.1)
        
        mean_MSE = MSE_final.groupby('k').mean()
        ax.plot(mean_MSE.index, mean_MSE.values, color='red')
        ax.scatter(k_final, mean_MSE[k_final], color='red', s=50)
        ax.set_xticks(ticks=mean_MSE.index)
        ax.set_yscale('log')
        ax.set_title(f'{study_name} {cell_type}, MSE across Optuna trials\nSelected L1_w and L1_h: {L1_w:.3f}, {L1_h:.3f}')
        ax.set_xlabel('k'), ax.set_ylabel('Mean MSE')
        ax.axis('on')

        plt.tight_layout()
        plt.savefig(f"results/MSE/{study_name}_plots_{flavour}.png", dpi=300)
            
        
for study_name in [study_names]:
    for cell_type in broad_cell_types:
        
        adata = sc.read(f'data/pseudobulk/{study_name}-broad.h5ad')
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        if 'SEAAD' in study_name:
            cols = ['Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'Last CASI Score',
                    'CERAD score', 'LATE-NC stage', 'Atherosclerosis', 'Arteriolosclerosis',
                    'Lewy body disease pathology', 'Microinfarct pathology',
                    'Continuous Pseudo-progression Score', 'APOE4 status',
                    'Neurotypical reference', 'ACT', 'ADRC Clinical Core',
                    'Age at Death', 'Age of onset cognitive symptoms',
                    'Age of Dementia diagnosis', 'Sex', 'PMI', 'Brain pH', 'RIN',
                    'Fresh Brain Weight', 'Years of education', 'self_reported_ethnicity']
            if cell_type == 'Excitatory':
                cols.extend(['L2/3 IT_num', 'L4 IT_num', 'L5 ET_num', 'L5 IT_num',
                             'L5/6 NP_num', 'L6 CT_num', 'L6 IT_num', 'L6 IT Car3_num', 'L6b_num'])
            if cell_type == 'Inhibitory':
                cols.extend(['Lamp5_num', 'Lamp5 Lhx6_num', 'Pax6_num', 'Pvalb_num', 'Sncg_num',
                             'Sst_num', 'Sst Chodl_num', 'Vip_num'])
                
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H_{flavour}.tsv', index_col=0)
        meta = adata.obs[cols].loc[H.index]
        
        from scipy.stats import pearsonr, chi2_contingency
        from itertools import product

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            return np.sqrt(chi2 / n / min(confusion_matrix.shape) - 1)

        def eta_squared(categorical, continuous):
            groups = [continuous[categorical == cat] for cat in set(categorical)]
            ss_between = sum(len(group) * (np.mean(group) - np.mean(continuous))**2 for group in groups)
            ss_total = (len(continuous) - 1) * np.var(continuous, ddof=1)
            return ss_between / ss_total
        
        # TODO just make everything pearson, hot-one-encode unordered categories 
        def calculate_associations(H, meta, absolute_R=True):
            # encoding categorical and boolean columns to integers
            meta = meta.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)
            meta = meta.apply(lambda x: x.astype(int) if x.dtype == bool else x)
            
            associations = pd.DataFrame(index=H.columns, columns=meta.columns)
            
            for col_H, col_meta in product(H.columns, meta.columns):
                data_H, data_meta = H[col_H], meta[col_meta]
                
                # when both columns are categorical
                if data_H.dtype.name == 'category' and data_meta.dtype.name == 'category':
                    associations.at[col_H, col_meta] = cramers_v(data_H, data_meta)
                
                # when one column is categorical and the other is numeric
                elif data_H.dtype.name == 'category' and pd.api.types.is_numeric_dtype(data_meta):
                    associations.at[col_H, col_meta] = eta_squared(data_H, data_meta)
                elif pd.api.types.is_numeric_dtype(data_H) and data_meta.dtype.name == 'category':
                    associations.at[col_H, col_meta] = eta_squared(data_meta, data_H)
                
                # when both columns are numeric
                elif pd.api.types.is_numeric_dtype(data_H) and pd.api.types.is_numeric_dtype(data_meta):
                    valid_indices = ~data_H.isna() & ~data_meta.isna()
                    data_H, data_meta = data_H[valid_indices], data_meta[valid_indices]
                    if len(data_H) > 0:  # To ensure there's data after filtering NaNs
                        r_value = pearsonr(data_H, data_meta)[0]
                        associations.at[col_H, col_meta] = abs(r_value) if absolute_R else r_value
                    else:
                        associations.at[col_H, col_meta] = np.nan
                        
            associations = associations.astype(float)
            return associations

        associations = calculate_associations(H, meta)

        row_link = linkage(pdist(associations.values), method='average', optimal_ordering=True)
        col_link = linkage(pdist(associations.values.T), method='average', optimal_ordering=True)

        sns.clustermap(associations, cmap='Reds', row_linkage=row_link, col_linkage=col_link,
                       rasterized=True)
        plt.gcf().set_size_inches(9, 11)
        os.makedirs('results/assoc', exist_ok=True)
        plt.savefig(f"results/assoc/{study_name}-{cell_type}_plot_{flavour}.png", dpi=300)


for study_name in study_names:
    for cell_type in broad_cell_types:

        adata = sc.read(f'data/pseudobulk/{study_name}-pseudobulk.h5ad')
        adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        if 'SEAAD' in study_name:
            meta = adata.obs[['Age at death','Cognitive status','ADNC','Braak stage','Thal phase',
                              'CERAD score','APOE4 status','Lewy body disease pathology','LATE-NC stage',
                              'Microinfarct pathology','PMI','disease','sex','ethnicity','num_cells']]
            
        if study_name == 'ROSMAP':
            meta = adata.obs[['apoe_genotype','amyloid','braaksc','ceradsc','gpath','tangles',
                              'cogdx','age_death','age_first_ad_dx','educ','msex','race7','pmi','num_cells']]

        W = pd.read_table(f'results/NMF/{study_name}-{cell_type}_W.tsv', index_col=0)
        H = pd.read_table(f'results/NMF/{study_name}-{cell_type}_H.tsv', index_col=0)

        # W.sort_values(by="Metagene 9", ascending=False).index[:30]


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
        savefig(f'results/coefficient/{study_name}-{cell_type}_basis_vectors_heatmap.png')

        # heatmap for basis vectors W (metagenes)
        cluster_grid = sns.clustermap(W, method='average', cmap='viridis', standard_scale=1, 
                                      yticklabels=False, figsize=(7, 10))
        plt.suptitle(f'Basis Vectors W (Metagenes), {study_name}-{cell_type}', y=1.02)
        cluster_grid.ax_heatmap.set_ylabel('Genes')
        cluster_grid.cax.yaxis.set_label_position('left')
        cluster_grid.cax.set_ylabel('Weight', rotation=90, labelpad=10, verticalalignment='center')
        # save
        os.makedirs('results/basis', exist_ok=True)
        savefig(f'results/basis/{study_name}-{cell_type}_basis_vectors_heatmap.png')


# plot mean MSE across folds vs k
for study_name in study_names:

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    [ax.axis('off') for ax in axes.flatten()]

    for i, cell_type in enumerate(broad_cell_types):

        MSE = pd.read_table(f'results/MSE/{study_name}-{cell_type}_MSE.tsv')\
            .astype({'k': int, 'rep': int})\
            .set_index(['k', 'rep'])\
            .squeeze()\
            .rename('MSE')

        mean_MSE = MSE.groupby('k').mean()
        k_best = int(mean_MSE.idxmin())
        k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + sem(MSE[k_best])][0])
        print(f'{study_name}-{cell_type}: {k_best=}, {k_1se=}')

        ax = axes[i // 3, i % 3]
        x = mean_MSE.index
        y = mean_MSE
        yerr = MSE.groupby('k').sem()
        ax.plot(x, y, c='darkgray', zorder=-1)
        ax.fill_between(x, y - yerr, y + yerr, color='darkgray', alpha=0.5, zorder=-1)
        ax.scatter(k_best, mean_MSE[k_best], c='b', s=30)
        ax.scatter(k_1se, mean_MSE[k_1se], c='r', s=30)
        ax.set_xlabel('$k$')
        ax.set_ylabel('Mean squared error')
        ax.set_xticks(range(kmin, kmax + 1))
        ax.set_xlim(kmin, kmax)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.axis('on')
        ax.set_title(f'{study_name} - {cell_type}')

    # save
    os.makedirs('results/MSE', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'results/MSE/{study_name}_grid_plot.png')

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
        