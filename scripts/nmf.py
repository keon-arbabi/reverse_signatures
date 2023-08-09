# export MPLCONFIGDIR=$SCRATCH
# export NUMBA_CACHE_DIR=$SCRATCH

import anndata as ad, gc, matplotlib.pyplot as plt, numpy as np, os, sys, \
    pandas as pd, scanpy as sc, seaborn as sns, warnings
from rpy2.robjects import r
from scipy.sparse import csr_matrix 
from scipy.stats import sem
from matplotlib import gridspec

sys.path.append('projects/reverse_signatures/scripts')

from utils import array_to_rmatrix, highly_variable_genes, rdf_to_df, rmatrix_to_df, \
      savefig, scientific_notation, Timer

r.library('RcppML')

os.chdir('projects/reverse_signatures')

# # load specific cell type and study
cell_type = 'Inhibitory'
study_name = 'SEAAD'

broad_cell_types = 'Excitatory','Inhibitory','Oligodendrocyte','Astrocyte','Microglia-PVM','OPC','Endothelial'
study_names = 'SEAAD','SZBDMulticohort'

for study_name in study_names:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15)) 
    [ax.axis('off') for ax in axes.flatten()]

    for i, cell_type in enumerate(broad_cell_types):

        with Timer(f'[{study_name} {cell_type}] Loading'):
            adata = sc.read(f'data/pseudobulks/{study_name}-pseudobulk.h5ad')
            adata.obs = adata.obs.assign(study_name=study_name)
            adata = adata[adata.obs['broad_cell_type'] == cell_type, :]

        # subset to the 2000 most highly variable genes 
        with Timer(f'[{study_name} {cell_type}] Subsetting to highly variable genes'):
            hvg = highly_variable_genes(adata).highly_variable
            adata = adata[:, hvg].copy()

        # convert counts to CPMs
        with Timer(f'[{cell_type}] Converting to CPMs'):
            #adata.X = adata.X * 1000000 / adata.X.sum(axis=1)[:, None]
            adata.X = np.log2((adata.X * 1000000 / adata.X.sum(axis=1)[:, None]) + 1)

        # convert to R
        # RcppML internally coerces to a dgcMatrix, so transpose the counts
        assert not np.any(adata.X < 0), "Array contains negative numbers"
        log_CPMs_R = array_to_rmatrix(adata.X.T)
        gene_names = adata.var_names
        samp_names = adata.obs_names

        # Run NMF with RcppML, selecting k via 3-fold cross-validation (3 is default):
        # - biorxiv.org/content/10.1101/2021.09.01.458620v1.full
        # - github.com/zdebruine/RcppML/blob/main/R/nmf.R
        # - github.com/zdebruine/RcppML/blob/main/R/crossValidate.R
        # - github.com/zdebruine/RcppML/blob/main/vignettes/getting_started.Rmd
        # - Install via devtools::install_github('zdebruine/RcppML')
        # - Use L1=0.01: github.com/zdebruine/singlet/blob/main/R/cross_validate_nmf.R

        kmin, kmax = 1, 10 # IMPORTANT: INCREASE KMAX IF BEST MSE IS CLOSE TO KMAX!!!
        r.options(**{'RcppML.verbose': True})
        MSE = r.crossValidate(log_CPMs_R, k=r.c(*range(kmin, kmax + 1)), seed=0, reps=3,
                            tol=1e-5, maxit=np.iinfo('int32').max, 
                            L1=r.c(0.01, 0.01))
        MSE = rdf_to_df(MSE)\
            .astype({'k': int, 'rep': int})\
            .set_index(['k', 'rep'])\
            .squeeze()\
            .rename('MSE')

        # choose the smallest k that has a mean MSE (across the three folds) within 1
        # standard error of the k with the lowest mean MSE (similar to glmnet's
        # lambda.1se), where the SE is taken across the three folds at the best k
        # - Motivation for the 1 SE rule: stats.stackexchange.com/a/138573
        # - glmnet code: github.com/cran/glmnet/blob/master/R/getOptcv.glmnet.R

        mean_MSE = MSE.groupby('k').mean()
        k_best = int(mean_MSE.idxmin())
        k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + sem(MSE[k_best])][0])
        print(f'{study_name}-{cell_type}: {k_best=}, {k_1se=}')

        # plot mean MSE across folds vs k
        x = mean_MSE.index
        y = mean_MSE
        yerr = MSE.groupby('k').sem()

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

        # save MSE results 
        os.makedirs('results/MSE', exist_ok=True)
        MSE.to_csv(f'results/MSE/{study_name}-{cell_type}_MSE.tsv', sep='\t')

        # re-run NMF at k_1se, without cross-validation
        NMF_results = r.nmf(log_CPMs_R, k=k_1se, seed=0, tol=1e-5, maxit=np.iinfo('int32').max, L1=r.c(0.01, 0.01))

        # get W and H matrices
        W = rmatrix_to_df(NMF_results.slots['w'])\
            .set_axis(gene_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))
        H = rmatrix_to_df(NMF_results.slots['h'])\
            .T\
            .set_axis(samp_names)\
            .rename(columns=lambda col: col.replace('nmf', 'Metagene '))

        # save W and H matrices, and MSE
        os.makedirs('results/NMF', exist_ok=True)
        W.to_csv(f'results/NMF/{study_name}-{cell_type}_W.tsv', sep='\t')
        H.to_csv(f'results/NMF/{study_name}-{cell_type}_H.tsv', sep='\t')

        # consensus matrix calculation
        # re-run NMF multiple times and calculate the consensus matrix

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

        # heatmap with clustering
        sns.clustermap(consensus_matrix, method='average', cmap="YlGnBu")
        plt.title(f'Consensus Matrix {study_name}-{cell_type}, nruns= {n_runs}')
        # save
        os.makedirs('results/consensus', exist_ok=True)
        savefig(f'results/consensus/{study_name}-{cell_type}_consensus_heatmap.pdf')

    # MSE plots
    plt.tight_layout()
    plt.savefig(f'results/MSE/{study_name}_grid_plot.pdf')