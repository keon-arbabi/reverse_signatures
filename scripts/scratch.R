suppressPackageStartupMessages({
    library("tidyverse")
    library("anndata")
    library("RcppML")
})
setwd("projects/reverse_signatures/")

sc = import('scanpy')
source_python("scripts/utils.py")

broad_cell_types = c('Excitatory','Inhibitory','Oligodendrocyte','Astrocyte','Microglia-PVM','OPC','Endothelial')
study_names = c('SEAAD','SZBDMulticohort')

cell_type = 'Excitatory'
study_name = 'SEAAD'

# load pseudobulks and subset by cell types
for (study_name in study_name){
    for (cell_type in broad_cell_types){
        message(sprintf('[%s-%s] Loading', study_name, cell_type))
        adata = read_h5ad(sprintf('./data/pseudobulks/%s-pseudobulk.h5ad', study_name))
        adata$obs$study_name = study_name
        adata = adata[adata$obs$broad_cell_type == cell_type,]

        # subset to 2000 most highly variable genes for each dataset 
        hvg = highly_variable_genes(x, n_top_genes = 2000)
        x = x[, hvg$highly_variable]
    }
}
    
