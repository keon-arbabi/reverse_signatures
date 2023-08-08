suppressPackageStartupMessages({
  library("reticulate")
  library("anndata")
  library("Matrix")
  library("RcppML")
  library("singlet") 
})
setwd('/home/s/shreejoy/karbabi/scratch/projects/reverse_signatures/')

sc = import('scanpy')
source_python("scripts/utils.py")

broad_cell_types = c('Excitatory','Inhibitory','Oligodendrocyte','Astrocyte','Microglia-PVM','OPC','Endothelial')
study_names = c('SEAAD','SZBDMulticohort')

# load pseudobulks and subset by cell types
adata = list()
for (study_name in study_names) {
  message(sprintf('[%s] Loading', study_name))
  adata_full = read_h5ad(sprintf('./data/pseudobulks/%s-pseudobulk.h5ad', study_name))
  adata_full$obs$study_name = study_name
  
  for(cell_type in broad_cell_types){
    adata_subset = adata_full[adata_full$obs$broad_cell_type == cell_type,]
    adata[[paste(study_name, cell_type, sep = "_")]] = adata_subset
  }
}
# subset to 2000 most highly variable genes for each dataset 
adata = lapply(adata, function(x) {
  hvg = highly_variable_genes(x, n_top_genes = 2000)
  x = x[, hvg$highly_variable]
  return(x)
})
# convert counts to CPMs (no log transformation)
# convert to sparse matrix 
sdata = lapply(adata, function(x) {
  m = x$X
  m = m / rowSums(m) * 1e6
  m = Matrix(t(m), sparse = TRUE)
  return(m)
})
# check if sparse 
class(sdata[[1]])[[1]] != "matrix"

kmin = 2
kmax = 200
MSE_ls = lapply(sdata, function(x) {
  cross_validate_nmf(x,
                     ranks = c(kmin, kmax),
                     maxit = 1000)
  #run_nmf(t(x$X), rank=5, tol=1e-5, maxit = 1000)

})
