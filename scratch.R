## scratch.R

setwd("/nethome/kcni/karbabi/r_projects/reverse_signatures")
# load libraries
suppressPackageStartupMessages({
    library("data.table")
    library("tidyverse")
    library("edgeR")
    library("RcppML")
    #library("singlet") 
})
counts = fread(file = "pseudobulk_subclass.csv", data.table = FALSE) %>%
    filter(num_cells > 5, Subclass == "L2/3 IT") %>%  
    select(-num_cells) %>%
    unite("ID", c('Donor ID', 'Subclass')) %>%
    column_to_rownames(var = "ID")
cpms = cpm(t(counts), log = FALSE)
metadata = fread(file = "pseudobulk_subclass_meta.csv", data.table = FALSE)

kmin = 2
kmax = 200
system.time(
    MSE = cross_validate_nmf(
        cpms, 
        ranks=kmin:kmax,
        maxit=.Machine$integer.max)
)

cross_validate_nmf
remotes::install_github("zdebruine/singlet")
