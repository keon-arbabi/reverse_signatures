import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr
os.chdir('/home/s/shreejoy/karbabi/projects/reverse_signatures')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 50)

df = sc.read('data/pseudobulk/SEAAD-DLPFC-broad.h5ad')
#df = sc.read('data/pseudobulk/p400-broad.h5ad').obs

df = df\
    .query('broad_cell_type == "Oligodendrocyte"')\
    .assign(norm_n_oligo=df['Oligodendrocyte_num'] / df['num_cells_total'])

df['Oligodendrocyte_num'].value_counts

continuous_vars = ['cogn_global_random_slope', 'cogn_globaln_lv', 'gpath']
categories = ['cogdx', 'braaksc', 'ceradsc'] + continuous_vars

# Reshape the DataFrame to long format
df_melted = df.melt(id_vars='norm_n_oligo', value_vars=categories)

# Create separate plots for each category
fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(20, 5))

for i, cat in enumerate(categories):
    order = df[cat].cat.categories if hasattr(df[cat], 'cat') else None
    
    if cat in continuous_vars:
        sns.regplot(x=cat, y='norm_n_oligo', data=df, ax=axes[i], scatter_kws={'s': 10}, line_kws={'color': 'red'}, ci=None)
        #axes[i].set_xticks([])
        #axes[i].set_xscale('log')
        
        # Calculate Pearson correlation
        valid_data = df[[cat, 'norm_n_oligo']].dropna()
        corr, p_val = pearsonr(valid_data[cat], valid_data['norm_n_oligo'])
        # Annotate the plot with the correlation value and p-value
        axes[i].annotate(f"R = {corr:.2f}, p = {p_val:.2g}", xy=(0.1, 0.9), xycoords='axes fraction')
    else:
        sns.boxplot(x=cat, y='norm_n_oligo', data=df, ax=axes[i], order=order, showfliers=False, palette="Set3")
        sns.swarmplot(x=cat, y='norm_n_oligo', data=df, color='grey', alpha=0.7, ax=axes[i], size=2.5, order=order)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    axes[i].set_title(cat)
    axes[i].grid(False)
    
plt.suptitle('ROSMAP (N = 436)')
plt.tight_layout()
plt.subplots_adjust(wspace=0.5) 
plt.savefig("data/georgie__plots.png", dpi=300)
