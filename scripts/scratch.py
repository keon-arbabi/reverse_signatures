import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

tmp = sc.read('projects/reverse_signatures/data/pseudobulk/p400-broad.h5ad')
df = tmp.obs[tmp.obs['broad_cell_type'] == 'Oligodendrocyte']
continuous_vars = ['gpath', 'amyloid', 'plaq_n', 'nft', 'tangles']

categories = ['cogdx', 'braaksc', 'ceradsc'] + continuous_vars

# Reshape the DataFrame to long format
df_melted = df.melt(id_vars='Oligodendrocyte_num', value_vars=categories)

# Create separate plots for each category
fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(20, 5))

for i, cat in enumerate(categories):
    order = df[cat].cat.categories if hasattr(df[cat], 'cat') else None
    
    if cat in continuous_vars:
        sns.regplot(x=cat, y='Oligodendrocyte_num', data=df, ax=axes[i], scatter_kws={'s': 10}, line_kws={'color': 'red'}, ci=None)
        axes[i].set_xticks([])
        axes[i].set_xscale('log')
    else:
        sns.boxplot(x=cat, y='Oligodendrocyte_num', data=df, ax=axes[i], order=order, palette="Set3")
        sns.swarmplot(x=cat, y='Oligodendrocyte_num', data=df, color='black', ax=axes[i], size=2.5, order=order)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    axes[i].set_title(cat)
    axes[i].grid(False)
    
plt.suptitle('SEAAD MTG')
plt.tight_layout()
plt.subplots_adjust(wspace=0.5) 
plt.savefig("projects/reverse_signatures/data/georgie_p400_plots.png", dpi=300)

