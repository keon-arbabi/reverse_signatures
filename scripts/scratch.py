import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tmp = sc.read('pseudobulk/SEAAD-DLPFC-broad.h5ad')
df = tmp.obs[tmp.obs['broad_cell_type'] == 'Oligodendrocyte']
categories = ['Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'CERAD score', 'LATE-NC stage']

# Reshape the DataFrame to long format
df_melted = df.melt(id_vars='Oligodendrocyte_num', value_vars=categories).drop_

# Create separate boxplots for each category
fig, axes = plt.subplots(nrows=1, ncols=len(['Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'CERAD score', 'LATE-NC stage']), figsize=(20, 5))

for i, cat in enumerate(['Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'CERAD score', 'LATE-NC stage']):
    order = df[cat].cat.categories if hasattr(df[cat], 'cat') else None
    sns.boxplot(x=cat, y='Oligodendrocyte_num', data=df, ax=axes[i], order=order, palette="Set3")
    sns.swarmplot(x=cat, y='Oligodendrocyte_num', data=df, color='black', ax=axes[i], size=2.5, order=order)
    
    axes[i].set_title(cat)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].grid(False)
plt.suptitle('SEAAD MTG')
plt.tight_layout()
plt.subplots_adjust(wspace=0.5) 
plt.savefig("georgie_SEAAD_MTG_plots.png", dpi=300)

