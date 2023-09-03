import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df =  data.obs[data.obs['broad_cell_type'] == 'Oligodendrocyte'].drop_duplicates(subset='donor_id')
categories = ['cognitive_status', 'adnc', 'braak_stage', 'thal_phase', 'cerad_score', 'late_nc_stage']

# Reshape the DataFrame to long format
df_melted = df.melt(id_vars='Oligodendrocyte_num', value_vars=categories)
df_melted['value'] = df_melted['value'].astype(str)

# Plot
plt.figure(figsize=(15, 6))
sns.boxplot(x='variable', y='Oligodendrocyte_num', hue='value', data=df_melted, palette="Set2")
sns.stripplot(x='variable', y='Oligodendrocyte_num', hue='value', data=df_melted, jitter=True, 
              dodge=True, marker='o', alpha=0.7, edgecolor='black', linewidth=0.5, palette="Set2")
plt.gca().legend().set_visible(False)

plt.savefig("boxplot_with_points.png")
