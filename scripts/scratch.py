import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

tmp = sc.read('data/pseudobulk/SEAAD-MTG-broad.h5ad')
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

def get_rpy2_vector_class(array_or_index_or_series_or_df: object):
    assert isinstance(array_or_index_or_series_or_df,
                      (np.ndarray, pd.Index, pd.Series, pd.DataFrame))
    from rpy2.robjects import FloatVector, IntVector, r, StrVector
    
    if isinstance(array_or_index_or_series_or_df, pd.DataFrame):
        assert array_or_index_or_series_or_df.dtypes.nunique() == 1, \
            array_or_index_or_series_or_df.dtypes.value_counts()
        dtype = array_or_index_or_series_or_df.dtypes[0]
    else:
        dtype = array_or_index_or_series_or_df.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return categorical_to_factor
    elif np.issubdtype(dtype, np.floating):
        vector_class = FloatVector
    elif np.issubdtype(dtype, np.integer):
        vector_class = IntVector
    elif dtype == bool:
        vector_class = lambda x: r['as.logical'](IntVector(x.astype('int32')))
    elif isinstance(array_or_index_or_series_or_df, np.ndarray) and dtype.type is np.str_ \
            or (isinstance(array_or_index_or_series_or_df, pd.DataFrame) and 
                array_or_index_or_series_or_df.apply(pd._libs.lib.infer_dtype).eq('string').all()) \
            or pd._libs.lib.infer_dtype(array_or_index_or_series_or_df) == 'string':
        vector_class = StrVector
    else:
        raise ValueError(f'Unsupported dtype "{dtype}"!')

    return vector_class

def df_to_rmatrix(df, convert_index=True, convert_columns=True):
    assert isinstance(df, pd.DataFrame)
    from rpy2.robjects import r
    Vector = get_rpy2_vector_class(df.values)
    rmatrix = r.matrix(Vector(df.values.ravel('F')), *df.shape)
    if convert_index:
        from rpy2.robjects import StrVector
        rmatrix.rownames = StrVector(astype_str(df.index))
    if convert_columns:
        from rpy2.robjects import StrVector
        rmatrix.colnames = StrVector(astype_str(df.columns))
    return rmatrix
