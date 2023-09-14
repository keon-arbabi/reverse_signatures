import math, numpy as np, os, pandas as pd, re, subprocess, sys
from contextlib import contextmanager
from functools import cache, reduce
from itertools import product

def run(cmd, pipefail=True, force_unbuffered=False, log_file=None, **kwargs):
    # force_unbuffered only works for single commands currently
    run_kwargs = dict(check=True, shell=True, executable='/bin/bash')
    run_kwargs.update(**kwargs)
    return subprocess.run(
        f'set -eu{"o pipefail" if pipefail else ""}; '
        f'{"stdbuf -i0 -o0 -e0 " if force_unbuffered else ""}'
        f'{cmd}'
        f'{f" 2>&1 | tee {log_file}" if log_file is not None else ""}',
        **run_kwargs)

def run_background(cmd, **kwargs):
    return run(f'{cmd} &', **kwargs)

def run_slurm(cmd, job_name, time, log_file='/dev/null', num_threads=1,
              mem_per_cpu='16000M'):
    # 16000 MiB = 15.625 GiB, the memory per CPU on the 12-CPU SCC nodes:
    # $ scontrol show node node03 | grep CfgTRES
    # CfgTRES=cpu=12,mem=187.50G,billing=58
    assert ' ' not in job_name
    from tempfile import NamedTemporaryFile
    try:
        with NamedTemporaryFile('w', dir='.', suffix='.sh',
                                delete=False) as temp_file:
            print(f'#!/bin/bash\n'
                  f'#SBATCH --job-name={job_name}\n'
                  f'#SBATCH --time={time}\n'
                  f'#SBATCH --cpus-per-task={num_threads}\n'
                  f'#SBATCH --mem-per-cpu={mem_per_cpu}\n'
                  f'#SBATCH --output={log_file}\n'
                  f'export MKL_NUM_THREADS={num_threads}\n'
                  f'set -euo pipefail; {cmd}\n',
                  file=temp_file)
        sbatch_message = run(f'sbatch {temp_file.name}',
                             capture_output=True).stdout.decode().rstrip('\n')
        print(f'{sbatch_message} ("{job_name}")')
    finally:
        try:
            os.unlink(temp_file.name)
        except NameError:
            pass

def run_function_slurm(function, job_name, time, log_file='/dev/null',
                       num_threads=1, mem_per_cpu='16000M', **function_kwargs):
    # function must be self-sufficient, e.g. importing everything it needs
    from dill.source import getname, getsource
    run_slurm(
        f'python -u << "EOF"\n{getsource(function)}\n{getname(function)}'
        f'({", ".join(f"{k}={repr(v)}" for k, v in function_kwargs.items())})\n'
        f'EOF', job_name=job_name, time=time, log_file=log_file,
        num_threads=num_threads, mem_per_cpu=mem_per_cpu)

def use_font(font_name):
    from matplotlib import rcParams
    from matplotlib.font_manager import fontManager, findSystemFonts
    for font_file in findSystemFonts(os.path.expanduser('~/.fonts')):
        fontManager.addfont(font_file)
    # Make sure the fond is there
    fontManager.findfont(font_name, fallback_to_default=False)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [font_name]

def savefig(filename, **kwargs):
    import matplotlib.pyplot as plt
    all_kwargs = dict(dpi=450, bbox_inches='tight', pad_inches=0,
                      transparent=filename.endswith('pdf'))
    all_kwargs.update(kwargs)
    plt.savefig(filename, **all_kwargs)
    plt.close()

def standardize(arr, axis=0):
    assert axis == 0 or axis == 1
    assert isinstance(arr, (np.ndarray, pd.Series, pd.DataFrame))
    if axis == 0:
        return (arr - arr.mean(axis=axis)) / arr.std(axis=axis)
    elif isinstance(arr, np.ndarray):
        return (arr - arr.mean(axis=1)[:, None]) / arr.std(axis=1)[:, None]
    else:
        return arr.sub(arr.mean(axis=1), axis=0).div(arr.std(axis=1), axis=0)

def get_covariate_file(use_baseline_age=False):
    covariate_file = 'covariates_baseline_age.tsv' \
        if use_baseline_age else 'covariates.tsv'
    if not os.path.exists(covariate_file):
        from phenotype import get_age_at_last_follow_up, get_phenotype
        covariates = pd.DataFrame({
            'age': standardize(
                get_phenotype('Age when attended assessment centre', instance=0)
                if use_baseline_age else get_age_at_last_follow_up()),
            'sex': get_phenotype('Sex')})  # 1 = male, 0 = female
        covariates['age_squared'] = covariates.age ** 2
        covariates['age_by_sex'] = covariates.age * covariates.sex
        covariates['age_squared_by_sex'] = \
            covariates.age_squared * covariates.sex
        covariates['array'] = get_phenotype(
            'Genotype measurement batch').lt(0).astype(int)
        PCs = get_phenotype('Genetic principal components').iloc[:, :10]
        PCs.columns = 'PC' + pd.Index(list(range(1, 11))).astype(str)
        covariates = covariates.join(PCs)
        covariates.dropna(axis=0, inplace=True)
        assert len(covariates) == 488172, len(covariates)
        # Standardize non-binary covariates
        for column in covariates.columns[covariates.nunique() > 2]:
            covariates[column] = standardize(covariates[column])
        covariates.insert(0, 'FID', covariates.index)
        covariates.insert(1, 'IID', covariates.index)
        # noinspection PyTypeChecker
        covariates.to_csv(covariate_file, sep='\t', index=False)
    return covariate_file

def bonferroni(p):
    return np.minimum(p * p.size, 1)

def fdr(p):
    series_input = isinstance(p, pd.Series)
    if series_input:
        index = p.index
        p = p.values
    q = np.empty_like(p)
    missing = np.isnan(p)
    any_missing = missing.any()
    if any_missing:
        non_missing = ~missing
        p = p[non_missing]
        q[missing] = np.nan
        reverse_order = np.argsort(-p)
        # q[non_missing][reverse_order] doesn't work due to chained assignment
        q[np.flatnonzero(non_missing)[reverse_order]] = np.minimum.accumulate(
            p[reverse_order] / np.linspace(1, 1 / len(p), len(p)))
    else:
        reverse_order = np.argsort(-p)
        q[reverse_order] = np.minimum.accumulate(
            p[reverse_order] / np.linspace(1, 1 / len(p), len(p)))
    if series_input:
        # noinspection PyUnboundLocalVariable
        q = pd.Series(q, index=index)
    return q

def global_fdr(p, significance_threshold=0.05):
    FDR = fdr(p.values.ravel())
    FDR = pd.DataFrame(FDR.reshape(p.shape), index=p.index, columns=p.columns)
    significant = FDR < significance_threshold
    return FDR, significant

def fisher(table):
    # fisher.test gives the conditional maximum liklelihood estimate of the OR,
    # while fisher_exact gives the unconditional maximum liklelihood estimate:
    # docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    from rpy2.robjects import r
    table = np.asarray(table)
    assert table.shape == (2, 2)
    result = r2py(r['fisher.test'](array_to_rmatrix(table)))
    OR, = result['estimate']
    LCI, UCI = result['conf.int']
    p, = result['p.value']
    return OR, LCI, UCI, p

def auPRC(Y_test, predictions):
    from sklearn.metrics import auc, precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(Y_test, predictions)
    auPRC = auc(recall, precision)
    return auPRC

def date_format(series):
    return series.dt.strftime('%a %b %d, %I:%M %p').str.replace(' 0', ' ')

def time_delta_format(series):
    components = series.dt.components
    components_str = components.astype(str)
    hour_str = components_str.hours + 'h'
    minute_str = components_str.minutes + 'm'
    return minute_str.where(components.hours == 0,
                            hour_str + ' ' + minute_str)

def print_blocks(blocks):
    pretty_blocks = pd.DataFrame({
        'activity': blocks.activity,
        'start': date_format(blocks.start),
        'end': date_format(blocks.end),
        'duration': time_delta_format(blocks.duration)})
    with pd.option_context('display.max_rows', None):
        print(pretty_blocks)

def distinct(series):
    return (series.shift() != series).cumsum()

def hours(series):
    return series.dt.total_seconds() / 3600

def get_season(series):
    # https://stackoverflow.com/a/60285720/1397061
    date_offset = (series.dt.month * 100 + series.dt.day - 320) % 1300
    season = pd.cut(date_offset, [0, 300, 602, 900, 1300],
                    labels=['spring', 'summer', 'autumn', 'winter'])
    return season

def format_time(time):
    minutes, hours = np.modf(time)
    minutes = int(minutes * 60)
    hours = int(hours % 24)
    if hours == 0: hours = 12; suffix = 'AM'
    elif hours < 12: suffix = 'AM'
    else: hours -= 12; suffix = 'PM'
    return f'{hours}:{minutes:02d} {suffix}'

def format_timedelta(timedelta):
    minutes, hours = np.modf(timedelta)
    minutes = int(minutes * 60)
    hours = int(hours)
    return f'{hours}:{minutes:02d}'

def escape(x):
    if hasattr(x, 'str'):
        return x.str.replace('\W+', '_', regex=True).str.strip('_')
    else:
        return re.sub('\W+', '_', x).strip('_')

def save_npy(filename, df, dtype=float):
    prefix = filename.removesuffix('.npy')
    np.save(f'{prefix}.npy', df.values.astype(dtype))
    index = df.index
    columns = df.columns
    if type(index) == pd.Index and index.name is None:
        index = index.rename('None')
    if type(columns) == pd.Index and columns.name is None:
        columns = columns.rename('None')
    index.to_frame().to_csv(f'{prefix}.rows.tsv', index=False, sep='\t')
    columns.to_frame().to_csv(f'{prefix}.columns.tsv', index=False, sep='\t')

def load_npy(filename):
    prefix = filename.removesuffix('.npy')
    index = pd.read_table(f'{prefix}.rows.tsv').squeeze('columns')
    columns = pd.read_table(f'{prefix}.columns.tsv').squeeze('columns')
    if isinstance(index, pd.DataFrame):
        index = pd.MultiIndex.from_frame(index)
    elif index.name == 'None':
        index.name = None
    if isinstance(columns, pd.DataFrame):
        columns = pd.MultiIndex.from_frame(columns)
    elif columns.name == 'None':
        columns.name = None
    return pd.DataFrame(np.load(f'{prefix}.npy'), index=index, columns=columns)

def get_Ensembl_to_gene_name(ENSP=False):
    # Map each ENSG (or ENSP, if ENSP=True) to the gene symbol associated with
    # its most recent entry in the Ensembl database
    os.makedirs('gene_annotations', exist_ok=True)
    mapping_file = 'gene_annotations/ENSP_to_gene_name.tsv' \
        if ENSP else 'gene_annotations/ENSG_to_gene_name.tsv'
    if not os.path.exists(mapping_file):
        ENSG_files = (
            'release-107/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.107.chr_patch_hapl_scaff',
            'release-106/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.106.chr_patch_hapl_scaff',
            'release-105/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.105.chr_patch_hapl_scaff',
            'release-104/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.104.chr_patch_hapl_scaff',
            'release-103/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.103.chr_patch_hapl_scaff',
            'release-102/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.102.chr_patch_hapl_scaff',
            'release-101/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.101.chr_patch_hapl_scaff',
            'release-100/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.100.chr_patch_hapl_scaff',
            'release-99/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.99.chr_patch_hapl_scaff',
            'release-98/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.98.chr_patch_hapl_scaff',
            'release-97/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.97.chr_patch_hapl_scaff',
            'release-96/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.96.chr_patch_hapl_scaff',
            'release-95/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.95.chr_patch_hapl_scaff',
            'release-94/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.94.chr_patch_hapl_scaff',
            'release-93/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.93.chr_patch_hapl_scaff',
            'release-92/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.92.chr_patch_hapl_scaff',
            'release-91/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.91.chr_patch_hapl_scaff',
            'release-90/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.90.chr_patch_hapl_scaff',
            'release-89/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.89.chr_patch_hapl_scaff',
            'release-88/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.88.chr_patch_hapl_scaff',
            'release-87/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.87.chr_patch_hapl_scaff',
            'grch37/release-87/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh37.87.chr_patch_hapl_scaff',
            'release-86/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.86.chr_patch_hapl_scaff',
            'grch37/release-85/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh37.85.chr_patch_hapl_scaff',
            'release-85/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.85.chr_patch_hapl_scaff',
            'release-84/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.84.chr_patch_hapl_scaff',
            'release-83/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.83.chr_patch_hapl_scaff',
            'grch37/release-82/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh37.82.chr_patch_hapl_scaff',
            'release-82/gtf/homo_sapiens/'
                'Homo_sapiens.GRCh38.82.chr_patch_hapl_scaff',
            'release-81/gtf/homo_sapiens/Homo_sapiens.GRCh38.81',
            'release-80/gtf/homo_sapiens/Homo_sapiens.GRCh38.80',
            'release-79/gtf/homo_sapiens/Homo_sapiens.GRCh38.79',
            'release-78/gtf/homo_sapiens/Homo_sapiens.GRCh38.78',
            'release-77/gtf/homo_sapiens/Homo_sapiens.GRCh38.77',
            'release-76/gtf/homo_sapiens/Homo_sapiens.GRCh38.76',
            'release-76/gtf/homo_sapiens/Homo_sapiens.GRCh38.76',
            'release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75',
            'release-74/gtf/homo_sapiens/Homo_sapiens.GRCh37.74',
            'release-73/gtf/homo_sapiens/Homo_sapiens.GRCh37.73',
            'release-72/gtf/homo_sapiens/Homo_sapiens.GRCh37.72',
            'release-71/gtf/homo_sapiens/Homo_sapiens.GRCh37.71',
            'release-70/gtf/homo_sapiens/Homo_sapiens.GRCh37.70',
            'release-69/gtf/homo_sapiens/Homo_sapiens.GRCh37.69',
            'release-68/gtf/homo_sapiens/Homo_sapiens.GRCh37.68',
            'release-67/gtf/homo_sapiens/Homo_sapiens.GRCh37.67',
            'release-66/gtf/homo_sapiens/Homo_sapiens.GRCh37.66',
            'release-65/gtf/homo_sapiens/Homo_sapiens.GRCh37.65',
            'release-64/gtf/homo_sapiens/Homo_sapiens.GRCh37.64',
            'release-63/gtf/homo_sapiens/Homo_sapiens.GRCh37.63',
            'release-62/gtf/homo_sapiens/Homo_sapiens.GRCh37.62',
            'release-61/gtf/homo_sapiens/Homo_sapiens.GRCh37.61',
            'release-60/gtf/homo_sapiens/Homo_sapiens.GRCh37.60',
            'release-59/gtf/homo_sapiens/Homo_sapiens.GRCh37.59',
            'release-58/gtf/homo_sapiens/Homo_sapiens.GRCh37.58',
            'release-57/gtf/homo_sapiens/Homo_sapiens.GRCh37.57',
            'release-56/gtf/homo_sapiens/Homo_sapiens.GRCh37.56',
            'release-55/gtf/homo_sapiens/Homo_sapiens.GRCh37.55',
            'release-54/gtf/homo_sapiens/Homo_sapiens.NCBI36.54',
            'release-53/gtf/homo_sapiens/Homo_sapiens.NCBI36.53',
            'release-52/gtf/homo_sapiens/Homo_sapiens.NCBI36.52',
            'release-51/gtf/homo_sapiens/Homo_sapiens.NCBI36.51',
            'release-50/gtf/homo_sapiens/Homo_sapiens.NCBI36.50',
            'release-49/gtf/homo_sapiens/Homo_sapiens.NCBI36.49',
            'release-48/gtf/homo_sapiens/Homo_sapiens.NCBI36.48',
            'release-47/gtf/Homo_sapiens.NCBI36.47',
            'release-46/homo_sapiens_46_36h/data/gtf/Homo_sapiens.NCBI36.46',
            'release-44/homo_sapiens_44_36f/data/gtf/Homo_sapiens.NCBI36.44',
            'release-43/homo_sapiens_43_36e/data/gtf/Homo_sapiens.NCBI36.43')
        assert len(ENSG_files) == 68, len(ENSG_files)
        if ENSP:
            run(f'curl -s ftp.ensembl.org/pub/{{{",".join(ENSG_files)}}}.gtf.'
                f'gz | zcat | sed -nr \'s/.*gene_name "(\S*)".*protein_id '
                f'"(\S*)".*/\\2\\t\\1/p\' | awk \'!seen[$1]++\' > '
                f'{mapping_file}')
        else:
            run(f'curl -s ftp.ensembl.org/pub/{{{",".join(ENSG_files)}}}.gtf.'
                f'gz | zcat | sed -nr \'s/.*gene_id "(\S*)".*gene_name '
                f'"(\S*)".*/\\1\\t\\2/p\' | awk \'!seen[$1]++\' > '
                f'{mapping_file}')
    return pd.read_table(mapping_file, index_col=0, header=None)\
        .squeeze().rename_axis(None).rename(None)

@cache
def get_alias_to_gene():
    gene_alias_file = 'gene_annotations/gene_aliases.tsv'
    if not os.path.exists(gene_alias_file):
        os.makedirs('gene_annotations', exist_ok=True)
        run(f'curl -s https://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/'
            f'locus_groups/protein-coding_gene.txt | cut -f2,11 | tail -n +2 > '
            f'{gene_alias_file}')
    alias_to_gene = pd.read_table(gene_alias_file, header=None)\
        .dropna()\
        .apply('|'.join, axis=1)\
        .str.split('|')\
        .map(set)
    alias_to_gene = pd.Series({gene: aliases.difference({gene})
                               for aliases in alias_to_gene.values
                               for gene in aliases}).explode()
    return alias_to_gene

def unalias(gene_list, target_gene_list, duplicate_policy='raise',
            Ensembl=False):
    # Returns an index of the same length as gene_list, containing each gene's
    # name in target_gene_list, or NA if not found.
    #
    # If there are duplicate genes after unaliasing, remove them if duplicate_
    # policy == 'remove', raise an error if 'raise', or ignore them if 'ignore'.
    #
    # If Ensembl == True, treat genes in gene_list as Ensembl IDs.
    assert isinstance(gene_list, pd.Index)
    assert isinstance(target_gene_list, pd.Index)
    assert duplicate_policy in {'remove', 'ignore', 'raise'}
    if Ensembl:
        assert gene_list.str.startswith('ENSG').all()
        gene_list = gene_list.str.split('.').str[0]  # remove numeric suffix
    # Get all aliases that map to genes in target_gene_list; drop ambiguous
    # aliases that map to multiple genes in target_gene_list
    alias_to_gene = (get_Ensembl_to_gene_name() if Ensembl else
                     get_alias_to_gene())\
        .pipe(lambda x: x[x.isin(target_gene_list)])\
        .pipe(lambda x: x[~x.index.duplicated(False)])
    # For each gene in gene_list:
    # - if gene is already in target_gene_list, leave it unchanged
    # - else if the gene in alias_to_gene, substitute it with its alias
    # - else set it to NA
    unaliased_gene_list = gene_list.where(gene_list.isin(target_gene_list),
                                          alias_to_gene.reindex(gene_list))
    if duplicate_policy == 'remove' or duplicate_policy == 'raise':
        duplicated = unaliased_gene_list.notna() & \
                     unaliased_gene_list.duplicated(False)
        if duplicated.any():
            if duplicate_policy == 'remove':
                unaliased_gene_list = unaliased_gene_list.where(~duplicated)
            else:
                duplicates = pd.DataFrame({
                    'Original': gene_list[duplicated],
                    'Unaliased': unaliased_gene_list[duplicated]})\
                    .sort_values(['Unaliased', 'Original'])
                raise ValueError(f'Duplicated genes after unaliasing:\n'
                                 f'{duplicates.to_string(index=False)}')
    return unaliased_gene_list

def astype_str(pandas_obj):
    # As an optimization, only convert to string if it's not already a string.
    # pd._libs.lib.infer_dtype() is the same function that raises an error when
    # you use the .str accessor with non-string dtypes.
    # noinspection PyUnresolvedReferences
    if pd._libs.lib.infer_dtype(pandas_obj) != 'string':
        pandas_obj = pandas_obj.astype(str)
    return pandas_obj

def categorical_to_factor(categorical):
    from rpy2.robjects import FactorVector, IntVector, r
    assert isinstance(categorical.dtype, pd.CategoricalDtype)
    # Convert codes from 0- to 1-based
    # R uses -2147483648 (np.iinfo('int32').min) for NaN, Python uses -1
    codes = np.array(categorical.cat.codes.values, dtype='int32')
    codes += 1
    codes[codes == 0] = np.iinfo('int32').min
    factor = IntVector(codes)
    factor.rclass = r.c('ordered', 'factor') \
        if categorical.dtype.ordered else 'factor'
    factor.slots['levels'] = index_to_rvector(categorical.cat.categories)
    factor = FactorVector(factor)
    return factor

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
    # handle new nullable types properly
    if isinstance(dtype, pd.CategoricalDtype):
        return categorical_to_factor
    elif pd.api.types.is_float_dtype(dtype):
        vector_class = FloatVector if dtype == float else \
            lambda x: FloatVector(x.astype(float))
    elif pd.api.types.is_integer_dtype(dtype) or dtype == bool:
        vector_class = IntVector if dtype == np.int32 else \
            lambda x: IntVector(x.astype(np.int32))
        if dtype == bool:
            vector_class = lambda x: r['as.logical'](IntVector(x.astype(np.int32)))
    elif isinstance(array_or_index_or_series_or_df, np.ndarray) and dtype.type is np.str_ \
            or (isinstance(array_or_index_or_series_or_df, pd.DataFrame) and 
                array_or_index_or_series_or_df.apply(pd._libs.lib.infer_dtype).eq('string').all()) \
            or pd._libs.lib.infer_dtype(array_or_index_or_series_or_df) == 'string':
        vector_class = StrVector
    else:
        raise ValueError(f'Unsupported dtype "{dtype}"!')
    # conversion only applied to DataFrames
    if not isinstance(array_or_index_or_series_or_df, np.ndarray):
        vector_class = lambda x, vector_class=vector_class: \
            vector_class(x.values) if isinstance(x, pd.DataFrame) else vector_class(x)

    return vector_class

def array_to_rvector(array):
    assert isinstance(array, np.ndarray)
    Vector = get_rpy2_vector_class(array)
    rvector = Vector(array)
    return rvector

def index_to_rvector(index):
    assert isinstance(index, pd.Index)
    Vector = get_rpy2_vector_class(index)
    rvector = Vector(index)
    return rvector

def series_to_rvector(series, convert_index=True):
    assert isinstance(series, pd.Series)
    Vector = get_rpy2_vector_class(series)
    rvector = Vector(series)
    if convert_index:
        from rpy2.robjects import StrVector
        rvector.names = StrVector(astype_str(series.index))  # coerce to string
    return rvector

def rvector_to_array(rvector):
    from rpy2.robjects import DataFrame, FactorVector, ListVector, Matrix, r, \
        Vector
    assert isinstance(rvector, Vector)
    assert not isinstance(rvector, (DataFrame, ListVector, Matrix))
    if isinstance(rvector, FactorVector):
        # Convert codes from 1- to 0-based
        # R uses -2147483648 (np.iinfo('int32').min) for NaN, Python uses -1
        codes = np.array(rvector)
        codes[codes == np.iinfo('int32').min] = 0
        codes -= 1
        array = pd.Categorical.from_codes(
            codes=codes, dtype=pd.CategoricalDtype(
                rvector.levels, ordered=r['is.ordered'](rvector)[0]))
    else:
        array = np.asarray(rvector)
    return array

def rvector_to_series(rvector):
    from rpy2.robjects import FactorVector, IntVector
    series = pd.Series(rvector_to_array(rvector),
                       index=np.asarray(rvector.names)
                       if rvector.names else None)
    if isinstance(rvector, IntVector) and not isinstance(rvector, FactorVector):
        # R uses -2147483648 (np.iinfo('int32').min) for NaN
        mask = series == np.iinfo('int32').min
        if mask.any():
            # noinspection PyUnresolvedReferences
            series = series.astype('Int32')
            series[mask] = pd.NA
    return series

def rvector_to_index(rvector):
    index = pd.Index(rvector_to_series(rvector))
    return index

def df_to_rdf(df, convert_index=True):
    assert isinstance(df, pd.DataFrame)
    from rpy2.robjects import DataFrame
    rdf = DataFrame({col: series_to_rvector(series, convert_index=False)
                     for col, series in df.items()})
    if convert_index:
        from rpy2.robjects import StrVector
        rdf.rownames = StrVector(astype_str(df.index))  # coerce to string
    return rdf

def rdf_to_df(rdf):
    from rpy2.robjects import DataFrame
    assert isinstance(rdf, DataFrame)
    df = pd.DataFrame({col: rvector_to_series(rvector)
                       for col, rvector in zip(rdf.names, rdf)})
    if rdf.rownames:
        df.index = np.asarray(rdf.rownames)
    return df

def array_to_rmatrix(array):
    assert isinstance(array, np.ndarray)
    if isinstance(array, np.matrix):
        array = np.asarray(array)
    from rpy2.robjects import r
    Vector = get_rpy2_vector_class(array)
    rmatrix = r.matrix(Vector(array.ravel('F')), *array.shape)
    return rmatrix

def rmatrix_to_array(rmatrix):
    from rpy2.robjects import Matrix
    assert isinstance(rmatrix, Matrix)
    array = np.asarray(rmatrix)
    return array

def df_to_rmatrix(df, convert_index=True, convert_columns=True):
    assert isinstance(df, pd.DataFrame)
    from rpy2.robjects import r
    Vector = get_rpy2_vector_class(df)
    rmatrix = r.matrix(Vector(df.values.ravel('F')), *df.shape)
    if convert_index:
        from rpy2.robjects import StrVector
        rmatrix.rownames = StrVector(astype_str(df.index))  # coerce to string
    if convert_columns:
        from rpy2.robjects import StrVector
        rmatrix.colnames = StrVector(astype_str(df.columns))  # coerce to string
    return rmatrix

def rmatrix_to_df(rmatrix):
    from rpy2.robjects import Matrix
    assert isinstance(rmatrix, Matrix)
    df = pd.DataFrame(
        np.asarray(rmatrix),
        index=np.asarray(rmatrix.rownames) if rmatrix.rownames else None,
        columns=np.asarray(rmatrix.colnames) if rmatrix.colnames else None)
    return df

def rsparse_to_pysparse(rsparse):
    # R/Python sparse matrix formats: github.com/theislab/anndata2ri/issues/8
    # Existing conversion code:
    # github.com/theislab/anndata2ri/blob/master/src/anndata2ri/scipy2ri/r2py.py
    from rpy2.robjects import r
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
    matrix_class = r['class'](rsparse)[0]
    if matrix_class.endswith('gCMatrix'):
        return csc_matrix((rsparse.slots['x'], rsparse.slots['i'],
                           rsparse.slots['p']), shape=rsparse.slots['Dim'])
    elif matrix_class.endswith('gRMatrix'):
        return csr_matrix((rsparse.slots['x'], rsparse.slots['j'],
                           rsparse.slots['p']), shape=rsparse.slots['Dim'])
    elif matrix_class.endswith('gTMatrix'):
        return coo_matrix((rsparse.slots['x'], (rsparse.slots['i'],
                                                rsparse.slots['j'])),
                          shape=rsparse.slots['Dim'])
    else:
        raise ValueError(f'Unsupported sparse matrix class "{matrix_class}"!')

def pysparse_to_rsparse(pysparse):
    # R/Python sparse matrix formats: github.com/theislab/anndata2ri/issues/8
    # Existing conversion code:
    # github.com/theislab/anndata2ri/blob/master/src/anndata2ri/scipy2ri/py2r.py
    from rpy2.robjects import r
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
    r.library('Matrix', quietly=True)
    assert isinstance(pysparse, (csc_matrix, csr_matrix, coo_matrix))
    if pysparse.dtype == np.float32:
        print('Casting float32 to float64 for much faster conversion to R...')
        pysparse = pysparse.astype(float)
    if pysparse.dtype in (bool, np.uint32, np.uint64, np.int64):
        print(f'Casting {pysparse.dtype.__name__} to int32 for much faster '
              f'conversion to R...')
        pysparse = pysparse.astype(np.int32)
    matrix_class = type(pysparse)
    if matrix_class == csc_matrix:
        return r.sparseMatrix(
            x=array_to_rvector(pysparse.data),
            i=array_to_rvector(pysparse.indices),
            p=array_to_rvector(pysparse.indptr),
            dims=r.c(*pysparse.shape), index1=False, repr='C')
    elif matrix_class == csr_matrix:
        return r.sparseMatrix(
            x=array_to_rvector(pysparse.data),
            j=array_to_rvector(pysparse.indices),
            p=array_to_rvector(pysparse.indptr),
            dims=r.c(*pysparse.shape), index1=False, repr='R')
    elif matrix_class == coo_matrix:
        return r.sparseMatrix(
            x=array_to_rvector(pysparse.data),
            i=array_to_rvector(pysparse.row),
            j=array_to_rvector(pysparse.col),
            dims=r.c(*pysparse.shape), index1=False, repr='T')
    else:
        raise ValueError(f'Unsupported sparse matrix class "{matrix_class}"!')

def r2py(robject, verbose=False, level=0):
    # Attempt to recursively convert an R object to Python.
    # Note: DataFrame subclasses ListVector, which subclasses Vector.
    # Matrix also subclasses Vector.
    import re
    from rpy2.robjects import DataFrame, ListVector, Matrix, r, Vector
    from rpy2.robjects.methods import RS4
    from rpy2.rinterface_lib.sexp import NULLType
    if isinstance(robject, DataFrame):
        return rdf_to_df(robject)
    elif isinstance(robject, ListVector):
        if robject.names:
            if not verbose:
                return dict(zip(robject.names, map(r2py, robject)))
            else:
                py = {}
                for name, obj in zip(robject.names, robject):
                    print('\t' * level + f'{name}:')
                    py[name] = r2py(obj, verbose=True, level=level + 1)
                return py
        else:
            return tuple(map(r2py, robject))
    elif isinstance(robject, Matrix):
        return rmatrix_to_df(robject)
    elif isinstance(robject, Vector):
        # noinspection PyTypeChecker
        return rvector_to_series(robject)
    elif re.match('[a-z]g[CRT]Matrix', r['class'](robject)[0]):
        return rsparse_to_pysparse(robject)
    elif r['class'](robject)[0] == 'DFrame':
        # Bioconductor data frame class
        return rdf_to_df(r['as.data.frame'](robject))
    elif isinstance(robject, RS4):
        # Iterate over slotnames() instead of slots because slots includes one
        # extra attribute at the end, "class" (the object's class)
        if not verbose:
            return {slot: r2py(robject.slots[slot])
                    for slot in robject.slotnames()}
        else:
            py = {}
            for slot in robject.slotnames():
                print('\t' * level + f'{slot}:')
                py[slot] = r2py(robject.slots[slot], verbose=True,
                                level=level + 1)
            return py
    elif isinstance(robject, NULLType):
        return None
    else:
        return robject  # unsupported type; return unchanged

def get_nested_robject(robject, node_names):
    # e.g. to get obj@assays@data@listData$counts, use
    # get_nested_robject(obj, ('assays', 'data', 'listData')).rx2('counts')
    for node_name in node_names:
        robject = robject.slots[node_name]
    return robject

def ACAT(pvalues):
    from scipy.stats import cauchy
    return cauchy.sf(np.tan((0.5 - pvalues.astype(np.float128)) * np.pi).mean())

def linear_regression(X, ys, covariance=None, whitening_matrix=None,
                      return_p=True):
    # Performs vectorized multiple linear regression (of each column of y on X)
    # Based on https://pingouin-stats.org/_modules/pingouin/regression.html
    # and https://pingouin-stats.org/generated/pingouin.linear_regression.html
    # Does not add an intercept!
    # If covariance or whitening_matrix is not None, use GLS instead of OLS
    if isinstance(X, pd.DataFrame): X = X.values
    if isinstance(ys, pd.DataFrame): ys = ys.values
    assert isinstance(X, np.ndarray), type(X)
    assert isinstance(ys, np.ndarray), type(ys)
    assert X.ndim == ys.ndim == 2, (X.ndim, ys.ndim)
    assert np.issubdtype(X.dtype, np.number), X.dtype
    assert np.issubdtype(ys.dtype, np.number), ys.dtype
    assert len(X) == len(ys), (X.shape, ys.shape)
    assert covariance is None or whitening_matrix is None
    assert not np.isnan(X).any()
    assert not np.isnan(ys).any()
    if covariance is not None:
        if isinstance(covariance, pd.DataFrame): covariance = covariance.values
        assert isinstance(covariance, np.ndarray), type(covariance)
        assert np.issubdtype(covariance.dtype, np.number), covariance.dtype
        assert covariance.shape == (len(X), len(X)), covariance.shape
        assert (covariance == covariance.T).all(), 'Covariance is not symmetric'
        assert not np.isnan(covariance).any()
        # dtrtri(chol, lower=True) can give an order of magnitude speed-up over
        # inv(chol) by exploiting the fact that the Cholesky matrix is lower
        # triangular.
        from scipy.linalg import cholesky
        from scipy.linalg.lapack import dtrtri
        whitening_matrix, info = dtrtri(cholesky(
            covariance, lower=True, check_finite=False),
            lower=True, overwrite_c=True)
        if info > 0:
            raise np.linalg.LinAlgError('Singular matrix')
        elif info < 0:
            raise ValueError(f'Invalid input to dtrtri (info = {info})')
        X = whitening_matrix @ X
        ys = whitening_matrix @ ys
    if whitening_matrix is not None:
        if isinstance(whitening_matrix, pd.DataFrame):
            whitening_matrix = whitening_matrix.values
        assert isinstance(whitening_matrix, np.ndarray), type(whitening_matrix)
        assert np.issubdtype(whitening_matrix.dtype, np.number), \
            whitening_matrix.dtype
        assert whitening_matrix.shape == (len(X), len(X)), \
            whitening_matrix.shape
        assert not np.isnan(whitening_matrix).any()
        X = whitening_matrix @ X
        ys = whitening_matrix @ ys
    coef, residues, rank, singular_values = np.linalg.lstsq(X, ys, rcond=None)
    # If residues are missing, could mean that:
    if len(residues) == 0:
        # a) the fit is perfect (if number of features <= number of examples)
        if len(X) <= X.shape[1]:
            assert not return_p, \
                f'Perfect fit since # features (len(X)={len(X)}) <= # ' \
                f'examples (X.shape[1]={X.shape[1]}), so p-values not defined'
        # or b) some columns are collinear (otherwise)
        else:
            X_rank = np.linalg.matrix_rank(X)
            assert X_rank < X.shape[1]
            raise ValueError(f'linear_regression: X has {X.shape[1]} features '
                             f'but only {X_rank} are linearly independent; are '
                             f'some columns collinear?')
    if return_p:
        from scipy.special import stdtr
        df = len(X) - rank
        se = np.sqrt(np.outer(np.linalg.pinv(X.T @ X).diagonal(),
                              residues / df))
        # pval = 2 * t.sf(np.abs(coef / se), df)
        pval = 2 * stdtr(df, -np.abs(coef / se))
        return coef, se, pval
    else:
        return coef

def regress_out(data, covariates, axis=0):
    assert axis == 0 or axis == 1
    if axis == 1:
        data = data.T
    assert data.index.equals(covariates.index)
    residuals = data - covariates.values @ linear_regression(
        covariates.values, data.values, return_p=False)
    if axis == 1:
        residuals = residuals.T
    return residuals

def cov(X, Y=None, *, rowvar=True):
    # np.cov but with proper support for Y
    if Y is not None:
        assert type(X) == type(Y), f'{type(X)=}, {type(Y)=}'
        assert len(X.shape) == len(Y.shape) == 2, f'{X.shape=}, {Y.shape=}'
        assert X.shape[rowvar] == Y.shape[rowvar], f'{X.shape=}, {Y.shape=}'
    dataframe_input = isinstance(X, pd.DataFrame)
    if dataframe_input:
        if Y is not None:
            assert isinstance(Y, pd.DataFrame)
            if rowvar:
                assert X.columns.equals(Y.columns)
                index, columns = X.index, Y.index
            else:
                assert X.index.equals(Y.index)
                index, columns = X.columns, Y.columns
            X = X.values
            Y = Y.values
        else:
            if rowvar:
                index, columns = X.index, X.index
            else:
                index, columns = X.columns, X.columns
            X = X.values    # a) Transpose if rowvar=False
    if not rowvar:
        X = X.T
        if Y is not None:
            Y = Y.T
    # b) De-mean
    X = X - X.mean(axis=1)[:, None]
    if Y is not None:
        Y = Y - Y.mean(axis=1)[:, None]
    # c) Calculate covariance
    if Y is None:
        c = X.dot(X.T) / (X.shape[1] - 1)
    else:
        c = X.dot(Y.T) / (X.shape[1] - 1)
    if dataframe_input:
        # noinspection PyUnboundLocalVariable
        c = pd.DataFrame(c, index=index, columns=columns)
    return c

def cor(X, Y=None, *, rowvar=True, return_p=False):
    # np.corrcoef but with proper support for Y and ability to return p-values
    if Y is not None:
        assert type(X) == type(Y), f'{type(X)=}, {type(Y)=}'
        assert len(X.shape) == len(Y.shape) == 2, f'{X.shape=}, {Y.shape=}'
        assert X.shape[rowvar] == Y.shape[rowvar], f'{X.shape=}, {Y.shape=}'
    dataframe_input = isinstance(X, pd.DataFrame)
    if dataframe_input:
        if Y is not None:
            assert isinstance(Y, pd.DataFrame)
            if rowvar:
                assert X.columns.equals(Y.columns)
                index, columns = X.index, Y.index
            else:
                assert X.index.equals(Y.index)
                index, columns = X.columns, Y.columns
            X = X.values
            Y = Y.values
        else:
            if rowvar:
                index, columns = X.index, X.index
            else:
                index, columns = X.columns, X.columns
            X = X.values
    # a) Calculate covariance
    c = cov(X, Y, rowvar=rowvar)
    # b) Convert to correlation
    if Y is None:
        stddev = np.sqrt(np.diag(c))
        c /= stddev[:, None]
        c /= stddev[None, :]
    else:
        c /= X.std(axis=int(rowvar), ddof=1)[:, None]
        c /= Y.std(axis=int(rowvar), ddof=1)
    # c) Clip to [-1, 1]
    np.clip(c, -1, 1, out=c)
    if dataframe_input:
        # noinspection PyUnboundLocalVariable
        c = pd.DataFrame(c, index=index, columns=columns)
    # d) Calculate p-value if specified
    if return_p:
        from scipy.special import stdtr
        df = X.shape[rowvar] - 2
        with np.errstate(divide='ignore'):
            p = 2 * stdtr(df, -np.abs(c * np.sqrt(df / (1 - c * c))))
        return c, p
    else:
        return c

def check_input(X, X_name, *, types=(np.ndarray, pd.DataFrame),
                dtypes=(np.number, bool), ndim=2, no_NaNs=True, min_rows=1,
                min_cols=1):
    X = X.values if hasattr(X, 'values') else X
    if types is not None and not isinstance(X, types):
        raise ValueError(
            f'{X_name}\'s type is ({type(X)}) but must instead be one '
            f'of these types: {", ".join(t.__name__ for t in types)}')
    if dtypes is not None and not any(np.issubdtype(X.dtype, dtype)
                                      for dtype in dtypes):
        raise ValueError(
            f'{X_name}\'s dtype is ({X.dtype}) but must instead be one '
            f'of these dtypes: {", ".join(map(str, dtypes))}')
    if ndim is not None and X.ndim != ndim:
        raise ValueError(f'{X_name} must have {ndim} dimensions, '
                         f'but instead has {X.ndim} (shape: {X.shape})')
    if no_NaNs and np.isnan(np.atleast_1d(X)).any():
        raise ValueError(f'{X_name} contains NaNs')
    if min_rows is not None and len(X) < min_rows:
        raise ValueError(
            f'{X_name} must have at least {min_rows} row'
            f'{"s" if min_rows != 1 else ""}, but instead has {len(X)}')
    if min_cols is not None and X.shape[1] < min_cols:
        raise ValueError(
            f'{X_name} must have at least {min_cols} column'
            f'{"s" if min_cols != 1 else ""}, but instead has {X.shape[1]}')

def check_input_pair(X, X_name, Y, Y_name, *, same_type=True, same_dtype=False,
                     same_ndim=True, same_nrows=False, same_ncols=False):
    if same_type and type(X) != type(Y):
        raise ValueError(f'{X_name} and {Y_name} have different types: '
                         f'{X_name}\'s is "{type(X)}" while {Y_name}\'s is '
                         f'"{type(Y)}"')
    if same_dtype and X.dtype != Y.dtype:
        raise ValueError(f'{X_name} and {Y_name} have different dtypes: '
                         f'{X_name}\'s is "{X.dtype}" while {Y_name}\'s is '
                         f'"{Y.dtype}"')
    if same_ndim and X.ndim != Y.ndim:
        raise ValueError(f'{X_name} and {Y_name} have different numbers of '
                         f'dimensions: {X_name} has {X.ndim} while {Y_name} '
                         f'has {Y.ndim}')
    if same_nrows and len(X) != len(Y):
        raise ValueError(
            f'{X_name} and {Y_name} have different numbers of rows: '
            f'{X_name} has {len(X)} while {Y_name} has {len(Y)}')
    if same_ncols and X.shape[1] != Y.shape[1]:
        raise ValueError(
            f'{X_name} and {Y_name} have different numbers of columns: '
            f'{X_name} has {X.shape[1]} while {Y_name} has {Y.shape[1]}')

def whiten(X, standardize=True, method='Cholesky'):
    # Based on rdrr.io/cran/whitening/src/R/getPhiPsiW.R
    if method not in ('Cholesky', 'ZCA', 'ZCA-cor', 'PCA', 'PCA-cor'):
        raise ValueError(f"Unknown whitening method {method} (valid methods: "
                         f"'Cholesky', 'ZCA', 'ZCA-cor', 'PCA', 'PCA-cor')")
    # Convert DataFrame inputs to NumPy arrays
    dataframe_input = isinstance(X, pd.DataFrame)
    if dataframe_input:
        X_index = X.index
        X_columns = X.columns
        X = X.values
    # Standardize X if standardize=True, otherwise just de-mean
    X = (X - X.mean(axis=1)[:, None])
    if standardize:
        X = X / X.std(axis=1)[:, None]
    # Estimate covariance from X using the OAS method, which is much
    # faster than Ledoit-Wolf and corpcor, e.g. 100s vs 540s vs 1702s
    # for np.random.RandomState(0).random(size=(10000, 10000)). The
    # choice of covariance method appears to have very little effect on
    # the final benchmark results.
    from sklearn.covariance import oas
    covariance = oas(X)[0]
    # Calculate whitening matrix
    if method == 'Cholesky':
        # dtrtri(chol, lower=True) can give an order of magnitude speed-up over
        # inv(chol) by exploiting the fact that the Cholesky matrix is lower
        # triangular.
        #
        # Note: both cholesky() and dtrtri() can be efficiently parallelized,
        # e.g. via the mkl-service conda package:
        # import mkl; mkl.set_num_threads(num_threads)
        from scipy.linalg import cholesky
        from scipy.linalg.lapack import dtrtri
        whitening_matrix, info = dtrtri(cholesky(
            covariance, lower=True, overwrite_a=True, check_finite=False),
            lower=True, overwrite_c=True)
        if info > 0:
            raise np.linalg.LinAlgError('Singular matrix')
        elif info < 0:
            raise ValueError(f'Invalid input to dtrtri (info = {info})')
    else:
        # Calculate eigenvalues and eigenvectors of the covariance (for ZCA and
        # PCA) or correlation (for ZCA-cor and PCA-cor). Use scipy.linalg.eigh
        # since np.linalg.eigh will rarely, and incorrectly, return all-zero
        # eigenvalues - possibly because the eigenvalue computation runs out of
        # memory?
        from scipy.linalg import eigh
        if method == 'ZCA' or method == 'PCA':
            eigenvalues, eigenvectors = eigh(covariance, check_finite=False)
        else:
            stddev = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(stddev, stddev)
            eigenvalues, eigenvectors = eigh(correlation, check_finite=False)
        # For ZCA, the whitening matrix is U * S^-0.5 * V; for PCA, it's just
        # S^-0.5 * V. For PCA and PCA-cor, the eigenvectors' sign ambiguity
        # affects the final result, so multiply eigenvectors by -1 as needed to
        # ensure that the eigenvector matrix has a positive diagonal.
        if method == 'ZCA' or method == 'ZCA-cor':
            whitening_matrix = eigenvectors @ \
                               np.diag(1 / np.sqrt(eigenvalues)) @ \
                               eigenvectors.T
        else:
            makePosDiag = lambda matrix: matrix * np.sign(np.diag(matrix))
            whitening_matrix = np.diag(1 / np.sqrt(eigenvalues)) @ \
                               makePosDiag(eigenvectors).T
        # For ZCA-cor and PCA-cor, right-multiply by a diagonal matrix of the
        # inverse standard deviations
        if method == 'ZCA-cor' or method == 'PCA-cor':
            whitening_matrix = whitening_matrix @ \
                               np.diag(1 / np.sqrt(np.diag(covariance)))
    # Whiten
    whitened_X = whitening_matrix.dot(X.T).T
    # Convert output to DataFrame if input was DataFrame
    if dataframe_input:
        # noinspection PyUnboundLocalVariable
        whitened_X = pd.DataFrame(whitened_X, index=X_index, columns=X_columns)
    return whitened_X

def correlate(X, Y=None, *, method='GLS', covariates=None, standardize=True,
              check_inputs=True):
    # If beta_1 is the beta from gls(y ~ x), then:
    # pearson_1 = beta_1 * sd(x) / sd(y)
    #
    # and if beta_2 is the beta from gls(x ~ y), then:
    # pearson_2 = beta_2 * sd(y) / sd(x)
    #
    # If you multiply them together:
    # pearson_1 * pearson_2 = beta_1 * beta_2
    #
    # But pearson_1 should be equal to pearson_2, so:
    # pearson^2 = beta_1 * beta_2
    #
    # Take the square root
    # |pearson| = sqrt(beta_1 * beta_2)
    #
    # But beta_1 and beta_2 and pearson should all have the same sign, so:
    # pearson = sqrt(beta_1 * beta_2) * sign(beta_1)  # or beta_2
    #
    from scipy.special import stdtr
    # Check inputs
    # X needs a bare minimum of 2 rows (variables) to calculate covariance from.
    # For Ledoit-Wolf and corpcor, the bare minimum is 3: Ledoit-Wolf gives a
    # non-positive-definite covariance matrix when there are only two variables,
    # and corpcor's cov.shrink() explicitly requires at least 3 rows.
    assert method in ('Pearson', 'Spearman', 'GLS', 'PCA', 'MMC') or \
           method.startswith('whiten'), method
    if check_inputs:
        check_input(X, 'X', min_rows=2, min_cols=2)
        if Y is not None:
            check_input(Y, 'Y', min_rows=1, min_cols=2)
            check_input_pair(X, 'X', Y, 'Y', same_ncols=True)
        if covariates is not None:
            check_input(covariates, 'covariates')
    # For methods starting with 'whiten', whiten the data and then run Pearson
    if method.startswith('whiten'):
        return correlate(whiten(X, method=method.split('_')[1],
                                standardize=standardize),
                         method='Pearson', standardize=False,
                         check_inputs=False)
    # If method == 'MMC', run MMC
    if method == 'MMC':
        assert Y is None  # not implemented
        assert covariates is None  # not implemented
        from rpy2.robjects import r
        r.source('mmc.R')
        X_r = df_to_rmatrix(X)
        # mmc.corsym() didn't finish for co-essentiality after 13 hours
        MMC_results = r['mmc.cor'](X_r, r['mmc.ncov'](X_r))
        correlation = rmatrix_to_df(MMC_results.rx2('rs'))
        p = 1 - correlation  # so that Spearman(correlation, p) = -1;
                             # note that correlation is always positive
        return correlation, p
    # Convert DataFrame inputs to NumPy arrays
    dataframe_input = isinstance(X, pd.DataFrame)
    if dataframe_input:
        X_index = X.index
        X = X.values
        if Y is not None:
            Y_index = Y.index
            Y = Y.values
    df = X.shape[1] - 2
    if method == 'Spearman':
        from scipy.stats import rankdata
        X = rankdata(X, axis=1)
        if Y is not None:
            Y = rankdata(Y, axis=1)
    if method == 'PCA':
        # Include the top N PCs as covariates, where N is the optimal number of
        # PCs according to PCAforQTL's runElbow(), and then run Pearson
        # correlation.
        # github.com/heatherjzhou/PCAForQTL
        # biorxiv.org/content/10.1101/2022.03.09.483661v1.full
        assert Y is None  # not implemented
        assert covariates is None  # not implemented
        from rpy2.robjects import r
        r.source('https://raw.githubusercontent.com/heatherjzhou/PCAForQTL/'
                 'master/R/22.01.04_main1.1_runElbow.R')
        # Standardize X
        if standardize:
            X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
        X_r_T = array_to_rmatrix(X.T)
        prcompResult = r.prcomp(X_r_T, center=False)
        numOfPCsChosen = int(r.runElbow(X_r_T, prcompResult)[0])
        print(f'PCA: {numOfPCsChosen} PCs chosen')
        PCs = rmatrix_to_array(prcompResult.rx2('x'))[:, :numOfPCsChosen]
        correlation, p = correlate(X, method='Pearson', covariates=PCs,
                                   standardize=False, check_inputs=False)
    elif (method == 'Pearson' or method == 'Spearman') and covariates is None:
        correlation, p = cor(X, Y, return_p=True)
    else:
        # Standardize data and covariates
        if standardize:
            X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
            if Y is not None:
                Y = (Y - Y.mean(axis=1)[:, None]) / Y.std(axis=1)[:, None]
            if covariates is not None:
                covariates = (covariates - covariates.mean()) / covariates.std()
        if method == 'GLS':
            # Estimate covariance from X using the OAS method, which is much
            # faster than Ledoit-Wolf and corpcor, e.g. 100s vs 540s vs 1702s
            # for np.random.RandomState(0).random(size=(10000, 10000)). The
            # choice of covariance method appears to have very little effect on
            # the final benchmark results.
            from sklearn.covariance import oas
            covariance = oas(X)[0]
            # Calculate whitening matrix. dtrtri(chol, lower=True) can give an
            # order of magnitude speed-up over inv(chol) by exploiting the fact
            # that the Cholesky matrix is lower triangular.
            from scipy.linalg import cholesky
            from scipy.linalg.lapack import dtrtri
            whitening_matrix, info = dtrtri(cholesky(
                covariance, lower=True, overwrite_a=True, check_finite=False),
                lower=True, overwrite_c=True)
            if info > 0:
                raise np.linalg.LinAlgError('Singular matrix')
            elif info < 0:
                raise ValueError(f'Invalid input to dtrtri (info = {info})')
            # Whiten everything. Make a design matrix with the whitened
            # intercept and covariates, plus an empty slot in the 1st position
            # (0-based) to copy each row of data to.
            whitened_intercept = whitening_matrix.sum(axis=1)
            whitened_X = whitening_matrix.dot(X.T).T
            if Y is not None:
                whitened_Y = whitening_matrix.dot(Y.T).T
            if covariates is not None:
                whitened_covariates = whitening_matrix.dot(covariates)
                design_matrix = np.concatenate((
                    whitened_intercept[:, None],
                    np.empty(len(whitened_intercept))[:, None],
                    whitened_covariates), axis=1)
            else:
                design_matrix = np.stack((
                    whitened_intercept,
                    np.empty(len(whitened_intercept))), axis=1)
        else:
            # Pearson or Spearman with covariates; don't whiten the data, just
            # make the design matrix
            whitened_X = X
            whitened_Y = Y
            intercept = np.ones(X.shape[1])
            design_matrix = np.concatenate((intercept[:, None],
                                            np.empty(len(intercept))[:, None],
                                            covariates), axis=1)
        # Run linear regression on the whitened data; get pseudocorrelation
        def linear_regression(X, Y, design_matrix):
            GLS_betas, GLS_SEs = (np.empty((len(X), len(Y))) for _ in range(2))
            for row_index in range(len(X)):
                design_matrix[:, 1] = X[row_index]
                coef, residues = np.linalg.lstsq(design_matrix, Y.T,
                                                 rcond=None)[:2]
                GLS_betas[row_index] = coef[1]
                GLS_SEs[row_index] = np.sqrt(np.linalg.pinv(
                    design_matrix.T.dot(design_matrix))[1, 1] * residues / df)
            return GLS_betas, GLS_SEs
        if Y is None:
            GLS_betas, GLS_SEs = linear_regression(whitened_X, whitened_X,
                                                   design_matrix)
            correlation = np.sqrt(GLS_betas * GLS_betas.T) * np.sign(GLS_betas)
            with np.errstate(divide='ignore'):
                p = 2 * stdtr(df, -np.abs(GLS_betas / GLS_SEs))
        else:
            # noinspection PyUnboundLocalVariable
            GLS_betas_X, GLS_SEs_X = linear_regression(whitened_X, whitened_Y,
                                                       design_matrix)
            GLS_betas_Y, GLS_SEs_Y = linear_regression(whitened_Y, whitened_X,
                                                       design_matrix)
            correlation = np.sqrt(GLS_betas_X * GLS_betas_Y.T) * \
                          np.sign(GLS_betas_X)
            with np.errstate(divide='ignore'):
                p = 2 * stdtr(df, -np.abs(GLS_betas_X / GLS_SEs_X))
    # Convert outputs to DataFrames if inputs were DataFrames
    if dataframe_input:
        if Y is not None:
            # noinspection PyUnboundLocalVariable
            correlation = pd.DataFrame(correlation,
                                       index=X_index, columns=Y_index)
            p = pd.DataFrame(p, index=X_index, columns=Y_index)
        else:
            # noinspection PyUnboundLocalVariable
            correlation = pd.DataFrame(correlation,
                                       index=X_index, columns=X_index)
            p = pd.DataFrame(p, index=X_index, columns=X_index)
    return correlation, p

def logistic_regression(X, y, return_model=False, report_intercept=False,
                        **kwargs):
    # Based on https://pingouin-stats.org/_modules/pingouin/regression.html
    # and https://pingouin-stats.org/generated/pingouin.logistic_regression.html
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    assert isinstance(X, pd.DataFrame), type(X)
    assert isinstance(y, pd.Series), type(y)
    assert np.issubdtype(X.values.dtype, np.number), X.dtypes
    assert y.dtype == bool, y.dtype
    model = LogisticRegression(penalty=None, solver='newton-cg', random_state=0,
                               **kwargs).fit(X, y)
    predictions = model.decision_function(X)
    denom = np.tile(2 * (1 + np.cosh(predictions)), (X.shape[1] + 1, 1)).T
    X_design = np.column_stack((np.ones(len(X)), X))
    variance_covariance_matrix = np.linalg.pinv((X_design / denom).T @ X_design)
    index = X.columns.insert(0, 'Intercept') if report_intercept else X.columns
    betas = pd.Series(np.concatenate([model.intercept_, model.coef_[0]])
                      if report_intercept else model.coef_[0], index=index)
    ORs = np.exp(betas)
    sigmas = pd.Series(np.sqrt(np.diag(variance_covariance_matrix)[
                               0 if report_intercept else 1:]), index=index)
    lower_CIs = np.exp(betas - norm.isf(0.025) * sigmas)
    upper_CIs = np.exp(betas + norm.isf(0.025) * sigmas)
    pvalues = pd.Series(2 * norm.sf(np.abs(betas / sigmas)), index=index)
    results = pd.DataFrame({'OR': ORs, 'lower_CI': lower_CIs,
                            'upper_CI': upper_CIs, 'p': pvalues})
    return (results, model) if return_model else results

def associate_linear(features, outcomes, covariates, return_results=False,
                     print_significant_only=False, print_p_values=False,
                     transpose_results=False):
    from scipy.stats import norm
    assert isinstance(features, pd.DataFrame), type(features)
    assert isinstance(outcomes, pd.DataFrame), type(outcomes)
    assert isinstance(covariates, pd.DataFrame), type(covariates)
    assert outcomes.nunique().gt(2).all(), outcomes.nunique().sort_values()
    betas, LCIs, UCIs, ps = (pd.DataFrame(
        index=features.columns, columns=outcomes.columns, dtype=float)
        for _ in range(4))
    for feature_name, outcome_name in product(features, outcomes):
        data = features[[feature_name]].dropna().join(
            outcomes[[outcome_name]].dropna(), how='inner').join(
            covariates, how='inner').assign(intercept=lambda df: 1)
        assert not any(series.hasnans for col, series in data.items()), \
            data.isna().sum().sort_values()
        X = data[[feature_name, 'intercept'] +
                 covariates.columns.tolist()].astype(float)
        y = data[[outcome_name]].astype(float)
        coef, se, df, pval = linear_regression(X.values, y.values)
        beta = coef[0, 0]
        se = se[0, 0]
        d = 2 * beta / se / np.sqrt(df)
        LCI = 2 * (beta - norm.isf(0.025) * se) / se / np.sqrt(df)
        UCI = 2 * (beta + norm.isf(0.025) * se) / se / np.sqrt(df)
        p = pval[0, 0]
        betas.at[feature_name, outcome_name] = d
        LCIs.at[feature_name, outcome_name] = LCI
        UCIs.at[feature_name, outcome_name] = UCI
        ps.at[feature_name, outcome_name] = p
    if transpose_results:
        betas, LCIs, UCIs, ps = betas.T, LCIs.T, UCIs.T, ps.T
    fdrs, significant = global_fdr(ps)
    if print_significant_only:
        print((betas.stack().apply('{:.2f}'.format) + ' [' +
               LCIs.stack().apply('{:.2f}'.format) + ', ' +
               UCIs.stack().apply('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.stack().replace({False: ' ', True: '*'}))
              [betas.stack().sort_values(ascending=False).index]
              .where(lambda x: x.str.endswith('*')).dropna()
              .to_string())
    else:
        print((betas.applymap('{:.2f}'.format) + ' [' +
               LCIs.applymap('{:.2f}'.format) + ', ' +
               UCIs.applymap('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.replace({False: ' ', True: '*'})).to_string())
    if return_results:
        return {'beta': betas, 'LCI': LCIs, 'UCI': UCIs, 'p': ps,
                'fdr': fdrs, 'significant': significant}

def associate_logistic(features, outcomes, covariates, return_results=False,
                       print_significant_only=False, print_p_values=False,
                       transpose_results=False):
    assert isinstance(features, pd.DataFrame), type(features)
    assert isinstance(outcomes, pd.DataFrame), type(outcomes)
    assert isinstance(covariates, pd.DataFrame), type(covariates)
    assert outcomes.nunique().eq(2).all(), outcomes.nunique().sort_values()
    ORs, LCIs, UCIs, ps = (pd.DataFrame(
        index=features.columns, columns=outcomes.columns, dtype=float)
        for _ in range(4))
    for feature_name, outcome_name in product(features, outcomes):
        data = features[[feature_name]].dropna().join(
            outcomes[[outcome_name]].dropna(), how='inner').join(
            covariates, how='inner')
        assert not any(series.hasnans for col, series in data.items()), \
            data.isna().sum().sort_values()
        X = data[[feature_name] + covariates.columns.tolist()].astype(float)
        y = data[outcome_name].astype(bool)
        lr_results = logistic_regression(X, y)
        OR, LCI, UCI, p = lr_results.loc[feature_name]
        ORs.at[feature_name, outcome_name] = OR
        LCIs.at[feature_name, outcome_name] = LCI
        UCIs.at[feature_name, outcome_name] = UCI
        ps.at[feature_name, outcome_name] = p
    if transpose_results:
        ORs, LCIs, UCIs, ps = ORs.T, LCIs.T, UCIs.T, ps.T
    fdrs, significant = global_fdr(ps)
    if print_significant_only:
        print((ORs.stack().apply('{:.2f}'.format) + ' [' +
               LCIs.stack().apply('{:.2f}'.format) + ', ' +
               UCIs.stack().apply('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.stack().replace({False: ' ', True: '*'}))
              [ORs.stack().sort_values(ascending=False).index]
              .where(lambda x: x.str.endswith('*')).dropna()
              .to_string())
    else:
        print((ORs.applymap('{:.2f}'.format) + ' [' +
               LCIs.applymap('{:.2f}'.format) + ', ' +
               UCIs.applymap('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.replace({False: ' ', True: '*'})).to_string())
    if return_results:
        return {'OR': ORs, 'LCI': LCIs, 'UCI': UCIs, 'p': ps,
                'fdr': fdrs, 'significant': significant}

def associate_semi_partial(features, outcomes, covariates, return_results=False,
                           print_significant_only=False, print_p_values=False,
                           transpose_results=False):
    import pingouin as pg
    assert isinstance(features, pd.DataFrame), type(features)
    assert isinstance(outcomes, pd.DataFrame), type(outcomes)
    assert isinstance(covariates, pd.DataFrame), type(covariates)
    assert features.nunique().gt(2).all(), features.nunique().sort_values()
    assert outcomes.nunique().gt(2).all(), outcomes.nunique().sort_values()
    rhos, LCIs, UCIs, ps = (pd.DataFrame(
        index=features.columns, columns=outcomes.columns, dtype=float)
        for _ in range(4))
    for feature_name, outcome_name in product(features, outcomes):
        data = features[[feature_name]].dropna().join(
            outcomes[[outcome_name]].dropna(), how='inner').join(
            covariates, how='inner')
        assert not any(series.hasnans for col, series in data.items()), \
            data.isna().sum().sort_values()
        corr_results = pg.partial_corr(
            data=data, x=feature_name, y=outcome_name,
            y_covar=covariates.columns.tolist(), method='pearson').squeeze()
        rho = corr_results.r
        LCI, UCI = corr_results['CI95%']
        p = corr_results['p-val']
        rhos.at[feature_name, outcome_name] = rho
        LCIs.at[feature_name, outcome_name] = LCI
        UCIs.at[feature_name, outcome_name] = UCI
        ps.at[feature_name, outcome_name] = p
    if transpose_results:
        rhos, LCIs, UCIs, ps = rhos.T, LCIs.T, UCIs.T, ps.T
    fdrs, significant = global_fdr(ps)
    if print_significant_only:
        print((rhos.stack().apply('{:.2f}'.format) + ' [' +
               LCIs.stack().apply('{:.2f}'.format) + ', ' +
               UCIs.stack().apply('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.stack().replace({False: ' ', True: '*'}))
              [rhos.stack().sort_values(ascending=False).index]
              .where(lambda x: x.str.endswith('*')).dropna()
              .to_string())
    else:
        print((rhos.applymap('{:.2f}'.format) + ' [' +
               LCIs.applymap('{:.2f}'.format) + ', ' +
               UCIs.applymap('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values else '') +
               significant.replace({False: ' ', True: '*'})).to_string())
    if return_results:
        return {'rho': rhos, 'LCI': LCIs, 'UCI': UCIs, 'p': ps,
                'fdr': fdrs, 'significant': significant}

def associate_lmer(features, outcomes, covariates, random_effects,
                   return_results=False, print_significant_only=False,
                   print_p_values=False, transpose_results=False,
                   Bonferroni=False):
    from scipy.stats import norm
    from rpy2.robjects import pandas2ri, r
    r('suppressMessages(library("lmerTest"))')  # mamba install r-lmertest
    assert isinstance(features, pd.DataFrame), type(features)
    assert isinstance(outcomes, pd.DataFrame), type(outcomes)
    assert isinstance(covariates, pd.DataFrame), type(covariates)
    assert outcomes.nunique().gt(2).all(), outcomes.nunique().sort_values()
    ds, LCIs, UCIs, ps = (pd.DataFrame(
        index=features.columns, columns=outcomes.columns, dtype=float)
        for _ in range(4))
    for feature_name, outcome_name in product(features, outcomes):
        data = features[[feature_name]].dropna().join(
            outcomes[[outcome_name]].dropna(), how='inner').join(
            covariates, how='inner').astype(float).join(
            random_effects, how='inner')
        assert not any(series.hasnans for col, series in data.items()), \
            data.isna().sum().sort_values()
        pandas2ri.activate()
        model = r.lmer(
            f'{escape(outcome_name)} ~ {escape(feature_name)} + '
            f'{"+".join(escape(covariates.columns))} + '
            f'{"+".join(escape(random_effects.columns).map("(1|{})".format))}',
            data=data.rename(columns=escape))
        pandas2ri.deactivate()
        coefficients = rmatrix_to_df(r.summary(model).rx2('coefficients'))
        beta = coefficients.at[escape(feature_name), 'Estimate']
        se = coefficients.at[escape(feature_name), 'Std. Error']
        df = coefficients.at[escape(feature_name), 'df']
        d = 2 * beta / se / np.sqrt(df)
        LCI = 2 * (beta - norm.isf(0.025) * se) / se / np.sqrt(df)
        UCI = 2 * (beta + norm.isf(0.025) * se) / se / np.sqrt(df)
        p = coefficients.at[escape(feature_name), 'Pr(>|t|)']
        ds.at[feature_name, outcome_name] = d
        LCIs.at[feature_name, outcome_name] = LCI
        UCIs.at[feature_name, outcome_name] = UCI
        ps.at[feature_name, outcome_name] = p
    if transpose_results:
        ds, LCIs, UCIs, ps = ds.T, LCIs.T, UCIs.T, ps.T
    if Bonferroni:
        fdrs = bonferroni(ps)  # not really FDR, actually FWER
        significant = fdrs < 0.05
    else:
        fdrs, significant = global_fdr(ps)
    if print_significant_only:
        print((ds.stack().apply('{:.2f}'.format) + ' [' +
               LCIs.stack().apply('{:.2f}'.format) + ', ' +
               UCIs.stack().apply('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values and not Bonferroni else
                (' (p = ' + ps.applymap('{:.0g}'.format) + ')')
                if print_p_values else '') +
               significant.stack().replace({False: ' ', True: '*'}))
              [ds.stack().sort_values(ascending=False).index]
              .where(lambda x: x.str.endswith('*')).dropna()
              .to_string())
    else:
        print((ds.applymap('{:.2f}'.format) + ' [' +
               LCIs.applymap('{:.2f}'.format) + ', ' +
               UCIs.applymap('{:.2f}'.format) + ']' +
               (' (p = ' + ps.applymap('{:.0g}'.format) +
                ', FDR = ' + fdrs.applymap('{:.0g}'.format) + ')'
                if print_p_values and not Bonferroni else
                (' (p = ' + ps.applymap('{:.0g}'.format) + ')')
                if print_p_values else '') +
               significant.replace({False: ' ', True: '*'})).to_string())
    if return_results:
        return {'d': ds, 'LCI': LCIs, 'UCI': UCIs, 'p': ps,
                'fdr': fdrs, 'significant': significant}

def mantissa_and_exponent(x, base):
    exponent = math.floor(math.log(x, base))
    mantissa = x / base ** exponent
    return mantissa, exponent

def scientific_notation(number, num_decimals=0, threshold=0.001, latex=False):
    # For matplotlib
    if (threshold is not None and number >= threshold) or number == 0:
        return f'{number:.{num_decimals}f}'
    else:
        mantissa, exponent = mantissa_and_exponent(number, base=10)
        if f'{mantissa:.{num_decimals}e}' == '1e+01':
            mantissa = 1
            exponent += 1
        if latex:
            return f'{mantissa:.{num_decimals}f} × ' \
                   f'10$^\mathregular{{{exponent}}}$'
        else:
            return f'{mantissa:.{num_decimals}f} × 10^{exponent}'

def grouped_barplot_with_errorbars(data: pd.DataFrame, lower_CIs: pd.DataFrame,
                                   upper_CIs: pd.DataFrame, **kwargs):
    # Input sizes are {# groups} rows x {# bars per group/# colors} columns
    # Rows = groups, columns = bars-in-groups = colors.
    # See pandas.pydata.org/pandas-docs/stable/user_guide/
    #     visualization.html#visualization-errorbars
    # Note: you can also do something like stackoverflow.com/a/58041227/1397061
    assert data.index.equals(lower_CIs.index)
    assert data.index.equals(upper_CIs.index)
    assert data.columns.equals(lower_CIs.columns)
    assert data.columns.equals(upper_CIs.columns)
    data.plot.bar(yerr=np.stack([data - lower_CIs, upper_CIs - data], axis=1).T,
                  **kwargs)

def upper_diagonal(array, include_diagonal=True):
    # Applies upper diagonal to the last two dimensions, e.g.:
    # upper_diagonal(np.ones((100, 100))).shape == (100 * 99 // 2 + 100,)
    #                                           == (5050,)
    # upper_diagonal(np.ones((70, 100, 100))).shape == (70, 5050)
    # upper_diagonal(np.ones((30, 70, 100, 100))).shape == (30, 70, 5050)
    # If include_diagonal=False, the last dimension would be 4950 not 5050
    assert isinstance(array, np.ndarray), type(array)
    assert array.ndim >= 2, array.ndim
    assert array.shape[-2] == array.shape[-1], array.shape
    upper_diagonal_mask = np.triu(np.ones(array.shape[-2:], dtype=bool),
                                  k=0 if include_diagonal else 1)
    return array[..., upper_diagonal_mask]

def upper_diagonal_to_square(array, include_diagonal=True, mirror=False):
    assert isinstance(array, np.ndarray), type(array)
    assert array.ndim == 1, array.ndim
    # Infer the size of the square matrix we're trying to reconstruct
    # If U is the size of the upper diagonal array, and N was the size of the
    # original square matrix, then:
    # 1) include_diagonal=True: U = N(N - 1) / 2 + N = 0.5N^2 + 0.5N
    #    --> by the quadratic formula, N = (sqrt(8U + 1) - 1) / 2
    # 2) include_diagonal=False: U = N(N - 1) / 2 = 0.5N^2 - 0.5N
    #    --> by the quadratic formula, N = (sqrt(8U + 1) + 1) / 2
    N = (np.sqrt(8 * len(array) + 1) + (-1 if include_diagonal else 1)) / 2
    assert N == int(N), f'Array is the wrong length ({len(array)}) to be the ' \
                        f'upper diagonal portion of a square matrix'
    N = int(N)
    square = np.full((N, N), np.nan)
    row_indices, col_indices = np.triu_indices_from(
        square, k=0 if include_diagonal else 1)
    square[row_indices, col_indices] = array
    if mirror:
        square[col_indices, row_indices] = array
    return square

def inflation_factor(ps):
    from scipy.stats import chi2
    return chi2.isf(np.median(ps), df=1) / chi2.isf(0.5, df=1)

def inverse_normal_transform(series, c=3/8):
    # Rank-based inverse normal transform
    # Multiple possible values for c, see
    # ncbi.nlm.nih.gov/pmc/articles/PMC2921808/#S2title
    from scipy.stats import norm
    rank = series.rank()
    transformed_rank = (rank - c) / (rank.notna().sum() - 2 * c + 1)
    return pd.Series(norm.ppf(transformed_rank), index=series.index)

class Timer(object):
    # See preshing.com/20110924/timing-your-code-using-pythons-with-statement
    def __init__(self, name=None):
        if name is not None:
            print(f'{name}...')
        self.name = name
    def __enter__(self):
        from timeit import default_timer
        self.start = default_timer()
        return self
    def __exit__(self, exception_type, value, traceback):
        from timeit import default_timer
        time_difference = default_timer() - self.start
        hours, remainder = divmod(time_difference, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'{self.name if self.name is not None else "Command"} '
              f'{"took" if exception_type is None else "aborted after"} '
              f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}')

@contextmanager
def cd(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

@contextmanager
def SuppressMessages():
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout = sys.__stdout__

def z_to_p(z, high_precision=False):
    if high_precision:
        if isinstance(z, pd.Series): z = z.values
        from rpy2.robjects import numpy2ri, r
        p = 2 * np.exp(np.float128(r.pnorm(numpy2ri.py2rpy(np.abs(z)),
                                           lower_tail=False, log_p=True)))
    else:
        from scipy.special import ndtr
        p = 2 * ndtr(-np.abs(z))  # = 2 * norm.sf(np.abs(z))
    return p

def t_to_p(t_score, df):
    from scipy.special import stdtr
    p = 2 * stdtr(df, -np.abs(t_score))  # = 2 * t.sf(np.abs(t_score), df=df)
    return p

def p_to_abs_z(p):
    from scipy.special import ndtri
    abs_z = -ndtri(p / 2)  # = norm.isf(p / 2)
    return abs_z

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # Equivalent to itertools.pairwise(), which is available in Python >= 3.10
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def ld_clump(sumstats, clump_prefix, bfile, sumstats_file=None,
             clump_p1=5e-8, clump_p2=0.001, clump_r2=0.01, clump_kb=5000,
             SNP_col='SNP', chrom_col='CHROM', bp_col='BP', p_col='P'):
    # https://www.cog-genomics.org/plink/1.9/postproc#clump
    # https://zzz.bwh.harvard.edu/plink/clump.shtml
    # explodecomputer.github.io/EEPE_2016/worksheets_win/bioinformatics.html
    #
    # If sumstats_file is not None, you take responsibility for ensuring that
    # sumstats_file is identical to sumstats (except for being a file rather
    # than in memory)
    #
    # bfile can be a string (assumed to be a single bfile for all chromosomes)
    # or a list/tuple of bfiles (assumed to be one per chromosome)
    #
    assert SNP_col in sumstats.columns or sumstats.index.name == SNP_col, \
        f'{SNP_col} not in sumstats'
    assert chrom_col in sumstats, f'{chrom_col} not in sumstats'
    assert bp_col in sumstats, f'{bp_col} not in sumstats'
    assert p_col in sumstats, f'{p_col} not in sumstats'
    assert sumstats[p_col].min() <= clump_p1
    if sumstats.index.name != SNP_col:
        sumstats = sumstats.set_index(SNP_col)
    # assert not sumstats.index.duplicated().any()
    #
    # a) Run the clumping. The default settings are:
    # --clump-p1 5e-8: start with genome-wide-significant SNPs as lead SNPs
    # --clump-p2 0.001: clumps will include all SNPs with p < 0.001...
    # --clump-r2 0.01: ...r2 > 0.01 with the lead SNP...
    # --clump-kb 5000: ...and within 5 MB of the lead SNP
    # Note: while we specify SNP_col with --clump-snp-field and p_col with
    # --clump-field, there doesn't seem to be a way to specify chrom_col or
    # bp_col to plink's --clump. However, it does seem to be able to auto-detect
    # both CHR and CHROM.
    # Note: LD clumping only uses 1 thread in plink 1.9, so no need to specify
    # --threads
    #
    if not os.path.exists(f'{clump_prefix}.clumped'):
        make_temp_sumstats_file = sumstats_file is None
        if make_temp_sumstats_file:
            sumstats_file = 'sumstats.temp'
            sumstats.to_csv(sumstats_file, sep='\t')
        if isinstance(bfile, str):
            run(f'plink '
                f'--seed 0 '
                f'--bfile {bfile} '
                f'--clump {sumstats_file} '
                f'--clump-p1 {clump_p1} '
                f'--clump-p2 {clump_p2} '
                f'--clump-r2 {clump_r2} '
                f'--clump-kb {clump_kb} '
                f'--clump-snp-field {SNP_col} '
                f'--clump-field {p_col} '
                f'--out {clump_prefix}')
        else:
            # Run on each chromosome and concatenate the results
            assert isinstance(bfile, (list, tuple))
            for bfile_index, chrom_bfile in enumerate(bfile, start=1):
                run(f'plink '
                    f'--seed 0 '
                    f'--bfile {chrom_bfile} '
                    f'--clump {sumstats_file} '
                    f'--clump-p1 {clump_p1} '
                    f'--clump-p2 {clump_p2} '
                    f'--clump-r2 {clump_r2} '
                    f'--clump-kb {clump_kb} '
                    f'--clump-snp-field {SNP_col} '
                    f'--clump-field {p_col} '
                    f'--out {clump_prefix}_{bfile_index}')
            if len(bfile) == 1:
                run(f'cat {clump_prefix}_1.clumped > {clump_prefix}.clumped')
            else:
                run(f'(cat {clump_prefix}_1.clumped; tail -qn+2 {clump_prefix}_'
                    f'{{2..{len(bfile)}}}.clumped) > {clump_prefix}.clumped')
            run(f'rm {clump_prefix}_{{1..{len(bfile)}}}.clumped')
        if make_temp_sumstats_file:
            run('rm sumstats.temp')
    #
    # b) Load clumping results; map each clumped variant to its lead variant;
    #   join with sumstats
    #
    clumped_variants = pd.read_table(
        f'{clump_prefix}.clumped', usecols=['SNP', 'SP2'], index_col='SNP',
        delim_whitespace=True).squeeze(axis=1).str.split(',').explode()
    clumped_variants = pd.concat([
        clumped_variants.where(lambda x: x != 'NONE')
            .dropna().str.split('(').str[0],
        # add a mapping from each lead variant to itself
        pd.Series(clumped_variants.index.unique(),
                  index=clumped_variants.index.unique())])\
        .reset_index().set_index(0).squeeze(axis=1).rename('lead_variant')\
        .rename_axis('clumped_variant')
    assert not clumped_variants.index.duplicated().any()
    clumped_variants = clumped_variants.to_frame().join(sumstats)
    #
    # c) Merge overlapping clumps
    # (Note: sort_values(p_col) ensures that agg 'first' gives the lowest-p SNP)
    #
    clump_extent = clumped_variants\
        .sort_values(p_col)\
        .groupby('lead_variant')\
        .agg({chrom_col: 'first', bp_col: ('min', 'max'), p_col: 'min'})\
        .set_axis(['chrom', 'start', 'end', 'p'], axis=1)\
        .sort_values(['chrom', 'start'])\
        .assign(group=lambda df: (df.chrom.shift().ne(df.chrom) |
                                  df.end.shift().lt(df.start)).cumsum())\
        .sort_values('p')
    lead_variant_to_new_lead_variant = clump_extent\
        .assign(lead_variant=lambda df: df.index)\
        .groupby('group')\
        ['lead_variant']\
        .transform('first')
    clump_extent = clump_extent\
        .reset_index()\
        .groupby('group', sort=False)\
        .agg({'chrom': 'first', 'start': 'min', 'end': 'max',
              'p': 'first', 'lead_variant': 'first'})\
        .set_index('lead_variant')
    clumped_variants['lead_variant'] = clumped_variants['lead_variant']\
        .replace(lead_variant_to_new_lead_variant.to_dict())
    #
    # d) Make a nice table of lead variants with the bp extent of each clump
    #
    lead_variants = clumped_variants\
        .loc[clump_extent.index]\
        .drop('lead_variant', axis=1)\
        .assign(LOCUS=lambda df: 'chr' + clump_extent.chrom.astype(str) + ':' +
                                 clump_extent.start.map('{:,}'.format) + '-' +
                                 clump_extent.end.map('{:,}'.format))
    assert lead_variants[p_col].min() <= clump_p1
    return lead_variants, clumped_variants

def manhattan_plot(savefig, sumstats, clumped_variants=None, lead_variants=None,
                   SNP_col='SNP', chrom_col='CHROM', bp_col='BP', p_col='P',
                   gene_col=None, p_thresholds={5e-8: ('#D62728', 'dashed')},
                   rasterize_clumps=False):
    # Inspired by ncbi.nlm.nih.gov/pmc/articles/PMC6481311/figure/F1
    import matplotlib.pyplot as plt, seaborn as sns
    from adjustText import adjust_text
    use_font('Myriad Pro')
    assert SNP_col in sumstats.columns or sumstats.index.name == SNP_col, \
        f'{SNP_col} not in sumstats'
    assert chrom_col in sumstats, f'{chrom_col} not in sumstats'
    assert bp_col in sumstats, f'{bp_col} not in sumstats'
    assert p_col in sumstats, f'{p_col} not in sumstats'
    if gene_col is not None:
        assert lead_variants is not None
        assert clumped_variants is not None
        assert gene_col in lead_variants, f'{gene_col} not in lead_variants'
    plt.figure(constrained_layout=True)
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(1.2 * w, h)
    # Subset sumstats to p < 1e-3
    sumstats = sumstats.query(f'{p_col} < 1e-3')
    # Convert chromosome numbers to strings, but strip the chr prefix if present
    if sumstats[chrom_col].dtype != str:
        sumstats = sumstats.astype({chrom_col: str})
    elif sumstats[chrom_col].iloc[0].startswith('chr'):
        sumstats[chrom_col] = sumstats[chrom_col].str[3:]
    # Alternate each chromosome in a different color (dark blue then light blue)
    sumstats['color'] = (sumstats[chrom_col].astype('category').cat.codes % 2)\
        .replace({0: '#0A6FA5', 1: '#008FCD'})
    # Get the total number of bps to the start and end of each chromosome
    cumulative_bp = sumstats.groupby(chrom_col, sort=False)[bp_col].max()\
        .cumsum().to_frame(name='end')\
        .assign(start=lambda df: df.end.shift().fillna(0).astype(int))
    # Get x and y coordinates to plot
    sumstats = sumstats.assign(x=lambda df: cumulative_bp.start.reindex(
        sumstats[chrom_col]).values + sumstats[bp_col],
                               y=lambda df: -np.log10(df[p_col]))
    with sns.plotting_context('notebook', font_scale=1.2):
        # Overplot three times: first non-clumped variants, then clumped
        # variants, then lead variants. Rasterize non-clumped and (optionally)
        # clumped variants for quick plotting and rendering.
        if clumped_variants is not None:
            non_clump_sumstats = sumstats.query(
                f'{SNP_col} not in @clumped_variants.index')
            clump_sumstats = sumstats.query(
                f'{SNP_col} in @clumped_variants.index and '
                f'{SNP_col} not in @clumped_variants.lead_variant')
            lead_sumstats = sumstats.query(
                f'{SNP_col} in @clumped_variants.lead_variant')
            plt.scatter(non_clump_sumstats.x, non_clump_sumstats.y,
                        c=non_clump_sumstats.color, s=1, rasterized=True)
            plt.scatter(clump_sumstats.x, clump_sumstats.y,
                        rasterized=rasterize_clumps, c='#DBA756', s=1)
            plt.scatter(lead_sumstats.x, lead_sumstats.y, c='#DBA756',
                        marker='D', edgecolors='k', s=10)
            if gene_col is not None:
                lead_variants = lead_variants.query('GENE.notna()')
                texts = [
                    plt.text(x=x, y=y, s=gene, fontsize='small')
                    for gene, x, y in zip(lead_variants.GENE,
                                          lead_sumstats.x[lead_variants.index],
                                          lead_sumstats.y[lead_variants.index])]
                adjust_text(texts, x=lead_sumstats.x.values,
                            y=lead_sumstats.y.values,
                            expand_points=(1.4, 1.2),
                            arrowprops=dict(arrowstyle="-", color='black',
                                            lw=0.5))
        else:
            plt.scatter(sumstats.x, sumstats.y, c=sumstats.color, s=1,
                        rasterized=True)
        # Also plot horizontal lines indicating significance thresholds
        for p_threshold, (color, linestyle) in p_thresholds.items():
            plt.axhline(y=-np.log10(p_threshold), color=color,
                        linestyle=linestyle, zorder=-1)
        # Put each chrom's label in the chrom's center; hide chr17/19/21 labels
        plt.xticks((cumulative_bp.start + cumulative_bp.end) / 2,
                   cumulative_bp.index.to_series().replace({
                       cumulative_bp.index[16]: '', cumulative_bp.index[18]: '',
                       cumulative_bp.index[20]: '', 23: 'X', '23': 'X'}).values)
        # Set x and y labels and limits, resize, despine and save
        plt.xlabel('Chromosome')
        plt.ylabel('-log$_{10}$(p)')
        padding = 10_000_000  # avoid clipping SNPs at the far left or right
        plt.xlim(sumstats.x.min() - padding, sumstats.x.max() + padding)
        plt.ylim(bottom=3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(savefig, dpi=600)
        plt.close()

def qqplot(ps, **kwargs):
    # Inspired by qqplot() from the qmplot package
    import matplotlib.pyplot as plt
    ax = kwargs.pop('ax') if 'ax' in kwargs else plt.gca()
    ppoints = lambda n, a=0.5: (np.arange(n) + 1 - a) / (n + 1 - 2 * a)
    ax.scatter(-np.log10(ppoints(len(ps))), -np.log10(np.sort(ps).clip(5e-324)),
               edgecolors='none', rasterized=True, **kwargs)
    ax.plot([0, (xmax := ax.get_xlim()[1])], [0, xmax], c='k', zorder=-1)
    ax.set_aspect('equal')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def harmonize_alleles(sumstats, alleles, beta_column=None, OR_column=None,
                      Z_column=None, MAF_column=None):
    # Harmonize sumstats (which must have A1 and A2 columns and SNPs as index)
    # with alleles (which must have ONLY A1 and A2 columns and SNPs as index),
    # discarding ambiguous variants and allele-flipping as appropriate
    assert 'A1' in sumstats.columns and 'A2' in sumstats.columns, \
        sumstats.columns
    assert alleles.columns.equals(pd.Index(['A1', 'A2'])), alleles.columns
    assert alleles.index.name == 'SNP', alleles.index.name
    assert sumstats.index.name == 'SNP', alleles.index.name
    assert alleles.index[0].startswith('rs'), alleles.index[0]
    assert sumstats.index[0].startswith('rs'), sumstats.index[0]
    assert not (beta_column is None and OR_column is None and Z_column is None)
    if beta_column is not None:
        assert beta_column in sumstats.columns, sumstats.columns
    if OR_column is not None:
        assert OR_column in sumstats.columns, sumstats.columns
    if Z_column is not None:
        assert Z_column in sumstats.columns, sumstats.columns
    if MAF_column is not None:
        assert MAF_column in sumstats.columns, sumstats.columns
    assert sumstats.A1.str.fullmatch('[ACGT]*').all(), \
        sumstats.query('not A1.str.fullmatch("[ACGT]*")')
    assert sumstats.A2.str.fullmatch('[ACGT]*').all(), \
        sumstats.query('not A2.str.fullmatch("[ACGT]*")')
    assert alleles.A1.str.fullmatch('[ACGT]*').all(), \
        alleles.query('not A1.str.fullmatch("[ACGT]*")')
    assert alleles.A2.str.fullmatch('[ACGT]*').all(), \
        alleles.query('not A2.str.fullmatch("[ACGT]*")')
    # Add reverse complements
    reverse_complement = lambda series: series.str[::-1].str.translate(
        str.maketrans({'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}))
    alleles = alleles.assign(A1_RC=lambda df: reverse_complement(df.A1),
                             A2_RC=lambda df: reverse_complement(df.A2))
    # Remove ambiguous variants
    alleles = alleles.query('A1 != A2_RC')  # or equivalently, A2 != A1_RC
    # Keep track of the original alleles
    alleles = alleles.assign(A1_ORIG=lambda df: df.A1,
                             A2_ORIG=lambda df: df.A2)
    # List all A1/A2 combinations we'll accept matches to: original, A1/A2 flip,
    # reverse complement, both A1/A2 flip and reverse complement
    alleles = pd.concat([
        alleles[['A1', 'A2', 'A1_ORIG', 'A2_ORIG']]
            .assign(A1_A2_flip=False),
        alleles[['A1', 'A2', 'A1_ORIG', 'A2_ORIG']]
            .rename(columns={'A1': 'A2', 'A2': 'A1'})
            .assign(A1_A2_flip=True),
        alleles[['A1_RC', 'A2_RC', 'A1_ORIG', 'A2_ORIG']]
            .rename(columns={'A1_RC': 'A1', 'A2_RC': 'A2'})
            .assign(A1_A2_flip=False),
        alleles[['A2_RC', 'A1_RC', 'A1_ORIG', 'A2_ORIG']]
            .rename(columns={'A1_RC': 'A2', 'A2_RC': 'A1'})
            .assign(A1_A2_flip=True)])
    # Merge sumstats with alleles on rs, ref, alt
    sumstats = sumstats.merge(alleles, on=['SNP', 'A1', 'A2'], how='inner')
    # Assign A1 and A2 to be the harmonized alleles
    sumstats = sumstats.assign(A1=lambda df: df.A1_ORIG,
                               A2=lambda df: df.A2_ORIG)\
        .drop(['A1_ORIG', 'A2_ORIG'], axis=1)
    # Sign-flip betas, ORs, Zs and MAFs
    with pd.option_context('mode.chained_assignment', None):
        if beta_column is not None:
            sumstats[beta_column][sumstats.A1_A2_flip] = \
                -sumstats[beta_column][sumstats.A1_A2_flip]
        if OR_column is not None:
            sumstats[OR_column][sumstats.A1_A2_flip] = \
                1 / sumstats[OR_column][sumstats.A1_A2_flip]
        if Z_column is not None:
            sumstats[Z_column][sumstats.A1_A2_flip] = \
                -sumstats[Z_column][sumstats.A1_A2_flip]
        if MAF_column is not None:
            sumstats[MAF_column][sumstats.A1_A2_flip] = \
                1 - sumstats[MAF_column][sumstats.A1_A2_flip]
    # Remove unnecessary A1_A2_flip column
    sumstats = sumstats.drop('A1_A2_flip', axis=1)
    return sumstats

def harmonize_sumstats(sumstats_dict, beta_column=None, OR_column=None,
                       Z_column=None, MAF_column=None):
    # Harmonize a dict of sumstats (which must have A1 and A2 columns and SNPs
    # as index) with each other (i.e. with the alleles of the first one in the
    # list), discarding ambiguous variants and allele-flipping as appropriate
    assert isinstance(sumstats_dict, dict)
    assert all('A1' in sumstats.columns and 'A2' in sumstats.columns
               for sumstats in sumstats_dict.values())
    assert all(sumstats.index[0].startswith('rs')
               for sumstats in sumstats_dict.values())
    assert not (beta_column is None and OR_column is None and Z_column is None)
    for column in beta_column, OR_column, Z_column, MAF_column:
        if column is not None:
            assert all(column in sumstats.columns
                       for sumstats in sumstats_dict.values()), column
    # Harmonize the second through last sumstats with the first sumstats
    alleles = next(iter(sumstats_dict.values()))[['A1', 'A2']]
    harmonized_sumstats = {trait: sumstats if index == 0 else harmonize_alleles(
        sumstats, alleles, beta_column=beta_column, OR_column=OR_column,
        Z_column=Z_column, MAF_column=MAF_column)
        for index, (trait, sumstats) in enumerate(sumstats_dict.items())}
    # Subset to (rs, A1, A2) combos found in all of the harmonized sumstats
    harmonized_sumstats = {trait: sumstats.set_index(['A1', 'A2'], append=True)
                           for trait, sumstats in harmonized_sumstats.items()}
    variants_in_common = reduce(pd.Index.intersection, (
        sumstats.index for sumstats in harmonized_sumstats.values()))
    harmonized_sumstats = {
        trait: sumstats.loc[variants_in_common].reset_index(level=['A1', 'A2'])
               [sumstats_dict[trait].columns]
        for trait, sumstats in harmonized_sumstats.items()}
    assert all(len(sumstats) == len(next(iter(harmonized_sumstats.values())))
               for sumstats in harmonized_sumstats.values())
    return harmonized_sumstats

def load_GO_terms():
    # Load mapping of GO term (e.g. GO:0000001) to name
    # (e.g. mitochondrion inheritance) and namespace
    # (biological_process, molecular_function, cellular_component)
    GO_term_cache = 'GO_terms/GO_terms.tsv.gz'
    if os.path.exists(GO_term_cache):
        return pd.read_table(GO_term_cache, index_col='GO_term').squeeze()
    import networkx as nx, obonet
    from functools import lru_cache
    from io import StringIO
    os.makedirs('GO_terms', exist_ok=True)
    obo_file = 'GO_terms/go-basic.obo'
    if not os.path.exists(obo_file):
        run('wget http://geneontology.org/ontology/go-basic.obo -P GO_terms')
    condensed_GO_to_name_file = 'GO_terms/go-condensed.txt'
    if not os.path.exists(condensed_GO_to_name_file):
        run(f"grep '\[Term\]' -A3 {obo_file} | grep -v '\[Term\]' | "
            f"grep -v '^\-\-' > {condensed_GO_to_name_file}")
    GO_to_name = pd.read_table(condensed_GO_to_name_file,
                               header=None).squeeze()
    split = GO_to_name.str.split(': ')
    GO_to_name = pd.Series(split.str.get(1).values, index=split.str.get(0))
    GO_to_name = pd.DataFrame(GO_to_name.values.reshape(-1, 3),
                            columns=['id', 'name', 'namespace'])
    GO_to_name.set_index('id', inplace=True)
    GO_to_name = GO_to_name.namespace.str.replace('_', ' ') + ':' + \
                 GO_to_name.name
    # Load gene-GO term pairs
    # Drop a tiny number of entries where the gene is missing
    GO_file = 'GO_terms/goa_human.gaf.gz'
    if not os.path.exists(GO_file):
        run('wget http://geneontology.org/gene-associations/goa_human.gaf.gz '
            '-P GO_terms')
    GO_terms = pd.read_table(GO_file, usecols=[2, 4], names=['gene', 'GO_term'],
                             comment='!', header=None).dropna()
    # Some GO terms are present in GO_terms.GO_term under alt_ids. Map these to
    # their canonical id.
    obo_file_contents = open(obo_file).read()
    alt_id_to_id_map = {}
    for line in obo_file_contents.splitlines():
        if line.startswith('id: '):
            current_id = line.removeprefix('id: ')
        elif line.startswith('alt_id: '):
            # noinspection PyUnboundLocalVariable
            alt_id_to_id_map[line.removeprefix('alt_id: ')] = current_id
    GO_terms['GO_term'] = GO_terms.GO_term.map(
        lambda x: alt_id_to_id_map.get(x, x))  # much faster than replace()
    # Remove obsolete GO terms (this is an extremely rare edge case: only 1
    # entry was obsolete the last time I checked, GO:0006286 for POLB)
    obsolete_terms = set()
    for line in obo_file_contents.splitlines():
        if line.startswith('id: '):
            current_id = line.removeprefix('id: ')
        elif line == 'is_obsolete: true':
            # noinspection PyUnboundLocalVariable
            obsolete_terms.add(current_id)
    GO_terms = GO_terms.query('GO_term not in @obsolete_terms')
    # If a gene has a GO term, also make it part of all parents of the GO term.
    # part_of, regulates, positively_regulates and negatively_regulates are all
    # indicators of parent-child relationships (just like is_a), but are not
    # handled by obonetread_obo(), so we need to handle them ourselves.
    obo_file_contents = re.sub(
        'relationship: (part_of|regulates|positively_regulates|'
        'negatively_regulates)', 'is_a:', obo_file_contents)
    obo_graph = obonet.read_obo(StringIO(obo_file_contents))
    get_parents = lru_cache(maxsize=None)(
        lambda GO_term: [GO_term] + list(nx.descendants(obo_graph, GO_term)))
    GO_terms = GO_terms.assign(GO_term=lambda df: df.GO_term.map(get_parents))\
        .set_index('gene').GO_term.explode().reset_index()
    # Convert GO term IDs to names
    GO_terms['GO_term'] = GO_terms['GO_term'].map(GO_to_name)
    # Drop duplicates
    GO_terms.drop_duplicates(inplace=True)
    # Sort by GO term, then gene
    GO_terms.sort_values(['GO_term', 'gene'], inplace=True)
    # Set GO term as index
    GO_terms.set_index('GO_term', inplace=True)
    GO_terms = GO_terms.gene
    GO_terms.to_csv(GO_term_cache, sep='\t', compression='gzip')
    return GO_terms

def get_n_clusters(affinity_matrix, max_n_clusters=10):
    # See docs.scipy.org/doc/scipy/reference/tutorial/arpack.html and
    # github.com/mingmingyang/auto_spectral_clustering/blob/master/autosp.py
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    lap = laplacian(affinity_matrix, normed=True)
    eigenvalues, eigenvectors = eigsh(-lap, k=max_n_clusters + 1, sigma=1)
    eigenvalues = -eigenvalues[::-1]
    eigengap = np.diff(eigenvalues)
    n_clusters_ranking = 2 + np.argsort(-eigengap[1:])
    best, second_best = n_clusters_ranking[:2]
    return best, second_best

def spectral_cluster(features, n_neighbors=20, max_n_clusters=10,
                     n_clusters=None):
    from sklearn.cluster import spectral_clustering
    from sklearn.neighbors import kneighbors_graph
    features = standardize(features)
    affinity_matrix = kneighbors_graph(features, n_neighbors=n_neighbors,
                                       include_self=True)
    affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
    if n_clusters is None:
        best, second_best = get_n_clusters(affinity_matrix,
                                           max_n_clusters=max_n_clusters)
        n_clusters = best
    cluster_labels = pd.Series(spectral_clustering(
        affinity_matrix, n_clusters=n_clusters, random_state=0),
        index=features.index)
    return cluster_labels

def liftOver(infile, outfile, input_assembly='hg38', output_assembly='hg19',
             log_file='/dev/null', bash_input=False):
    os.makedirs('gene_annotations', exist_ok=True)
    # Download from hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver
    chain_file = f'gene_annotations/{input_assembly}To' \
                 f'{output_assembly.capitalize()}.over.chain.gz'
    if not os.path.exists(chain_file):
        run(f'rsync -za rsync://hgdownload.cse.ucsc.edu/goldenPath/'
            f'{input_assembly}/liftOver/{chain_file} .')
    infile = f"<({infile})" if bash_input else f"\"{infile}\""
    outfile = f'"{outfile}"'
    run(f'liftOver -bedPlus=3 {infile} {chain_file} {outfile} {log_file}')

def get_column_indices_1_based(filename, sep='\t', comment=None,
                               delim_whitespace=False, skip_blank_lines=False):
    # Map each column to its 1-based numeric index, suitable for passing to awk.
    # Use bash operations + BytesIO, so that filename can be a bash command.
    from io import BytesIO
    if comment is not None or skip_blank_lines is not None:
        grep_flags = f'-v "^{comment}"' if skip_blank_lines is None else \
            f'-v "^$"' if comment is None else f'-ve "^{comment}" -e "^$"'
        header_command = f'{"z" if filename.endswith(".gz") else ""}grep ' \
                         f'{grep_flags} {filename} | head -1'
    elif filename.endswith('.gz'):
        header_command = f'zcat {filename} | head -1'
    else:
        header_command = f'head -1 {filename}'
    columns = pd.read_table(BytesIO(run(header_command, capture_output=True,
                                        pipefail=False).stdout),
                            **({'delim_whitespace': delim_whitespace}
                               if delim_whitespace else {'sep': sep})).columns
    return pd.Series(range(1, len(columns) + 1), index=columns)

def munge_sumstats(raw_sumstats_files, munged_sumstats_file, SNP, A1, A2, P,
                   CHROM=None, BP=None, BETA=None, OR=None, SE=None, N=None,
                   N_CASES=None, N_CONTROLS=None, NEFF=None, A1FREQ=None,
                   A1FREQ_CASES=None, A1FREQ_CONTROLS=None, INFO=None,
                   quantitative=False, preamble='', sep='\t', comment=None,
                   delim_whitespace=False, skip_blank_lines=False,
                   skip_missing_SNP_or_P=True, verbose=True):
    # Check inputs
    if verbose:
        print(f'{raw_sumstats_files=}; {munged_sumstats_file=}; {SNP=}; '
              f'{A1=}; {A2=}; {P=}; {CHROM=}; {BP=}; {BETA=}; {OR=}; {SE=}; '
              f'{N=}; {N_CASES=}; {N_CONTROLS=}; {NEFF=}; {A1FREQ=}; '
              f'{A1FREQ_CASES=}; {A1FREQ_CONTROLS=}; {INFO=}; {quantitative=}; '
              f'{preamble=}; {sep=}; {comment=}; {delim_whitespace=}; ' 
              f'{skip_blank_lines=}; {skip_missing_SNP_or_P=}; {verbose=}')
    assert SNP is not None
    assert A1 is not None
    assert A2 is not None
    assert P is not None
    assert BETA is not None or OR is not None  # must have either BETA or OR
    assert BETA is None or OR is None  # can't have both BETA and OR
    if quantitative: assert N is not None
    if quantitative: assert N_CASES is None and N_CONTROLS is None
    if not quantitative: assert NEFF is not None or A1FREQ is not None or (
        N_CASES is not None and N_CONTROLS is not None)  # need for NEFF calc.
    if preamble:
        if preamble.endswith(';'):
            preamble += ' '
        elif not preamble.endswith('; '):
            preamble += '; '
    # If SE is missing, back-calculate it from BETA and P: |Z| = |BETA| / SE,
    # so SE = |BETA| / |Z|. We can get |Z| from P via p_to_abs_z(). Skip SNPs
    # with P = 1 to avoid division by 0 errors.
    if SE is None:
        SE = f'abs({BETA if BETA is not None else f"log({OR})"}) / ' \
             f'p_to_abs_z({P})'
        preamble += f'if (({P}) == 1) {{next}}; '
        # Translated from scipy's C implementation of ndtri:
        # github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c
        # github.com/scipy/scipy/blob/main/scipy/special/cephes/polevl.h
        # docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html
        # See reference/ndtri_awk.txt for the unminified, commented version.
        # Need to split up functions and constants because constants have to be
        # defined inside a block like BEGIN, while functions can't be.
        functions = (
            'function polevl(x, coef, N) {ans = coef[1]; i = 2; L = N + 1; do {'
            'ans = ans * x + coef[i]} while(i++ < L); return ans}; function '
            'p1evl(x, coef, N) {ans = x + coef[1]; i = 2; do {ans = ans * x + '
            'coef[i]} while(i++ < N); return ans}; function ndtri(y) {if (y == '
            '0) return log(0); if (y == 1) return -log(0); if (y < 0 || y > 1) '
            'return "nan"; code = 1; if (y > 1.0 - 0.13533528323661269189) {y '
            '= 1.0 - y; code = 0}; if (y > 0.13533528323661269189) {y = y - '
            '0.5; y2 = y * y; x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, '
            'Q0, 8)); x = x * s2pi; return x}; x = sqrt(-2.0 * log(y)); x0 = '
            'x - log(x) / x; z = 1.0 / x; if (x < 8) x1 = z * polevl(z, P1, 8)'
            ' / p1evl(z, Q1, 8); else x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, '
            '8); x = x0 - x1; if (code != 0) x = -x; return x}; function '
            'p_to_abs_z(p) {return -ndtri(p / 2)}; function abs(x) {return '
            'x > 0 ? x : -x}')
        constants = (
            's2pi = 2.50662827463100050242; P0[1] = -59.9633501014107895267; '
            'P0[2] = 98.0010754185999661536; P0[3] = -56.6762857469070293439; '
            'P0[4] = 13.9312609387279679503; P0[5] = -1.23916583867381258016; '
            'Q0[1] = 1.95448858338141759834; Q0[2] = 4.67627912898881538453; '
            'Q0[3] = 86.3602421390890590575; Q0[4] = -225.462687854119370527; '
            'Q0[5] = 200.260212380060660359; Q0[6] = -82.0372256168333339912; '
            'Q0[7] = 15.9056225126211695515; Q0[8] = -1.18331621121330003142; '
            'P1[1] = 4.05544892305962419923; P1[2] = 31.5251094599893866154; '
            'P1[3] = 57.1628192246421288162; P1[4] = 44.0805073893200834700; '
            'P1[5] = 14.6849561928858024014; P1[6] = 2.18663306850790267539; '
            'P1[7] = -0.140256079171354495875; '
            'P1[8] = -0.0350424626827848203418; '
            'P1[9] = -8.57456785154685413611E-4; '
            'Q1[1] = 15.7799883256466749731; Q1[2] = 45.3907635128879210584; '
            'Q1[3] = 41.3172038254672030440; Q1[4] = 15.0425385692907503408; '
            'Q1[5] = 2.50464946208309415979; Q1[6] = -0.142182922854787788574; '
            'Q1[7] = -0.0380806407691578277194; '
            'Q1[8] = -9.33259480895457427372E-4; '
            'P2[1] = 3.23774891776946035970; P2[2] = 6.91522889068984211695; '
            'P2[3] = 3.93881025292474443415; P2[4] = 1.33303460815807542389; '
            'P2[5] = 0.201485389549179081538; '
            'P2[6] = 0.0123716634817820021358; '
            'P2[7] = 3.01581553508235416007E-4; '
            'P2[8] = 2.65806974686737550832E-6; '
            'P2[9] = 6.23974539184983293730E-9; '
            'Q2[1] = 6.02427039364742014255; Q2[2] = 3.67983563856160859403; '
            'Q2[3] = 1.37702099489081330271; Q2[4] = 0.216236993594496635890; '
            'Q2[5] = 0.0134204006088543189037; '
            'Q2[6] = 3.28014464682127739104E-4; '
            'Q2[7] = 2.89247864745380683936E-6; '
            'Q2[8] = 6.79019408009981274425E-9; ')
    else:
        functions = constants = ''
    # Try to infer N and NEFF if not specified
    #
    # For quantitative traits, NEFF = N. For binary traits, NEFF can be given in
    # two mathematically equivalent ways: 4/(1/N_CASES + 1/N_CONTROLS)
    # (cell.com/ajhg/pdfExtended/S0002-9297(21)00145-2) and 4v(1-v)N where v =
    # N_CASES/N (medrxiv.org/content/10.1101/2021.09.22.21263909v1.full-text).
    #
    # But this doesn't account for varying case-control ratios across the
    # cohorts in a meta-analysis, which biases estimates that depend on N
    # (medrxiv.org/content/10.1101/2021.09.22.21263909v1.full).
    #
    # Instead, biorxiv.org/content/10.1101/2021.03.29.437510v4.full suggests
    # a per-variant Neff = ((4 / (2 * MAF * (1 - MAF) * INFO)) - BETA^2) / SE^2,
    # bounded to between 0.5 and 1.1 times "the total (effective) sample size".
    # This seems to be max(4/(1/N_CASES + 1/N_CONTROLS)) across SNPs:
    # github.com/privefl/paper-misspec/blob/main/code/investigate-misspec-N.R
    # To get a global Neff, they take the 80th %ile of the per-variant Neffs.
    # This formula is based on Equation 1 of the "New formula used in LDpred2"
    # section of sciencedirect.com/science/article/abs/pii/S0002929721004201.
    #
    # github.com/GenomicSEM/GenomicSEM/wiki/2.1-Calculating-Sum-of-Effective-
    # Sample-Size-and-Preparing-GWAS-Summary-Statistics drops the BETA^2 and
    # INFO, i.e. Neff = 4 / (2 * MAF * (1 - MAF)) / SE^2, bounded to between 0.5
    # and 1.1 times 4/(1/N_CASES + 1/N_CONTROLS), where N_CASES and N_CONTROLS
    # are again global rather than per-SNP. This formula is based on an old
    # version of the above formula that was used in the original LDpred2 paper.
    #
    # Here, we use Neff = ((4 / (2 * A1FREQ * (1 - A1FREQ) * INFO)) - BETA^2) /
    # SE^2 (dropping the INFO part if not available), without bounding (because
    # honestly it seems like a hack). If MAF not available, but N_CASES and
    # N_CONTROLS are available, fall back to using 4/(1/N_CASES + 1/N_CONTROLS).
    # Note: even if A1FREQ = 1 - MAF instead of MAF, A1FREQ * (1 - A1FREQ) is
    # always equal to MAF * (1 - MAF). Note: "((4 / (2 * ..." can be simplified
    # to "((2 / (...".
    if N is None and N_CASES is not None and N_CONTROLS is not None:
        N = f'({N_CASES}) + ({N_CONTROLS})'
    if NEFF is None:
        if quantitative and N is not None:
            NEFF = N
        elif A1FREQ is not None:
            NEFF = f'((2 / (({A1FREQ}) * (1 - ({A1FREQ})' \
                   f'{f" * ({INFO})" if INFO is not None else ""}))) - ' \
                   f'{f"({BETA})" if BETA is not None else f"log({OR})"} ' \
                   f'^ 2) / ({SE}) ^ 2'
        elif N_CASES is not None and N_CONTROLS is not None:
            NEFF = f'4 / (1 / ({N_CASES}) + 1 / ({N_CONTROLS}))'
    # For case-control traits, output ORs even if input has betas, & vice versa
    if not quantitative and OR is None:
        OR = f'exp({BETA})'
    if quantitative and BETA is None:
        BETA = f'log({OR})'
    # Ensure A1 and A2 are capitalized
    A1 = f'toupper({A1})'
    A2 = f'toupper({A2})'
    # Get 1-based indices of each column (e.g. SNP -> $1, CHROM -> $2, ...).
    # If multiple sumstats files, require all to have the same columns.
    if isinstance(raw_sumstats_files, str):
        raw_sumstats_files = [raw_sumstats_files]
    column_indices = get_column_indices_1_based(
        raw_sumstats_files[0], sep=sep, comment=comment,
        skip_blank_lines=skip_blank_lines, delim_whitespace=delim_whitespace)
    assert len(column_indices) > 1, column_indices  # delimiter issue?
    assert all(column_indices.equals(get_column_indices_1_based(
        sumstats_file, sep=sep, comment=comment, skip_blank_lines=
        skip_blank_lines)) for sumstats_file in raw_sumstats_files[1:])
    # Enumerate columns to include, and the formula for each column
    column_formulas = pd.Series({
        'SNP': SNP, 'CHROM': CHROM, 'BP': BP, 'A1': A1, 'A2': A2,
        'A1FREQ': A1FREQ, 'A1FREQ_CASES': A1FREQ_CASES,
        'A1FREQ_CONTROLS': A1FREQ_CONTROLS, 'INFO': INFO,
        'BETA' if quantitative else 'OR': BETA if quantitative else OR,
        'SE': SE, 'P': P, 'N': N, 'N_CASES': N_CASES, 'N_CONTROLS': N_CONTROLS,
        'NEFF': NEFF})\
        .dropna()\
        .astype(str)  # so we can regex numbers (e.g. N_CASES=150000)
    # Substitute each column referenced in each formula (including the preamble)
    # with its column index.
    #
    # Don't verify that all referenced columns are present in the sumstats,
    # since unreferenced "columns" could be awk built-ins.
    #
    # To avoid e.g. "N_CASES" being replaced by "$10_CASES" when N is column 10,
    # use negative lookahead/lookbehind to only match column names that are not
    # immediately preceeded (?<!\w) or followed (?!\w) by a word character \w
    # (A-Z, a-z, 0-9, _, and alphanumeric characters in other languages).
    #
    # Make substitutions in descending order of column name length, in case one
    # column contains a word that's the name of another column, e.g. "SNP ID"
    # and "SNP" or "&BETA;" and "BETA".
    sorted_column_indices = column_indices\
        .to_frame(name='column_index')\
        .assign(length=lambda df: df.index.str.len())\
        .sort_values('length', ascending=False)\
        .column_index
    def substitute(formula):
        for column_name, column_index in sorted_column_indices.items():
            formula = re.sub(f'(?<!\w){re.escape(column_name)}(?!\w)',
                             f'${column_index}', formula)
        return formula
    column_formulas = column_formulas.map(substitute)
    if preamble:
        preamble = substitute(preamble)
    # Handle missing data: if any column in a formula is NA or "", output NA.
    # As an optimization, only check for "", not NA, when the formula is a
    # single, bare field (like $8), because if it's NA then the output will be
    # NA anyway. That's what the re.fullmatch part is doing. Also, if the
    # formula is just a constant (no "$"), keep it unchanged.
    #
    # Do NOT do the same to the preamble; the caller is responsible for ensuring
    # mssing data is handled correctly in the preamble.
    #
    # If skip_missing_SNP_or_P, add a condition to the preamble skipping lines
    # with the SNP or P column equal to NA or "". (These columns are more or
    # less the bare minimum of what a downstream method might need.) As an
    # optimization, don't include NA/"" checks in the SNP and P formulas.
    #
    # Note: this could be extended to handle other NA values; NA and "" are just
    # the most common ones (and the only ones included by default in tidyverse's
    # readr.tidyverse.org/reference/read_delim.html), but pandas.read_table()
    # has a lot more.
    if skip_missing_SNP_or_P:
        preamble += f'if ({column_formulas["SNP"]} == "NA" || ' \
                    f'{column_formulas["SNP"]} == "" || ' \
                    f'{column_formulas["P"]} == "NA" || ' \
                    f'{column_formulas["P"]} == "") {{next}}; '
    column_formulas = column_formulas.map(
        lambda formula: formula if '$' not in formula or (
                        skip_missing_SNP_or_P and (
                        formula == column_formulas['SNP'] or
                        formula == column_formulas['P'])) else
                        f'{formula} == "" ? "NA" : {formula}'
                        if re.fullmatch('\$\d+', formula) else
                        ' || '.join(f'{index} == "NA" || {index} == ""'
                                    for index in re.findall('\$\d+', formula)) +
                        f' ? "NA" : {formula}')
    # Prepare and run awk command to munge the sumstats
    # Use OFMT="%.12g" to reduce rounding: stackoverflow.com/questions/59070401
    # Do the skipping of blank lines (skip_blank_lines) and/or commented lines
    # (comment is not None) outside the main awk program, to ensure that
    # FNR == 1 for header lines.
    # FS = ' ' by default, and treats all whitespace as a delimiter.
    FS = ' ' if delim_whitespace else repr(sep).replace("'", '')
    set_options = f'FS="{FS}"; OFS="\\t"; OFMT="%.12g"'
    header = ', '.join('"' + column_formulas.index + '"')
    print_header = f'print {header}'
    print_row = f'print {", ".join(column_formulas)}'
    if comment is not None or skip_blank_lines:
        grep_flags = f'-v "^{comment}"' if not skip_blank_lines else \
            f'-v "^$"' if comment is None else f'-ve "^{comment}" -e "^$"'
        awk_inputs = ' '.join(
            f'<({"z" if sumstats_file.endswith(".gz") else ""}grep '
            f'{grep_flags} {sumstats_file})'
            for sumstats_file in raw_sumstats_files)
    else:
        awk_inputs = ' '.join(
            f'<(zcat {sumstats_file})'
            if sumstats_file.endswith('.gz') else sumstats_file
            for sumstats_file in raw_sumstats_files)
    awk_output = f'| gzip > {munged_sumstats_file}' \
        if munged_sumstats_file.endswith('.gz') else f'> {munged_sumstats_file}'
    awk_command = f'gawk \'{functions}BEGIN {{{constants}{set_options}; ' \
                  f'{print_header}}} FNR > 1 {{{preamble}{print_row}}}\' ' \
                  f'{awk_inputs} {awk_output}'
    if verbose:
        print(awk_command)
    # Run the 1st 10 rows and check that no columns are blank; if verbose, print
    first_ten_lines = run(awk_command.replace(awk_output, '| head'),
                         pipefail=False, capture_output=True).stdout.decode()
    assert '\t\t' not in first_ten_lines, first_ten_lines
    if verbose:
        print(first_ten_lines)
    # Run the full command
    run(awk_command)

def munge_regenie_sumstats(raw_sumstats_files, munged_sumstats_file,
                           quantitative=False):
    munge_sumstats(raw_sumstats_files, munged_sumstats_file, CHROM='CHROM',
                   BP='GENPOS', SNP='ID', A2='ALLELE0', A1='ALLELE1',
                   A1FREQ='A1FREQ',
                   A1FREQ_CASES=None if quantitative else 'A1FREQ_CASES',
                   A1FREQ_CONTROLS=None if quantitative else 'A1FREQ_CONTROLS',
                   INFO='INFO', N='N',
                   N_CASES=None if quantitative else 'N_CASES',
                   N_CONTROLS=None if quantitative else 'N_CONTROLS',
                   BETA='BETA', SE='SE', P='P', quantitative=quantitative,
                   preamble='P = 10^-LOG10P; if (CHROM == 23) {CHROM = "X"}',
                   sep=' ')

def get_sumstat_column_indices_1_based(sumstats_file: str, ldsc_flags: str) -> \
        tuple[dict[str, int], bool]:
    # Given a sumstats file and a string containing some flags for LDSC (like
    # '--snp ID --a1 ALT --a2 REF'), return a dict mapping SNP, NSTUDY, P, A1,
    # A2, N (or N_CAS and N_CON), SIGNED_SUMSTAT, INFO, and MAF to their
    # respective 1-based column indices within the sumstats file USING THE SAME
    # LOGIC AS LDSC, suitable for extracting these columns with awk.
    # Minor devation from LDSC: also support optional CHROM, BP and SE columns,
    # via --chrom and --bp.
    # Also return a bool specifying whether the SIGNED_SUMSTAT column has ORs.
    # Priority is:
    # (1) ignore everything in --ignore
    # (2) use everything in flags that is not in --ignore
    # (3) use everything in default that is not in --ignore or in flags
    # The entries of flags and default are cleaned with clean_header;
    # the entries of ignore are not
    default_column_names = {
        # This is from github.com/bulik/ldsc/blob/master/munge_sumstats.py#L31
        # except that I removed a duplicated entry ('N_CASE': 'N_CAS') and
        # made all the BETA, OR and Z aliases map to 'SIGNED_SUMSTAT' rather
        # than to 'BETA', 'OR' and 'Z'. I also added CHROM, BP and SE.
        # CHROM
        'CHROM': 'CHROM', 'CHR': 'CHROM', 'CHROMOSOME': 'CHROM',
        # BP
        'BP': 'BP', 'POS': 'BP', 'POSITION': 'BP', 'BASE_PAIR': 'BP',
        'BASE_PAIR_LOCATION': 'BP', 'BP_LOCATION': 'BP',
        # SE
        'SE': 'SE', 'SIGMA': 'SE', 'STDERR': 'SE', 'STANDARD_ERROR': 'SE',
        'STDERRLOGOR': 'SE',
        # RS NUMBER
        'SNP': 'SNP', 'MARKERNAME': 'SNP', 'SNPID': 'SNP', 'RS': 'SNP',
        'RSID': 'SNP', 'RS_NUMBER': 'SNP', 'RS_NUMBERS': 'SNP',
        # NSTUDY
        'NSTUDY': 'NSTUDY', 'N_STUDY': 'NSTUDY', 'NSTUDIES': 'NSTUDY',
        'N_STUDIES': 'NSTUDY',
        # P-VALUE
        'P': 'P', 'PVALUE': 'P', 'P_VALUE':  'P', 'PVAL': 'P', 'P_VAL': 'P',
        'GC_PVALUE': 'P',
        # ALLELE 1
        'A1': 'A1', 'ALLELE1': 'A1', 'ALLELE_1': 'A1', 'EFFECT_ALLELE': 'A1',
        'REFERENCE_ALLELE': 'A1', 'INC_ALLELE': 'A1', 'EA': 'A1',
        # ALLELE 2
        'A2': 'A2', 'ALLELE2': 'A2',  'ALLELE_2': 'A2', 'OTHER_ALLELE': 'A2',
        'NON_EFFECT_ALLELE': 'A2', 'DEC_ALLELE': 'A2', 'NEA': 'A2',
        # N
        'N': 'N', 'NCASE': 'N_CAS', 'CASES_N': 'N_CAS', 'N_CASE': 'N_CAS',
        'N_CASES': 'N_CAS', 'N_CONTROLS': 'N_CON', 'N_CAS': 'N_CAS',
        'N_CON': 'N_CON', 'NCONTROL': 'N_CON', 'CONTROLS_N': 'N_CON',
        'N_CONTROL': 'N_CON', 'WEIGHT': 'N',  # metal does this. possibly risky.
        # SIGNED STATISTICS
        'ZSCORE': 'SIGNED_SUMSTAT', 'Z-SCORE': 'SIGNED_SUMSTAT',
        'GC_ZSCORE': 'SIGNED_SUMSTAT', 'Z': 'SIGNED_SUMSTAT',
        'OR': 'SIGNED_SUMSTAT', 'B': 'SIGNED_SUMSTAT', 'BETA': 'SIGNED_SUMSTAT',
        'LOG_ODDS': 'SIGNED_SUMSTAT', 'EFFECTS': 'SIGNED_SUMSTAT',
        'EFFECT': 'SIGNED_SUMSTAT', 'SIGNED_SUMSTAT': 'SIGNED_SUMSTAT',
        # INFO
        'INFO': 'INFO',
        # MAF
        'EAF': 'FRQ', 'FRQ': 'FRQ', 'MAF': 'FRQ', 'FRQ_U': 'FRQ', 'F_U': 'FRQ'}
    sumstats_header = pd.read_table(sumstats_file, nrows=0,
                                    delim_whitespace=True).columns
    ldsc_flag_map = {flag: value.replace('"', '').replace("'", '')
                     for flag, value in
                     zip(ldsc_flags.split()[::2], ldsc_flags.split()[1::2])}
    ignore_set = set(ldsc_flag_map['--ignore'].split(',')) \
        if '--ignore' in ldsc_flag_map else set()
    clean_header = lambda column_name: column_name.upper().replace('-', '_')\
        .replace('.', '_').replace('\n', '')
    flag_column_names = {
        clean_header(ldsc_flag_map[flag]): column_type for flag, column_type in
        {'--chrom': 'CHROM', '--bp': 'BP', '--snp': 'SNP', '--N-col': 'N',
         '--N-cas-col': 'N_CAS', '--N-con-col': 'N_CON', '--a1': 'A1',
         '--a2': 'A2', '--p': 'P', '--frq': 'FRQ', '--info': 'INFO',
         '--nstudy': 'NSTUDY'}.items() if flag in ldsc_flag_map}
    if '--signed-sumstats' in ldsc_flag_map:
        signed_sumstats_cname, center = \
            ldsc_flag_map['--signed-sumstats'].split(',')
        assert center == '0' or center == '1', center
        is_odds_ratio = center == '1'
        flag_column_names[clean_header(signed_sumstats_cname)] = \
            'SIGNED_SUMSTAT'
    else:
        is_odds_ratio = 'OR' in sumstats_header
    column_indices = {}
    for column_index, column_name in enumerate(sumstats_header, start=1):
        if column_name in ignore_set: continue
        column_name = clean_header(column_name)
        if column_name in flag_column_names:
            column_indices[flag_column_names[column_name]] = column_index
        elif column_name in default_column_names:
            column_indices[default_column_names[column_name]] = column_index
    assert 'SNP' in column_indices
    assert 'P' in column_indices
    assert 'A1' in column_indices
    assert 'A2' in column_indices
    assert 'SIGNED_SUMSTAT' in column_indices
    return column_indices, is_odds_ratio

def get_minimal_representation(bp, ref, alt):
    # From github.com/ericminikel/minimal_representation/blob/master/
    # minimal_representation.py
    min_len = min(len(ref), len(alt))
    if min_len == 1:
        return bp, ref, alt
    else:
        # strip off identical suffixes
        while alt[-1] == ref[-1] and min_len > 1:
            alt = alt[:-1]
            ref = ref[:-1]
            min_len -= 1
        # strip off identical prefixes and increment position
        while alt[0] == ref[0] and min_len > 1:
            alt = alt[1:]
            ref = ref[1:]
            bp += 1
            min_len -= 1
        return bp, ref, alt

def get_minimal_representations_python(df, bp_col='BP', ref_col='REF',
                                       alt_col='ALT'):
    bp = df[bp_col].values
    ref = df[ref_col].values
    alt = df[alt_col].values
    for i in range(len(df)):
        bp[i], ref[i], alt[i] = \
            get_minimal_representation(bp[i], ref[i], alt[i])

def cythonize(cython_code, function_name=None):
    import pyximport
    pyximport.install(setup_args=dict(include_dirs=np.get_include()))
    from importlib import import_module
    from textwrap import dedent
    module_name = ''.join(chr(ord(c) + ord('a') - ord('0'))
                          for c in str(abs(hash(cython_code))))
    code_file = f'{module_name}.pyx'
    try:
        with open(code_file, 'w') as f:
            print(dedent(cython_code).lstrip('\n'), file=f)
        if function_name is not None:
            function = import_module(module_name, function_name)
        else:
            from inspect import getmembers, isbuiltin
            module = import_module(module_name)
            builtins = [member for member in getmembers(module, isbuiltin)
                        if member[0] != '__pyx_unpickle_Enum']
            assert len(builtins) == 1, builtins
            function = builtins[0][1]
        return function
    finally:
        try:
            os.unlink(code_file)
        except NameError:
            pass

def get_minimal_representations(df, bp_col='bp', ref_col='REF', alt_col='ALT'):
    bp_dtype = df[bp_col].dtype
    assert bp_dtype == 'int32' or bp_dtype == int, bp_dtype
    int_type = 'int' if bp_dtype == 'int32' else 'long'
    return cythonize(rf'''
    #cython: language_level=3
    cimport cython
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def get_minimal_representation({int_type}[:] bp_vector, str[:] ref_vector, 
                                   str[:] alt_vector):
        cdef size_t i, min_len, ref_len, alt_len, ref_start, ref_end, \
            alt_start, alt_end
        cdef {int_type} bp
        cdef str ref, alt
        for i in range(bp_vector.shape[0]):
            ref = ref_vector[i]
            alt = alt_vector[i]
            ref_len = len(ref)
            alt_len = len(alt)
            min_len = ref_len if ref_len < alt_len else alt_len
            if min_len == 1:
                continue
            bp = bp_vector[i]
            ref_start = 0
            alt_start = 0
            ref_end = ref_len - 1
            alt_end = alt_len - 1
            while ref[ref_end] == alt[alt_end]:
                ref_end -= 1
                alt_end -= 1
                min_len -= 1
                if min_len == 1: break
            else:
                while ref[ref_start] == alt[alt_start]:
                    ref_start += 1
                    alt_start += 1
                    bp += 1
                    min_len -= 1
                    if min_len == 1: break
            bp_vector[i] = bp
            ref_vector[i] = ref[ref_start:ref_end + 1]
            alt_vector[i] = alt[alt_start:alt_end + 1]
    ''')(bp_vector=df[bp_col].values, ref_vector=df[ref_col].values,
         alt_vector=df[alt_col].values)

@cache
def get_dbSNP_chromosome_IDs(assembly='hg38'):
    if assembly == 'hg38':
        return {
            'chr1': 'NC_000001.11', 'chr2': 'NC_000002.12',
            'chr3': 'NC_000003.12', 'chr4': 'NC_000004.12',
            'chr5': 'NC_000005.10', 'chr6': 'NC_000006.12',
            'chr7': 'NC_000007.14', 'chr8': 'NC_000008.11',
            'chr9': 'NC_000009.12', 'chr10': 'NC_000010.11',
            'chr11': 'NC_000011.10', 'chr12': 'NC_000012.12',
            'chr13': 'NC_000013.11', 'chr14': 'NC_000014.9',
            'chr15': 'NC_000015.10', 'chr16': 'NC_000016.10',
            'chr17': 'NC_000017.11', 'chr18': 'NC_000018.10',
            'chr19': 'NC_000019.10', 'chr20': 'NC_000020.11',
            'chr21': 'NC_000021.9', 'chr22': 'NC_000022.11',
            'chr23': 'NC_000023.11', 'chrX': 'NC_000023.11',
            'chr24': 'NC_000024.10', 'chrY': 'NC_000024.10',
            'chrM': 'NC_012920.1'}
    elif assembly == 'hg19':
        return {
            'chr1': 'NC_000001.10', 'chr2': 'NC_000002.11',
            'chr3': 'NC_000003.11', 'chr4': 'NC_000004.11',
            'chr5': 'NC_000005.9', 'chr6': 'NC_000006.11',
            'chr7': 'NC_000007.13', 'chr8': 'NC_000008.10',
            'chr9': 'NC_000009.11', 'chr10': 'NC_000010.10',
            'chr11': 'NC_000011.9', 'chr12': 'NC_000012.11',
            'chr13': 'NC_000013.10', 'chr14': 'NC_000014.8',
            'chr15': 'NC_000015.9', 'chr16': 'NC_000016.9',
            'chr17': 'NC_000017.10', 'chr18': 'NC_000018.9',
            'chr19': 'NC_000019.9', 'chr20': 'NC_000020.10',
            'chr21': 'NC_000021.8', 'chr22': 'NC_000022.10',
            'chr23': 'NC_000023.10', 'chrX': 'NC_000023.10',
            'chr24': 'NC_000024.9', 'chrY': 'NC_000024.9',
            'chrM': 'NC_012920.1'}
    else:
        raise ValueError(f'Unknown assembly "{assembly}"!')

def get_coding_genes(assembly, gencode_version=41):
    # Get coding genes (autosomal/chrX, at least one non-readthrough transcript)
    os.makedirs('gene_annotations', exist_ok=True)
    gencode_URL = \
        f'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_' \
        f'{gencode_version}/GRCh37_mapping/gencode.v{gencode_version}lift37.' \
        f'annotation.gtf.gz' if assembly == 'hg19' else \
        f'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_' \
        f'{gencode_version}/gencode.v{gencode_version}.annotation.gtf.gz'
    non_readthrough_file = f'gene_annotations/non_readthrough_genes_' \
                           f'{assembly}_gencode_v{gencode_version}.txt'
    if not os.path.exists(non_readthrough_file):
        run(f'curl -s {gencode_URL} | zcat | tr -d ";\\"" | awk \'$3 == '
            f'"transcript" && $0 !~ /readthrough_transcript/ '
            f'{{for (x=1; x<=NF; x++) if ($x=="gene_name") print $(x+1)}}\' | '
            f'sort -u > {non_readthrough_file}')
    # noinspection PyUnusedLocal
    non_readthrough_genes = pd.read_table(non_readthrough_file, header=None,
                                          index_col=0).index
    coding_genes_file = f'gene_annotations/coding_genes_{assembly}_' \
                        f'gencode_v{gencode_version}.bed'
    if not os.path.exists(coding_genes_file):
        run(f'curl -s {gencode_URL} | zcat | tr -d ";\\"" | awk \'$3 == "gene" '
            f'&& $0 ~ /protein_coding|IG_.*_gene|TR_.*_gene/ {{for (x=1; '
            f'x<=NF; x++) if ($x=="gene_name") print $1, $4, $5, $(x+1)}}\' '
            f'OFS="\t" | sort -k1,1V -k2,3n > {coding_genes_file}')
    coding_genes = pd.read_table(coding_genes_file, header=None,
                                 names=['chrom', 'start', 'end', 'gene'])\
        .query('chrom != "chrY" and chrom != "chrM" and '
               'gene in @non_readthrough_genes')\
        .groupby('gene')\
        .agg(dict(chrom='first', start='min', end='max'))
    if gencode_version == 41 and assembly == 'hg38':
        assert len(coding_genes) == 19978, len(coding_genes)
    return coding_genes

def get_nearest_genes(sumstats, assembly, gencode_version=41, chrom_col='CHROM',
                      bp_col='BP'):
    # Sumstats should have variant IDs as index (not used, just there for
    # reference) and, at a minimum, chrom and bp columns
    assert assembly in ('hg19', 'hg38')
    assert chrom_col in sumstats, f'{chrom_col} not in sumstats'
    assert bp_col in sumstats, f'{bp_col} not in sumstats'
    if np.issubdtype(sumstats[chrom_col].dtype, np.integer):
        sumstats[chrom_col] = sumstats[chrom_col].astype(str)
    has_chr_prefix = sumstats[chrom_col].iloc[0].startswith('chr')
    # Get coding genes (autosomal/chrX, at least one non-readthrough transcript)
    coding_genes = get_coding_genes(assembly, gencode_version).reset_index()
    # Get each variant's nearest gene; if multiple genes are equidistant (e.g.
    # the variant is inside two overlapping genes), print a comma-separated list
    return pd.Series([
        coding_genes[['gene', 'chrom', 'start', 'end']]
            .query(f'chrom == @chrom' if has_chr_prefix else
                   f'chrom == "chr{chrom}"')
            .assign(distance=lambda df: (df[['start', 'end']] - bp).abs()
                        .min(axis=1).where((bp < df.start) | (df.end < bp), 0),
                    gene_length=lambda df: df.end - df.start)
            .sort_values(['distance', 'gene_length'])
            .pipe(lambda df: ', '.join(df.query(
                f'distance == {df.distance.iloc[0]}').gene))
        if chrom == 'X' or chrom == 'chrX' or
           int(chrom[3:] if has_chr_prefix else chrom) in range(1, 23)
        else pd.NA
        for chrom, bp in sumstats[[chrom_col, bp_col]].values],
        index=sumstats.index).dropna()

def get_nearby_genes(sumstats, assembly, gencode_version=41, chrom_col='CHROM',
                     bp_col='BP', max_distance=500_000):
    # Sumstats should have variant IDs as index (not used, just there for
    # reference) and, at a minimum, chrom and bp columns
    assert assembly in ('hg19', 'hg38')
    assert chrom_col in sumstats, f'{chrom_col} not in sumstats'
    assert bp_col in sumstats, f'{bp_col} not in sumstats'
    if np.issubdtype(sumstats[chrom_col].dtype, np.integer):
        sumstats[chrom_col] = sumstats[chrom_col].astype(str)
    has_chr_prefix = sumstats[chrom_col].iloc[0].startswith('chr')
    # Get coding genes (autosomal/chrX, at least one non-readthrough transcript)
    coding_genes = get_coding_genes(assembly, gencode_version).reset_index()
    # Get each variant's nearby genes within max_distance
    return pd.concat([
        coding_genes[['gene', 'chrom', 'start', 'end']]
        .query(f'chrom == @chrom' if has_chr_prefix else
               f'chrom == "chr{chrom}"')
        .assign(distance=lambda df: (df[['start', 'end']] - bp).abs()
                .min(axis=1).where((bp < df.start) | (df.end < bp), 0),
                gene_length=lambda df: df.end - df.start)
        .query(f'distance < {max_distance}')
        .sort_values(['distance', 'gene_length'])
        [['gene', 'distance']]
        .assign(rs=rs)
        .set_index('rs')
        for rs, chrom, bp in sumstats[[chrom_col, bp_col]].itertuples()
        if chrom == 'X' or chrom == 'chrX' or
           int(chrom[3:] if has_chr_prefix else chrom) in range(1, 23)])

def cache_df(function, filename, index_col):
    if os.path.exists(filename):
        return pd.read_table(filename, index_col=index_col)
    else:
        df = function()
        df.to_csv(filename, sep='\t')
        return df

def nsmallest(series, n):
    # Much faster version of pd.Series.nsmallest(), but sorted differently
    return series.iloc[np.argpartition(series, n)[:n]]

def nlargest(series, n):
    # Much faster version of pd.Series.nlargest(), but sorted differently
    return series.iloc[np.argpartition(series, -n)[-n:]]

def nth_smallest(df, n, axis=0):
    # Get the nth-smallest values of df along axis
    # assert nth_smallest(df, n=1, axis=axis).equals(df.min(axis=axis))
    assert axis == 0 or axis == 1
    if axis == 0:
        return pd.Series(
            df.values[np.argpartition(df.values, kth=n - 1, axis=0)[n - 1],
                      range(df.shape[1])],
            index=df.columns)
    else:
        return pd.Series(
            df.values[range(df.shape[0]),
                      np.argpartition(df.values, kth=n - 1, axis=1)[:, n - 1]],
            index=df.index)

def nth_largest(df, n, axis=0):
    # Get the nth-largest values of df along axis
    # assert nth_largest(df, n=1, axis=axis).equals(df.max(axis=axis))
    return nth_smallest(df, n=df.shape[axis] - (n - 1), axis=axis)

def nth_smallest_index(df, n, axis=0):
    # Get the indices of the nth-smallest values of df along axis
    # assert nth_smallest_index(df, n=1, axis=axis).equals(df.idxmin(axis=axis))
    assert axis == 0 or axis == 1
    if axis == 0:
        return pd.Series(
            df.index[np.argpartition(df.values, kth=n - 1, axis=0)[n - 1]],
            index=df.columns)
    else:
        return pd.Series(
            df.columns[np.argpartition(df.values, kth=n - 1, axis=1)[:, n - 1]],
            index=df.index)

def nth_largest_index(df, n, axis=0):
    # Get the indices of the nth-largest values of df along axis
    # assert nth_largest_index(df, n=1, axis=axis).equals(df.idxmax(axis=axis))
    return nth_smallest_index(df, n=df.shape[axis] - (n - 1), axis=axis)

def percentile_transform(array_or_series, method='average'):
    from scipy.stats import rankdata
    if isinstance(array_or_series, pd.Series):
        index = array_or_series.index
        array_or_series = array_or_series.values
    else:
        index = None
    ranks = rankdata(array_or_series, method=method)
    quantiles = 100 * (ranks - 1) / (len(ranks) - 1)
    if index is not None:
        quantiles = pd.Series(quantiles, index=index)
    return quantiles

def CPM(counts, groups=None):
    from anndata import AnnData
    from rpy2.robjects import r
    r.library('edgeR', quietly=True)
    assert isinstance(counts, AnnData)
    if groups is not None:
        assert isinstance(groups, pd.Series)
    counts_r_T = array_to_rmatrix(counts.X.T if type(counts.X) == np.ndarray
                                  else counts.X.T.toarray())
    if groups is not None:
        groups_r = series_to_rvector(groups)
        dge_list = r.DGEList(counts_r_T, group=groups_r)
    else:
        dge_list = r.DGEList(counts_r_T)
    normalized_counts = r.calcNormFactors(dge_list)
    CPMs_r = r.cpm(normalized_counts)
    CPMs = rmatrix_to_df(CPMs_r)
    CPMs = AnnData(CPMs.T.values, obs=counts.obs, var=counts.var, dtype='float')
    return CPMs

def calculate_qc_metrics(anndata):
    # A subset of the functionality from scanpy.pp.calculate_qc_metrics()
    anndata.obs = anndata.obs.assign(
        n_genes_by_counts=lambda df: anndata.X.getnnz(axis=1),
        total_counts=lambda df: anndata.X.sum(axis=1).A1,
        pct_counts_mt=lambda df: anndata.X[
                                 :, anndata.var_names.str.startswith('MT-')]
                                 .sum(axis=1).A1 / df.total_counts * 100)

class KNeighborsClassifier(object):
    # Inspired by Scikit's KNeighborsClassifier,
    # github.com/shankarpm/faiss_knn/blob/master/faiss_knn.py and
    # towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-
    # 20-lines-5e29d74e76bb
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.y_categories = None
        self.k = k

    def fit(self, X, y, use_gpu=True):
        assert len(X) == len(y), f'{len(X)=}, {len(y)=}'
        import faiss  # mamba install faiss
        self.index = faiss.IndexFlatL2(X.shape[1])
        if use_gpu:
            assert faiss.get_num_gpus() > 0
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(np.ascontiguousarray(X))
        if y.dtype == 'category':
            self.y = y.cat.codes.values
            self.y_categories = y.cat.categories.values
        else:
            self.y = np.asarray(y)
        return self

    def predict(self, X):
        from scipy.stats import mode
        indices = self.index.search(np.ascontiguousarray(X), k=self.k)[1]
        votes = self.y[indices]
        predictions, confidences = mode(votes, axis=1, keepdims=False)
        if self.y_categories is not None:
            predictions = self.y_categories[predictions]
        confidences = confidences / self.k
        return predictions, confidences

def read_obs(adata_file):
    import h5py
    from anndata._io.h5ad import read_dataframe
    with h5py.File(adata_file) as f:
        obs = read_dataframe(f['obs'])
    return obs

def read_var(adata_file):
    import h5py
    from anndata._io.h5ad import read_dataframe
    with h5py.File(adata_file) as f:
        var = read_dataframe(f['var'])
    return var

def sparse_to_cugraph(adjacency):
    # Converts a sparse adjacency matrix (e.g. obsp['connectivities'] from
    # sc.pp.neighbors) to a cugraph
    from scipy.sparse import spmatrix
    assert isinstance(adjacency, spmatrix)
    # noinspection PyUnresolvedReferences
    import cugraph
    nn_graph = cugraph.Graph()
    # Broken for now due to github.com/rapidsai/cugraph/issues/3016
    # nn_graph.from_cudf_adjlist(cudf.Series(adjacency.indptr),
    #                            cudf.Series(adjacency.indices),
    #                            cudf.Series(adjacency.data))
    # Instead, use:
    coo = adjacency.tocoo()
    nn_graph.from_pandas_edgelist(pd.DataFrame({
        'source': coo.row, 'destination': coo.col, 'weight': coo.data}),
        edge_attr='weight', renumber=False)
    assert not nn_graph.is_directed()
    assert nn_graph.is_weighted()
    return nn_graph

def log_ss_ratio(X, labels):
    # github.com/elixir-code/craved/blob/master/craved/internal_indices.py#L182
    from sklearn.utils import check_X_y
    from sklearn.metrics.cluster._unsupervised import check_number_of_labels
    from sklearn.preprocessing import LabelEncoder

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = cluster_k.mean(axis=0)
        extra_disp += len(cluster_k) * ((mean_k - mean) ** 2).sum()
        intra_disp += ((cluster_k - mean_k) ** 2).sum()

    return np.log(extra_disp / intra_disp)

def ball_hall(X, labels):
    # github.com/elixir-code/craved/blob/master/craved/internal_indices.py#L142
    from sklearn.utils import check_X_y
    from sklearn.metrics.cluster._unsupervised import check_number_of_labels
    from sklearn.preprocessing import LabelEncoder

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    sum_mean_dispersions = 0

    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = cluster_k.mean(axis=0)
        intra_disp_cluster = ((cluster_k - mean_k) ** 2).sum()
        sum_mean_dispersions += intra_disp_cluster / len(cluster_k)

    return sum_mean_dispersions / n_labels

def banfeld_raftery(X, labels):
    # github.com/elixir-code/craved/blob/master/craved/internal_indices.py#L155
    from sklearn.utils import check_X_y
    from sklearn.metrics.cluster._unsupervised import check_number_of_labels
    from sklearn.preprocessing import LabelEncoder

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    br_index = 0
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = cluster_k.mean(axis=0)
        intra_disp_cluster = ((cluster_k - mean_k) ** 2).sum()
        br_index += len(cluster_k) * np.log(intra_disp_cluster / len(cluster_k))

    return br_index

def ray_turi(X, labels):
    # github.com/elixir-code/craved/blob/master/craved/internal_indices.py#L348
    from itertools import combinations
    from sklearn.utils import check_X_y
    from sklearn.metrics.cluster._unsupervised import check_number_of_labels
    from sklearn.preprocessing import LabelEncoder

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    intra_disp = 0.0

    cluster_means = []
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = cluster_k.mean(axis=0)
        intra_disp += ((cluster_k - mean_k) ** 2).sum()
        cluster_means.append(mean_k)

    min_cluster_mean_diff = min(
        ((mean_i - mean_j) ** 2).sum()
        for mean_i, mean_j in combinations(cluster_means, 2))

    return intra_disp / (min_cluster_mean_diff * n_samples)

def minimize_scalar(fun, bounds, args=(), method='hybrid-brent', tol=None,
                    options=dict(maxiter=np.iinfo('int32').max)):
    assert method in ('optuna', 'hybrid-brent', 'hybrid-golden'), method
    # Start with optuna
    # github.com/optuna/optuna, dl.acm.org/doi/10.1145/3292500.3330701,
    # medium.com/criteo-engineering/
    # hyper-parameter-optimization-algorithms-2fe447525903
    import optuna
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda trial: fun(
        trial.suggest_float('x', *bounds), *args), n_trials=100)
    x = study.best_trial.params['x']
    if method == 'optuna':
        from scipy.optimize import OptimizeResult
        return OptimizeResult(x=x, fun=study.best_trial.value)
    else:
        # Fine-tune with brent or golden, using the best resolution from
        # optuna as the middle element of the bracket
        from scipy.optimize import minimize_scalar, OptimizeResult
        bracket = bounds[0], x, bounds[1]
        try:
            return minimize_scalar(fun, bracket=bracket, args=args,
                                   method=method.split('-')[1], tol=tol,
                                   options=options)
        except ValueError as error:
            if str(error) != 'Not a bracketing interval.': raise
            fun_left, fun_right = fun(bounds[0], *args), fun(bounds[1], *args)
            assert fun_left != fun_right, \
                f'f({bounds[0]}) = f({bounds[1]}) = {fun_left}'
            return OptimizeResult(x=bounds[0], fun=fun_left) \
                if fun_left < fun_right else \
                OptimizeResult(x=bounds[1], fun=fun_right)

def leiden(nn_graph, resolution):
    # noinspection PyUnresolvedReferences
    import cugraph
    assert isinstance(nn_graph, cugraph.Graph), type(nn_graph)
    assert resolution >= 0, resolution
    return cugraph.leiden(nn_graph, max_iter=np.iinfo('int32').max,
                          resolution=resolution)[0]\
        .partition.to_pandas().rename(None)

def leiden_cv(nn_graph, resolution=None, n_folds=5, batch_labels=None,
              bounds=(0, 1), method=None, random_state=0, verbose=False):
    from scipy.sparse import spmatrix
    if not isinstance(nn_graph, spmatrix):
        nn_graph = nn_graph.sparse
        assert isinstance(nn_graph, spmatrix)
    from sklearn.model_selection import KFold, StratifiedKFold
    # If resolution is None, call recursively to find the resolution at which as
    # close to 90% as possible of test cells are confidently (>=90%) predicted,
    # and return this resolution and the clusters at this resolution
    if resolution is None:
        from scipy.optimize import root_scalar
        def objective(resolution):
            fraction_confidently_predicted = leiden_cv(
                nn_graph, resolution, n_folds, batch_labels, random_state)
            if verbose:
                print(f'{resolution}: '
                      f'{np.sort(fraction_confidently_predicted)}')
            return np.median(fraction_confidently_predicted) - 0.9
        best_resolution = root_scalar(objective, method=method,
                                      bracket=bounds).x
        best_clusters = batch_labels.to_frame()\
            .assign(leiden_cluster=leiden(sparse_to_cugraph(nn_graph),
                                          best_resolution).values)
        return best_clusters, best_resolution
    # For each fold...
    CV_class = StratifiedKFold if batch_labels is not None else KFold
    folds = CV_class(n_folds, shuffle=True, random_state=random_state)\
        .split(nn_graph, batch_labels)
    fraction_confidently_predicted = np.empty(n_folds)
    for fold_index, (train_fold, test_fold) in enumerate(folds):
        # Run Leiden on the training fold (without rerunning knn for simplicity)
        leiden_clusters = leiden(sparse_to_cugraph(
            nn_graph[train_fold][:, train_fold]), resolution)
        # Figure out which test cells are connected to which train cells
        train_test_connections = nn_graph[test_fold][:, train_fold].tocoo()
        # Get the Leiden labels of the test cells' nearest train cell neighbors
        test_nearest_labels = pd.Series(
            leiden_clusters.values[train_test_connections.col],
            index=pd.Index(train_test_connections.row, name='cell_index'),
            name='leiden_cluster')
        # For each test cell, get the fraction of its nearest train cell
        # neighbors that share the same label as each other
        # The below code is equivalent to:
        # mode_fraction = test_nearest_labels\
        #     .groupby(test_nearest_labels.index)\
        #     .agg(lambda x: scipy.stats.mode(x, keepdims=False)[1] / len(x))
        mode_fraction = test_nearest_labels\
            .reset_index()\
            .value_counts()\
            .reset_index('leiden_cluster')\
            .leiden_cluster\
            .groupby('cell_index', sort=False)\
            .first()\
            .loc[test_nearest_labels.index]\
            .eq(test_nearest_labels)\
            .groupby('cell_index')\
            .mean()
        # Finally, get the fraction of test cells confidently (>= 90%) predicted
        fraction_confidently_predicted[fold_index] = \
            mode_fraction.ge(0.9).mean()
    return fraction_confidently_predicted

def best_leiden(nn_graph, underlying_data, batch_labels=None, metric='ray_turi',
                method='hybrid-brent', bounds=(0, 1), verbose=True):
    # Pick the leiden resolution with the best value of metric, and return that
    # resolution and the Leiden clustering at that resolution.
    # - Definitions of clustering metrics:
    #   cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    # - Comparison of clustering metrics: stats.stackexchange.com/a/358937
    # - List of lists of Python implementations:
    #   github.com/scikit-learn/scikit-learn/discussions/21164
    # - Particularly useful ones:
    #   - github.com/elixir-code/craved/blob/master/craved/internal_indices.py
    #   - github.com/Simon-Bertrand/Clusters-Features/blob/main/
    #     ClustersFeatures/src/_score_index.py
    if metric == 'calinski_harabasz':
        from sklearn.metrics import calinski_harabasz_score as scorer
        higher_is_better = True
    elif metric == 'davies_bouldin':
        from sklearn.metrics import davies_bouldin_score as scorer
        higher_is_better = False
    elif metric == 'log_ss_ratio':
        scorer = log_ss_ratio
        higher_is_better = True
    elif metric == 'ball_hall':
        scorer = ball_hall
        higher_is_better = True
    elif metric == 'banfeld_raftery':
        scorer = banfeld_raftery
        higher_is_better = False
    elif metric == 'ray_turi':
        scorer = ray_turi
        higher_is_better = False
    elif metric == 'silhouette':
        # noinspection PyUnresolvedReferences
        from cuml.metrics.cluster.silhouette_score import \
            cython_silhouette_score as scorer  # very slow for large datasets!
        higher_is_better = True
    elif metric == 'distortion':
        # noinspection PyUnresolvedReferences
        from yellowbrick.cluster.elbow import distortion_score as scorer
        higher_is_better = False
    elif metric == 'cv':
        return leiden_cv(nn_graph, batch_labels=batch_labels, bounds=bounds,
                         verbose=verbose)
    else:
        raise ValueError(f'Unknown metric "{metric}"!')
    from sklearn.metrics import adjusted_rand_score
    def clustering_score(resolution, underlying_data):
        leiden_clusters = leiden(nn_graph, resolution)
        num_clusters = leiden_clusters.nunique()
        score = (-np.inf if higher_is_better else np.inf) \
            if num_clusters == 1 else scorer(underlying_data, leiden_clusters)
        if verbose:
            message = f'{resolution}: {num_clusters=}, {metric}=' \
                      f'{score:.{8 if metric == "silhouette" else 2}f}'
            if batch_labels is not None:
                batch_ARI = adjusted_rand_score(leiden_clusters, batch_labels)
                message += f', {batch_ARI=}'
            print(message)
        return -score if higher_is_better else score
    best_resolution = minimize_scalar(clustering_score, bounds=bounds,
                                      args=(underlying_data,),
                                      method=method).x
    best_clusters = leiden(nn_graph, best_resolution)
    if batch_labels is not None:
        best_clusters = batch_labels.to_frame()\
            .assign(leiden_cluster=best_clusters.values)
    return best_clusters, best_resolution

def highly_variable_genes(adata, n_top_genes=2000, batch_key=None, span=0.3):
    # Roughly equivalent to sc.pp.highly_variable_genes(adata, inplace=False,
    # flavor='seurat_v3', check_values=False), with minor changes to reduce
    # runtime and peak memory usage
    import gc  # my addition
    # noinspection PyUnresolvedReferences
    from scanpy.pp._utils import _get_mean_var
    from scipy.sparse import csr_matrix, issparse
    # noinspection PyUnresolvedReferences
    from skmisc.loess import loess
    X = adata.X
    df = pd.DataFrame(index=adata.var_names)
    batch_info = adata.obs[batch_key].values if batch_key is not None else \
        pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    norm_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = X[batch_info == b]
        mean, var = _get_mean_var(X_batch)
        not_const = var > 0
        estimat_var = np.zeros(X.shape[1], dtype=np.float64)
        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)
        batch_counts = X_batch.astype(np.float64).copy()
        # clip large values as in Seurat
        N = X_batch.shape[0]
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean
        if issparse(batch_counts):
            batch_counts = csr_matrix(batch_counts)
            mask = batch_counts.data > clip_val[batch_counts.indices]
            batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]
            squared_batch_counts_sum = \
                np.array(batch_counts.power(2).sum(axis=0))
            batch_counts_sum = np.array(batch_counts.sum(axis=0))
        else:
            clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
            np.putmask(batch_counts, batch_counts > clip_val_broad,
                       clip_val_broad)
            squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
            batch_counts_sum = batch_counts.sum(axis=0)
        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean)) + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean)
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))
        del batch_counts; gc.collect()  # my addition
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = \
        np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = \
        np.sum((ranked_norm_gene_vars < n_top_genes).astype(int), axis=0)
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    # noinspection PyUnresolvedReferences
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
    df['highly_variable_nbatches'] = num_batches_high_var
    df['highly_variable_rank'] = median_ranked
    sorted_index = df[['highly_variable_rank', 'highly_variable_nbatches']]\
        .sort_values(['highly_variable_rank', 'highly_variable_nbatches'],
                     ascending=[True, False], na_position='last')\
        .index
    df['highly_variable'] = False
    df.loc[sorted_index[:int(n_top_genes)], 'highly_variable'] = True
    if batch_key is None:
        df = df.drop(['highly_variable_nbatches'], axis=1)
    return df

def pseudobulk(dataset, ID_column, cell_type_column):
    # Use observed=True to skip groups where any of the columns is NaN
    grouped = dataset.obs.groupby([cell_type_column, ID_column], observed=True)
    # Fill with 0s to avoid auto-conversion to float when filling with NaNs
    pseudobulk = pd.DataFrame(
        0, index=pd.MultiIndex.from_frame(
            grouped.size().rename('num_cells').reset_index()),
        columns=dataset.var_names, dtype=dataset.X.dtype)
    for row_index, group_indices in enumerate(grouped.indices.values()):
        group_counts = dataset[group_indices].X
        pseudobulk.values[row_index] = group_counts.sum(axis=0).A1
    return pseudobulk

@cache
def generate_palette(num_colors, lightness_range=(40, 80),
                     chroma_range=(40, 100)):
    # Generate maximally perceptually distinct color palette with glasbey
    # - Install: pip install git+https://github.com/taketwo/glasbey
    # - Useful discussion: github.com/holoviz/colorcet/issues/11
    # - Parameter examples: colorcet.holoviz.org/assets/images/named.png
    # - For all i < j, generate_palette(i) == generate_palette(j)[:i]
    #
    # Box num_colors (if not an iterable) and convert to tuple
    from collections.abc import Iterable
    num_colors = tuple(num_colors) if isinstance(num_colors, Iterable) else \
        (num_colors,)
    # Generate colormap
    from glasbey import Glasbey
    palette = Glasbey(no_black=True, lightness_range=lightness_range,
                      chroma_range=chroma_range)\
        .generate_palette(size=max(num_colors) + 1).clip(0, 1)
    # Exclude white
    assert (palette[0] == (1, 1, 1)).all(), palette[0]
    palette = palette[1:]
    # Order by hue; could also use matplotlib.colors.rgb_to_hsv()
    from colorspacious import cspace_convert
    hue = cspace_convert(palette, 'sRGB255', 'JCh')[:, 2]
    palette = [palette[:num_color][hue[:num_color].argsort()]
               for num_color in num_colors]
    # Convert rgb to hex
    from matplotlib.colors import to_hex
    palette = [np.apply_along_axis(to_hex, 1, p) for p in palette]
    # Unbox if only one num_color
    if len(num_colors) == 1:
        palette = palette[0]
    return palette

def print_large_objects(N=20):
    # There are two main options for accurately calculating Python object sizes:
    # - objsize.get_deep_size from pypistats.org/packages/objsize
    # - pympler.asizeof.asizeof from pypistats.org/packages/pympler
    # noinspection PyUnresolvedReferences
    from pympler.asizeof import asizeof
    mem = pd.Series({key: asizeof(eval(key)) for key in globals()})\
        .sort_values(ascending=False)
    print(mem.head(N).to_string())

def save_gzipped(df, filename, **kwargs):
    assert filename.endswith('.gz')
    ungzipped = filename.removesuffix('.gz')
    df.to_csv(ungzipped, sep='\t', **kwargs)
    run(f'gzip -f {ungzipped}')

def ordered_legend(order, ax=None, **kwargs):
    # Adapted from stackoverflow.com/a/35926913/1397061
    import matplotlib.pyplot as plt
    if ax is None: ax = plt.gca()
    ax.legend(**kwargs)
    handles, labels = ax.get_legend_handles_labels()
    keys = dict(zip(order, range(len(order))))
    labels, handles = zip(*sorted(
        zip(labels, handles), key=lambda t, keys=keys: keys.get(t[0], np.inf)))
    ax.legend(handles, labels, **kwargs)


###########################################################

def get_available_ram():
    page_size = os.sysconf('SC_PAGE_SIZE')
    av_phys_pages = os.sysconf('SC_AVPHYS_PAGES')
    available_ram = (av_phys_pages * page_size) / (1024 * 1024 * 1024) 
    return available_ram