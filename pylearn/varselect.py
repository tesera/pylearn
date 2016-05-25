from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats
import math
import hashlib

def count_xvars(xvarselv1=None):
    return len(xvarselv1[xvarselv1['XVARSEL'] == 'X'])

def rank_xvars(varselect=None):
    ncases = len(varselect)
    by_varname = varselect.groupby(by=['VARNAME'])
    data = []

    for varname, group in by_varname:
        nvars = len(group)
        rank = group['SOLTYPE'].min()
        p = (nvars / float(ncases)) ** (1 / 2.0)
        importance = (p * (1 / float(rank)) ** (1 / 2.0)) ** (1 / 2.0)
        data.append({'VARNAME': varname,'RANK': rank,
                     'P': p, 'IMPORTANCE': importance})

    ranks = pd.DataFrame(data, columns=['VARNAME', 'IMPORTANCE', 'P', 'RANK'])
    return ranks

def extract_xvar_combos(varselect=None):
    by_model_id = varselect.groupby(by='MODELID')

    data = []
    for modelid, model in by_model_id:
        xvars = list(model['VARNAME'].sort_values())
        xvars_data = {'XVAR' + str(i+1): var for i, var in enumerate(xvars)}
        model_hash = hashlib.sha1(''.join(xvars)).hexdigest()
        row_data = {'MODELHASH': model_hash, 'MODELID': modelid,
                    'NMODELS': 0, 'NVAR': len(xvars)}
        data.append(dict(row_data.items() + xvars_data.items()))

    by_model_hash = pd.DataFrame(data).groupby(by='MODELHASH')

    models = []
    varset = 0
    for model_hash, model in by_model_hash:
        varset = varset + 1
        model['NMODELS'] = len(model.index)
        models.append(model.iloc[0])

    xvarselv = pd.DataFrame(models)
    xvarselv.sort_values('MODELID', inplace=True)
    xvarselv['VARSET'] = xvarselv['NVAR'].rank(method='first').astype(int)
    del xvarselv['MODELHASH']

    cols = ['VARSET', 'MODELID', 'NMODELS', 'NVAR']
    nvarcols = len(xvarselv.columns.values)-4
    varcols = ['XVAR' + str(i+1) for i in range(0, nvarcols)]
    cols.extend(varcols)

    xvarselv = xvarselv.reindex(columns=cols)
    xvarselv.set_index(['VARSET'], inplace=True)
    xvarselv.replace(np.nan, 'N', inplace=True)

    uniq_xvars = varselect['VARNAME'].unique()
    uniq_xvars.sort()
    xcols = ['X' + str(i+1) for i in range(0, len(uniq_xvars))]
    cols = ['MYID']
    cols.extend(xcols)
    data = {'X' + str(i+1): var for i, var in enumerate(uniq_xvars)}
    data['MYID'] = 1
    uniquevar = pd.DataFrame(data=[data], columns=cols)
    uniquevar.set_index(['MYID'], inplace=True)

    return (xvarselv, uniquevar)

def remove_high_corvar(varrank=None, xvarselv1=None, ucorcoef=None):
    for var1, var2 in combinations(varrank['VARNAME'], 2):
        corr_coef = ucorcoef[(ucorcoef['VARNAME1'] == var1) &
                             (ucorcoef['VARNAME2'] == var2)]['CORCOEF'].item()

        if abs(corr_coef) < 0.8:
            continue

        varrank_selection = (varrank['VARNAME'] == var1) | (varrank['VARNAME'] == var2)
        varrank_pair = varrank[varrank_selection]
        ix_least_important = varrank_pair['IMPORTANCE'].idxmin()
        least_important_var = varrank_pair.ix[ix_least_important, 'VARNAME']
        xvarselv1.ix[xvarselv1['VARNAME'] == least_important_var, 'XVARSEL'] = 'N'

    return xvarselv1
