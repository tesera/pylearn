import pandas as pd
import numpy as np
from scipy import stats

from pylearn.varset import get_param

def rate(xy, x_filtered, param):
    param_const = param[param['PARAMNAME'] == 'CONSTANT']['PARAMVALUE'][0]
    param_values = param[param['PARAMNAME'] != 'CONSTANT'].set_index('ATTR')['PARAMVALUE']
    param_attrs = param[param['PARAMNAME'] != 'CONSTANT']['ATTR']

    xy = xy.filter(param_attrs)

    zs = param_const + (xy * param_values).sum(axis=1)
    zs = zs.value_counts().to_frame().sort_index()
    zs.columns = ['N']

    z_max_ref = zs.index.max() + 0.001
    z_min_ref = zs.index.min() - 0.001

    zs['RANK'] = zs['N'].rank(method='first')
    zs['pRANK'] = zs['RANK'] / (zs['RANK'].max() + 1.0)
    zs['ODDS'] = zs['pRANK'] / (1.0 - zs['pRANK'])
    zs['LODDS'] = np.log(zs['ODDS'])
    zs['ZTRANS'] = (zs.index - z_min_ref) / (z_max_ref - z_min_ref)

    xys = zs[(zs['pRANK'] >= 0.1) & (zs['pRANK'] <= 0.9)]
    x = list(np.log(xys['ZTRANS']))
    y = list(xys['LODDS'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    idfcoef_data = [slope, intercept, r_value, p_value, std_err]
    idfcoef_columns = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
    idfcoef = pd.Series(idfcoef_data, idfcoef_columns)

    sumz = x_filtered.set_index('UID').filter(param_attrs)
    sumz['ZSCORE'] = param_const + (sumz * param_values).sum(axis=1)
    sumz['ZTRANS'] = (sumz['ZSCORE'] - z_min_ref) / (z_max_ref - z_min_ref)
    sumz['ODDS'] = np.exp(intercept + slope * np.log(sumz['ZTRANS']))
    sumz['p'] = sumz['ODDS'] / (1 + sumz['ODDS'])

    return (idfcoef, sumz)

def predict(xy, x_filtered, dfunct, varset, yvar, idf):
    param = get_param(dfunct, varset)

    idfpvt = pd.pivot_table(idf, columns=['PERIOD'], values=['TRMM'], aggfunc=np.sum)
    divisor = idfpvt.ix['TRMM','HISTORICAL']
    idfpvt.ix['RATIO'] = idfpvt.ix['TRMM'] / divisor

    idfcoef, sumz = rate(xy, x_filtered, param)

    sumz.ix[sumz['ZTRANS'] > 0, 'ZTRANS'] = np.log(sumz['ZTRANS'])

    logz = sumz['ZTRANS']
    intercept = idfcoef['intercept']
    slope = idfcoef['slope']

    for period in idfpvt:
        idfratio = idfpvt.ix['RATIO', period]
        fx = np.exp((intercept * idfratio) + (slope * logz))
        sumz[period] = fx / (1 + fx)

    xvars = param[~(param['PARAMNAME'] == 'CONSTANT')]['ATTR']

    sumz['ZOU'] = 0
    sumz['CHK'] = 0
    sumz.ix[sumz[xvars].sum(axis=1) == 0.0, 'ZOU'] = 1
    sumz.ix[sumz['ZTRANS'] == 0.0, 'CHK'] = 1

    return sumz

