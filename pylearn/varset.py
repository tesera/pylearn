import os
from functools import partial
import logging

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger('pylearn')

def rank_varset(row, rank_coefficient=200):
    khat = float(row['KHAT'])
    nvar = int(row['NVAR'])
    return khat - (1 - khat) * nvar / (rank_coefficient - nvar - 1)

def rank_varset_assess(assess=None, rank_coefficient=200):
    rank_with = partial(rank_varset, rank_coefficient=rank_coefficient)
    assess['VARSETRANK'] = assess.apply(rank_with, axis=1)

    return assess

def get_xvars(dfunct=None, varset=None):
    """Function returns variables names in specified varset.

    Args:
        dfunct (DataFrame): DFUNCT data from lda.
        varset (Int): The integer representing the varset to extract.

    Returns: Pandas Series
    """
    if varset:
        return dfunct[dfunct['VARSET3'] == varset]['VARNAMES3'].sort_values()
    else:
        return dfunct['VARNAMES3'].sort_values().unique()

def get_xy_summary(xy=None, dfunct=None, yvar=None):
    """Function returns mean values for each unique variable in DFUNCT.

    Args:
        xy (DataFrame): The xy reference data.
        dfunct (DataFrame): DFUNCT data from lda.
        yvar (String): Variable Name representing the Y Class.

    Returns: Pandas DataFrame
    """
    xvars = get_xvars(dfunct)
    data = []

    for xvar in xvars:
        column = xy[xvar]
        summary = pd.pivot_table(xy, values=column.name, index=[yvar], aggfunc=np.mean)
        data.extend([{'ATTR': column.name, yvar: idx, 'MEAN': val} for idx, val in summary.iteritems()])

    return pd.DataFrame(data=data)

def get_rloadrank(xy=None, dfunct=None, yvar=None):
    """Function Extends DFUNCT with rank based relative loading per varset.

    Args:
        xy (DataFrame): The xy reference data.
        dfunct (DataFrame): DFUNCT data from lda.
        yvar (String): Variable Name representing the Y Class.

    Returns: Pandas DataFrame
    """
    xy_summary = get_xy_summary(xy, dfunct, yvar)
    ranks = []
    dfunct_by_varset = dfunct.groupby(by='VARSET3')

    for vset, df in dfunct_by_varset:
        vdfunct = df.copy()
        vdfunct.set_index('VARNAMES3', inplace=True)
        xvars = vdfunct.index.values
        xyref = xy.filter(items=xvars, axis=1)

        vdfunct['SD'] = xyref.std()
        vdfunct['B*'] = vdfunct['DFCOEF3'] * vdfunct['SD']
        vdfunct['LOADING'] = np.sum(xyref.corr().mul(vdfunct['B*'].dropna(), axis=0))
        vdfunct['RL'] = vdfunct['LOADING'] / vdfunct['LOADING'].max()

        for xvar in xvars:
            xvar_summary = xy_summary[xy_summary['ATTR'] == xvar]
            vdfunct.loc[xvar, 'Y0'] = xvar_summary[xvar_summary[yvar] == 0]['MEAN'].iat[0]
            vdfunct.loc[xvar, 'Y1'] = xvar_summary[xvar_summary[yvar] == 1]['MEAN'].iat[0]

        z0 = np.sum(vdfunct['DFCOEF3'].mul(vdfunct['Y0'], axis=0))
        z1 = np.sum(vdfunct['DFCOEF3'].mul(vdfunct['Y1'], axis=0))
        vdfunct['Z0'] = z0
        vdfunct['Z1'] = z1
        vdfunct['DFCOEF3_ADJ'] = vdfunct['DFCOEF3'] if z1 > z0 else vdfunct['DFCOEF3'] * -1
        vdfunct['Z0_ADJ'] = np.sum(vdfunct['DFCOEF3_ADJ'].mul(vdfunct['Y0'], axis=0))
        vdfunct['Z1_ADJ'] = np.sum(vdfunct['DFCOEF3_ADJ'].mul(vdfunct['Y1'], axis=0))

        vdfunct['RANK'] = vdfunct['RL'].rank(ascending=False)
        ranks.append(vdfunct)

    labels = list(dfunct.axes[1])
    labels.extend(['Y0', 'Y1', 'Z0', 'Z1','DFCOEF3_ADJ', 'Z0_ADJ', 'Z1_ADJ', 'SD', 'B*', 'LOADING', 'RL', 'RANK'])

    return pd.concat(ranks).reset_index().reindex_axis(labels=labels, axis=1)

def get_avgp(xy, xvars):
    """Function returns unique values, rank and average probability
    for each x variable in lda dfunct.

    Args:
        xy (DataFrame): The xy reference data.
        xvars (List): List of variables to get average pprobabilities for

    Returns: multi-index Pandas DataFrame

                     VAL    RANK      AVGP
        SSH7_25   0  1.0   270.0  0.477876
                  1  2.0   552.5  0.977876
        SPDIASTRC 0  1.0   155.0  0.074699
                  1  2.0   972.0  0.468434
                  2  3.0  1855.0  0.893976
    """
    avgps = []

    for xvar in xvars:
        vals = xy[xvar]
        vals = vals[vals > 0].to_frame()

        vals['RANK'] = vals.rank()
        vals['P'] = vals.apply(lambda row: row['RANK'] / len(vals), axis=1)

        avg = vals.groupby(xvar).mean()
        avg.reset_index(level=0, inplace=True)
        avg.rename(columns={xvar: 'VAL'}, inplace = True)
        avg.rename(columns={'P': 'AVGP'}, inplace = True)

        avgps.append(avg)

    return pd.concat(avgps, keys=xvars)

def get_param(dfunct, varset):
    """Function Returns a PARAM file representing the varset chosen.

    Args:
        dfunct (DataFrame): DFUNCT data from lda.
        varset (Int): The integer representing the varset of interest.

    Returns: Pandas DataFrame
    """
    xvars = dfunct[dfunct['VARSET3'] == varset].reset_index()

    data = [['CONSTANT','EL6YT1YN',0,'NA','Constant value can be ignored']]
    for idx, xvar in xvars.iterrows():
        data.append(['b' + str(idx), 'EL6YT1YN', xvar['DFCOEF3'], xvar['VARNAMES3'], 'Available variable'])

    return pd.DataFrame(data=data, columns=['PARAMNAME','EQUATION','PARAMVALUE','ATTR','DESCRIPTION'])

