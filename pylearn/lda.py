# TODO: logging
# TODO: module naming - do module names represent what they do/contain?
import pandas as pd
import numpy as np

from varset import rank_varset_assess
from .scores import khat, overall_accuracy, producer_accuracy, user_accuracy

# TODO: change name
def cohens_khat(ctabulation):
    """Calculate classification accuracy scores

    Includes Cohen's khat, overall accuracy, and the min and max producer and
    user accuracy

    :param ctabulation: class tabulation data frame

    """
    by_varset_ref = ctabulation.groupby(by=['VARSET'])

    ctabsum_cols = ['OA', 'KHAT', 'MINPA', 'MAXPA', 'MINUA', 'MAXUA']
    ix = set(ctabulation['VARSET'].values.tolist())
    ctabsum = pd.DataFrame(index=ix, columns=ctabsum_cols)

    # TODO: replace with function that accepts tab and
    #       by_varset_ref.agg(calculate_classification_scores_from_ctab)
    for varset, tab in by_varset_ref:
        del tab['VARSET']
        # transform to confusion matrix
        matrix = tab.pivot(index='PREDCLASS', columns='REFCLASS')
        # remove CTAB from nested columns index
        matrix.columns = matrix.columns.droplevel()

        oa = overall_accuracy(matrix)
        pa = producer_accuracy(matrix)
        ua = user_accuracy(matrix)
        min_pa = pa.min(axis=1)[0]
        max_pa = pa.max(axis=1)[0]
        min_ua = ua.min(axis=1)[0]
        max_ua = ua.max(axis=1)[0]
        kappa = khat(matrix)

        ctabsum.ix[varset,'OA'] = oa
        ctabsum.ix[varset,'KHAT'] = kappa
        ctabsum.ix[varset,'MINPA'] = min_pa
        ctabsum.ix[varset,'MAXPA'] = max_pa
        ctabsum.ix[varset,'MINUA'] = min_ua
        ctabsum.ix[varset,'MAXUA'] = max_ua

    # set index name to VARSET in order to join in combine
    ctabsum.index.name = 'VARSET'
    return ctabsum.astype(np.float)


def combine_evaluation_datasets(lda_ctabsum, posterior, vsel_x, merge_on='VARSET'):
    """Combine evaluation Datasets

    Merges class tabulation, posterior, and selected x variable data.

    :param lda_ctabsum: lda_ctabsum data frame
    :param posterior: posteriors data frame
    :param vsel_x: model variables data frame
    :param merge_on: name of column to merge inputs on. any other columns
    common to inputs will be repeated with deafult suffix

    """
    lda_ctabsum.reset_index(level=0, inplace=True)
    posterior.reset_index(level=0, inplace=True)

    ctab_posterior = pd.merge(posterior, lda_ctabsum,
                              how='inner', sort=True, on=[merge_on])
    combined = pd.merge(ctab_posterior, vsel_x, sort=True, on=[merge_on])

    ranked = rank_varset_assess(combined)
    columns = ['VARSET','OA','KHAT','MINPA','MAXPA','MINUA','MAXUA','NVAR','UERROR','MODELID','NMODELS','XVAR1','XVAR2','XVAR3','XVAR4','XVAR5','XVAR6','XVAR7','XVAR8','XVAR9','XVAR10','VARSETRANK']
    ranked = ranked.reindex_axis(columns, axis=1)

    return ranked
