from __future__ import division
import pandas as pd
import numpy as np


def overall_accuracy(confusion_matrix):
    """Calculate overall accuracy

    Overall accuracy is the proportion of correctly classified observations.

    :param confusion_matrix:
    :rtype: float

    """
    total = confusion_matrix.sum(axis=1).sum(axis=0)
    correct = np.trace(confusion_matrix)

    return correct/total


def producer_accuracy(confusion_matrix):
    """Calculate producer accuracy

    For a class, A, producer accuracy is the ratio of observations correctly
    classified as A to all observations which are actually class A.

    .. warning:: Assumes confusion_matrix columns represent observations and
    rows represent predictions

    :param confusion_matrix:
    :return: 1 row data frame with columns corresponding to classes and values
    corresponding to producer accuracy

    """
    diags = np.diag(confusion_matrix)
    column_totals = confusion_matrix.sum(axis=1).values
    df = pd.DataFrame(columns=confusion_matrix.columns.values, index=[0])
    df.ix[0] = diags/column_totals
    return df


def user_accuracy(confusion_matrix):
    """Calculate producer accuracy

    For a class, A, user accuracy is the ratio of observations correctly
    classified as A to all observations classified as A.

    .. warning:: Assumes confusion_matrix columns represent observations and
    rows represent predictions

    :param confusion_matrix:
    :return: 1 row data frame with columns corresponding to classes and values
    corresponding to user accuracy

    """
    diags = np.diag(confusion_matrix)
    row_totals = confusion_matrix.sum(axis=0).values
    df = pd.DataFrame(columns=confusion_matrix.index.values, index=[0])
    df.ix[0] = diags/row_totals
    return df


def khat(confusion_matrix):
    """Calculate Cohen's KHat

    Calculate the KHAT Coefficient (Cohen, 1960). KHAT is a measure of
    classification accuracy which accounts for change agreement in reference
    data and classifications.

    .. warning:: Assumes confusion_matrix columns represent observations and
    rows represent predictions

    :param confusion_matrix:
    :rtype: float

    """
    total = confusion_matrix.sum(axis=1).sum(axis=0)
    diag_sum = np.trace(confusion_matrix)
    row_totals = confusion_matrix.sum(axis=1).values
    column_totals = confusion_matrix.sum(axis=0).values

    col_tot_row_tot_prod = np.sum((row_totals * column_totals))
    numerator = (total*diag_sum) - col_tot_row_tot_prod
    denominator = (total*total) - col_tot_row_tot_prod

    return numerator/denominator
