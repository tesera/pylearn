import os
import shutil
from unittest import TestCase
import filecmp

import pandas as pd
import numpy as np

from pylearn.varset import *


class TestLdaVarset(TestCase):

    def setUp(self):
        self.output = './tests/output'
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def assert_file_same(self, filename):
        expected = os.path.join('./tests/data/expected', filename)
        actual = os.path.join('./tests/output', filename)
        return filecmp.cmp(expected, actual)

    def test_rank_varset_with_default_coefficient(self):
        rank = rank_varset({'KHAT': '0.248', 'NVAR': '20'})
        self.assertEqual(rank, 0.16397765363128491)

    def test_rank_varset_with_coefficient(self):
        rank = rank_varset({'KHAT': '0.248', 'NVAR': '20'}, 100)
        self.assertEqual(rank, 0.05762025316455696)

    def test_rank_varset(self):
        lda_x_assess = pd.read_csv('./tests/data/expected/lda_x_assess.csv')
        lda_x_assess_ranked = rank_varset_assess(lda_x_assess)

        self.assertTrue('VARSETRANK' in lda_x_assess_ranked.columns.values)

    def test_get_xvars(self):
        dfunct = pd.read_csv('./tests/data/expected/lda_x_dfunct.csv')
        varset = get_xvars(dfunct, 18)
        expected = ['VAR1091', 'VAR1092', 'VAR197', 'VAR2', 'VAR466', 'VAR477', 'VAR544', 'VAR775', 'VAR913', 'VAR915']

        self.assertListEqual(varset.tolist(), expected)

    def test_get_xvars_without_varset(self):
        dfunct = pd.read_csv('./tests/data/expected/lda_x_dfunct.csv')
        varset = get_xvars(dfunct)
        expected = ['VAR1091', 'VAR1092', 'VAR11', 'VAR197', 'VAR2', 'VAR20', 'VAR445', 'VAR465', 'VAR466', 'VAR477',
        'VAR544', 'VAR647', 'VAR775', 'VAR80', 'VAR872', 'VAR913', 'VAR915', 'VAR931']

        self.assertListEqual(varset.tolist(), expected)

    def test_get_xy_summary_columns_names(self):
        xy = pd.read_csv('./tests/data/data_xy.csv')
        dfunct = pd.read_csv('./tests/data/expected/lda_x_dfunct.csv')

        summary = get_xy_summary(xy, dfunct, 'VAR47')
        names = ['ATTR', 'MEAN', 'VAR47']
        self.assertListEqual(list(summary.columns.values), names)

    def test_get_rloadrank_column_names(self):
        xy = pd.read_csv('./tests/data/data_xy.csv')
        dfunct = pd.read_csv('./tests/data/expected/lda_x_dfunct.csv')

        rloadrank = get_rloadrank(xy, dfunct, 'VAR47')
        expected = ['VARSET3', 'VARNAMES3', 'FUNCLABEL3', 'DFCOEF3', 'Y0', 'Y1', 'Z0', 'Z1','DFCOEF3_ADJ', 'Z0_ADJ', 'Z1_ADJ','SD', 'B*', 'LOADING', 'RL', 'RANK']
        self.assertListEqual(list(rloadrank.axes[1]), expected)

    def test_get_avgp(self):
        xy = pd.read_csv('./tests/data/data_xy.csv')
        xvars = ['VAR1091', 'VAR1092', 'VAR197', 'VAR464']

        avgp = get_avgp(xy, xvars)

        index = avgp.index.get_level_values(0).unique()
        self.assertListEqual(list(index), xvars)

    def test_export_param(self):
        dfunct = pd.read_csv('./tests/data/expected/lda_x_dfunct.csv')
        param = get_param(dfunct, 18)

        expected_columns = ['PARAMNAME','EQUATION','PARAMVALUE','ATTR','DESCRIPTION']
        self.assertListEqual(list(param.columns.values), expected_columns)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

if __name__ == '__main__':
    unittest.main()
