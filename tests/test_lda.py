import os
import shutil
import filecmp
from unittest import TestCase

from pylearn.lda import cohens_khat, combine_evaluation_datasets

import pandas as pd


class TestLDAnalysis(TestCase):

    def setUp(self):
        self.output = './tests/output'
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def assert_file_same(self, filename):
        expected = os.path.join('./tests/data/expected', filename)
        actual = os.path.join('./tests/output', filename)
        return filecmp.cmp(expected, actual)

    def test_cohens_khat(self):
        ctabulation = pd.read_csv('./tests/data/rlearn/CTABULATION.csv')
        ctabsum = cohens_khat(ctabulation)

        ctabsum.to_csv('./tests/output/lda_ctabsum.csv', index=True,
                       index_label='VARSET', float_format='%.3f')

        self.assertTrue(self.assert_file_same('lda_ctabsum.csv'))

    def test_combine_evaluation_datasets(self):
        vsel_x = pd.read_csv('./tests/data/expected/vsel_x.csv')
        lda_ctabsum = pd.read_csv('./tests/data/expected/lda_ctabsum.csv', index_col='VARSET')
        posterior = pd.read_csv('./tests/data/rlearn/POSTERIOR.csv', index_col='VARSET')

        assess = combine_evaluation_datasets(lda_ctabsum, posterior, vsel_x)

        # Hack formatting and na_rep to match legacy output
        # the cause is missing values in POSTERIOR.csv from subselect R routine
        columns = ['VARSET','OA','KHAT','MINPA','MAXPA','MINUA','MAXUA','NVAR','UERROR','MODELID','NMODELS','XVAR1','XVAR2','XVAR3','XVAR4','XVAR5','XVAR6','XVAR7','XVAR8','XVAR9','XVAR10','VARSETRANK']
        assess = assess.reindex_axis(columns, axis=1)
        assess.to_csv('./tests/output/lda_x_assess.csv', index=False,
                      float_format='%.3f', na_rep='-1.000')

        self.assertTrue(self.assert_file_same('lda_x_assess.csv'))

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

if __name__ == '__main__':
    unittest.main()
