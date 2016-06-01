import os
import shutil
import filecmp
from unittest import TestCase, expectedFailure

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
        lda_ctabsum = pd.read_csv('./tests/data/expected/lda_ctabsum.csv',
                                  index_col='VARSET')
        posterior = pd.read_csv('./tests/data/rlearn/POSTERIOR.csv',
                                index_col='VARSET')

        posterior.drop('NVAR', axis=1, inplace=True)
        assess = combine_evaluation_datasets(lda_ctabsum, posterior, vsel_x)
        assess.to_csv('./tests/output/lda_x_assess.csv')
        self.assertTrue(self.assert_file_same('lda_x_assess.csv'))

    @expectedFailure
    def test_combine_evaluation_datasets_old(self):
        # preserved for historical record, but underlying method and data was incorrect
        vsel_x = pd.read_csv('./tests/data/expected/vsel_x.csv')
        lda_ctabsum = pd.read_csv('./tests/data/expected/lda_ctabsum.csv', index_col='VARSET')
        posterior = pd.read_csv('./tests/data/rlearn/POSTERIOR.csv', index_col='VARSET')

        posterior.drop('NVAR', axis=1, inplace=True)
        assess = combine_evaluation_datasets(lda_ctabsum, posterior, vsel_x)
        # formatting and na_rep to match old output
        assess.to_csv('./tests/output/lda_x_assess_old.csv', index=False,
                      float_format='%.3f', na_rep='-1.000')
        self.assertTrue(self.assert_file_same('lda_x_assess_old.csv'))

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

if __name__ == '__main__':
    unittest.main()
