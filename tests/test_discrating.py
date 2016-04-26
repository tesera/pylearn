import os
import shutil
from unittest import TestCase

import pandas as pd

from pylearn.discrating import predict


class TestDiscRating(TestCase):

    def setUp(self):
        self.output = './tests/output'
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def test_predict(self):
        args = {
            'xy': pd.read_csv('./tests/data/data_xy.csv'),
            'x_filtered': pd.read_csv('./tests/data/data_xy.csv'),
            'dfunct': pd.read_csv('./tests/data/expected/lda_x_dfunct.csv'),
            'varset': 18,
            'yvar': 'CLPRDP',
            'idf': pd.read_csv('./tests/data/data_idf.csv'),
        }

        args['idf'] = args['idf'][args['idf']['CITY'] == 'BUSYTOWN']

        forecasts = predict(**args)

        expected_columns = ['VAR1091', 'VAR1092', 'VAR197', 'VAR2', 'VAR466', 'VAR477',
        'VAR544', 'VAR775', 'VAR913', 'VAR915', 'ZSCORE', 'ZTRANS', 'ODDS', 'p', 'HISTORICAL',
        'L2020', 'L2050', 'U2020', 'U2050', 'ZOU', 'CHK']

        self.assertListEqual(list(forecasts.columns.values), expected_columns)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

if __name__ == '__main__':
    unittest.main()
