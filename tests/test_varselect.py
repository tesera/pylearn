import os
import shutil
import filecmp
from unittest import TestCase

import pandas as pd

from pylearn.varselect import count_xvars, rank_xvars, extract_xvar_combos, remove_high_corvar


class TestVariableSelect(TestCase):

    def setUp(self):
        self.output = './tests/output'
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def assert_file_same(self, filename):
        expected = os.path.join('./tests/data/expected', filename)
        actual = os.path.join('./tests/output', filename)
        return filecmp.cmp(expected, actual)

    def test_count_xvars(self):
        vsel_xy_config = pd.read_csv('./tests/data/vsel_xy_config.csv')
        count = count_xvars(vsel_xy_config)
        self.assertEqual(count, 708)

    def test_rank_xvars(self):
        varselect = pd.read_csv('./tests/data/rlearn/VARSELECT.csv')
        ranks = rank_xvars(varselect)

        columns = ranks.columns.values.tolist()
        self.assertListEqual(columns, ['VARNAME', 'IMPORTANCE', 'P', 'RANK'])

    def test_extract_xvar_combos(self):
        varselect = pd.read_csv('./tests/data/rlearn/VARSELECT.csv')
        vsel_x, vsel_uniquevar = extract_xvar_combos(varselect)

        vsel_x.to_csv('./tests/output/vsel_x.csv')
        vsel_uniquevar.to_csv('./tests/output/vsel_uniquevar.csv')

        expected_files = ['vsel_x.csv', 'vsel_uniquevar.csv']

        for expected_file in expected_files:
            self.assertTrue(self.assert_file_same(expected_file))

    def test_remove_high_corvar(self):
        varrank = pd.read_csv('./tests/data/vsel_varrank.csv')
        vsel_xy_config = pd.read_csv('./tests/data/vsel_xy_config.csv')
        ucorcoef = pd.read_csv('./tests/data/rlearn/UCORCOEF.csv')

        adjusted = remove_high_corvar(varrank, vsel_xy_config, ucorcoef)

        VAR466 = adjusted[adjusted['VARNAME'] == 'VAR466']['XVARSEL'].item()
        VAR915 = adjusted[adjusted['VARNAME'] == 'VAR915']['XVARSEL'].item()
        VAR913 = adjusted[adjusted['VARNAME'] == 'VAR913']['XVARSEL'].item()
        self.assertEqual(VAR466, 'N')
        self.assertEqual(VAR915, 'X')
        self.assertEqual(VAR913, 'N')

    def test_remove_high_corvar_fake_data(self):
        varrank = pd.DataFrame(data={
            'VARNAME': pd.Series(['included', 'excluded']),
            'IMPORTANCE': pd.Series([1,0])
        })
        vsel_xy_config = pd.DataFrame(data={
            'VARNAME': pd.Series(['included', 'excluded']),
            'XVARSEL': pd.Series(['X', 'X'])
        })
        ucorcoef = pd.DataFrame(data={
            'VARNAME1': pd.Series(['included', 'included', 'excluded', 'excluded']),
            'VARNAME2': pd.Series(['included', 'excluded', 'excluded', 'included']),
            'CORCOEF': pd.Series([1, .9, 1, .9])
        })

        adjusted = remove_high_corvar(varrank, vsel_xy_config, ucorcoef)

        excluded = adjusted[adjusted['VARNAME'] == 'excluded']['XVARSEL'].item()
        included = adjusted[adjusted['VARNAME'] == 'included']['XVARSEL'].item()
        original_included = vsel_xy_config[
            vsel_xy_config['VARNAME'] == 'included'
        ]['XVARSEL'] .item()
        self.assertEqual(excluded, 'N')
        self.assertEqual(included, original_included)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

if __name__ == '__main__':
    unittest.main()
