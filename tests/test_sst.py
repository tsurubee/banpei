import unittest
import pandas as pd
from banpei import sst


class TestSst(unittest.TestCase):

    def setUp(self):
        self.raw_data = pd.read_csv('tests/test_data/periodic_wave.csv')
        self.data = self.raw_data['y']


    def test_sst(self):
        result = sst.sst(self.data, 50, 2)
        self.assertEqual(len(self.data), len(result))


if __name__ == "__main__":
    unittest.main()
