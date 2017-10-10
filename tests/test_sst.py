import unittest
import pandas as pd
from banpei.sst import SST


class TestSST(unittest.TestCase):

    def setUp(self):
        self.raw_data = pd.read_csv('tests/test_data/periodic_wave.csv')
        self.data = self.raw_data['y']

    def test_sst(self):
        model = SST()
        results = model.detect(self.data, 50)
        self.assertEqual(len(self.data), len(results))


if __name__ == "__main__":
    unittest.main()
