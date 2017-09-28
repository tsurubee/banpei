import unittest
import pandas as pd
from banpei import hotelling


class TestHotelling(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('tests/test_data/davis.csv')
        self.data_1d = self.data['weight']

    def test_hotelling_1d(self):
        expected = 2
        result = hotelling.hotelling_1d(self.data_1d, 0.01)
        self.assertEqual(expected, len(result))


if __name__ == "__main__":
    unittest.main()
