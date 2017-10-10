import unittest
import pandas as pd
from banpei.hotelling import Hotelling


class TestHotelling(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('tests/test_data/davis.csv')
        self.data_1d = self.data['weight']

    def test_hotelling(self):
        expected = 2
        model = Hotelling()
        results = model.detect(self.data_1d, 0.01)
        self.assertEqual(expected, len(results))


if __name__ == "__main__":
    unittest.main()
