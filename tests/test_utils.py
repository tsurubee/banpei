import unittest
import numpy as np
from banpei.utils import power_method


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.A = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]])

    def test_power_method(self):
        u, s, v = power_method(self.A, iter_num=1)
        self.assertEqual(len(u), self.A.shape[0])
        self.assertEqual(len(v), self.A.shape[1])

if __name__ == "__main__":
    unittest.main()
