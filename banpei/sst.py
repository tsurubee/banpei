import numpy as np
from banpei.base.model import Model


class SST(Model):
    def __init__(self, data, w):
        super().__init__(data)
        self.w = w

    def _extract_matrix(self, data, start, end, w):
        row = w
        column = end - start + 1
        matrix = np.empty((row, column))
        i = 0
        for t in range(start, end+1):
            matrix[:, i] = data[t-1:t-1+row]
            i += 1

        return matrix

    def detect(self, m=2, k=None, L=None):
        """
        Parameters
        ----------
        data : array_like
               Input array or object that can be converted to an array.
        w    : int
               Window size
        m    : int
               Number of basis vectors

        Returns
        -------
        Numpy array contains the degree of change.
        """
        # Set variables
        data = self.data
        w = self.w
        if k is None:
            k = w // 2
        if L is None:
            L = k // 2
        T = len(data)

        # Calculation range
        start_cal = k + w
        end_cal = T - L + 1

        # Calculate the degree of change
        change_scores = np.zeros(len(data))
        for t in range(start_cal, end_cal + 1):
            # Trajectory matrix
            start_tra = t - w - k + 1
            end_tra = t - w
            tra_matrix = self._extract_matrix(data, start_tra, end_tra, w)

            # Test matrix
            start_test = start_tra + L
            end_test = end_tra + L
            test_matrix = self._extract_matrix(data, start_test, end_test, w)

            # Singular value decomposition(SVD)
            U_tra, _, _  = np.linalg.svd(tra_matrix, full_matrices=False)
            U_test, _, _ = np.linalg.svd(test_matrix, full_matrices=False)
            U_tra_m  = U_tra[:, :m]
            U_test_m = U_test[:, :m]
            _, s, _ = np.linalg.svd(np.dot(U_tra_m.T, U_test_m), full_matrices=False)
            change_scores[t] = 1 - s[0] ** 2

        return change_scores
