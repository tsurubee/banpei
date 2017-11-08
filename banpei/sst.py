import numpy as np
from banpei.base.model import Model


class SST(Model):
    def __init__(self):
        pass

    def detect(self, data, w, m=2, k=None, L=None):
        """
        Parameters
        ----------
        data : array_like
               Input array or object that can be converted to an array.
        w    : int
               Window size
        m    : int
               Number of basis vectors
        k    : int
               Number of columns for the trajectory and test matrices
        L    : int
               Lag time

        Returns
        -------
        Numpy array contains the degree of change.
        """
        # Set variables
        data = self.convert_to_nparray(data)
        if k is None:
            k = w // 2
        if L is None:
            L = k // 2
        T = len(data)

        # Check the size of input data
        if not len(data) > L + w + k - 2:
            raise ValueError("Input data is to small.")

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

            # Calculate the score by singular value decomposition(SVD)
            change_scores[t] = self._calculate_score(tra_matrix, test_matrix, m)

        return change_scores

    def _calculate_score(self, tra_matrix, test_matrix, m):
        U_tra, _, _ = np.linalg.svd(tra_matrix, full_matrices=False)
        U_test, _, _ = np.linalg.svd(test_matrix, full_matrices=False)
        U_tra_m = U_tra[:, :m]
        U_test_m = U_test[:, :m]
        s = np.linalg.svd(np.dot(U_tra_m.T, U_test_m), full_matrices=False, compute_uv=False)
        return 1 - s[0]

    def _extract_matrix(self, data, start, end, w):
        row = w
        column = end - start + 1
        matrix = np.empty((row, column))
        i = 0
        for t in range(start, end+1):
            matrix[:, i] = data[t-1:t-1+row]
            i += 1

        return matrix
