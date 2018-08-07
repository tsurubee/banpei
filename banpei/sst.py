import numpy as np
from banpei.base.model import BaseModel
from banpei.utils import rolling_window

class SST(BaseModel):
    def __init__(self, w, m=2, k=None, L=None):
        """
        Parameters
        ----------
        w    : int
               Window size
        m    : int
               Number of basis vectors
        k    : int
               Number of columns for the trajectory and test matrices
        L    : int
               Lag time
        """
        self.w = w
        self.m = m
        if k is None:
            self.k = self.w // 2
        else:
            self.k = k
        if L is None:
            self.L = self.k // 2
        else:
            self.L = L

    def detect(self, data):
        """
        Batch mode detection

        Parameters
        ----------
        data : array_like
               Input array or object that can be converted to an array.

        Returns
        -------
            Numpy array contains the degree of change.
            The size of Numpy array is the same as input array.
        """
        # Set variables
        data = self.convert_to_nparray(data)
        T = len(data)

        # Check the size of input data
        if not len(data) > self.L + self.w + self.k - 2:
            raise ValueError("Input data is too small.")

        # Calculation range
        start_cal = self.k + self.w
        end_cal = T - self.L + 1

        # Calculate the degree of change
        change_scores = np.zeros(len(data))
        for t in range(start_cal, end_cal + 1):
            # Trajectory matrix
            start_tra = t - self.w - self.k + 1
            end_tra = t - self.w
            tra_matrix = self._extract_matrix(data, start_tra, end_tra, self.w)

            # Test matrix
            start_test = start_tra + self.L
            end_test = end_tra + self.L
            test_matrix = self._extract_matrix(data, start_test, end_test, self.w)

            # Calculate the score by singular value decomposition(SVD)
            change_scores[t] = self._calculate_score(tra_matrix, test_matrix)

        return change_scores

    def stream_detect(self, data):
        """
        Stream mode detection for live monitoring.

        Parameters
        ----------
        data : array_like
               Input array or object that can be converted to an array.

        Returns
        -------
        tuple: (score, delay)
              score means the degree of change in the latest we can calculate.
              delay means the time lag between the latest data point and calculation point.
        """
        # Set variables
        data = self.convert_to_nparray(data)
        T = len(data)

        # Check the size of input data
        if not len(data) > self.L + self.w + self.k - 2:
            return 0

        # Calculation range
        t = T - self.L + 1

        # Trajectory matrix
        start_tra = t - self.w - self.k + 1
        end_tra = t - self.w
        tra_matrix = self._extract_matrix(data, start_tra, end_tra, self.w)

        # Test matrix
        start_test = start_tra + self.L
        end_test = end_tra + self.L
        test_matrix = self._extract_matrix(data, start_test, end_test, self.w)

        # Calculate the score by singular value decomposition(SVD)
        score = self._calculate_score(tra_matrix, test_matrix)

        return score

    def _calculate_score(self, tra_matrix, test_matrix):
        U_tra, _, _ = np.linalg.svd(tra_matrix, full_matrices=False)
        U_test, _, _ = np.linalg.svd(test_matrix, full_matrices=False)
        U_tra_m = U_tra[:, :self.m]
        U_test_m = U_test[:, :self.m]
        s = np.linalg.svd(np.dot(U_tra_m.T, U_test_m), full_matrices=False, compute_uv=False)
        return 1 - s[0]

    def _extract_matrix(self, data, start, end, w):
        matrix = rolling_window(data[start - 1:end - 1 + w], w).T
        return matrix
