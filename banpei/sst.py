import numpy as np


def extract_matrix(data, start, end, w):
    row = w
    column = end - start + 1
    matrix = np.empty((row, column))
    i = 0
    for t in range(start, end+1):
        matrix[:, i] = data[t-1:t-1+row]
        i += 1

    return matrix


def sst(data, w, m):
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
    # initialize variables
    k = w // 2
    L = k // 2
    T = len(data)
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # calculation range
    start_cal = k + w
    end_cal = T - L + 1

    change_scores = np.zeros(len(data))
    for t in range(start_cal, end_cal + 1):
        # trajectory matrix
        start_tra = t - w - k + 1
        end_tra = t - w
        tra_matrix = extract_matrix(data, start_tra, end_tra, w)

        # test matrix
        start_test = start_tra + L
        end_test = end_tra + L
        test_matrix = extract_matrix(data, start_test, end_test, w)

        # singular value decomposition(SVD)
        U1, s1, V1 = np.linalg.svd(tra_matrix, full_matrices=True)
        U2, s2, V2 = np.linalg.svd(test_matrix, full_matrices=True)
        U1_m = U1[:, 0:m]
        U2_m = U2[:, 0:m]
        U, s, V = np.linalg.svd(np.dot(U1_m.T, U2_m), full_matrices=True)
        change_scores[t] = 1 - s[0] ** 2

    return change_scores