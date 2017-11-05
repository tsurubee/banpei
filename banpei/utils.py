import numpy as np


def power_method(A, iter_num=1):
    """
    Calculate the first singular vector/value of a target matrix based on the power method.
    Parameters
    ----------
    A : numpy array
        Target matrix
    iter_num : int
               Number of iterations

    Returns
    -------
    u : numpy array
        first left singular vector of A
    s : float
        first singular value of A
    v : numpy array
        first right singular vector of A
    """
    # set initial vector q
    q = np.random.normal(size=A.shape[1])
    q = q / np.linalg.norm(q)

    for i in range(iter_num):
        q = np.dot(np.dot(A.T, A), q)

    v = q / np.linalg.norm(q)
    Av = np.dot(A, v)
    s = np.linalg.norm(Av)
    u = Av / s

    return u, s, v
