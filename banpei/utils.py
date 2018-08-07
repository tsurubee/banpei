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


def _rolling_window(a, window):
    """
    Usage:
    a = np.random.rand(30, 5)
    for 2d array:
        roll aling axis=0: rolling_window(a.T, 3).transpose(1, 2, 0)
        roll along axis=1: rolling_window(a, 3).transpose(1, 0, 2)
    for 3d array:
        roll along height(axis=0): rolling_window(a.transpose(2, 1, 0), 3).transpose(2, 3, 1, 0)
        roll along width(axis=1): rolling_window(a, 3).transpose(2, 0, 1, 3)
        roll along depth(axis=2): rolling_window(a.transpose(0, 2, 1), 3).transpose(3, 0, 2, 1)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(arr, window, axis=0):
    if arr.ndim == 1:
        return _rolling_window(arr, window)
    elif arr.ndim == 2:
        if axis == 0:
            return _rolling_window(arr.T, window).transpose(1, 2, 0)
        elif axis == 1:
            return _rolling_window(arr, window).transpose(1, 0, 2)
        else:
            raise Exception('AxisError: axis {} is out of bounds for array of dimension {}'.format(axis, arr.ndim))
    elif arr.ndim == 3:
        if axis == 0:
            return _rolling_window(arr.transpose(0, 2, 1), window).transpose(3, 0, 2, 1)
        elif axis == 1:
            return _rolling_window(arr, window).transpose(2, 0, 1, 3)
        elif axis == 2:
            return _rolling_window(arr.transpose(2, 1, 0), window).transpose(2, 3, 1, 0)
        else:
            raise Exception('AxisError: axis {} is out of bounds for array of dimension {}'.format(axis, arr.ndim))
    else:
        return _rolling_window(arr, window)
