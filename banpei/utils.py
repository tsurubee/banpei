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

def tridiagonalize_by_lanczos(P, m, k):
    """
    Tridiagonalize matrix by lanczos method
    Parameters
    ----------
    P : numpy array
        Target matrix
    q : numpy array
        Initial vector
    k : int
        Size of the tridiagonal matrix

    Returns
    -------
    T : numpy array
        tridiagonal matrix
    """
    # Initialize variables
    T = np.zeros((k, k))
    r0 = m
    beta0 = 1
    q0 = np.zeros(m.shape)

    for i in range(k):
        q1 = r0 / beta0
        C = np.dot(P, q1)
        alpha1 = np.dot(q1, C)
        r1 = C - alpha1 * q1 - beta0 * q0
        beta1 = np.linalg.norm(r1)

        T[i, i] = alpha1
        if i + 1 < k:
          T[i, i + 1] = beta1
          T[i + 1, i] = beta1

        q0 = q1
        beta0 = beta1
        r0 = r1

    return T

def tridiag_eigen(T, iter_num=1, tol=1e-3):
    """
    Calculate eigenvalues and eigenvectors of tridiagonal matrix
    Parameters
    ----------
    P : numpy array
        Target matrix (tridiagonal)
    iter_num : int
        Number of iterations
    tol : float
        Stop iteration if the target matrix converges to a diagonal matrix with acceptable tolerance `tol`

    Returns
    -------
    eigenvalue : numpy array
                 Calculated eigenvalues
    eigenvectors : numpy array
                 Calculated eigenvectors
    """
    eigenvectors = np.identity(T.shape[0])

    for i in range(iter_num):
        Q, R = tridiag_qr_decomposition(T)
        T = np.dot(R, Q)
        eigenvectors = np.dot(eigenvectors, Q)
        eigenvalue = np.diag(T)
        if np.all((T - np.diag(eigenvalue) < tol)):
            break

    return eigenvalue, eigenvectors

def tridiag_qr_decomposition(T):
    """
    QR decomposition for a tridiagonal matrix
    Ref. http://www.ericmart.in/blog/optimizing_julia_tridiag_qr
    Parameters
    ----------
    T : numpy array
        Target matrix (tridiagonal)

    Returns
    -------
    Qt.T : numpy array
    R    : numpy array
    """
    R = T.copy()
    Qt = np.eye(T.shape[0])

    for i in range(T.shape[0] - 1):
        u = householder(R[i:i + 2, i])
        M = np.outer(u, u)
        R[i:i + 2, :(i + 3)] -= 2 * np.dot(M, R[i:i + 2, :(i + 3)])
        Qt[i:i + 2, :(i + 3)] -= 2 * np.dot(M, Qt[i:i + 2, :(i + 3)])

    return Qt.T, R

def householder(x):
    """
    Householder projection for vector.
    Parameters
    ----------
    x : numpy array
        Target vector

    Returns
    -------
    x : numpy array
    """
    x[0] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
    x = x / np.linalg.norm(x)
    return x
