import numpy as np
from .common import get_rng


def shuffle_by_dim(X, dim, seed=None, rng=None):
    '''Shuffle a matrix by dimension

    dim:
        0, shuffle by rows
        1, shuffle by columns

    same as:
        np.take(X, idx, axis=dim, out=X)
    '''
    rng = get_rng(seed, rng)
    idx = rng.rand(X.shape[dim]).argsort()
    X = X[idx, :] if dim == 0 else X[:, idx]
    return idx, X, rng


def shuffle_matrix(X, seed=None, rng=None):
    '''Shuffle a matrix
    '''
    rng = get_rng(seed, rng)
    U_idx, X, rng = shuffle_by_dim(X=X, dim=0, rng=rng)
    V_idx, X, rng = shuffle_by_dim(X=X, dim=0, rng=rng)
    return U_idx, V_idx, X, rng


def add_noise(X, noise, seed=None, rng=None):
    '''Add noise to a matrix.

    Parameters
    ----------
    X : ndarray, spmatrix
    noise : list of 2 float in [0, 1]
        Probabilities for false negative (p_pos) and false positive (p_neg).
    seed : optional
        Random seed.
    rng : optional
        Random number generator.
    '''
    rng = get_rng(seed, rng)
    p_pos, p_neg = noise
    n = rng.binomial(size=X.shape, n=1, p=p_pos)
    X = np.maximum(X - n, 0)
    n = rng.binomial(size=X.shape, n=1, p=p_neg)
    X = np.minimum(X + n, 1)
    return X, rng

    
def reverse_index(idx):
    '''Reverse index

    Example
    -------
    idx = np.array([0, 1, 2, 4, 5, 3])
    inv = reverse_index(idx) # inv = [0, 1, 2, 5, 3, 4]
    '''
    inv = [np.where(idx==i)[0][0] for i in range(len(idx))]
    return inv


