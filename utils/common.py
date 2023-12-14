import numpy as np
import time
import numbers
from scipy.sparse import isspmatrix
from .sparse_utils import sparse_indexing


def get_rng(seed, rng):
    '''Get random number generator
    '''
    if isinstance(rng, np.random.RandomState):
        print("[I] Using RandomState.")
        return rng
    if isinstance(seed, (numbers.Integral, np.integer)):
        print("[I] Using seed   :", seed)
        return np.random.RandomState(seed)
    else:
        seed = int(time.time())
        print("[I] Using seed   :", seed)
        return np.random.RandomState(seed)


def safe_indexing(X, indices):
    """Return items or rows from X using indices

    Allows simple indexing of lists or arrays.
    Modified from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis
    """
    if hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            if isspmatrix(X):
                return sparse_indexing(X, indices=indices)
            else:
                return X[indices]
    else:
        return [X[idx] for idx in indices]


def step_function(X, threshold):
    '''Heaviside step function
    '''
    X[X >= threshold] = 1
    X[X < threshold] = 0
    return X


def sigmoid_function(X, lamda=None):
    '''Sigmoid function
    '''
    if lamda is None:
        X = 1 / (1 + np.exp(-X))
    else:
        X = 1 / (1 + np.exp(-lamda * X))
    return X