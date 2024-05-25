import numpy as np
import time
import numbers
from scipy.sparse import isspmatrix
from .sparse_utils import sparse_indexing
from scipy.sparse import spmatrix
from .sparse_utils import to_sparse
from .boolean_utils import multiply
from .decorator_utils import ignore_warnings

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
    '''Return items or rows from X using indices

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
    '''
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
    

def binarize(X, threshold=0.5):
    '''To binarize a matrix. Also known as Heaviside step function.

    Parameters
    ----------
    X : float ndarray, spmatrix
    threshold : float, default: 0.5

    Returns
    -------
    result : int ndarray, spmatrix
    '''
    Y = (X > threshold).astype(int)
    if isinstance(X, spmatrix):
        Y = to_sparse(Y, type=X.format)
    return Y

@ignore_warnings
def sigmoid(X):
    '''Sigmoid function.
    '''
    X = X.astype(np.float64)
    Y = np.zeros(X.shape)
    Y[X >= 0] = 1.0 / (1.0 + np.exp( - X[X >= 0]) )
    Y[X < 0] = np.exp(X[X < 0]) / (1 + np.exp(X[X < 0]))
    return Y


def d_sigmoid(X):
    Y = sigmoid(X)
    Z = multiply(Y, 1 - Y)
    return Z