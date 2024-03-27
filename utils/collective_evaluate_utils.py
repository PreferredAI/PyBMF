from .metrics import cover
import numpy as np
from .boolean_utils import matmul
import pandas as pd


def collective_cover(gt, pd, w, axis, starts=None):
    '''The collective wrapper for cover function.

    Parameters
    ----------
    gt : spmatrix
        The concatenated ground-truth matrix.
    pd : spmatrix
        The concatenated predicted matrix.
    w : list of float
        The allowed ratio of false positives in each matrices.
    axis : int in {0, 1}, default: None
        The dimension of the basis.
    starts : list of int
        The starting point of each matrix on the other dimension rather than the dimension of the basis.

    Returns
    -------
    scores : (n_submat, n_basis) array
        The scores of each basis over each submatrix.
    '''
    n_submat = len(w)
    assert len(starts) == n_submat + 1, "[E] Starting points and the number of sub-matrices don't match."

    scores = np.zeros((n_submat, gt.shape[1-axis]))
    for i in range(n_submat):
        a, b = starts[i], starts[i+1]
        s = cover(gt=gt[:, a:b] if axis else gt[a:b, :], 
                  pd=pd[:, a:b] if axis else pd[a:b, :], 
                  w=w[i], axis=axis)
        scores[i] = s

    return scores


def weighted_score(scores, weights):
    '''Weighted score(s) of `n` sets of scores.

    Parameters
    ----------
    scores : (n, k) array
    weights : (1, n) array

    Returns
    -------
    s : float or (1, k) array
    '''
    n = scores.shape[0]
    weights = np.array(weights).reshape(1, n)
    s = matmul(U=weights, V=scores)
    return s


def harmonic_score(scores):
    '''Harmonic score(s) of `n` sets of scores.

    Parameters
    ----------
    scores : (n, k) array

    Returns
    -------
    s : float or (1, k) array
    '''
    n, k = scores.shape
    s = np.zeros((1, k))
    # # debug
    # zero_counter = 0
    for i in range(k):
        if (scores[:, i] == 0).any():
            s[0, i] = 0
            # zero_counter += 1
        else:
            denom = (1 / scores[:, i]).sum(axis=0)
            s[0, i] = n / denom
    # if zero_counter > 0:
    #     print(f"[W] {zero_counter} out of {k} zeros encountered in harmonic score.")
    return s
    