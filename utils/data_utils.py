import numpy as np
import time
from .generator_utils import reverse_index


def summarize(X):
    """To show the summary of a matrix.

    Parameters
    ----------
    X : ndarray, spmatrix
    """
    u_num, v_num = X.shape
    u_sum, v_sum = sum(X)
    u_min, u_max = min(u_sum), max(u_sum)
    v_min, v_max = min(v_sum), max(v_sum)
    u_mean, v_mean = mean(X)
    u_median, v_median = median(X)
    print("[I] Summary of matrix {}:".format(X.shape))
    print("[I]      num / min / median / mean / max")
    print("[I] rows {} / {} / {} / {:.1f} / {}".format(u_num, u_min, u_median, u_mean, u_max))
    print("[I] cols {} / {} / {} / {:.1f} / {}".format(v_num, v_min, v_median, v_mean, v_max))


def sum(X, axis=None):
    '''Row and column-wise sum.

    Parameters
    ----------
    X : ndarray, spmatrix
    axis : int, optional

    Returns
    -------
    result : tuple, ndarray
    '''
    sum_u = np.squeeze(np.array(X.sum(axis=1)))
    sum_v = np.squeeze(np.array(X.sum(axis=0)))
    result = (sum_u, sum_v)
    return result if axis is None else result[1-axis]


def mean(X, axis=None):
    '''Row and column-wise mean.

    Parameters
    ----------
    X : ndarray, spmatrix
    axis : int, optional

    Returns
    -------
    result : tuple, ndarray
    '''
    sum_u, sum_v = sum(X)
    result = (np.mean(sum_u), np.mean(sum_v))
    return result if axis is None else result[1-axis]


def median(X, axis=None):
    '''Row and column-wise median.

    Parameters
    ----------
    X : ndarray, spmatrix
    axis : int, optional

    Returns
    -------
    result : tuple, ndarray
    '''
    sum_u, sum_v = sum(X)
    result = (np.median(sum_u), np.median(sum_v))
    return result if axis is None else result[1-axis]


def sample(X, axis, factor_info=None, idx=None, n_samples=None, seed=None):
    '''Sample a matrix by its row or column. Update factor_info if provided.

    axis : int
        which dimension to down-sample.
        0, sample rows.
        1, sample columns.
    factor_info : list, tuple 
        factor_info for single matrix X or collective matrices Xs.
        for X, factor_info is a list of tuples.
        for Xs, factor_info is a tuple.
    idx:
        the indices to sample with.
    n_samples:
        randomly down-sample to this length.
    seed:
        seed for down-sampling.
    '''
    if idx is not None:
        print("[I] sampling axis {} with given indices".format(axis))
        assert X.shape[axis] >= len(idx), "[E] Target length exceeds the original."
    elif n_samples is not None:
        print("[I] Sampling axis {} to size {}".format(axis, n_samples))
        assert X.shape[axis] >= n_samples, "[E] Target length exceeds the original."
        
        seed = int(time.time()) if seed is None else seed
        rng = np.random.RandomState(seed)
        print("[I]   sampling seed    :", seed)
        
        idx = [True] * n_samples + [False] * (X.shape[axis] - n_samples)
        rng.shuffle(idx)

    print("[I]   sampling from    :", X.shape)
    X = X[idx, :] if axis == 0 else X[:, idx]
    print("[I]              to    :", X.shape)

    if isinstance(factor_info, list): # single matrix
        for i in [0, 1, 2]: # order, idmap, alias
            factor_info[axis][i] = factor_info[axis][i][idx]
        factor_info[axis][0] = sort_order(factor_info[axis][0])
    elif isinstance(factor_info, tuple): # collective matrices
        factor_info = list(factor_info)
        for i in [0, 1, 2]: # order, idmap, alias
            factor_info[i] = factor_info[i][idx]        
        factor_info[0] = sort_order(factor_info[0])
        factor_info = tuple(factor_info)
    return (idx, factor_info, X)


def sort_order(order):
    '''Fix the gap after down-sampling.

    E.g. [1, 6, 4, 2] will be turned into [0, 3, 2, 1].
    '''
    n = 0
    for i in range(max(order) + 1):
        if i in order:
            if isinstance(order, list):
                order[order.index(i)] = n
            elif isinstance(order, np.ndarray):
                order[order == i] = n
            n += 1

    return order









def get_settings(Xs, factors, factor_info, Us=None, all_factors=None):
    '''Get display settings, used in show_matrix() wrapper for CMF models

    factors: 
        may not contain factor relations for all matrices, e.g., when a part of matrices are used in validation and testing. all_factors is required when the Xs, factors and factor_info do not have all matrices and factors.
    all_factors:
        factors list that contains factor relations for all matrices, especially that the factor id starts from 0.

    a, b, f: factor id
        note that factor id may not be equal to the index in Us and factor_info, especially when the factor id does not start from 0.
        split_factor_list only accepts the compete factors list.
    '''
    all_factors = factors if all_factors is None else all_factors
    rows, cols = split_factor_list(all_factors)
    factor_list = get_factor_list(factors)

    settings = []

    for i, X in enumerate(Xs):
        # 1st and 2nd factor index
        a, b = factors[i]
        # reorder X by factor order
        a_info = factor_info[factor_list.index(a)]
        b_info = factor_info[factor_list.index(b)]
        a_order = reverse_index(a_info[0])
        b_order = reverse_index(b_info[0])
        X = X[a_order, :]
        X = X[:, b_order]
        # get row and column
        if a in rows and b in cols:
            r, c = rows.index(a), cols.index(b)
            settings.append((X, [r, c], f"Xs[{i}]"))
        elif a in cols and b in rows:
            r, c = rows.index(b), cols.index(a)
            settings.append((X.T, [r, c], f"Xs[{i}].T"))

    if Us is None:
        return settings

    for i, U in enumerate(Us):
        # factor index
        f = factor_list[i]
        # reorder factor
        f_info = factor_info[i]
        f_order = reverse_index(f_info[0])
        U = U[f_order, :]
        # get row and column
        if f in rows:
            r, c = rows.index(f), len(cols)
            settings.append((U, [r, c], f"Us[{i}]"))
        elif f in cols:
            r, c = len(rows), cols.index(f)
            settings.append((U.T, [r, c], f"Us[{i}].T"))
        
    return settings