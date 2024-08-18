from itertools import accumulate
import numpy as np


def get_factor_list(factors):
    '''Get sorted factor list.

    Parameters
    ----------
    factors : list of int list
        List of factor id pairs, indicating the row and column factors of each matrix.
        Please follow the convention that factors are numbered consecutively and starting from 0.
        There must exist a matrix with its factors numbered as [0, 1].

    Returns
    -------
    factor_list : list
        List of sorted factor ids.
    '''    
    factor_list = []
    for f in factors:
        factor_list.extend(f)
    factor_list = sorted(list(set(factor_list)))
    return factor_list


def get_matrices(factors):
    '''List of related matrices given factors.

    This is the reversion of 'factors', the list of related factors given matrices.
    '''
    factor_list = get_factor_list(factors)
    matrices = []

    for f in factor_list:
        matrix_list = []
        for i, fs in enumerate(factors):
            if f in fs:
                matrix_list.append(i)
        matrices.append(matrix_list)
    return matrices


def get_factor_dims(Xs, factors):
    '''The dimensions of each factor.
    '''
    factor_list = get_factor_list(factors)
    matrices = get_matrices(factors)
    factor_dims = []
    for f in factor_list:
        m = matrices[f][0] # pick 1 related matrix
        d = factors[m].index(f) # 0 or 1
        dim = Xs[m].shape[d]
        factor_dims.append(dim)
    return factor_dims


def get_factor_starts(Xs, factors):
    '''The starting point of each factor when multiple factors Us are concatenated into a pair of row and column factor U.
    '''
    rows, cols = split_factor_list(factors=factors)
    factor_dims = get_factor_dims(Xs, factors)
    heights = [factor_dims[r] for r in rows]
    widths = [factor_dims[c] for c in cols]
    row_starts = [0] + list(accumulate(heights))
    col_starts = [0] + list(accumulate(widths))
    factor_starts = [row_starts, col_starts]
    return factor_starts


def get_dummy_factor_info(Xs, factors):
    '''Get dummy factor_info for collective matrices.
    '''
    factor_list = get_factor_list(factors)
    factor_dims = get_factor_dims(Xs, factors)
    factor_info = []
    for i, f in enumerate(factor_list):
        dim = factor_dims[i]
        f_order = np.arange(dim).astype(int)
        f_idmap = np.arange(dim).astype(int)
        f_alias = np.arange(dim).astype(str)
        f_info = (f_order, f_idmap, f_alias)
        factor_info.append(f_info)
    return factor_info


def split_factor_list(factors):
    '''Classify factors into row and column factors.

    Please follow the convention that factors are numbered consecutively and starting from 0.
    There must exist a matrix with its factors numbered as [0, 1].
    Factor 0 and those on the same side as 0 are regraded as row factors. Factor 1 and those on the same side as 1 are regraded as column factors. 

    List `f` stores the type of each factor with 0 for unclassified, 1 for row factor and 2 for column factor.
    '''
    factor_list = get_factor_list(factors)

    f = [0] * len(factor_list)
    f[0], f[1] = 1, 2
    for i in range(len(factors)):
        a, b = factors[i]
        if f[a] + f[b] != 3:
            f[a] = 3 - f[b] if f[a] == 0 else f[a]
            f[b] = 3 - f[a] if f[b] == 0 else f[b]
        
    row_factors = sorted([i for i, v in enumerate(f) if v == 1])
    col_factors = sorted([i for i, v in enumerate(f) if v == 2])
    return row_factors, col_factors
