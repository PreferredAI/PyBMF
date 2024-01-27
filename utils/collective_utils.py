from itertools import accumulate
from scipy.sparse import lil_matrix, vstack
import numpy as np


def get_factor_list(factors):
    '''Get sorted factor list
    '''
    factor_list = []
    for f in factors:
        factor_list.extend(f)
    factor_list = sorted(list(set(factor_list)))
    return factor_list


def get_matrices(factors):
    '''Related matrices of a factor

    This is the reverse of 'factors', the related factors of a matrix.
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
    factor_list = get_factor_list(factors)
    matrices = get_matrices(factors)
    factor_dims = []
    for i, f in enumerate(factor_list):
        # todo: 
        # i and f should be the same when factors are properly numbered
        m = matrices[i][0]
        d = factors[m].index(f) # 0 or 1
        dim = Xs[m].shape[d]
        factor_dims.append(dim)
    return factor_dims


def get_factor_starts(Xs, factors):
    rows, cols = split_factor_list(factors=factors)
    factor_dims = get_factor_dims(Xs, factors)
    heights = [factor_dims[r] for r in rows]
    widths = [factor_dims[c] for c in cols]
    row_starts = [0] + list(accumulate(heights))
    col_starts = [0] + list(accumulate(widths))
    factor_starts = [row_starts, col_starts]
    return factor_starts


def get_dummy_factor_info(Xs, factors):
    """Get dummy factor_info

    For collective matrices.
    """
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
    '''Split factors into row and column factors

    The factor_list must begin with 0.
    We regard 0 as the rows of Xs[0], and 1 as the columns of Xs[0].
    '''
    factor_list = get_factor_list(factors)

    f = [0] * len(factor_list)
    f[0], f[1] = 1, 2 # 1 for row, 2 for column
    for i in range(len(factors)):
        a, b = factors[i]
        if f[a] + f[b] != 3:
            f[a] = 3 - f[b] if f[a] == 0 else f[a]
            f[b] = 3 - f[a] if f[b] == 0 else f[b]
        
    rows = sorted([i for i, v in enumerate(f) if v == 1])
    cols = sorted([i for i, v in enumerate(f) if v == 2])
    return (rows, cols)


# transformations


def concat_factor_info(factor_info, factors):
    '''Concatenate factor_info of collective matrices Xs into a length-2 factor_info of single matrix X
    '''
    rows, cols = split_factor_list(factors=factors)
    concat_factor_info = [] # [(row_order, row_idmap, row_alias), (col_order, col_idmap, col_alias)]
    for side in [rows, cols]:
        for i, s in enumerate(side):
            if i == 0:
                order, idmap, alias = factor_info[s]
            else:
                order = np.append(order, factor_info[s][0] + order.max() + 1) # preserve order
                idmap = np.append(idmap, factor_info[s][1])
                alias = np.append(alias, factor_info[s][2])
        concat_factor_info.append((order, idmap, alias))
    return concat_factor_info


def concat_Xs_into_X(Xs, factors):
    '''Concatenate collective matrices Xs into a single matrix X

    This is used in BaseData and some collective models.
    Can also create concatenated factor_info.
    '''
    rows, cols = split_factor_list(factors=factors)
    row_starts, col_starts = get_factor_starts(Xs=Xs, factors=factors)
    X = lil_matrix(np.zeros((row_starts[-1], col_starts[-1])))
    for i in range(len(factors)):
        x = Xs[i]
        a, b = factors[i]
        if a in cols and b in rows: # to transpose
            x, a, b = x.T, b, a
        # position in the concatenated matrix
        r, c = rows.index(a), cols.index(b)
        # fill in the concatenated matrix
        X[row_starts[r]:row_starts[r+1], col_starts[c]:col_starts[c+1]] = x
    return X

    
def concat_Us_into_U(Us, factors):
    '''Concatenate factors of collective matrices Us into a single pair of factors U

    This is used in some collective models.
    '''
    rows, cols = split_factor_list(factors=factors)
    k = Us[0].shape[1]
    U = [Us[i] for i in rows]
    V = [Us[i] for i in cols]
    U = vstack(U, format='lil')
    V = vstack(V, format='lil')
    return (U, V)


def split_X_into_Xs(X, factors, factor_starts):
    '''Seperate concatenated single matrix X into collective matrices Xs

    This is used in some collective models.
    '''
    rows, cols = split_factor_list(factors=factors)
    row_starts, col_starts = factor_starts

    Xs = [None] * len(factors)
    for i in range(len(factors)):
        a, b = factors[i]
        needs_transpose = a in cols and b in rows
        if needs_transpose:
            a, b = b, a
        # position in the concatenated matrix
        r, c = rows.index(a), cols.index(b)
        # sub-matrix
        x = X[row_starts[r]:row_starts[r+1], col_starts[c]:col_starts[c+1]]
        if needs_transpose:
            Xs[i] = x.T
        else:
            Xs[i] = x
    return Xs


def split_U_into_Us(U, V, factors, factor_starts):
    '''Seperate concatenated factors (U, V) into collective factors Us

    This is used in some collective models.
    '''
    factor_list = get_factor_list(factors=factors)
    rows, cols = split_factor_list(factors=factors)
    row_starts, col_starts = factor_starts

    Us = [None] * len(factor_list)
    for i in factor_list:
        if i in rows:
            r = rows.index(i)
            Us[i] = U[row_starts[r]:row_starts[r+1], :]
        elif i in cols:
            c = cols.index(i)
            Us[i] = V[col_starts[c]:col_starts[c+1], :]
    return Us
