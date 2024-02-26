from scipy.sparse import lil_matrix, vstack
import numpy as np
from .collective_utils import split_factor_list, get_factor_starts, get_factor_list
from .collective_display_utils import sort_matrices


def concat_factor_info(factor_info, factors):
    '''Concatenate factor_info of collective matrices Xs into a length-2 factor_info of a single matrix X.
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
    '''Concatenate collective matrices Xs into a single matrix X.

    Used in BaseData and some collective models.
    '''
    Xs_transpose, Xs_positions = sort_matrices(Xs, factors)
    row_starts, col_starts = get_factor_starts(Xs=Xs, factors=factors)
    X = lil_matrix(np.zeros((row_starts[-1], col_starts[-1])))
    for i in range(len(factors)):
        x = Xs_transpose[i] # x with transposition if necessary
        r, c = Xs_positions[i] # position in the concatenated matrix
        X[row_starts[r]:row_starts[r+1], col_starts[c]:col_starts[c+1]] = x # load the matrix
    return X

    
def concat_Us_into_U(Us, factors):
    '''Concatenate factors of collective matrices Us into a single pair of factors U.

    Used in some collective models.
    '''
    rows, cols = split_factor_list(factors=factors)
    k = Us[0].shape[1]
    U = [Us[i] for i in rows]
    V = [Us[i] for i in cols]
    U = vstack(U, format='lil')
    V = vstack(V, format='lil')
    return (U, V)


def split_X_into_Xs(X, factors, factor_starts):
    '''Split concatenated single matrix X into collective matrices Xs.

    Used in some collective models.
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
    '''Seperate concatenated factors (U, V) into collective factors Us.

    Used in some collective models.
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