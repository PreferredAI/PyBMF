from .collective_utils import split_factor_list


def sort_matrices(Xs, factors):
    '''Sort out matrices.
    
    Transpose the matrices when necessary and return the positions.
    '''
    rows, cols = split_factor_list(factors=factors)
    Xs_transpose = []
    Xs_positions = []
    for i in range(len(factors)):
        x = Xs[i]
        a, b = factors[i]
        needs_transpose = a in cols and b in rows
        if needs_transpose:
            x, a, b = x.T, b, a
        # position in the concatenated matrix
        r, c = rows.index(a), cols.index(b)
        Xs_transpose.append(x)
        Xs_positions.append([r, c])
    return Xs_transpose, Xs_positions


def get_settings(Xs, factors, Us=None):
    '''Get display settings.
    
    Used in the show_matrix() wrapper for CMF models.

    Parameters
    ----------
    Xs : list of spmatrix or ndarray
    factors : list of int list
    Us : list of spmatrix or ndarray, optional

    a, b, f: factor id
        note that factor id may not be equal to the index in Us and factor_info, especially when the factor id does not start from 0.
        split_factor_list only accepts the compete factors list.
    '''
    settings = []
    Xs_transpose, Xs_positions = sort_matrices(Xs, factors)
    for i, X in enumerate(Xs):
        x = Xs_transpose[i]
        r, c = Xs_positions[i]
        if X.shape == x.shape:
            settings.append((x, [r, c], f"Xs[{i}]"))
        elif X.shape == x.shape[::-1]:
            settings.append((x, [r, c], f"Xs[{i}].T"))
        else:
            raise ValueError("X should be either transposed or not.")

    if Us is None:
        return settings
    rows, cols = split_factor_list(factors)
    for i, U in enumerate(Us):
        if i in rows:
            r, c = rows.index(i), len(cols)
            settings.append((U, [r, c], f"Us[{i}]"))
        elif i in cols:
            r, c = len(rows), cols.index(i)
            settings.append((U.T, [r, c], f"Us[{i}].T"))
        
    return settings