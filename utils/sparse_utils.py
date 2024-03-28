from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, issparse, isspmatrix
import numpy as np
from typing import Union


def to_sparse(X, type='csr'):
    '''Convert to sparse matrix.

    Guide for choosing sparsity types:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    '''
    assert type in ['coo', 'csr', 'csc', 'lil'], "Matrix type not available"
    if type == 'coo':
        X = coo_matrix(X)
    elif type == 'csr':
        X = csr_matrix(X)
    elif type == 'csc':
        X = csc_matrix(X)
    elif type == 'lil': # LIst of Lists
        X = lil_matrix(X)
    return X


def to_dense(X, squeeze=False, keep_nan=False):
    '''Convert to dense array
    '''
    if keep_nan and issparse(X):
        rows, cols, values = to_triplet(X)
        X = np.empty(shape=X.shape)
        X.fill(np.nan)
        for i in range(len(values)):
            X[rows[i], cols[i]] = values[i]
    if issparse(X):
        X = X.toarray()
    elif isinstance(X, np.matrix):
        X = np.asarray(X)
    return X.squeeze() if squeeze else X


def to_triplet(X):
    '''Convert a dense or sparse matrix to a UIR triplet
    '''
    coo = coo_matrix(X)
    X = (np.asarray(coo.row, dtype='int'),
         np.asarray(coo.col, dtype='int'),
         np.asarray(coo.data, dtype='float'))
    return X

    
def check_sparse(X, sparse=None):
    if sparse == True and not issparse(X):
        return to_sparse(X)
    elif sparse == False and issparse(X):
        return to_dense(X)
    else:
        return X


def sparse_indexing(X, indices):
    type = X.getformat()
    coo = X.tocoo()
    r = coo.row[indices]
    c = coo.col[indices]
    v = coo.data[indices]
    X = coo_matrix((v, (r, c)), shape=X.shape)
    X = to_sparse(X, type=type)
    return X


def bool_to_index(x):
    bool_array = to_dense(x, squeeze=True).astype(bool)
    idx = np.arange(len(bool_array))
    idx = idx[bool_array]
    return idx


def index_to_bool(x):
    pass