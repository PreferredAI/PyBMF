from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dok_matrix, lil_matrix, issparse
import numpy as np


def to_sparse(X, type='csr'):
    '''Convert to sparse matrix
    '''
    assert type in ['coo', 'csr', 'csc', 'dok', 'lil'], "Not an available sparse matrix format"
    if type == 'coo':
        X = coo_matrix(X)
    elif type == 'csr':
        X = csr_matrix(X)
    elif type == 'csc':
        X = csc_matrix(X)
    elif type == 'dok':
        X = dok_matrix(X)
    elif type == 'lil':
        X = lil_matrix(X)
    return X


def to_dense(X):
    '''Convert to dense matrix
    '''
    if issparse(X):
        X = X.toarray()
        X = X.squeeze()
    return X


def to_triplet(X):
    '''Convert a dense or sparse matrix to a UIR triplet
    '''
    coo = coo_matrix(X)
    X = (
        np.asarray(coo.row, dtype='int'),
        np.asarray(coo.col, dtype='int'),
        np.asarray(coo.data, dtype='float')
    )
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

