import numpy as np
from scipy.sparse import csr_matrix, issparse
from .sparse_utils import check_sparse


def multiply(U, V, sparse=None, boolean=None):
    '''Point-wise multiplication for both dense and sparse cases

    For vector-vector or matrix-matrix Hadamard product.
    '''
    if issparse(U) or issparse(V) or sparse == True:
        U = csr_matrix(U)
        V = csr_matrix(V)
        assert U.shape == V.shape, "U and V should have the same shape"
        X = U.multiply(V)
    else:
        assert U.shape == V.shape, "U and V should have the same shape"
        if boolean == True: # replace multiplication with logical
            X = np.logical_and(U, V).astype(int) # same as u & v
        else:
            X = np.multiply(U, V) # same as u * v
    return check_sparse(X, sparse=sparse)


def dot(u, v, boolean=None):
    '''Dot product for both dense and sparse cases
    
    For vector-vector inner product only.
    '''
    if issparse(u) or issparse(v):
        u = csr_matrix(u)
        v = csr_matrix(v)
        assert u.shape == v.shape, "U and V should have the same shape"
        x = multiply(u, v).sum()
        if boolean == True:
            x = (x > 0).astype(int) # Boolean product
    else:
        assert u.shape == v.shape, "U and V should have the same shape"
        if boolean == True: # replace multiplication with logical
            x = np.any(np.logical_and(u, v), axis=-1).astype(int) # Boolean product
        else:
            x = np.dot(u, v)
    return x


def matmul(U, V, sparse=None, boolean=None):
    '''Matrix multiplication for both dense and sparse cases
    '''
    if issparse(U) or issparse(V) or sparse == True:
        U = csr_matrix(U)
        V = csr_matrix(V)
        assert U.shape[1] == V.shape[0], "U and V should be multiplicable"
        X = U @ V
        if boolean == True:
            X = X.minimum(1).astype(int)
    else:
        assert U.shape[1] == V.shape[0], "U and V should be multiplicable"
        X = U @ V # same as np.matmul(u, v)
        if boolean == True:
            X = np.minimum(X, 1).astype(int) # Boolean product
    return check_sparse(X, sparse=sparse)


def add(X, Y):
    sum = np.add(X, Y).astype(bool).astype(int)
    return sum


def subtract(X, Y):
    return np.subtract(X, Y).astype(bool).astype(int)