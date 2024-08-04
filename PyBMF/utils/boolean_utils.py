import numpy as np
from scipy.sparse import csr_matrix, issparse, spmatrix
from .sparse_utils import check_sparse, to_sparse, to_dense


def multiply(U, V, sparse=None, boolean=False):
    '''Point-wise multiplication for both dense and sparse input with Boolean logic support.

    For vector-vector or matrix-matrix Hadamard product.
    Also support regular const-vector and const-matrix product.

    Parameters
    ----------
    X : ndarray, spmatrix, int, float
    Y : ndarray, spmatrix, int, float
    sparse : bool, default: None
        Whether to enforce a sparse output. If `None`, keep the same dtype as input.
    boolean : bool, default: False
        Whether to use Boolean logic on the binary input.
    '''
    if ismat(U) and ismat(V):
        assert U.shape == V.shape, "U and V should have the same shape"
        if issparse(U) or issparse(V) or sparse:
            U, V = csr_matrix(U), csr_matrix(V)
            X = U.multiply(V)
        else:
            if boolean: # replace multiplication with logical
                X = np.logical_and(U, V).astype(int) # same as u & v
            else:
                X = np.multiply(U, V) # same as u * v
    else: # regular multiplication
        X = U * V
    return check_sparse(X, sparse=sparse)


def dot(u, v, boolean=False):
    '''Vector-vector inner product for both dense and sparse input with Boolean logic support.
    
    Parameters
    ----------
    U : ndarray, spmatrix
    V : ndarray, spmatrix
    boolean : bool, default: False
    '''
    if issparse(u) or issparse(v):
        u = csr_matrix(u)
        v = csr_matrix(v)
        assert u.shape == v.shape, "U and V should have the same shape"
        x = multiply(u, v).sum()
        if boolean:
            x = (x > 0).astype(int) # Boolean product
    else:
        assert u.shape == v.shape, "U and V should have the same shape"
        if boolean: # replace multiplication with logical
            x = np.any(np.logical_and(u, v), axis=-1).astype(int) # Boolean product
        else:
            x = np.dot(u, v)
    return x


def matmul(U, V, sparse=None, boolean=False):
    '''Matrix-matrix multiplication for both dense and sparse input with Boolean logic support.

    Parameters
    ----------
    U : ndarray, spmatrix
    V : ndarray, spmatrix
    sparse : bool, default: None
    boolean : bool, default: False
    '''
    sparse = sparse or (issparse(U) or issparse(V))
    if sparse:
        U = csr_matrix(U)
        V = csr_matrix(V)
        assert U.shape[1] == V.shape[0], "U and V should be multiplicable"
        X = U @ V
        if boolean:
            X = X.minimum(1).astype(int)
    else:
        assert U.shape[1] == V.shape[0], "U and V should be multiplicable"
        X = U @ V # same as np.matmul(u, v)
        if boolean:
            X = np.minimum(X, 1).astype(int) # Boolean product
    return check_sparse(X, sparse=sparse)


def add(X, Y, sparse=None, boolean=False):
    '''Matrix-matrix addition for both dense and sparse input with Boolean logic support.

    Also support regular matrix-const addition.

    Parameters
    ----------
    X : ndarray, spmatrix, int, float
    Y : ndarray, spmatrix, int, float
    sparse : bool, default: False
    boolean : bool, default: False
    '''
    if isnum(X) or isnum(Y): # const-matrix addition
        X = to_dense(X) if issparse(X) else X
        Y = to_dense(Y) if issparse(Y) else Y
    if boolean: # boolean matrix-matrix addition
        Z = np.add(X, Y).astype(bool).astype(float)
    else:
        Z = X + Y
    sparse = sparse or (issparse(X) or issparse(Y))
    return check_sparse(Z, sparse=sparse)


def subtract(X, Y, sparse=False, boolean=False):
    '''Matrix-matrix subtraction for both dense and sparse input with Boolean logic support.

    Also support regular matrix-const subtraction.

    Parameters
    ----------
    X : ndarray, spmatrix, int, float
    Y : ndarray, spmatrix, int, float
    sparse : bool, default: False
    boolean : bool, default: False
    '''  
    if isnum(X) or isnum(Y): # const-matrix subtraction
        _X = to_dense(X) if issparse(X) else X
        _Y = to_dense(Y) if issparse(Y) else Y

    if boolean: # boolean matrix-matrix subtraction
        Z = np.subtract(_X, _Y).astype(bool).astype(float)
    else:
        Z = _X - _Y

    sparse = sparse or (issparse(_X) or issparse(_Y))

    return check_sparse(Z, sparse=sparse)


def power(X, n):
    '''Matrix power for both dense and sparse input.
    '''
    if issparse(X):
        return X.power(n).astype(np.float64)
    else:
        return np.power(X, n).astype(np.float64)
    

def isnum(X):
    '''Whether X is a number or not.
    '''
    return isinstance(X, (int, float))


def ismat(X):
    '''Whether X is a matrix or not.

    ```{attention}
    Note that ``np.matrix`` is NOT a supported matrix type in PyBMF.
    ```
    
    '''
    return isinstance(X, (np.ndarray, spmatrix))