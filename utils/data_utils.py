from typing import Union
from scipy.sparse import spmatrix
import numpy as np
import time

import sys
sys.path.append('../')

from generators import BaseBooleanMatrix
from datasets import BaseData


def binarize(X: Union[np.ndarray, spmatrix], threshold=0.5):
    return (X >= threshold).astype(int)

def sum(X: Union[np.ndarray, spmatrix]):
    '''Row and column-wise sum
    '''
    sum_u = np.squeeze(np.array(X.sum(axis=1)))
    sum_v = np.squeeze(np.array(X.sum(axis=0)))
    return sum_u, sum_v

def mean(X: Union[np.ndarray, spmatrix]):
    sum_u, sum_v = sum(X)
    return np.mean(sum_u), np.mean(sum_v)

def median(X: Union[np.ndarray, spmatrix]):
    sum_u, sum_v = sum(X)
    return np.median(sum_u), np.median(sum_v)

def sample(X, axis, factor_info=None, idx=None, n_samples=None, seed=None):
    '''Sample a matrix by its row or column

    axis: which dimension to down-sample.
        0, sample rows.
        1, sample columns.
    factor_info: factor info for X.
    idx: sample with given indices.
    n_samples: randomly down-sample to this length.
    seed: seed for down-sampling.
    '''
    if idx is not None:
        print("[I] Sampling with given indices")
        assert X.shape[axis] >= len(idx), "[E] Target length is greater than the original."
    elif n_samples is not None:
        print("[I] Sampling to size", n_samples)
        assert X.shape[axis] >= n_samples, "[E] Target length is greater than the original."
        
        seed = int(time.time()) if seed is None else seed
        rng = np.random.RandomState(seed)
        print("[I]   Sampling seed    :", seed)
        
        idx = [True] * n_samples + [False] * (X.shape[axis] - n_samples)
        rng.shuffle(idx)

    X = X[idx, :] if axis == 0 else X[:, idx]

    if factor_info is not None:
        for i in [0, 1, 2]: # order, idmap, alias
            factor_info[axis][i] = factor_info[axis][i][idx]
        factor_info[axis][0] = sort_order(factor_info[axis][0])

    return idx


def sort_order(order):
    '''Fix the gap after down-sampling

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


def get_factor_info(X):
    """Returns the factor_info list given input X
    
    X can be any of the following types:
        np.ndarray, for custom data.
        spmatrix, for custom data.
        BaseBooleanMatrix, for generated data.
        BaseData, for datasets, e.g. NetflixData.
    """
    if isinstance(X, BaseBooleanMatrix):
        U_info = (X.U_order, X.U_order, X.U_order.astype(str))
        V_info = (X.V_order, X.V_order, X.V_order.astype(str))
        factor_info = [U_info, V_info]
    elif isinstance(X, np.ndarray) or isinstance(X, spmatrix):
        U_order = np.array([i for i in range(X.shape[0])]).astype(int)
        V_order = np.array([i for i in range(X.shape[1])]).astype(int)
        U_info = (U_order, U_order, U_order.astype(str))
        V_info = (V_order, V_order, V_order.astype(str))
        factor_info = [U_info, V_info]
    elif isinstance(X, BaseData):
        factor_info = [X.U_info, X.V_info]
    return factor_info
    

# # remove below

# class Factor:
#     '''
#     factor: a Factor object
#     order: an array of shuffled indices
#     idmap: an array of user or item IDs
#     alias: an array of user or item names
#     matrices: a list of related matrices
#     name: a string
#     '''
#     def __init__(self, factor=None, order=None, idmap=None, alias=None, matrices=None, name=None):
#         self.order = None
#         self.idmap = None
#         self.alias = None
#         self.matrices = []
#         self.name = None
#         self.update(factor=factor, order=order, idmap=idmap, alias=alias, matrices=matrices, name=name, ignore=True)
        

#     def update(self, factor=None, order=None, idmap=None, alias=None, matrices=None, name=None, ignore=False):
#         '''Check conflicts upon updating

#         Conflict check is carried out when importing multiple matrices, together with their factors, to form a multi-matrix dataset. It's a tool for sanity check.

#         Conflict happens when the factors of corresponding matrices have inconsist order or dimensions. Update will be stopped when it fails the conflict check.

#         ignore: force re-write the attributes.
#         '''
#         if isinstance(matrices, list):
#             if len(matrices) == 2:
#                 self.matrices = matrices
#             else:
#                 print("[W] Factor.matrices should be a list of len 2.")
#         elif isinstance(matrices, Matrix):
#             self.matrices.append(matrices)
#             if len(self.matrices) > 2:
#                 print("[E] Factor.matrices should be a list of len 2.")

#         if isinstance(name, str):
#             self.name = name
            
#         name = '' if name is None else name

#         if isinstance(factor, Factor):
#             order, idmap, alias = factor.order, factor.idmap, factor.alias

#         order = np.array(order) if isinstance(order, list) else order
#         idmap = np.array(idmap) if isinstance(idmap, list) else idmap
#         alias = np.array(alias) if isinstance(alias, list) else alias

#         self.order = safe_update(self.order, order, var_name='order ' + name, ignore=ignore)
#         self.idmap = safe_update(self.idmap, idmap, var_name='idmap ' + name, ignore=ignore)
#         self.alias = safe_update(self.alias, alias, var_name='alias ' + name, ignore=ignore)


#     def sort_order(self):
#         '''Fix the gap after down-sampling

#         E.g. [1, 6, 4, 2] will be turned into [0, 3, 2, 1]
#         '''
#         # self.order = np.arange(len(self.order))
#         n = 0
#         for i in range(max(self.order) + 1):
#             if i in self.order:
#                 if isinstance(self.order, list):
#                     self.order[self.order.index(i)] = n
#                 elif isinstance(self.order, np.ndarray):
#                     self.order[self.order == i] = n
#                 n += 1


# class Matrix:
#     '''
#     matrix: a Matrix object or a sparse matrix
#     factors: a list of related factors
#     name: a string
#     '''
#     def __init__(self, matrix=None, factors=None, name=None):
#         self.matrix = None
#         self.factors = []
#         self.name = None
#         self.update(matrix=matrix, factors=factors, name=name)

        
#     def update(self, matrix=None, factors=None, name=None):
#         if isinstance(matrix, Matrix):
#             self.matrix = matrix.matrix
#         elif issparse(matrix):
#             self.matrix = matrix

#         if isinstance(factors, list):
#             if len(factors) == 2:
#                 self.factors = factors
#             else:
#                 print("[W] Matrix.factors should be a list of len 2.")

#         if isinstance(name, str):
#             self.name = name

#         self.reset()


#     def reset(self):
#         self.__triplet = None
#         self.__csr_matrix = None
#         self.__csc_matrix = None
#         self.__coo_matrix = None
#         self.__dok_matrix = None
#         self.__lil_matrix = None


#     @property
#     def shape(self):
#         return None if self.matrix is None else self.matrix.shape

#     @property
#     def r(self):
#         return None if self.matrix is None else self.matrix.nnz

#     @property
#     def m(self):
#         return None if self.matrix is None else self.matrix.shape[0]

#     @property
#     def n(self):
#         return None if self.matrix is None else self.matrix.shape[1]

#     @property
#     def triplet(self):
#         if self.__triplet is None:
#             self.__triplet = to_triplet(self.matrix)
#         return self.__triplet

#     @property
#     def csr_matrix(self):
#         if self.__csr_matrix is None:
#             self.__csr_matrix = to_sparse(self.matrix, type='csr')
#         return self.__csr_matrix
    
#     @property
#     def csc_matrix(self):
#         if self.__csc_matrix is None:
#             self.__csc_matrix = to_sparse(self.matrix, type='csc')
#         return self.__csc_matrix
    
#     @property
#     def coo_matrix(self):
#         if self.__coo_matrix is None:
#             self.__coo_matrix = to_sparse(self.matrix, type='coo')
#         return self.__coo_matrix

#     @property
#     def dok_matrix(self):
#         if self.__dok_matrix is None:
#             self.__dok_matrix = to_sparse(self.matrix, type='dok')
#         return self.__dok_matrix
    
#     @property
#     def lil_matrix(self):
#         if self.__lil_matrix is None:
#             self.__lil_matrix = to_sparse(self.matrix, type='lil')
#         return self.__lil_matrix
    
#     @property
#     def mean(self):
#         sum_u, sum_v = self.sum
#         return np.mean(sum_u), np.mean(sum_v)
    
#     @property
#     def median(self):
#         sum_u, sum_v = self.sum
#         return np.median(sum_u), np.median(sum_v)
    
#     @property
#     def sum(self):
#         sum_u = np.squeeze(np.array(self.matrix.sum(axis=1)))
#         sum_v = np.squeeze(np.array(self.matrix.sum(axis=0)))
#         return sum_u, sum_v