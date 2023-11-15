import numpy as np
from utils import matmul, add, to_sparse, TP, FP
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool, cpu_count


class Asso(BaseModel):
    '''The Asso algorithm
    
    From the paper 'The discrete basis problem'.
    '''
    def __init__(self, k, tau=None, w=None):
        self.check_params(k=k, tau=tau, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        # check threshold tau
        if "tau" in kwargs:
            self.tau = kwargs.get("tau")
            if self.tau == None:
                self.tau = 0.5
            print("[I] tau          :", self.tau)
        # check reward and penalty parameters
        if "w" in kwargs:
            self.w = kwargs.get("w")
            if self.w == None:
                self.w = [0.5, 0.5]
            print("[I] weights      :", self.w)


    def _fit_prepare(self, train_set, val_set=None, display=False):
        self.check_params(display=display)
        self.check_dataset(train_set=train_set, val_set=val_set)
        self.U = lil_matrix(np.zeros((self.m, self.k)), dtype=np.float32)
        self.V = lil_matrix(np.zeros((self.k, self.n)), dtype=np.float32)
        self.assoc = None # real-valued association matrix
        self.basis = None # binary-valued basis candidates
        self.build_assoc()
        self.build_basis()
        self.show_matrix(settings=self.assoc, title='assoc', colorbar=True)
        self.show_matrix(settings=self.basis, title='basis', colorbar=True)


    def fit(self, train_set, val_set=None, display=False):
        self._fit_prepare(train_set=train_set, val_set=val_set, display=False)
        self.check_params(display=display)
        self.start_trial()
        self.show_matrix(title="tau: {}, w: {}".format(self.tau, self.w))


    def build_assoc(self):
        '''Build the integer-valued association matrix

        Confidence is the coverage made by vec_j within vec_i.

        Equivalent to:

        for i in range(self.n):
            col_sum = self.X_train[:, i].sum()
            self.assoc[i, :] = self.assoc[i, :] / col_sum if col_sum > 0 else 0
        '''
        self.assoc = self.X_train.T @ self.X_train
        self.assoc = self.assoc.astype(float)
        # col_sum = self.X_train.sum(axis=0)
        # idx = col_sum != 0
        # self.assoc[idx] = self.assoc[idx] / col_sum[np.newaxis, idx]
        for i in range(self.n):
            col_sum = self.X_train[:, i].sum()
            self.assoc[i, :] = self.assoc[i, :] / col_sum if col_sum > 0 else 0
        

    def build_basis(self):
        '''Get the binary-valued basis candidates
        '''
        self.basis = (self.assoc >= self.tau).astype(int)
        self.basis = to_sparse(self.basis)


    def start_trial(self):
        for k in tqdm(range(self.k), leave=False):
            # ver 1: vectorized
            best_basis = lil_matrix(np.zeros((1, self.n)))
            best_column = lil_matrix(np.zeros((self.m, 1)))

            best_score = 0 if k == 0 else best_score
            for c in tqdm(range(self.n), leave=False):
                score, column = self.get_col_candidate(c)
                if score > best_score:
                    best_score = score
                    best_basis = self.basis[c]
                    best_column = column

            self.V[k, :] = best_basis
            self.U[:, k] = best_column
            self.basis[c] = 0 # remove this basis

            self.show_matrix(title="tau: {}, w: {}, step: {}".format(self.tau, self.w, k+1))


    def get_col_candidate(self, c):
        '''Return the optimal column candidate given c-th basis
        '''
        before = matmul(self.U, self.V, sparse=True, boolean=True)
        
        U = lil_matrix(np.ones([self.m, 1]))
        V = self.basis[c]
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after)

        before_cover = self.cover(Y=before, axis=1)
        after_cover = self.cover(Y=after, axis=1)
        col_candidate = (after_cover > before_cover) * 1
        col_candidate = lil_matrix(col_candidate).T

        U = col_candidate
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after)
        cover = self.cover(Y=after)

        return cover, col_candidate
    
        
    def cover(self, X=None, Y=None, w=None, axis=None) -> Union[float, np.ndarray]:
        '''Measure the coverage of X using Y
        '''
        if X is None:
            X = self.X_train
        if Y is None:
            Y = matmul(self.U, self.V, sparse=True, boolean=True)
        covered = TP(X, Y, axis=axis)
        overcovered = FP(X, Y, axis=axis)
        w = self.w if w is None else w
        return w[0] * covered - w[1] * overcovered
