import numpy as np
from utils import matmul, add, to_sparse, add_log
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd


class Asso(BaseModel):
    '''The Asso algorithm
    
    Reference:
        The discrete basis problem
    '''
    def __init__(self, k, tau=None, w=None):
        """
        k:
            rank.
        tau:
            threshold.
        w:
            reward and penalty parameters.
        """
        self.check_params(k=k, tau=tau, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "tau" in kwargs:
            tau = kwargs.get("tau")
            if tau is None:
                tau = 0.5
            self.tau = tau
            print("[I] tau          :", self.tau)
        if "w" in kwargs:
            w = kwargs.get("w")
            if w is None:
                w = [0.2, 0.2]
            self.w = w
            print("[I] weights      :", self.w)


    def fit(self, X_train, X_val=None, **kwargs):
        self._fit_prepare(X_train=X_train, X_val=X_val)
        self.check_params(**kwargs)

        self._fit()
        self.show_matrix(title="tau: {}, w: {}".format(self.tau, self.w))


    def _fit_prepare(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.check_dataset(X_train=X_train, X_val=X_val)

        self.assoc = None # real-valued association matrix
        self.basis = None # binary-valued basis candidates
        self.build_assoc()
        self.build_basis()
        self.show_matrix(matrix=self.assoc, title='assoc, tau: {}'.format(self.tau), colorbar=True)
        self.show_matrix(matrix=self.basis, title='basis, tau: {}'.format(self.tau), colorbar=True)


    def build_assoc(self):
        '''Build the real-valued association matrix

        Confidence is the coverage made by vec_j within vec_i.

        # col_sum = self.X_train.sum(axis=0)
        # idx = col_sum != 0
        # self.assoc[idx] = self.assoc[idx] / col_sum[np.newaxis, idx]
        '''
        self.assoc = to_sparse(self.X_train.T @ self.X_train, 'lil').astype(float)
        for i in range(self.n):
            col_sum = self.X_train[:, i].sum()
            self.assoc[i, :] = (self.assoc[i, :] / col_sum) if col_sum > 0 else 0
        

    def build_basis(self):
        '''Get the binary-valued basis candidates
        '''
        self.basis = (self.assoc >= self.tau).astype(int)
        self.basis = to_sparse(self.basis, 'lil') # will be modified, thus lil is used


    def _fit(self):
        for k in tqdm(range(self.k), position=0):
            best_basis = None
            best_column = None
            best_cover = 0 if k == 0 else best_cover
            b = self.basis.shape[0]

            # early stop detection
            if b == 0:
                self.early_stop(msg="No more basis left", k=k)
                break

            for i in tqdm(range(b), leave=False, position=0, desc=f"[I] k = {k+1}"):
                score, column = self.get_optimal_column(i)
                if score > best_cover:
                    best_cover = score
                    best_basis = self.basis[i].T
                    best_column = column

            # early stop detection
            if best_basis is None:
                self.early_stop(msg="Coverage stops improving", k=k)
                break

            # update factors
            self.V[:, k] = best_basis
            self.U[:, k] = best_column

            # debug: remove this basis
            idx = np.array([j for j in range(b) if i != j])
            self.basis = self.basis[idx]

            # debug: show matrix at every step, when verbose=True and display=True
            if self.verbose:
                self.show_matrix(title="step: {}, tau: {}, w: {}".format(k+1, self.tau, self.w))

            # debug: validation, and print results when verbose=True
            self.validate(names=['time', 'k', 'tau', 'p_pos', 'p_neg'], 
                          values=[pd.Timestamp.now(), k+1, self.tau, self.w[0], self.w[1]], 
                          metrics=['Recall', 'Precision', 'Accuracy', 'F1'],
                          verbose=self.verbose)


    def get_optimal_column(self, i):
        '''Return the optimal column given i-th basis candidate

        Vectorized comparison on whether to set a column factor value to 1.
        '''
        X_before = matmul(self.U, self.V.T, sparse=True, boolean=True)
        
        U = lil_matrix(np.ones([self.m, 1]))
        V = self.basis[i]

        X_after = matmul(U, V, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover_before = self.cover(Y=X_before, axis=1)
        cover_after = self.cover(Y=X_after, axis=1)

        U = lil_matrix(np.array(cover_after > cover_before, dtype=int)).T

        X_after = matmul(U, V, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover = self.cover(Y=X_after)

        return cover, U
    
        
