import numpy as np
from utils import matmul, add, to_sparse, add_log
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from p_tqdm import p_map


class AssoOri(BaseModel):
    '''The Asso algorithm
    
    Reference:
        The discrete basis problem
    '''
    def __init__(self, k, tau=None, w=None):
        """
        k:
            rank.
        tau:
            threshold for binarization when building basis.
        w:
            weights for coverage reward and overcoverage penalty.
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
            self.w = w
            print("[I] weights      :", self.w)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val)
        self.init_model()

        self.build_assoc() # real-valued association matrix
        self.build_basis() # binary-valued basis candidates

        self.show_matrix([(self.assoc, [0, 0], 'assoc'), 
                          (self.basis, [0, 1], 'basis')], 
                          colorbar=True, clim=[0, 1], 
                          title='tau: {}'.format(self.tau))
        self._fit()
        self.show_matrix(title="result")


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
            best_basis, best_column = None, None
            best_cover = 0 if k == 0 else best_cover # inherit from coverage of previous factors
            basis_num = self.basis.shape[0] # number of basis candidates

            # early stop detection
            if basis_num == 0:
                self.early_stop(msg="No basis left.", k=k)
                break

            for i in tqdm(range(basis_num), leave=False, position=0, desc=f"[I] k = {k+1}"):
                score, column = self.get_optimal_column(i)
                if score > best_cover:
                    best_cover = score
                    best_basis = self.basis[i].T
                    best_column = column
            
            if best_basis is None:
                self.early_stop(msg="Coverage stops improving.", k=k)
                break

            # update factors
            self.V[:, k] = best_basis
            self.U[:, k] = best_column

            # remove this basis
            idx = np.array([j for j in range(basis_num) if i != j])
            self.basis = self.basis[idx]

            # show matrix at every step, when verbose=True and display=True
            if self.verbose:
                self.show_matrix(title="step: {}, tau: {}, w: {}".format(k+1, self.tau, self.w))

            self.evaluate(X_gt=self.X_train, 
                df_name="train_results", 
                verbose=self.verbose, task=self.task, 
                metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
                extra_metrics=['cover_score'], 
                extra_results=[self.cover()])
            
            if self.X_val is not None:
                self.evaluate(X_gt=self.X_val, 
                    df_name="val_results", 
                    verbose=self.verbose, task=self.task, 
                    metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
                    extra_metrics=['cover_score'], 
                    extra_results=[self.cover()])


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
        