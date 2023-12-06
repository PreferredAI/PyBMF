import numpy as np
from utils import matmul, add, to_sparse, add_log
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd


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


    def _fit_prepare(self, X_train, X_val=None, display=False):
        self.check_params(display=display)
        self.check_dataset(X_train=X_train, X_val=X_val)

        self.assoc = None # real-valued association matrix
        self.basis = None # binary-valued basis candidates
        self.build_assoc()
        self.build_basis()
        self.show_matrix(settings=self.assoc, title='assoc, tau: {}'.format(self.tau), colorbar=True, scaling=0.3)
        self.show_matrix(settings=self.basis, title='basis, tau: {}'.format(self.tau), colorbar=True, scaling=0.3)


    def fit(self, X_train, X_val=None, display=False):
        self._fit_prepare(X_train=X_train, X_val=X_val, display=False)
        self.check_params(display=display)

        self.start_trial()
        self.show_matrix(title="tau: {}, w: {}".format(self.tau, self.w), scaling=0.1)


    def build_assoc(self):
        '''Build the real-valued association matrix

        Confidence is the coverage made by vec_j within vec_i.

        # col_sum = self.X_train.sum(axis=0)
        # idx = col_sum != 0
        # self.assoc[idx] = self.assoc[idx] / col_sum[np.newaxis, idx]
        '''
        self.assoc = self.X_train.T @ self.X_train
        self.assoc = self.assoc.astype(float)

        for i in range(self.n):
            col_sum = self.X_train[:, i].sum()
            self.assoc[i, :] = self.assoc[i, :] / col_sum if col_sum > 0 else 0
        

    def build_basis(self):
        '''Get the binary-valued basis candidates
        '''
        self.basis = (self.assoc >= self.tau).astype(int)
        self.basis = to_sparse(self.basis, 'lil') # will be modified, thus lil is used


    def start_trial(self):
        for k in tqdm(range(self.k)):
            best_basis = None
            best_column = None

            best_cover = 0 if k == 0 else best_cover
            for i in tqdm(range(self.n), leave=False):
                score, column = self.get_optimal_column(i)
                if score > best_cover:
                    best_cover = score
                    best_basis = self.basis[i].T
                    best_column = column

            # update factors
            self.V[:, k] = best_basis
            self.U[:, k] = best_column
            self.basis[i] = 0 # remove this basis

            # self.show_matrix(title="step: {}, tau: {}, w: {}".format(k+1, self.tau, self.w), scaling=0.1)

            # debug: eval at every step
            if self.X_val is not None:
                if not hasattr(self, 'df_eval'):
                    metrics = ['Recall', 'Precsion', 'Error', 'Accuracy', 'F1']
                    columns = ['time', 'k', 'tau', 'p_pos', 'p_neg'] + metrics
                    self.df_eval = pd.DataFrame(columns=columns)

                results = self.eval(self.X_val, metrics=metrics, task='prediction')
                add_log(self.df_eval, [pd.Timestamp.now(), k+1, self.tau, self.w[0], self.w[1]] + results, verbose=False)


    def get_optimal_column(self, i):
        '''Return the optimal column given i-th basis candidate
        '''
        before = matmul(self.U, self.V.T, sparse=True, boolean=True)
        
        U = lil_matrix(np.ones([self.m, 1]))
        V = self.basis[i]
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after)

        before_cover = self.cover(Y=before, axis=1)
        after_cover = self.cover(Y=after, axis=1)
        optimal_col = (after_cover > before_cover) * 1
        optimal_col = lil_matrix(optimal_col).T

        U = optimal_col
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after)
        cover = self.cover(Y=after)

        return cover, optimal_col
    
        
