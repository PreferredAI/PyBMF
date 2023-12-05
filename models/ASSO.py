import numpy as np
from utils import matmul, add, to_sparse
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
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
        self.V = lil_matrix(np.zeros((self.n, self.k)), dtype=np.float32)
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
        self.basis = to_sparse(self.basis, 'csr')


    def start_trial(self):
        for k in tqdm(range(self.k), leave=False):
            # best_basis = lil_matrix(np.zeros((1, self.n)))
            # best_column = lil_matrix(np.zeros((self.m, 1)))

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

            print("[I] Asso step: {}, tau: {}, w: {}".format(self.tau, self.w, k+1))
            self.show_matrix()


    def get_optimal_column(self, i):
        '''Return the optimal column given i-th basis candidate
        '''
        before = matmul(self.U, self.V, sparse=True, boolean=True)
        
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
    
        
