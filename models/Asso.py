import numpy as np
from utils import matmul, add, to_sparse, to_dense, step, invert, show_matrix
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from p_tqdm import p_map


class Asso(BaseModel):
    '''The Asso algorithm.
    
    Reference
    ---------
    The discrete basis problem.
    '''
    def __init__(self, k, tau=None, w=None):
        """
        Parameters
        ----------
        k : int
            Rank.
        tau : float
            The binarization threshold when building basis.
        w : float in [0, 1]
            The ratio of true positives.
        """
        self.check_params(k=k, tau=tau, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "tau" in kwargs:
            tau = kwargs.get("tau")
            assert tau is not None, "Missing tau."
            self.tau = tau
            print("[I] tau          :", self.tau)
        if "w" in kwargs:
            w = kwargs.get("w")
            assert w is not None, "Missing w."
            self.w = w
            print("[I] weights      :", self.w)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val)
        self.init_model()

        # real-valued association matrix
        self.assoc = self.build_assoc(X=self.X_train, dim=1)
        # binary-valued basis candidates
        self.basis = self.build_basis(assoc=self.assoc, tau=self.tau)

        show_matrix([(self.assoc, [0, 0], 'assoc'), 
                     (self.basis, [0, 1], 'basis')], 
                    colorbar=True, clim=[0, 1], title='tau: {}'.format(self.tau))

        self._fit()
        self.show_matrix(title="result")


    @staticmethod
    def build_assoc(X, dim):
        '''Build the real-valued association matrix.

        Parameters
        ----------
        X : ndarray, spmatrix
        dim : int
            The dimension which `basis` belongs to.
            If `dim` == 0, `basis` is treated as a column vector and `vector` as a row vector.
        '''
        assoc = X @ X.T if dim == 0 else X.T @ X
        assoc = to_sparse(assoc, 'lil').astype(float)
        s = X.sum(axis=1-dim) # col sum when axis == 0
        s = to_dense(s, squeeze=True)
        for i in range(X.shape[dim]):
            assoc[i, :] = (assoc[i, :] / s[i]) if s[i] > 0 else 0
            # s = X[:, i].sum() if dim == 1 else X[i, :].sum()
            # assoc[i, :] = (assoc[i, :] / s) if s > 0 else 0
        return assoc
        

    @staticmethod
    def build_basis(assoc, tau):
        '''Get the binary-valued basis candidates.

        Parameters
        ----------
        basis : spmatrix
            Each row of `basis` is a candidate basis.
        '''
        basis = step(assoc.copy(), tau)
        basis = to_sparse(basis, 'lil').astype(int)
        return basis


    def _fit(self):
        for k in tqdm(range(self.k), position=0):
            best_row, best_col = None, None
            best_score = 0 if k == 0 else best_score # best coverage score is inherited from previous factors
            n_basis = self.basis.shape[0] # number of basis candidates

            # early stop detection
            if n_basis == 0:
                self.early_stop(msg="No basis left.", k=k)
                break

            X_before = matmul(self.U, self.V.T, sparse=True, boolean=True)

            for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k+1}"):
                row = self.basis[i]
                score, col = self.get_vector(X_gt=self.X_train, X_before=X_before, basis=row, dim=1)
                if score > best_score:
                    best_score, best_row, best_col = score, row, col
            
            if best_row is None:
                self.early_stop(msg="Coverage stops improving.", k=k)
                break

            # update factors
            self.U[:, k], self.V[:, k] = best_col.T, best_row.T

            # remove this basis
            idx = np.array([j for j in range(n_basis) if i != j])
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


    def get_vector(self, X_gt, X_before, basis, dim):
        '''Return the optimal column/row vector given a row/column basis candidate.

        Parameters
        ----------
        X_gt : spmatrix
        X_before : spmatrix
        basis : (1, n) spmatrix
        dim : int
            The dimension which `basis` belongs to.
            If `dim` == 0, `basis` is treated as a column vector and `vector` as a row vector.

        Returns
        -------
        cover : float
            The coverage score.
        vector : (1, n) spmatrix
        '''
        vector = lil_matrix(np.ones((1, X_gt.shape[1-dim])))
        X_after = matmul(basis.T, vector, sparse=True, boolean=True)
        X_after = X_after if dim == 0 else X_after.T
        X_after = add(X_before, X_after)

        cover_before = self.cover(X=X_gt, Y=X_before, axis=dim)
        cover_after = self.cover(X=X_gt, Y=X_after, axis=dim)

        vector = lil_matrix(np.array(cover_after > cover_before, dtype=int))

        a = cover_before[to_dense(invert(vector), squeeze=True).astype(bool)]
        b = cover_after[to_dense(vector, squeeze=True).astype(bool)]
        cover = a.sum() + b.sum()
        return cover, vector
        