import numpy as np
from utils import matmul, add, to_sparse, add_log
from utils import concat_Xs_into_X
from .Asso import Asso
from .BaseCollectiveModel import BaseCollectiveModel
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd


class AssoExCollective(BaseCollectiveModel, Asso):
    '''The Asso algorithm for collective MF (experimental)
    '''
    def __init__(self, k, tau=None, w=None, p=None):
        """
        Parameters
        ----------
        k : int
            Rank.
        tau : float
            The binarization threshold when building basis.
        w : list of float in [0, 1]
            The ratio of true positives.
        p : list of float
            Importance weights that sum up tp 1.
        """
        self.check_params(k=k, tau=tau, w=w, p=p)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "p" in kwargs:
            p = kwargs.get("p")
            assert p is not None, "Missing p."
            self.p = p
            print("[I] p            :", self.p)


    def fit(self, Xs_train, factors, Xs_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(Xs_train=Xs_train, factors=factors, Xs_val=Xs_val)
        self.init_model()

        # can have initial basis for multiple independent factors
        self.starting_factors = [0]
        self.get_independent_factors()
        self.init_basis()
        self._fit()


    def get_independent_factors(self):
        independent_factors = []
        for a in self.starting_factors:
            is_independent = True
            for b in independent_factors:
                if [a, b] in self.factors or [b, a] in self.factors:
                    is_independent = False
                    break
            if is_independent:
                independent_factors.append(a)
        if len(self.starting_factors) != len(independent_factors):
            print("[I] starting factors : {}".format(independent_factors))
        self.starting_factors = independent_factors


    def init_basis(self):
        self.basis = [None] * self.n_factors
        for f in self.starting_factors:
            is_row = f in self.row_factors
            i = self.row_factors.index(f) if is_row else self.col_factors.index(f)
            a = self.row_starts[i] if is_row else self.col_starts[i]
            b = self.row_starts[i+1] if is_row else self.col_starts[i+1]
            X = self.X_train[a:b, :] if is_row else self.X_train[:, a:b]
            A = self.build_assoc(X=X, dim=0 if is_row else 1)
            B = self.build_basis(assoc=A, tau=self.tau)
            self.basis[f] = B


    def _fit(self):
        for k in tqdm(range(self.k), position=0):

            best_basis = [None] * self.n_factors
            best_cover = 0 if k == 0 else best_cover

            is_first_iter = True

            while True:

                if is_first_iter:
                    factor_list = [i for i in self.factor_list if i != self.starting_factor] # no starting factor
                    is_first_iter = False
                else:
                    factor_list = self.factor_list

                for f in factor_list:
                    for m in self.matrices[f]:
                        if self.factors[m].index(f) == 0:
                            another = self.factors[m][1]
                            
                        elif self.factors[m].index(f) == 1:
                    


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


    # def get_optimal_column(self, i):
    #     '''Return the optimal column given i-th basis candidate

    #     Vectorized comparison on whether to set a column factor value to 1.
    #     '''
    #     X_before = matmul(self.U, self.V.T, sparse=True, boolean=True)
        
    #     U = lil_matrix(np.ones([self.m, 1]))
    #     V = self.basis[i]

    #     X_after = matmul(U, V, sparse=True, boolean=True)
    #     X_after = add(X_before, X_after)

    #     cover_before = self.cover(Y=X_before, axis=1)
    #     cover_after = self.cover(Y=X_after, axis=1)

    #     U = lil_matrix(np.array(cover_after > cover_before, dtype=int)).T

    #     X_after = matmul(U, V, sparse=True, boolean=True)
    #     X_after = add(X_before, X_after)

    #     cover = self.cover(Y=X_after)

    #     return cover, U
    
        
