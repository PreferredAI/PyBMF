import numpy as np
from utils import matmul, add, to_sparse, add_log
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd


class AssoExIterate(Asso):
    '''The Asso algorithm with iterative update among the factors (experimental)
    '''
    def _fit(self):

        self.scores = [None] * self.k
        
        for k in tqdm(range(self.k), position=0):
            best_basis = None
            best_column = None
            best_cover = 0 if k == 0 else best_cover
            b = self.basis.shape[0]

            # early stop detection
            if b == 0:
                self.early_stop(msg="No more basis left", k=k)
                break
            
            # debug
            # to record the scores of each pattern during iterative update
            self.scores[k] = [[] for _ in range(b)]
            # to record the columns of corresponding basis
            self.columns = lil_matrix(np.zeros((b, self.m)))

            while True:

                cover_before = best_cover

                for i in tqdm(range(b), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, column = self.get_optimal_column(i)
                    # debug
                    self.scores[k][i].append(score)
                    self.columns[i] = column.T

                    if score > best_cover:
                        best_cover = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T

                # # early stop detection
                # if best_basis is None:
                #     self.early_stop(msg="Coverage stops improving", k=k)
                #     break
                        
                print(f"k = {k}, after getting cols, best_cover = {best_cover}")

                # debug: iterative update
                for i in tqdm(range(b), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, basis = self.get_optimal_row(i)
                    # debug
                    self.scores[k][i].append(score)
                    self.basis[i] = basis.T

                    if score > best_cover:
                        best_cover = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T

                # # early stop detection
                # if best_basis is None:
                #     self.early_stop(msg="Coverage stops improving", k=k)
                #     break

                # debug: if stopped updating (this may be unnecessary)
                if cover_before == best_cover:
                    print("[W] best cover score is not improving.")
                    break
                else:
                    print(f"k = {k}, after getting rows, best_cover = {best_cover}")

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
        
        U = lil_matrix(np.ones((self.m, 1)))
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
    
    
    def get_optimal_row(self, i):
        X_before = matmul(self.U, self.V.T, sparse=True, boolean=True)
        
        U = self.columns[i].T
        V = lil_matrix(np.ones((1, self.n)))

        X_after = matmul(U, V, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover_before = self.cover(Y=X_before, axis=0)
        cover_after = self.cover(Y=X_after, axis=0)

        V = lil_matrix(np.array(cover_after > cover_before, dtype=int)).T

        X_after = matmul(U, V.T, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover = self.cover(Y=X_after)

        return cover, V
