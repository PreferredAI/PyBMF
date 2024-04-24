import numpy as np
from utils import matmul, add, to_sparse, cover, show_matrix
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
import pandas as pd
from utils import record


class AssoExAlternate(Asso):
    '''The Asso algorithm with iterative update between the two factors (experimental)
    '''
    def _fit(self):
        self.n_basis = self.basis.shape[0]
        self.basis = [lil_matrix(np.zeros((self.n_basis, self.m))), self.basis]
        self.scores = np.zeros(self.n_basis)
        
        for k in tqdm(range(self.k), position=0):
            print("[I] k   : {}".format(k))

            # index and score of the best basis
            best_idx = None
            best_score = 0 if k == 0 else best_score

            n_iter = 0
            n_stop = 0
            is_improving = True

            while is_improving:
                for basis_dim in [1, 0]:

                    # update basis of basis_dim
                    self.update_basis(basis_dim)

                    # highest score among all patterns
                    score = np.max(self.scores)

                    # check if it improves
                    if score > best_score:
                        # counting iterations and stagnations
                        n_iter += 1
                        n_stop = 0
                        print("[I] iter: {}, dim: {}, score: {:.2f} -> {:.2f}".format(n_iter, basis_dim, best_score, score))
                    else:
                        # break if it stops improving in the last 2 updates
                        n_stop += 1
                        print("[I] iter: {}, dim: {}, stop: {}".format(n_iter, basis_dim, n_stop))
                        if n_stop == 2:
                            is_improving = False
                            break
                        else:
                            continue

                    # index and score of the best basis
                    best_score = score
                    best_idx = np.argmax(self.scores)

                    # evaluate
                    self.update_factors(k, u=self.basis[0][best_idx, :].T, v=self.basis[1][best_idx, :].T)
                    self.predict_X()
                    self.evaluate(df_name='updates', head_info={'k': k, 'iter': n_iter, 'factor': 1-basis_dim, 'index': best_idx}, train_info={'score': best_score})
                    self.update_factors(k, u=0, v=0)

                    # debug
                    if self.verbose:
                        self.show_matrix([(self.basis[0], [0, 0], f'k = {k}'), (self.basis[1], [0, 1], f'k = {k}')])

                    # record the scores of all patterns
                    record(df_dict=self.logs, df_name='scores', columns=np.arange(self.n_basis).tolist(), records=self.scores.tolist(), verbose=self.verbose)

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                self.update_factors(k, u=self.basis[0][best_idx, :].T, v=self.basis[1][best_idx, :].T)
            
            self.evaluate(df_name='results', head_info={'k': k, 'iter': n_iter, 'index': best_idx}, train_info={'score': best_score})


    def update_basis(self, basis_dim):
        '''Use the basis of `basis_dim` to update its counterpart's basis.

        Parameters
        ----------
        basis_dim : int
        '''
        self.predict_X()
        target_dim = 1 - basis_dim
        cover_before = cover(gt=self.X_train, pd=self.X_pd, w=self.w, axis=basis_dim)

        for i in range(self.n_basis):
            self.scores[i], self.basis[target_dim][i] = Asso.get_vector(
                X_gt=self.X_train, 
                X_old=self.X_pd, 
                s_old=cover_before, 
                basis=self.basis[basis_dim][i], 
                basis_dim=basis_dim, 
                w=self.w)