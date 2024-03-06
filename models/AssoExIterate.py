import numpy as np
from utils import matmul, add, to_sparse, cover, show_matrix
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import pickle
from utils import record


class AssoExIterate(Asso):
    '''The Asso algorithm with iterative update between the two factors (experimental)
    '''
    def _fit(self):
        self.n_basis = self.basis.shape[0]
        self.basis = [lil_matrix(np.zeros((self.n_basis, self.m), dtype=int)), self.basis]
        self.scores = np.zeros(self.n_basis)
        
        for k in tqdm(range(self.k), position=0):
            best_idx = None
            best_score = 0 if k == 0 else best_score

            n_iter = 0
            break_counter = 0
            print("[I] k   : {}".format(k))
            while True:
                for basis_dim in [1, 0]:
                    self.update_basis(basis_dim)

                    score = np.max(self.scores)
                    if score > best_score:
                        print("[I] iter: {}, score: {:.2f} ---> {:.2f}".format(n_iter, best_score, score))

                        best_score = score
                        best_idx = np.argmax(self.scores)

                        # evaluate
                        self.U[:, k] = self.basis[0][best_idx, :].T
                        self.V[:, k] = self.basis[1][best_idx, :].T
                        self.evaluate(
                            names=['k', 'iter', 'factor', 'index', 'score'], 
                            values=[k, n_iter, 1 - basis_dim, best_idx, best_score], 
                            df_name='updates')
                        self.U[:, k] = 0
                        self.V[:, k] = 0

                        # record scores
                        record(
                            df_dict=self.logs, df_name='scores',
                            columns=np.arange(self.n_basis).tolist(), 
                            records=self.scores.tolist(), verbose=self.verbose)

                        n_iter += 1
                        break_counter = 0
                    else:
                        break_counter += 1
                        print("[I] iter: {}, break_counter: {}".format(n_iter, break_counter))
                        if break_counter == 2:
                            break
                    
                if break_counter == 2:
                    break

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                self.U[:, k] = self.basis[0][best_idx, :].T
                self.V[:, k] = self.basis[1][best_idx, :].T
            
            self.evaluate(
                names=['k', 'iter', 'index', 'score'], 
                values=[k, n_iter, best_idx, best_score], 
                df_name='results')


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