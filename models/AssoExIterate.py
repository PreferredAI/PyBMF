import numpy as np
from utils import matmul, add, to_sparse, cover, show_matrix
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import pickle


class AssoExIterate(Asso):
    '''The Asso algorithm with iterative update among the factors (experimental)
    '''
    def _fit(self):
        self.n_basis = self.basis.shape[0]
        self.basis = [lil_matrix(np.zeros((self.n_basis, self.m))), self.basis]
        self.scores = np.zeros(self.n_basis)

        # self.logs['scores'] = [None] * self.k
        
        for k in tqdm(range(self.k), position=0):
            best_idx = None
            best_score = 0 if k == 0 else best_score

            n_iter = 0
            break_counter = 0

            while True:
                print("[I] k: {}, n_iter: {}".format(k, n_iter))



            # best_basis, best_column, best_idx = None, None, None
            
            # n_basis = self.basis.shape[0]

            # # early stop detection
            # if n_basis == 0:
            #     self.early_stop(msg="No basis left.", k=k)
            #     break
            
            # debug:
            # to record the scores of each pattern during iterative update
            self.logs['scores'][k] = [[] for _ in range(n_basis)]
            # to record the columns of corresponding basis
            self.columns = lil_matrix(np.zeros((n_basis, self.m)))

            

            candidates = []

            while True:

                last_cover = best_score

                N_ITER = 1

                ###### original Asso ######
                for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, column = self.get_optimal_column(i)
                    # debug
                    self.logs['scores'][k][i].append(score)
                    self.columns[i] = column.T
                    if score > best_score:
                        best_score = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T
                        best_idx = i

                show_matrix([(self.columns, [0, 0], 'basis[0]'), (self.basis, [0, 1], 'basis[1]')], title="iter: {}".format(n_iter))
                if n_iter == N_ITER:
                    return
                

                if last_cover == best_score:
                    print("[I] k: {}, break_counter: {}".format(k, break_counter))
                    break_counter += 1
                    if break_counter == 2:
                        break
                else:
                    print("[I] k: {}, updated cols: {} => {}, best_idx: {}".format(k, last_cover, best_score, best_idx))
                    break_counter = 0
                    last_cover = best_score

                    self.U[:, k], self.V[:, k] = best_column, best_basis

                    self.evaluate(names=['k', 'iter', 'score'], values=[k, n_iter, best_score], df_name='updates')
                    n_iter += 1

                    self.U[:, k], self.V[:, k] = 0, 0 # reset
                # recorder
                candidates.append([self.columns.copy(), self.basis.copy()])
                
                                
                ###### iterative update ######
                for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, basis = self.get_optimal_row(i)
                    # debug
                    self.logs['scores'][k][i].append(score)
                    self.basis[i] = basis.T
                    if score > best_score:
                        best_score = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T
                        best_idx = i
                
                # show_matrix([(self.columns, [0, 0], 'basis[0]'), (self.basis, [0, 1], 'basis[1]')], title="iter: {}".format(n_iter))
                # if n_iter == N_ITER:
                #     return
                
                if last_cover == best_score:
                    print("[I] k: {}, break_counter: {}".format(k, break_counter))
                    break_counter += 1
                    if break_counter == 2:
                        break
                else:
                    print("[I] k: {}, updated rows: {} => {}, best_idx: {}".format(k, last_cover, best_score, best_idx))
                    break_counter = 0
                    last_cover = best_score

                    self.U[:, k], self.V[:, k] = best_column, best_basis
                    
                    self.evaluate(names=['k', 'iter', 'score'], values=[k, n_iter, best_score], df_name='updates')
                    n_iter += 1

                    self.U[:, k], self.V[:, k] = 0, 0
                # recorder
                candidates.append([self.columns.copy(), self.basis.copy()])

            # # recorder
            # from datetime import datetime
            # now = datetime.now()
            # current_time = now.strftime("%H_%M_%S")
            # with open(f'{current_time}_tau{self.tau}_candidates_{k}.pkl', 'wb') as f:  # open a text file
            #     pickle.dump(candidates, f) # serialize the list

            if best_basis is None or best_column is None:
                self.early_stop(msg="Coverage stops improving.", k=k)
                break

            # update factors
            self.V[:, k] = best_basis
            self.U[:, k] = best_column

            # remove this basis and column
            idx = np.array([j for j in range(n_basis) if i != j])
            self.basis = self.basis[idx]
            self.columns = self.columns[idx]

            self.evaluate(names=['k', 'iter', 'score'], values=[k, n_iter, best_score], df_name='results')
            

    def update_basis(self, basis_dim):
        '''Use the basis of `basis_dim` to update its counterpart's basis.

        Parameters
        ----------
        basis_dim : int
        '''
        self.predict()
        target_dim = 1 - basis_dim
        cover_before = cover(gt=self.X_train, pd=self.X_pd, w=self.w, axis=basis_dim)

        for i in tqdm(range(self.n_basis), leave=True, position=0, desc="[I] Updating basis"):
            self.scores[i], self.basis[target_dim][i] = Asso.get_vector(
                X_gt=self.X_train, 
                X_before=self.X_pd, 
                cover_before=cover_before, 
                basis=self.basis[basis_dim][i], 
                basis_dim=basis_dim, 
                w=self.w)