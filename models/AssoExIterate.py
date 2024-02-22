import numpy as np
from utils import matmul, add, to_sparse, add_log
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

        self.logs['scores'] = [None] * self.k
        
        for k in tqdm(range(self.k), position=0):
            best_basis, best_column = None, None
            best_cover = 0 if k == 0 else best_cover
            n_basis = self.basis.shape[0]

            # early stop detection
            if n_basis == 0:
                self.early_stop(msg="No basis left.", k=k)
                break
            
            # debug:
            # to record the scores of each pattern during iterative update
            self.logs['scores'][k] = [[] for _ in range(n_basis)]
            # to record the columns of corresponding basis
            self.columns = lil_matrix(np.zeros((n_basis, self.m)))

            break_counter = 0

            candidates = []

            while True:

                last_cover = best_cover


                ###### original Asso ######
                for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, column = self.get_optimal_column(i)
                    score, column = self.get_vector
                    # debug
                    self.logs['scores'][k][i].append(score)
                    self.columns[i] = column.T
                    if score > best_cover:
                        best_cover = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T
                if last_cover == best_cover:
                    print(f"k = {k+1} break_counter = {break_counter}")
                    break_counter += 1
                    if break_counter == 2:
                        break
                else:
                    print(f"k = {k+1} updated cols: {last_cover} -> {best_cover}")
                    break_counter = 0
                    last_cover = best_cover

                    self.U[:, k], self.V[:, k] = best_column, best_basis
                    self._evaluate()
                    self.U[:, k], self.V[:, k] = 0, 0
                # recorder
                candidates.append([self.columns.copy(), self.basis.copy()])
                

                ###### iterative update ######
                for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k+1}"):
                    score, basis = self.get_optimal_row(i)
                    # debug
                    self.logs['scores'][k][i].append(score)
                    self.basis[i] = basis.T
                    if score > best_cover:
                        best_cover = score
                        best_basis = self.basis[i].T
                        best_column = self.columns[i].T
                if last_cover == best_cover:
                    print(f"k = {k+1} break_counter = {break_counter}")
                    break_counter += 1
                    if break_counter == 2:
                        break
                else:
                    print(f"k = {k+1} updated rows: {last_cover} -> {best_cover}")
                    break_counter = 0
                    last_cover = best_cover

                    self.U[:, k], self.V[:, k] = best_column, best_basis
                    self._evaluate()
                    self.U[:, k], self.V[:, k] = 0, 0
                # recorder
                candidates.append([self.columns.copy(), self.basis.copy()])


            # recorder
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            with open(f'{current_time}_tau{self.tau}_candidates_{k}.pkl', 'wb') as f:  # open a text file
                pickle.dump(candidates, f) # serialize the list

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

            # show matrix at every step, when verbose=True and display=True
            if self.verbose:
                self.show_matrix(title="step: {}, tau: {}, w: {}".format(k+1, self.tau, self.w))

            # evaluation, and print the last few results when verbose=True
            self.evaluate(X_gt=self.X_val, df="val_results_each_k", 
                verbose=self.verbose, task=self.task, 
                metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
                extra_metrics=['cover_score'], 
                extra_results=[self.cover()])
            

    def _evaluate(self):
        # evaluation, and print the last few results when verbose=True
        self.evaluate(X_gt=self.X_train, df="train_results", 
            verbose=self.verbose, task=self.task, 
            metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            extra_metrics=['cover_score'], 
            extra_results=[self.cover()])

        # evaluation, and print the last few results when verbose=True
        self.evaluate(X_gt=self.X_val, df="val_results", 
            verbose=self.verbose, task=self.task, 
            metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            extra_metrics=['cover_score'], 
            extra_results=[self.cover()])


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
