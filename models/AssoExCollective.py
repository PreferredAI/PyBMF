import numpy as np
import pandas as pd
from utils import cover, matmul, add, to_dense, invert
from utils import collective_cover, weighted_score, harmonic_score
from .Asso import Asso
from .BaseCollectiveModel import BaseCollectiveModel
from scipy.sparse import lil_matrix, vstack, hstack
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import accumulate
import time
from utils import show_matrix


class AssoExCollective(BaseCollectiveModel, Asso):
    '''The Asso algorithm for collective MF (experimental)
    '''
    def __init__(self, k, tau=None, w=None, p=None, n_basis=None):
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
        self.check_params(k=k, tau=tau, w=w, p=p, n_basis=n_basis)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "p" in kwargs:
            p = kwargs.get("p")
            if p is None:
                print("[E] Missing p.")
                return
            self.p = p
            print("[I] p            :", self.p)
        if not hasattr(self, "n_basis") or "n_basis" in kwargs:
            n_basis = kwargs.get("n_basis")
            if n_basis is None:
                print("[W] Missing n_basis, using all basis.")
            self.n_basis = n_basis
            print("[I] n_basis      :", self.n_basis)


    def fit(self, Xs_train, factors, Xs_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(Xs_train=Xs_train, factors=factors, Xs_val=Xs_val)
        self.init_model()

        # the starting factor and the root vertex for BFS
        self.root = 1 # Asso is 1
        self.init_basis()
        self._fit()


    def init_basis(self):
        '''Initialize basis list and basis score recordings.
        '''
        is_row = self.root in self.row_factors
        i = self.row_factors.index(self.root) if is_row else self.col_factors.index(self.root)
        a = self.row_starts[i] if is_row else self.col_starts[i]
        b = self.row_starts[i+1] if is_row else self.col_starts[i+1]
        X = self.X_train[a:b, :] if is_row else self.X_train[:, a:b]
        A = self.build_assoc(X=X, dim=0 if is_row else 1)
        B = self.build_basis(assoc=A, tau=self.tau)

        # # debug
        # X = self.Xs_train[0]
        # A = self.build_assoc(X=X, dim=0 if is_row else 1)
        # B = self.build_basis(assoc=A, tau=self.tau)

        r = B.shape[0]

        if self.n_basis is None or self.n_basis > r:
            self.n_basis = r
            print("[I] n_basis is updated to: {}".format(r))

        self.basis = [lil_matrix(np.zeros((self.n_basis, d))) for d in self.factor_dims]

        if self.n_basis == r:
            self.basis[self.root] = B
        else:
            idx = np.random.choice(a=r, size=self.n_basis, replace=False)
            self.basis[self.root] = B[idx]
        self.scores = np.zeros((self.n_matrices, self.n_basis))


    def set_init_order(self, order='bfs'):
        '''Use a known factor `f0` to update the neighboring factor `f1` in matrix `m`.
        '''
        visited = [False] * self.n_factors
        queue = []
        self.init_order = []
        visited[self.root] = True
        queue.append(self.root)
        while queue:
            f0 = queue.pop(0)
            for m in self.matrices[f0]:
                f0_dim = self.factors[m].index(f0)
                f1_dim = 1 - f0_dim
                f1 = self.factors[m][f1_dim]
                if visited[f1] == False:
                    queue.append(f1)
                    visited[f1] = True
                    self.init_order.append((f0, f1, m))


    def set_update_order(self, factors=None):
        '''Use a (list of) known factor(s) `f0` to update the factor `f1` using (a list of) matrix(es) `m`.

        factors : list of int, optional
            Update order provided by the user.
        '''
        if factors is None:
            factors = [self.root]
            for _, f1, _ in self.init_order:
                factors.append(f1)
        self.update_order = []
        for f1 in factors:
            f0, m = [], []
            for i in self.matrices[f1]:
                f1_dim = self.factors[i].index(f1)
                f0_dim = 1 - f1_dim
                f0.append(self.factors[i][f0_dim])
                m.append(i)
            self.update_order.append((f0, f1, m))     


    def _fit(self):
        for k in tqdm(range(self.k), position=0):
            best_idx = None
            best_ws = 0 if k == 0 else best_ws
            best_hs = 0 if k == 0 else best_hs
            
            self.predict_Xs()
            self.set_init_order()
            self.set_update_order()

            n_iter = 0
            break_counter = 0
            while True:
                print("[I] k: {}, n_iter: {}".format(k, n_iter))
                update_order = self.init_order if n_iter == 0 else self.update_order

                # update each factor
                for f0, f1, m in update_order:
                    self.update_basis(f0, f1, m)

                    ws_list = weighted_score(scores=self.scores, weights=self.p)
                    hs_list = harmonic_score(scores=self.scores)
                    ws = np.max(ws_list)
                    hs = np.max(hs_list)

                    if hs > best_hs:
                        print("[I]     harmonic     : {:.2f} ---> {:.2f}".format(best_hs, hs))
                        best_hs = hs

                    if ws > best_ws:
                        print("[I]     weighted     : {:.2f} ---> {:.2f}".format(best_ws, ws))
                        best_ws = ws
                        best_idx = np.argmax(ws_list)
                        print("[I]     best_idx     : {}".format(best_idx))

                        # evaluate
                        for f in range(self.n_factors):
                            self.Us[f][:, k] = self.basis[f][best_idx].T
                        self.evaluate(df_name='updates', 
                            info={'k': k, 'iter': n_iter, 'index': best_idx, 'w. score': best_ws}, 
                            t_info={'score': self.scores[:, best_idx].tolist()})
                        for f in range(self.n_factors):
                            self.Us[f][:, k] = 0

                        n_iter += 1
                        break_counter = 0
                    else:
                        break_counter += 1
                        print("[I] iter: {}, break_counter: {}".format(n_iter, break_counter))
                        if break_counter == self.n_factors:
                            break
                    
                if break_counter == self.n_factors:
                    break

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                for f in range(self.n_factors):
                    self.Us[f][:, k] = self.basis[f][best_idx].T

            self.evaluate(df_name='results', 
                info={'k': k, 'iter': n_iter, 'index': best_idx, 'w. score': best_ws}, 
                t_info={'score': self.scores[:, best_idx].tolist()})

    
    def update_basis(self, f0, f1, m):
        '''Update f1's basis with factor f0's basis.

        Parameters
        ----------
        f0 : int or list of int
        f1 : int
        m : int or list of int
        '''
        desc = "[I]     [ f0: {} ]----[ m: {} ]--->[ f1: {} ]".format(f0, m, f1)
        self.predict_Xs()

        if isinstance(m, int): # f1 is invloved in only 1 matrix, during init basis
            X_gt = self.Xs_train[m]
            X_pd = self.Xs_pd[m]
            basis = self.basis[f0]
            basis_dim = self.factors[m].index(f0)
            w = self.w[m]
            s_old = cover(gt=X_gt, pd=X_pd, w=w, axis=basis_dim)

            for i in tqdm(range(self.n_basis), leave=True, position=0, desc=desc):
                self.scores[m, i], self.basis[f1][i] = Asso.get_vector(
                    X_gt=X_gt, 
                    X_old=X_pd, 
                    s_old=s_old, 
                    basis=basis[i], 
                    basis_dim=basis_dim, 
                    w=w)

        elif isinstance(m, list): # f1 is involved in multiple matrices
            f0_list, m_list = f0.copy(), m.copy()
            assert len(f0_list) == len(m_list), "[E] Number of basis factors and matrices don't match."

            gt_list, pd_list, basis_list, w_list, starts = [], [], [], [], []
            for f0, m in zip(f0_list, m_list):
                f0_dim = self.factors[m].index(f0)
                gt = self.Xs_train[m].T if f0_dim else self.Xs_train[m]
                pd = self.Xs_pd[m].T if f0_dim else self.Xs_pd[m]

                gt_list.append(gt)
                pd_list.append(pd)
                basis_list.append(self.basis[f0])
                w_list.append(self.w[m])
                starts.append(self.factor_dims[f0])
            
            X_gt = vstack(gt_list, 'lil')
            X_pd = vstack(pd_list, 'lil')
            basis = hstack(basis_list, 'lil')
            basis_dim = 0
            w = w_list
            starts = [0] + list(accumulate(starts))
            s_old = collective_cover(gt=X_gt, pd=X_pd, w=w, axis=basis_dim, starts=starts)

            for i in tqdm(range(self.n_basis), leave=True, position=0, desc=desc):
                scores, self.basis[f1][i] = self.get_vector(
                    X_gt=X_gt, 
                    X_old=X_pd, 
                    s_old=s_old, 
                    basis=basis[i], 
                    basis_dim=basis_dim, 
                    w=w, 
                    starts=starts)
                
                for j, m in enumerate(m_list):
                    self.scores[m][i] = scores[j]


    @staticmethod
    def get_vector(X_gt, X_old, s_old, basis, basis_dim, w, starts):
        '''CMF wrapper for `Asso.get_vector()`.
        
        With additional parameter `starts` to indicate the split of `X`, and ``list`` `w` to indicate the ratio of true positives..

        Parameters
        ----------
        X_gt : spmatrix
        X_old : spmatrix
        basis : (1, n) spmatrix
        basis_dim : int
            The dimension which `basis` belongs to.
            If `basis_dim == 0`, a pattern is considered `basis.T * vector`. Otherwise, it's considered `vector.T * basis`. Note that here both `basis` and `vector` are row vectors.
        w : float or list of float in [0, 1]
        starts : list of int

        Returns
        -------
        score : float
            The coverage score.
        vector : (1, n) spmatrix
        '''
        if starts is None: # non-collective
            score, vector = Asso.get_vector(X_gt, X_old, s_old, basis, basis_dim, w)
            return score, vector
        
        vector_dim = 1 - basis_dim
        vector = lil_matrix(np.ones((1, X_gt.shape[vector_dim])))
        X_new = matmul(basis.T, vector, sparse=True, boolean=True)
        X_new = X_new if basis_dim == 0 else X_new.T
        X_new = add(X_old, X_new)

        # collective cover
        s_new = collective_cover(gt=X_gt, pd=X_new, w=w, axis=basis_dim, starts=starts)

        vector = lil_matrix(np.array(s_new.sum(axis=0) > s_old.sum(axis=0), dtype=int))
        s_old = s_old[:, to_dense(invert(vector), squeeze=True).astype(bool)]
        s_new = s_new[:, to_dense(vector, squeeze=True).astype(bool)]
        score = s_old.sum(axis=1) + s_new.sum(axis=1)

        return score, vector

    #     # X_new = matmul(basis.T, vector, sparse=True, boolean=True)
    #     # X_new = X_new if basis_dim == 0 else X_new.T
    #     # X_new = add(X_old, X_new)
    #     # score = cover(gt=X_gt, pd=X_new, w=w[0]) # ?
    #     # return [score], vector
        

    # def _evaluate(self, k, n_iter, best_idx, prefix='updates'):
    #     '''Scripts to run at each update.

    #     The 4 parts are:
    #     1. evaluation of individual training matrices.
    #     2. evaluation of individual validation matrices.
    #     3. evaluation of collective training matrices.
    #     4. evaluation of collective validation matrices.
    #     5. display.
    #     '''
    #     self.predict_Xs()    # update the predictions
    #     results_train = []  # evaluation outcomes on training matrices
    #     results_val = []    # evaluation outcomes on validation matrices

    #     for m in range(self.n_matrices):
    #         # 1
    #         score = self.scores[m, best_idx] # cover score of the best basis on m-th matrix
    #         results = self.collective_evaluate(
    #             X_gt=self.Xs_train[m], m=m, df_name="{}_train_{}".format(prefix, m), 
    #             verbose=self.verbose, task=self.task, 
    #             metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
    #             extra_metrics=['k', 'iter', 'score'], 
    #             extra_results=[k, n_iter, score])
    #         results_train.append([score]+results)

    #         # 2
    #         if self.Xs_val is not None:
    #             results = self.collective_evaluate(
    #                 X_gt=self.Xs_val[m], m=m, df_name="{}_val_{}".format(prefix, m), 
    #                 verbose=self.verbose, task=self.task, 
    #                 metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
    #                 extra_metrics=['k', 'iter'], 
    #                 extra_results=[k, n_iter])
    #             results_val.append(results)
        
    #     # 3
    #     results_train = np.array(results_train).reshape(self.n_matrices, -1)
    #     results_train_weighted = list(weighted_score(results_train, self.p).squeeze())
    #     results_train_harmonic = list(harmonic_score(results_train).squeeze())
    #     columns = pd.MultiIndex.from_tuples([('k'), 
    #         ('weighted scores', 'score'), ('weighted scores', 'Recall'), ('weighted scores', 'Precision'), ('weighted scores', 'Accuracy'), ('weighted scores', 'F1'), 
    #         ('harmonic scores', 'score'), ('harmonic scores', 'Recall'), ('harmonic scores', 'Precision'), ('harmonic scores', 'Accuracy'), ('harmonic scores', 'F1')])
    #     self.add_log(
    #         df_name="{}_train_collective".format(prefix), 
    #         # metrics=['k', 'w-score', 'w-Recall', 'w-Precision', 'w-Accuracy', 'w-F1', 
    #         #               'h-score', 'h-Recall', 'h-Precision', 'h-Accuracy', 'h-F1'], 
    #         metrics=columns, 
    #         results=[k]+results_train_weighted+results_train_harmonic)
        
    #     # 4
    #     if self.Xs_val is not None:
    #         results_val = np.array(results_val).reshape(self.n_matrices, -1)
    #         results_val_weighted = list(weighted_score(results_val, self.p).squeeze())
    #         results_val_harmonic = list(harmonic_score(results_val).squeeze())
    #         self.add_log(
    #             df_name="{}_val_collective".format(prefix), 
    #             metrics=['k', 'w-Recall', 'w-Precision', 'w-Accuracy', 'w-F1', 
    #                           'h-Recall', 'h-Precision', 'h-Accuracy', 'h-F1'], 
    #             results=[k]+results_val_weighted+results_val_harmonic)

    #     # 5
    #     self.show_matrix(title="k: {}, n_iter: {}".format(k, n_iter))


    #     super().predict_X()