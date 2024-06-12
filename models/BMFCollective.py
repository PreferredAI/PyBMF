import numpy as np
import pandas as pd
from utils import coverage_score, matmul, add, to_dense, invert, record
from utils import collective_cover, weighted_score, harmonic_score, concat_Xs_into_X
from .Asso import Asso
from .BaseCollectiveModel import BaseCollectiveModel
from scipy.sparse import lil_matrix, vstack, hstack
from tqdm import tqdm
from itertools import accumulate


class BMFCollective(BaseCollectiveModel, Asso):
    '''The Asso algorithm for collective MF (experimental)
    '''
    def __init__(self, k, tau=None, w=None, p=None, n_basis=None):
        '''
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
        '''
        self.check_params(k=k, tau=tau, w=w, p=p, n_basis=n_basis)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "p" in kwargs:
            p = kwargs.get("p")
            if p is None:
                print("[E] Missing p.")
                return
            self.alpha = p
            print("[I] p            :", self.alpha)
        if not hasattr(self, "n_basis") or "n_basis" in kwargs:
            n_basis = kwargs.get("n_basis")
            if n_basis is None:
                print("[W] Missing n_basis, using all basis.")
            self.n_basis = n_basis
            print("[I] n_basis      :", self.n_basis)


    def fit(self, Xs_train, factors, Xs_val=None, Xs_test=None, **kwargs):
        super().fit(Xs_train, factors, Xs_val, Xs_test, **kwargs)

        # the starting factor and the root vertex for BFS
        self.root = 1 # in Asso is 1
        self.init_basis()
        self._fit()


    def init_basis(self):
        '''Initialize basis list and basis score recordings.
        '''
        is_row = self.root in self.row_factors
        i = self.row_factors.index(self.root) if is_row else self.col_factors.index(self.root)
        a = self.row_starts[i] if is_row else self.col_starts[i]
        b = self.row_starts[i+1] if is_row else self.col_starts[i+1]

        # X = self.X_train[a:b, :] if is_row else self.X_train[:, a:b]
        self.update_cover()
        X = self.X_uncovered[a:b, :] if is_row else self.X_uncovered[:, a:b]

        A = self.build_assoc(X=X, dim=0 if is_row else 1)
        B = self.build_basis(assoc=A, tau=self.tau)

        r = B.shape[0]

        if self.n_basis is None or self.n_basis > r:
            self.n_basis = r
            print("[I] n_basis is updated to: {}".format(r))

        self.basis = [lil_matrix((self.n_basis, d)) for d in self.factor_dims]

        if self.n_basis == r: # use all possible basis
            self.basis[self.root] = B
        else: # use n_basis basis
            idx = np.random.choice(a=r, size=self.n_basis, replace=False)
            self.basis[self.root] = B[idx]
        self.scores = np.zeros((self.n_matrices, self.n_basis))


    def update_cover(self):
        self.predict_Xs()
        self.X_pd = concat_Xs_into_X(self.Xs_pd, self.factors)
        self.X_uncovered = self.X_train.copy()
        self.X_uncovered[self.X_pd.astype(bool)] = 0



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
            n_stop = 0
            while True:
                # print("[I] k: {}, n_iter: {}".format(k, n_iter))
                update_order = self.init_order if n_iter == 0 else self.update_order

                # update each factor
                for f0, f1, m in update_order:
                    self.update_basis(f0, f1, m, k, n_iter)

                    ws_list = weighted_score(scores=self.scores, weights=self.alpha)
                    hs_list = harmonic_score(scores=self.scores)
                    ws = np.max(ws_list)
                    hs = np.max(hs_list)

                    if hs > best_hs:
                        # print("[I]     harmonic     : {:.2f} ---> {:.2f}".format(best_hs, hs))
                        best_hs = hs

                    if ws > best_ws:
                        # print("[I]     weighted     : {:.2f} ---> {:.2f}".format(best_ws, ws))
                        best_ws = ws
                        best_idx = np.argmax(ws_list)
                        # print("[I]     best_idx     : {}".format(best_idx))

                        # evaluate
                        for f in range(self.n_factors):
                            self.Us[f][:, k] = self.basis[f][best_idx].T
                        self.evaluate(df_name='updates', head_info={'k': k, 'iter': n_iter, 'factor': f1, 'index': best_idx, 'w. score': best_ws}, 
                            train_info={'score': self.scores[:, best_idx].tolist()})
                        for f in range(self.n_factors):
                            self.Us[f][:, k] = 0

                        # record (weighted) scores
                        record(df_dict=self.logs, df_name='scores', columns=np.arange(self.n_basis).tolist(), records=ws_list.flatten().tolist(), verbose=self.verbose)

                        n_iter += 1
                        n_stop = 0
                    else:
                        n_stop += 1
                        # print("[I] iter: {}, n_stop: {}".format(n_iter, n_stop))
                        if n_stop == self.n_factors:
                            break
                    
                if n_stop == self.n_factors:
                    break

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                for f in range(self.n_factors):
                    self.Us[f][:, k] = self.basis[f][best_idx].T

            self.evaluate(
                df_name='results', 
                head_info={'k': k, 'iter': n_iter, 'index': best_idx, 'w. score': best_ws}, 
                # train_info={'score': self.scores[:, best_idx].tolist()}
                )

    
    def update_basis(self, f0, f1, m, k, n_iter):
        '''Update f1's basis with factor f0's basis.

        Parameters
        ----------
        f0 : int or list of int
        f1 : int
        m : int or list of int
        '''
        desc = "[I] k: {}, iter: {}, f0: {} ---- m: {} ---> f1: {}".format(k, n_iter, f0, m, f1)
        self.predict_Xs()

        if isinstance(m, int): # f1 is invloved in only 1 matrix, during init basis
            X_gt = self.Xs_train[m]
            X_pd = self.Xs_pd[m]
            basis = self.basis[f0]
            basis_dim = self.factors[m].index(f0)
            w = self.w[m]
            s_old = cover(gt=X_gt, pd=X_pd, w=w, axis=basis_dim)

            for i in tqdm(range(self.n_basis), leave=False, position=0, desc=desc):
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

            for i in tqdm(range(self.n_basis), leave=False, position=0, desc=desc):
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
        
        With additional parameter `starts` to indicate the split of `X`, and `list` `w` to indicate the ratio of true positives..

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
        X_new = add(X_old, X_new, sparse=True, boolean=True)

        # collective cover
        s_new = collective_cover(gt=X_gt, pd=X_new, w=w, axis=basis_dim, starts=starts)

        vector = lil_matrix(np.array(s_new.sum(axis=0) > s_old.sum(axis=0), dtype=int))
        s_old = s_old[:, to_dense(invert(vector), squeeze=True).astype(bool)]
        s_new = s_new[:, to_dense(vector, squeeze=True).astype(bool)]
        score = s_old.sum(axis=1) + s_new.sum(axis=1)

        return score, vector
