import numpy as np
from utils import cover, matmul, add, to_dense, invert, collective_cover
from .Asso import Asso
from .BaseCollectiveModel import BaseCollectiveModel
from scipy.sparse import lil_matrix, vstack, hstack
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import accumulate
import time


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
        if "n_basis" in kwargs:
            n_basis = kwargs.get("n_basis")
            if n_basis is None:
                print("[W] Missing n_basis. Will be using all basis.")
            self.n_basis = n_basis
            print("[I] n_basis      :", self.n_basis)


    def fit(self, Xs_train, factors, Xs_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(Xs_train=Xs_train, factors=factors, Xs_val=Xs_val)
        self.init_model()

        # the starting factor and the root vertex for BFS
        self.root = 0
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
        r = B.shape[0]

        if self.n_basis is None or self.n_basis > r:
            self.n_basis = r
            print("[I] n_basis is updated to: {}".format(r))

        self.basis = [lil_matrix(np.zeros((self.n_basis, d)), dtype=int) for d in self.factor_dims]

        if self.n_basis == r:
            self.basis[self.root] = B
        else:
            idx = np.random.choice(a=r, size=self.n_basis, replace=False)
            self.basis[self.root] = B[idx]
        self.scores = np.zeros((self.n_matrices, self.n_basis))


    @staticmethod
    def weighted_sum_score(scores, weights):
        '''Weighted score(s) of the `n` scores and weights.

        Parameters
        ----------
        scores : (n, k) array
        weights : (1, n) array

        Returns
        -------
        s : float or (1, k) array
        '''
        n = scores.shape[0]
        weights = np.array(weights).reshape(1, n)
        s = matmul(U=weights, V=scores)
        return s


    @staticmethod
    def harmonic_mean_score(scores):
        n = scores.shape[0]
        if (scores == 0).any():
            print("[W] Zero score encountered.")
            return 0
        s = (1 / scores).sum(axis=0)
        s = n / s
        return s
    

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
            best_weighted_score = 0 if k == 0 else best_weighted_score
            best_harmonic_score = 0 if k == 0 else best_harmonic_score
            
            self.update_Xs()
            self.set_init_order()
            self.set_update_order()

            n_iter = 0
            while True:
                print("[I] k:{}, iter:{}".format(k+1, n_iter+1))
                update_order = self.init_order if n_iter == 0 else self.update_order

                for f0, f1, m in update_order:
                    self.update_basis(f0, f1, m)

                    weighted_scores = self.weighted_sum_score(scores=self.scores, weights=self.p)
                    harmonic_scores = self.harmonic_mean_score(scores=self.scores)
                    
                    weighted_score = np.max(weighted_scores)
                    harmonic_score = np.max(harmonic_scores)

                if harmonic_score > best_harmonic_score:
                    print("[I]     harmonic: {:.2f} ---> {:.2f}".format(best_harmonic_score, harmonic_score))
                    best_harmonic_score = harmonic_score

                if weighted_score > best_weighted_score:
                    print("[I]     weighted: {:.2f} ---> {:.2f}".format(best_weighted_score, weighted_score))
                    best_weighted_score = weighted_score

                    best_idx = np.argmax(weighted_score)
                    print("[I]     best_idx: {}".format(best_idx))
                    n_iter += 1
                else:
                    print("[I] Score stops improving after iter: {}".format(n_iter))
                    break

            # update factors
            for f in range(self.n_factors):
                if best_idx is None:
                    print("[W] Score stops improving at k: {}".format(k))
                else:
                    self.Us[f][:, k] = self.basis[f][best_idx].T

                    # # debug: remove this basis
                    # idx = np.array([j for j in range(self.n_basis) if i != j])
                    # self.basis = self.basis[idx]

            # debug
            self.show_matrix(title="k: {}".format(k+1))

            # debug: validation, and print results when verbose=True
            # self.evaluate(X_gt=self.X_train, 
            #     df_name="train_results", 
            #     verbose=self.verbose, task=self.task, 
            #     metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            #     extra_metrics=['cover_score'], 
            #     extra_results=[self.cover()])


    
    def update_basis(self, f0, f1, m):
        '''Update f1's basis with factor f0's basis.

        Parameters
        ----------
        f0 : int or list of int
        f1 : int
        m : int or list of int
        '''
        desc = "[I]   f0 {} ---- m {} ---> f1 {}".format(f0, m, f1)

        if isinstance(m, int): # f1 is invloved in only 1 matrix
            X_gt = self.Xs_train[m]
            X_pd = self.Xs[m]
            basis = self.basis[f0]
            basis_dim = self.factors[m].index(f0)
            w = self.w[m]
            cover_before = cover(gt=X_gt, pd=X_pd, w=w, axis=basis_dim)

            for i in tqdm(range(self.n_basis), leave=True, position=0, desc=desc):
                self.scores[m][i], self.basis[f1][i] = Asso.get_vector(
                    X_gt=X_gt, 
                    X_before=X_pd, 
                    cover_before=cover_before, 
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
                pd = self.Xs[m].T if f0_dim else self.Xs[m]

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
            cover_before = collective_cover(gt=X_gt, pd=X_pd, w=w, axis=basis_dim, starts=starts)

            for i in tqdm(range(self.n_basis), leave=True, position=0, desc=desc):
                scores, self.basis[f1][i] = self.get_vector(
                    X_gt=X_gt, 
                    X_before=X_pd, 
                    cover_before=cover_before, 
                    basis=basis[i], 
                    basis_dim=basis_dim, 
                    w=w, 
                    starts=starts)
                
                for j, m in enumerate(m_list):
                    self.scores[m][i] = scores[j]

        


    @staticmethod
    def get_vector(X_gt, X_before, cover_before, basis, basis_dim, w, starts):
        '''CMF wrapper for Asso.get_vector()

        Parameters
        ----------
        X_gt : spmatrix
        X_before : spmatrix
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
            score, vector = Asso.get_vector(X_gt, X_before, cover_before, basis, basis_dim, w)
            return score, vector
        
        vector_dim = 1 - basis_dim
        vector = lil_matrix(np.ones((1, X_gt.shape[vector_dim])))
        X_after = matmul(basis.T, vector, sparse=True, boolean=True)
        X_after = X_after if basis_dim == 0 else X_after.T
        X_after = add(X_before, X_after)

        # collective cover
        cover_after = collective_cover(gt=X_gt, pd=X_after, w=w, axis=basis_dim, starts=starts)

        vector = lil_matrix(np.array(cover_after.sum(axis=0) > cover_before.sum(axis=0), dtype=int))

        cover_before = cover_before[:, to_dense(invert(vector), squeeze=True).astype(bool)]
        cover_after = cover_after[:, to_dense(vector, squeeze=True).astype(bool)]
        score = cover_before.sum(axis=1) + cover_after.sum(axis=1)

        return score, vector
        