import numpy as np
from ..utils import matmul, add, to_sparse, coverage_score, show_matrix, to_dense, invert, multiply, binarize
from .Asso import Asso
from .BMFTools import BMFTools, w_schedulers
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
import pandas as pd
from ..utils import record, isnum, ignore_warnings, bool_to_index


class BMFInterleave(BMFTools):
    '''The BMFAlternate alternative that lazily updates covered and uncovered data, so that parallelism is possible.

    TODO:
    1. weight updating scheme
    '''
    def __init__(self, k, w, w_list, init_method='asso', tau=None, p=None, n_basis=None, re_init=True, seed=None):
        '''
        Parameters
        ----------
        k : int
            The rank.
        w : float in [0, 1]
            The overcoverage (FP) penalty weight in the objective function during factorization. 
            It's also the lower bound of true positive ratio at each iteration of factorization. 
        w_list : list of float in [0, 1]
            Update trajectory of weights during fitting.
        init_method : {'asso', 'random_rows', 'random_bits'}, default 'asso'
            'asso' : build basis from real-valued association matrix, like in `Asso`.
            'random_rows' : build basis from random rows of `X_train`.
            'random_bits' : build basis using random binary vector with density `p`.
        tau : float
            The binarization threshold when building basis with `init_method='asso'`.
        p : float in [0, 1]
            The density of random bits when `init_method='random_bits'`.
        n_basis : int
            The number of basis candidates.
            If `None`, use the number of columns of `X_train`.
            If `init_method='asso'`, when `n_basis` is less than the number of columns of `X_train`, it will randomly pick `n_basis` candidates from the `basis` of `Asso`.
        re_init : bool
            Re-initialize basis candidates. Effective when `init_method='asso'` or `init_method='random_rows'`.
        '''
        self.check_params(k=k, w=w, w_list=w_list, init_method=init_method, tau=tau, p=p, n_basis=n_basis, re_init=re_init, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        # check init_method
        assert self.init_method in ['asso', 'random_rows', 'random_bits']


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        self._init_factors()
        self._init_logs()

        self.init_cover()
        self.init_basis()

        self.scores = np.zeros(self.n_basis)

        # init schedulers
        self.w_schedulers = w_schedulers(w_list=self.w_list, n_basis=self.n_basis)
        # w_now maintains a list of wieghts used by each of the n_basis patterns during update
        self.w_now = [self.w_schedulers.step(basis_id) for basis_id in range(self.n_basis)]

        # current basis dim
        self.basis_dim = [1] * self.n_basis


    def _fit(self):
        '''Fitting process of the model. ===================================================
        '''
        n_iter = 0
        # n_stop = 0
        is_improving = True

        while is_improving:

            if n_iter > 0:
                print(self.scores)
                print(scores_last)
                converged_ids = self.check_converge(now=self.scores, last=scores_last, diff_threshold=0.05)

                print(f"[D] iter: {n_iter}, converged_ids: {converged_ids}")

                ########## to run upon convergence under each weight ##########

                # converged_ids = self.check_remove(basis_ids=converged_ids, lazy_update=False)

                # converged_ids = self.check_merge(basis_ids=converged_ids, lazy_update=False, greater_id_only=True)

                # converged_ids = self.check_overlap(basis_ids=converged_ids, lazy_update=False, greater_id_only=False)

                self.update_weights(basis_ids=converged_ids)
            scores_last = self.scores.copy()
            
            '''Start updating each basis, updating factors one by one.
            Lazy update used and basis_list is collected in the beginning of each step.
            '''
            self.update_basis(lazy_update=True)

            self._record(n_iter)

            self._show_patterns(basis_list=self.basis_list, n_iter=n_iter)

            ########## to run after each (batch) update of basis ##########

            self.update_basis_dim()

            n_iter += 1
                
            if n_iter == 30:
                is_improving = False # break the outer loop
                break # break the inner loop





    def update_weights(self, basis_ids):
        '''Update weights of each basis, based on the change of pattern.

        Here we take exclusive score as the indicator of pattern change.
        '''
        for i in basis_ids:
            self.w_now[i] = self.w_schedulers.step(i)


    def update_basis(self, lazy_update=True):
        '''Update one basis at target_dim using basis_dim.
        
        Lazy update: every iteration each pattern can only see the last update of other patterns, simulating a parallel set up.
        '''

        if lazy_update:
            basis_list = self.basis_list.copy()

        for i in range(self.n_basis):
            X_pd_rest = self.get_X_pd_rest(basis_list if lazy_update else self.basis_list, basis_id=i)

            basis_dim, target_dim = self.get_basis_dim(basis_id=i)
            
            # coverage without i-th pattern, over target_dim, with i-th weight
            cover_before = cover(
                gt=self.X_train, 
                pd=X_pd_rest, 
                w=self.w_now[i], 
                axis=basis_dim
            )
            # update basis
            self.scores[i], self.basis_list[target_dim][i] = self.get_vector(
                X_gt=self.X_train, 
                X_old=X_pd_rest, 
                s_old=cover_before, 
                basis=self.basis_list[basis_dim][i], 
                basis_dim=basis_dim, 
                w=self.w_now[i]
            )

            # self.evaluate(df_name='results', head_info={'k': k, 'iter': n_iter, 'index': best_idx}, train_info={'score': best_score})







    def get_overlappings(self, basis_list, basis_id, greater_id_only=True, overlap_dims=2):
        '''Find overlappings on any or both dimensions.
        '''
        overlaps = [] # overlapping basis ids of basis_id
        overlap_ratio = []

        if greater_id_only:
            possible_overlaps = range(basis_id, self.n_basis)
        else:
            possible_overlaps = range(self.n_basis)

        for j in possible_overlaps:
            if j == basis_id:
                continue

            ol_rows = basis_list[0][j][basis_list[0][basis_id].astype(bool)]
            ol_cols = basis_list[1][j][basis_list[1][basis_id].astype(bool)]

            is_ol_rows, is_ol_cols = ol_rows.sum() > 0, ol_cols.sum() > 0

            if overlap_dims == 1:
                is_overlapped = is_ol_rows or is_ol_cols
            else:
                is_overlapped = is_ol_rows and is_ol_cols

            if is_overlapped:
                overlaps.append(j)

                ratio = ol_rows.sum() * ol_cols.sum()
                ratio /= basis_list[0][basis_id].sum() * basis_list[1][basis_id].sum()
                overlap_ratio.append(ratio)

        return overlaps, overlap_ratio


    def refine_overlapping(self, basis_id, overlap_id):
        '''
        '''
        i_rows = bool_to_index(self.basis_list[0][basis_id])
        i_cols = bool_to_index(self.basis_list[1][basis_id])

        j_rows = bool_to_index(self.basis_list[0][overlap_id])
        j_cols = bool_to_index(self.basis_list[1][overlap_id])

        ol_rows = np.intersect1d(i_rows, j_rows)
        ol_cols = np.intersect1d(i_cols, j_cols)

        i_ex_rows = np.setdiff1d(i_rows, ol_rows)
        i_ex_cols = np.setdiff1d(i_cols, ol_cols)

        j_ex_rows = np.setdiff1d(j_rows, ol_rows)
        j_ex_cols = np.setdiff1d(j_cols, ol_cols)

        # if 

        rest_rows = [r for r in range(self.m) if r not in i_rows and r not in j_rows]
        rest_cols = [c for c in range(self.n) if c not in i_cols and c not in j_cols]
        X_pd_rest = lil_matrix(self.X_train.shape)
        X_pd_rest[rest_rows, :] = 1
        X_pd_rest[:, rest_cols] = 1

        i = basis_id
    
        # coverage without i-th pattern, over target_dim, with i-th weight
        cover_before = cover(
            gt=self.X_train, 
            pd=X_pd_rest, 
            w=self.w_now[i], 
            axis=basis_dim
        )
        # update basis
        self.scores[i], self.basis_list[target_dim][i] = self.get_vector(
            X_gt=self.X_train, 
            X_old=X_pd_rest, 
            s_old=cover_before, 
            basis=self.basis_list[basis_dim][i], 
            basis_dim=basis_dim, 
            w=self.w_now[i]
        )


    # @staticmethod
    def get_vector(self, X_gt, X_old, s_old, basis, basis_dim, w):
        '''Return the optimal column/row vector given a row/column basis candidate.

        Parameters
        ----------
        X_gt : spmatrix
            The ground-truth matrix.
        X_old : spmatrix
            The prediction matrix before adding the current pattern.
        s_old : array
            The column/row-wise cover scores of previous prediction `X_pd`.
        basis : (1, n) spmatrix
            The basis vector.
        basis_dim : int
            The dimension which `basis` belongs to.
            If `basis_dim == 0`, a pattern is considered `basis.T * vector`. Otherwise, it's considered `vector.T * basis`. Note that both `basis` and `vector` are row vectors.
        w : float in [0, 1]

        Returns
        -------
        score : float
            The exclusive coverage score.
        vector : (1, n) spmatrix
            The matched vector.
        '''
        vector_dim = 1 - basis_dim
        vector = lil_matrix(np.ones((1, X_gt.shape[vector_dim])))
        pattern = matmul(basis.T, vector, sparse=True, boolean=True)
        pattern = pattern if basis_dim == 0 else pattern.T
        X_new = add(X_old, pattern, sparse=True, boolean=True)
        s_new = cover(gt=X_gt, pd=X_new, w=w, axis=basis_dim)

        vector = lil_matrix(np.array(s_new > s_old, dtype=int))
        s_old = s_old[to_dense(invert(vector), squeeze=True).astype(bool)]
        s_new = s_new[to_dense(vector, squeeze=True).astype(bool)]
        score = s_old.sum() + s_new.sum()

        # THE SCORE NOW WILL DETERMINE IF THE PATTERN IS CHANGING
        # here we use exclusive score
        pattern = matmul(basis.T, vector, sparse=True, boolean=True)
        pattern = pattern if basis_dim == 0 else pattern.T

        pattern[X_old.astype(bool)] = 0
        score = cover(
            gt=X_gt[pattern.astype(bool)], 
            pd=pattern[pattern.astype(bool)], 
            w=0.5)

        return score * 2, vector


    # def reinit_basis(self, basis_id, basis_dim):
    #     print(f"[I] Re-initializing basis {basis_id}, dim {basis_dim}")

    #     other_id = [j for j in range(self.n_basis) if basis_id != j]
    #     U = self.basis_list[0][other_id].T
    #     V = self.basis_list[1][other_id].T

    #     self.update_cover(U=U, V=V)
    #     basis, _ = self.init_basis_random_rows(
    #         X=self.X_uncovered, n_basis=1, axis=basis_dim)

    #     self.basis_list[basis_dim][basis_id] = basis
    #     self.basis_list[1 - basis_dim][basis_id] = 0


    def exclusive_dl(self, basis_ids, U=None, V=None):
        '''Description length of the exclusive area.

        The lower the better.
        '''
        # other patterns
        other_ids = [j for j in range(self.n_basis) if j not in basis_ids]
        u, v = self.basis_list[0][other_ids].T, self.basis_list[1][other_ids].T
        X_pd_rest = matmul(u, v.T, sparse=True, boolean=True)

        # load pattern
        if U is None and V is None:
            u, v = self.basis_list[0][basis_ids].T, self.basis_list[1][basis_ids].T
        else:
            u, v = U, V
        X_pattern = matmul(u, v.T, sparse=True, boolean=True)
        
        # merged pattern
        u_merged = binarize(u.sum(axis=1), threshold=0.5)
        v_merged = binarize(v.sum(axis=1), threshold=0.5)
        X_merged = matmul(u_merged, v_merged.T, sparse=True, boolean=True)

        X_pattern[X_pd_rest.astype(bool)] = 0
        X_merged[X_pd_rest.astype(bool)] = 0

        # description length of none/pattern/merged
        dl_none = self.X_train[X_merged.astype(bool)].sum()

        dl_pattern = u.sum() + v.sum()
        dl_pattern += self.X_train[X_merged.astype(bool)].sum() - self.X_train[X_pattern.astype(bool)].sum() # under-covered
        dl_pattern += X_pattern.sum() - self.X_train[X_pattern.astype(bool)].sum() # over-covered

        dl_merged = u_merged.sum() + v_merged.sum()
        dl_merged += X_merged.sum() - self.X_train[X_merged.astype(bool)].sum() # over-covered

        return dl_none, dl_pattern, dl_merged
    

    def exclusive_score(self, basis_ids, U=None, V=None):
        '''Score of the exclusive area.

        The higher the better.
        '''
        pass