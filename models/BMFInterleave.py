import numpy as np
from utils import matmul, add, to_sparse, cover, show_matrix, to_dense, invert, multiply
from .Asso import Asso
from .BMFAlternateTools import BMFAlternateTools, w_schedulers
from scipy.sparse import lil_matrix
from tqdm import tqdm
import pandas as pd
from utils import record, isnum, ignore_warnings, bool_to_index


class BMFInterleave(BMFAlternateTools):
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

        self.predict_X()
        self.finish()


    def init_model(self):
        self._init_factors()
        self._init_logs()

        self.init_cover()
        self.init_basis()

        self.init_basis_list()
        self.scores = np.zeros(self.n_basis)

        # init schedulers
        self.w_schedulers = w_schedulers(w_list=self.w_list, n_basis=self.n_basis)

        # w_now maintains a list of wieghts used by each of the n_basis patterns during update
        self.w_now = [self.w_schedulers.step(basis_id) for basis_id in range(self.n_basis)]


    def _fit(self):
        '''Fitting process of the model.
        '''
        n_iter = 0
        # n_stop = 0
        is_improving = True

        while is_improving:
            '''Update weights of each basis, based on the change of pattern.

            Here we take exclusive score as the indicator of pattern change.
            '''
            scores_new = self.scores

            if n_iter > 0:
                # check diff and update ws
                diff_threshold = 0.1
                for i in range(self.n_basis):
                    scores_diff = abs(scores_new[i] - scores_old[i]) / scores_old[i]
                    if scores_diff < diff_threshold:
                        self.w_now[i] = self.w_schedulers.step(i)

            scores_old = self.scores.copy()
            
            '''Start updating each basis, updating factors one by one.
            '''

            for basis_dim in [1, 0]:

                self.update_basis(basis_dim, n_iter)
                n_iter += 1
                
                if n_iter == 40:
                    is_improving = False # break the outer loop
                    break # break the inner loop
            

    def update_basis(self, basis_dim, n_iter):
        '''Update one basis at target_dim using basis_dim.
        '''
        target_dim = 1 - basis_dim

        '''Lazy update.

        Every iteration each pattern can only see the last update of other patterns.
        '''
        basis_list = self.basis_list.copy()

        for i in range(self.n_basis):

            '''Make X_pd_rest with other patterns using the copy from last lazy update.'''
            other_id = [j for j in range(self.n_basis) if i != j]
            U=basis_list[0][other_id].T
            V=basis_list[1][other_id].T
            X_pd_rest = matmul(U, V.T, sparse=True, boolean=True)
            
            '''Coverage without i-th pattern, over target_dim, with i-th weight.'''
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

        ''' Recording. ################################################################################

        - weight
        - exclusive desc len (decrease of desc len with i-th pattern)
        - exclusive score upon update
        - exclusive score after updating all patterns
        - density of exclusive pattern
        '''
        records = [n_iter, target_dim]
        columns = ['n_iter', 'target_dim']

        for i in range(self.n_basis):
            dl_0, dl_1, u, v, tp, fp = self.exclusive_dl(basis_id=i)
            columns += [str(i) + suffix for suffix in ["_w", "_ex_dl", "_ex_s", "_ex_s", "_dl1v0", ' ']]
            records += [self.w_now[i], dl_0 - dl_1, self.scores[i], tp - fp, dl_1 / dl_0, ' ']

        record(df_dict=self.logs, df_name='updates', columns=columns, records=records, verbose=self.verbose)

        '''############################################################################################'''

        ### Plot all patterns. ########################################################################
        self._show_matrix(self, basis_list, basis_dim, target_dim, n_iter)
        ###############################################################################################

        ''' Find overlappings. ########################################################################'''

        # self.

        # # plot again
        # self.X_pd = lil_matrix(self.X_train.shape)
        # for i in range(self.n_basis):
        #     # pattern after update
        #     pattern = matmul(U=basis_list[0][i].T, V=basis_list[1][i], sparse=True, boolean=True)
        #     pattern = pattern * (i + 1)
        #     self.X_pd[pattern > 0] = pattern[pattern > 0]

        # # self.show_matrix(colorbar=True, clim=[0, self.n_basis], discrete=True, center=True, 
        # #     title=f'n_iter: {n_iter} ({basis_dim} -> {target_dim}), basis: {i+1}/{self.n_basis}')
        # self.show_matrix(
        #     settings=[(self.X_pd, [0, 0], 'patterns')], 
        #     colorbar=True, 
        #     cmap='tab20', 
        #     clim=[0, self.n_basis], 
        #     discrete=True, 
        #     center=True, 
        #     title=f'n_iter: {n_iter} ({basis_dim} -> {target_dim}), basis: {i+1}/{self.n_basis}')


    # def find_overlappings(self):
    #     pairs = []
    #     for i in range(self.n_basis - 1):
    #         _pairs = []
    #         for j in range(i + 1, self.n_basis):
    #             ol_rows = self.basis_list[0][j][self.basis_list[0][i].astype(bool)]
    #             ol_cols = self.basis_list[1][j][self.basis_list[1][i].astype(bool)]
    #             if ol_rows.sum() > 0 and ol_cols.sum() > 0:
    #                 pairs.append((i, j))
    #                 _pairs.append(j)
    #         print(f"overlappings of {i}: ", _pairs)

    # def group_and_split(self, pair):
    #     i, j = pair
    #     ol_rows = self.basis_list[0][j][self.basis_list[0][i].astype(bool)]
    #     ol_cols = self.basis_list[1][j][self.basis_list[1][i].astype(bool)]

    #     ol_rows = bool_to_index(ol_rows)
    #     ol_cols = bool_to_index(ol_cols)

    #     i_all_rows = bool_to_index(self.basis_list[0][i])
    #     i_all_cols = bool_to_index(self.basis_list[1][i])
    #     i_ex_rows = np.setdiff1d(i_all_rows, ol_rows)
    #     i_ex_cols = np.setdiff1d(i_all_cols, ol_cols)

    #     j_all_rows = bool_to_index(self.basis_list[0][j])
    #     j_all_cols = bool_to_index(self.basis_list[1][j])
    #     j_ex_rows = np.setdiff1d(j_all_rows, ol_rows)
    #     j_ex_cols = np.setdiff1d(j_all_cols, ol_cols)


    # def merge_and_split(self, pair):


    # def just_split(self, basis_id):
    #     pass
            

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
            The coverage score.
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


    def reinit_basis(self, basis_id, basis_dim):
        print(f"[I] Re-initializing basis {basis_id}, dim {basis_dim}")

        other_id = [j for j in range(self.n_basis) if basis_id != j]
        U = self.basis_list[0][other_id].T
        V = self.basis_list[1][other_id].T

        self.update_cover(U=U, V=V)
        basis, _ = self.init_basis_random_rows(
            X=self.X_uncovered, n_basis=1, axis=basis_dim)

        self.basis_list[basis_dim][basis_id] = basis
        self.basis_list[1 - basis_dim][basis_id] = 0


    def exclusive_dl(self, basis_id):
        '''Description length of the exclusive area.
        '''
        other_id = [j for j in range(self.n_basis) if basis_id != j]
        U = self.basis_list[0][other_id].T
        V = self.basis_list[1][other_id].T
        X_pd_rest = matmul(U, V.T, sparse=True, boolean=True)

        U = self.basis_list[0][basis_id].T
        V = self.basis_list[1][basis_id].T
        X_pattern = matmul(U, V.T, sparse=True, boolean=True)
        X_pattern[X_pd_rest.astype(bool)] = 0
        
        # exclusive pattern size
        u = (X_pattern.sum(axis=1) > 0).sum()
        v = (X_pattern.sum(axis=0) > 0).sum()

        # tp
        tp = self.X_train[X_pattern.astype(bool)].sum()

        # fp
        fp = X_pattern.sum() - tp

        # dl
        dl_0 = tp
        dl_1 = u + v + fp

        return dl_0, dl_1, u, v, tp, fp


    def get_neighbors(self, basis_id):
        other_id = [j for j in range(self.n_basis) if basis_id != j]
        U = self.basis_list[0][other_id].T
        V = self.basis_list[1][other_id].T
        X_pd_rest = matmul(U, V.T, sparse=True, boolean=True)


    def _show_matrix(self, basis_list, basis_dim, target_dim, n_iter):
        X = lil_matrix(self.X_train.shape)
        for i in range(self.n_basis):
            pattern = matmul(U=basis_list[0][i].T, V=basis_list[1][i], sparse=True, boolean=True)
            pattern = pattern * (i + 1)
            X[pattern > 0] = pattern[pattern > 0]

        name = f'n_iter: {n_iter} ({basis_dim} -> {target_dim}), basis: {i+1}/{self.n_basis}'

        self.show_matrix(
            settings=[(X, [0, 0], name)], 
            cmap="tab20", 
            colorbar=True, 
            clim=[0, self.n_basis], 
            discrete=True, 
            center=True)
        