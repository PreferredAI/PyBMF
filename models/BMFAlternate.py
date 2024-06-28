import numpy as np
from utils import matmul, add, to_sparse, coverage_score, show_matrix, binarize, bool_to_index
from .Asso import Asso
from .BMFTools import BMFTools, w_scheduler
from scipy.sparse import lil_matrix
from tqdm import tqdm
import pandas as pd
from utils import record, isnum, ignore_warnings, get_prediction


class BMFAlternate(BMFTools):
    '''The Asso algorithm with alternative update between the two factors (experimental).
    '''
    def __init__(self, k, w_list, w=0.5, init_method='asso', tau=None, p=None, n_basis=None, re_init=True, seed=None):
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

        self.w_scheduler = w_scheduler(w_list=self.w_list)


    def _fit(self):

        # backup initialized factors
        self.basis_list_backup = self.basis_list.copy()

        k = 0
        
        while k < self.k:
            print("[I] k   : {}".format(k))

            if k > 0 and self.re_init: # restrore initialized factors
                self.update_cover(U=self.U, V=self.V)
                self.init_basis() # re-init basis?
            else: # use the same basis as in `Asso`
                self.basis_list = self.basis_list_backup.copy()
            
            # index and score of the best basis
            best_idx = None
            best_score = 0 if k == 0 else best_score

            n_iter = 0
            n_stop = 0

            self.w_scheduler.reset()
            self.w_now = self.w_scheduler.step()

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
                        print("[I] iter: {}, basis_dim: {}, w: {}, score: {:.2f} -> {:.2f}".format(n_iter, basis_dim, self.w_now, best_score, score))
                    else:
                        # break if it stops improving in the last 2 updates
                        n_stop += 1
                        print("[I] iter: {}, basis_dim: {}, w: {}, stop: {}".format(n_iter, basis_dim, self.w_now, n_stop))
                        # original
                        if n_stop == 2:
                            if self.w_now == self.w: # reached last w in w_list
                                is_improving = False
                                break
                            else: # can update to next w
                                self.w_now = self.w_scheduler.step()
                                n_stop = 0
                                best_score = 0
                                continue
                        else:
                            continue

                        # ex02
                        # if n_stop == 10:
                        #     is_improving = False
                        #     break
                        # else:
                        #     continue

                    # index and score of the best basis
                    best_score = score
                    best_idx = np.argmax(self.scores)

                    # evaluate
                    self.set_factors(k, u=self.basis_list[0][best_idx, :].T, v=self.basis_list[1][best_idx, :].T)
                    self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
                    if self.verbose:
                        self.show_matrix(colorbar=True, clim=[0, 1], title=f'k: {k}, w: {self.w_now}')
                    self.evaluate(df_name='updates', head_info={'k': k, 'iter': n_iter, 'factor': 1-basis_dim, 'index': best_idx}, train_info={'score': best_score})
                    self.set_factors(k, u=0, v=0)

                    # debug
                    if self.verbose:
                        self.show_matrix([(self.basis_list[0], [0, 0], f'k = {k}'), (self.basis_list[1], [0, 1], f'k = {k}')])

                    # record the scores of all patterns
                    record(df_dict=self.logs, df_name='scores', columns=np.arange(self.n_basis).tolist(), records=self.scores.tolist(), verbose=self.verbose)

            # update factors ######################################################################

            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                self.set_factors(k, u=self.basis_list[0][best_idx, :].T, v=self.basis_list[1][best_idx, :].T)
            
            self.evaluate(df_name='results', head_info={'k': k, 'iter': n_iter, 'index': best_idx}, train_info={'score': best_score})

            k += 1

            # # post processing #####################################################################

            # k -= 1

            # # check_merge

            # overlaps = self.get_overlaps(
            #     U_all=self.U[:, 0:k+1], 
            #     V_all=self.V[:, 0:k+1], 
            #     basis_id=k, 
            #     overlap_dims=1
            # )
            # print("[I] overlaps (any dim) for k = {}: {}".format(k, overlaps))

            # has_merged = self.check_merge(
            #     U_all=self.U[:, 0:k+1], 
            #     V_all=self.V[:, 0:k+1], 
            #     basis_id=k, 
            #     overlaps=overlaps
            # )

            # if has_merged:
            #     continue

            # # check_overlap

            # self.check_overlap(
            #     U_all=self.U[:, 0:k+1],
            #     V_all=self.V[:, 0:k+1],
            #     basis_id=k,
            # )

            # k += 1


    def check_overlap(self, U_all, V_all, basis_id):
        k = basis_id
        overlaps = self.get_overlaps(
            U_all=U_all, 
            V_all=V_all, 
            basis_id=k, 
            overlap_dims=2
        )
        print("[I] overlaps (both dim) for k = {}: {}".format(k, overlaps))

        if len(overlaps) == 0:
            return
        
        i = basis_id
        j = overlaps[0]

        # refine with i and j

        X_i = matmul(U_all[:, i], V_all[:, i].T, sparse=True, boolean=True)
        X_j = matmul(U_all[:, j], V_all[:, j].T, sparse=True, boolean=True) * 2
        settings = ([(X_i + X_j, [0, 0], f'i: {i}, j: {j}')])
        show_matrix(settings, colorbar=True, discrete=True, center=True, clim=[0, 3], scaling=0.5)

        # build exclusive area

        i_rows = bool_to_index(U_all[:, i])
        i_cols = bool_to_index(V_all[:, i])
        j_rows = bool_to_index(U_all[:, j])
        j_cols = bool_to_index(V_all[:, j])

        ol_rows = np.intersect1d(i_rows, j_rows)
        ol_cols = np.intersect1d(i_cols, j_cols)

        i_ex_rows = np.setdiff1d(i_rows, ol_rows)
        i_ex_cols = np.setdiff1d(i_cols, ol_cols)

        j_ex_rows = np.setdiff1d(j_rows, ol_rows)
        j_ex_cols = np.setdiff1d(j_cols, ol_cols)

        # masking others

        other_rows = [r for r in range(self.m) if r not in i_rows and r not in j_rows]
        other_cols = [c for c in range(self.n) if c not in i_cols and c not in j_cols]
        X_pd_other = lil_matrix(self.X_train.shape)
        X_pd_other[other_rows, :] = 1
        X_pd_other[:, other_cols] = 1

        settings = ([(X_pd_other, [0, 0], f'X_pd_other')])
        show_matrix(settings, colorbar=True, discrete=True, center=True, clim=[0, 3], scaling=0.5)

        # merge basis_id and best_id

        u0 = lil_matrix((self.m, 1))
        u0[i_ex_rows] = 1

        v0 = lil_matrix((self.n, 1))
        v0[i_ex_cols] = 1

        u1 = lil_matrix((self.m, 1))
        u1[j_ex_rows] = 1

        v1 = lil_matrix((self.n, 1))
        v1[j_ex_cols] = 1

        basis_list = [[u0.T, v0.T], [u1.T, v1.T]]

        dimension_order = [1, 0]

        if u0.sum() == 0 or u1.sum() == 0:
            dimension_order = [1, 0]
        elif v0.sum() == 0 or v1.sum() == 0:
            dimension_order = [0, 1]

        pattern_order = [0, 1]

        # self.set_factors(basis_id, u=0, v=0)
        # self.set_factors(best_id, u=0, v=0)

        # self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)

        # is_improving = True
        # best_score = 0
        # n_iter = 0
        # n_stop = 0

        dimension_order.extend(dimension_order)

        for basis_dim in dimension_order:

            for t in pattern_order:

                target_dim = 1 - basis_dim

                cover_before = coverage_score(
                    gt=self.X_train, 
                    pd=X_pd_other, 
                    w=self.w_now, 
                    axis=basis_dim
                )

                score, basis_list[t][target_dim] = Asso.get_vector(
                    X_gt=self.X_train, 
                    X_old=X_pd_other, 
                    s_old=cover_before, 
                    basis=basis_list[t][basis_dim], 
                    basis_dim=basis_dim, 
                    w=self.w_now)

                # # check if it improves
                # if score > best_score:
                #     # counting iterations and stagnations
                #     n_iter += 1
                #     n_stop = 0
                #     print("[I]   iter: {}, basis_dim: {}, w: {}, score: {:.2f} -> {:.2f}".format(n_iter, basis_dim, self.w_now, best_score, score))
                # else:
                #     # break if it stops improving in the last 2 updates
                #     n_stop += 1
                #     print("[I]   iter: {}, basis_dim: {}, w: {}, stop: {}".format(n_iter, basis_dim, self.w_now, n_stop))
                #     # original
                #     if n_stop == 2:
                #         is_improving = False
                #         break

                X_i = matmul(basis_list[0][0].T, basis_list[0][1], sparse=True, boolean=True)
                X_j = matmul(basis_list[1][0].T, basis_list[1][1], sparse=True, boolean=True)

                settings = ([(X_i + X_j, [0, 0], f'basis_dim: {basis_dim}, pattern: {t}')])
                
                show_matrix(settings, colorbar=True, discrete=True, center=True, clim=[0, 2], scaling=0.5)

        self.set_factors(i, u=basis_list[0][0].T, v=basis_list[0][1].T)
        self.set_factors(j, u=basis_list[1][0].T, v=basis_list[1][1].T)



    def check_merge(self, U_all, V_all, basis_id, overlaps):
        has_merged = False
        if len(overlaps) == 0:
            return has_merged
        
        sc = np.zeros((len(overlaps), 3))
        dl = np.zeros((len(overlaps), 3))
        
        for i, j in enumerate(overlaps):
            sc_none, sc_pattern, sc_merged, dl_none, dl_pattern, dl_merged = self.exclusive_cost(
                U_all=U_all, 
                V_all=V_all, 
                basis_ids=[basis_id, j]
            )
            print(f"[I] if merge {basis_id}, {j}:", sc_none, sc_pattern, sc_merged, dl_none, dl_pattern, dl_merged)

            sc[i] = [sc_none, sc_pattern, sc_merged]
            dl[i] = [dl_none, dl_pattern, dl_merged]

        # find best id
        sc_diff = sc[:, 1] - sc[:, 2]
        dl_diff = dl[:, 1] - dl[:, 2]
        best_id = overlaps[np.argmax(dl_diff)]

        if np.max(dl_diff) < 0:
            return has_merged

        # merge basis_id and best_id

        u0, v0 = U_all[:, basis_id], V_all[:, basis_id]
        u1, v1 = U_all[:, best_id], V_all[:, best_id]
        u_merged = add(u0, u1, sparse=True, boolean=True)
        v_merged = add(v0, v1, sparse=True, boolean=True)
        basis_list = [u_merged.T, v_merged.T]

        self.set_factors(basis_id, u=0, v=0)
        self.set_factors(best_id, u=0, v=0)

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)

        is_improving = True
        best_score = 0
        n_iter = 0
        n_stop = 0
        while is_improving:
            for basis_dim in [1, 0]:
                target_dim = 1 - basis_dim
                cover_before = coverage_score(
                    gt=self.X_train, 
                    pd=self.X_pd, 
                    w=self.w_now, 
                    axis=basis_dim, 
                )

                score, basis_list[target_dim] = Asso.get_vector(
                    X_gt=self.X_train, 
                    X_old=self.X_pd, 
                    s_old=cover_before, 
                    basis=basis_list[basis_dim], 
                    basis_dim=basis_dim, 
                    w=self.w_now, 
                )

                # check if it improves
                if score > best_score:
                    # counting iterations and stagnations
                    n_iter += 1
                    n_stop = 0
                    print("[I]   iter: {}, basis_dim: {}, w: {}, score: {:.2f} -> {:.2f}".format(n_iter, basis_dim, self.w_now, best_score, score))
                else:
                    # break if it stops improving in the last 2 updates
                    n_stop += 1
                    print("[I]   iter: {}, basis_dim: {}, w: {}, stop: {}".format(n_iter, basis_dim, self.w_now, n_stop))
                    # original
                    if n_stop == 2:
                        is_improving = False
                        break


        self.set_factors(best_id, u=basis_list[0].T, v=basis_list[1].T)
        has_merged = True

        return has_merged



    def exclusive_cost(self, U_all, V_all, basis_ids):
        assert U_all.shape[1] == V_all.shape[1]
        k = U_all.shape[1]

        other_ids = [j for j in range(k) if j not in basis_ids]
        if len(other_ids) == 0:
            X_pd_others = lil_matrix(self.X_train.shape)
        else:
            X_pd_others = matmul(
                U=U_all[:, other_ids],
                V=V_all[:, other_ids].T,
                sparse=True,
                boolean=True
            )

        u, v = U_all[:, basis_ids], V_all[:, basis_ids]
        X_pattern = matmul(U=u, V=v.T, sparse=True, boolean=True)

        u_merged = lil_matrix(X_pattern.sum(axis=1) > 0, dtype=int)
        v_merged = lil_matrix(X_pattern.sum(axis=0) > 0, dtype=int).T
        X_merged = matmul(
            U=u_merged, 
            V=v_merged.T, 
            sparse=True, 
            boolean=True
        )

        X_pattern[X_pd_others.astype(bool)] = 0
        X_merged[X_pd_others.astype(bool)] = 0

        # description length of none/pattern/merged, score when w_uc = w_oc = 1
        sc_none = self.X_train[X_merged.astype(bool)].sum() # under-covered
        dl_none = 0 + sc_none

        sc_pattern = self.X_train[X_merged.astype(bool)].sum() - self.X_train[X_pattern.astype(bool)].sum() # under-covered
        sc_pattern += X_pattern.sum() - self.X_train[X_pattern.astype(bool)].sum() # over-covered
        dl_pattern = u.sum() + v.sum() + sc_pattern

        sc_merged = X_merged.sum() - self.X_train[X_merged.astype(bool)].sum() # over-covered
        dl_merged = u_merged.sum() + v_merged.sum() + sc_merged

        return sc_none, sc_pattern, sc_merged, dl_none, dl_pattern, dl_merged


    def get_overlaps(self, U_all, V_all, basis_id, overlap_dims):
        '''Find overlappings on any or both dimensions.

        U_all : m-by-k' matrix
        V_all : n-by-k' matrix
        '''
        assert U_all.shape[1] == V_all.shape[1]
        k = U_all.shape[1]

        overlaps = []
        u, v = U_all[:, basis_id], V_all[:, basis_id]

        for i in range(k):
            if i == basis_id:
                continue

            ol_rows = U_all[:, i][u.astype(bool)]
            ol_cols = V_all[:, i][v.astype(bool)]

            is_ol_rows, is_ol_cols = ol_rows.sum() > 0, ol_cols.sum() > 0

            if overlap_dims == 1:
                is_overlapped = is_ol_rows or is_ol_cols
            else:
                is_overlapped = is_ol_rows and is_ol_cols

            if is_overlapped:
                overlaps.append(i)

        return np.array(overlaps)

    def update_basis(self, basis_dim):
        '''Use the basis of `basis_dim` to update its counterpart's basis.

        Parameters
        ----------
        basis_dim : int
        '''
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        target_dim = 1 - basis_dim
        cover_before = coverage_score(
            gt=self.X_train, 
            pd=self.X_pd, 
            w=self.w_now, 
            axis=basis_dim
        )

        for i in range(self.n_basis):
            self.scores[i], self.basis_list[target_dim][i] = Asso.get_vector(
                X_gt=self.X_train, 
                X_old=self.X_pd, 
                s_old=cover_before, 
                basis=self.basis_list[basis_dim][i], 
                basis_dim=basis_dim, 
                w=self.w_now, 
            )