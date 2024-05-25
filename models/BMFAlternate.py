import numpy as np
from utils import matmul, add, to_sparse, cover, show_matrix
from .Asso import Asso
from .BMFAlternateTools import BMFAlternateTools, w_scheduler
from scipy.sparse import lil_matrix
from tqdm import tqdm
import pandas as pd
from utils import record, isnum, ignore_warnings


class BMFAlternate(BMFAlternateTools):
    '''The Asso algorithm with alternative update between the two factors (experimental).

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
        self.finish(save_model=False)


    def init_model(self):
        self._init_factors()
        self._init_logs()

        self.init_cover()
        self.init_basis()

        self.init_basis_list()
        self.scores = np.zeros(self.n_basis)

        self.w_scheduler = w_scheduler(w_list=self.w_list)


    # def distribute(self):
    #     n_weights = len(self.w_now)

    #     self.basis_trials = []
    #     self.scores_trials = np.tile(self.scores, (n_weights, 1))

    #     for _ in self.w_now:
    #         self.basis_trials.append(self.basis.copy())


    # def collect(self, i):
    #     '''
    #     Let i-th weigth be the main track.
    #     '''
    #     self.basis = self.basis_trials[i]
    #     self.scores = self.scores_trials[i]


    def _fit(self):

        # backup initialized factors
        self.basis_list_backup = self.basis_list.copy()
        
        for k in tqdm(range(self.k), position=0):
            print("[I] k   : {}".format(k))

            
            if k > 0 and self.re_init: # restore initialized factors
                self.update_cover(U=self.U, V=self.V)
                self.init_basis()
                self.init_basis_list()
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
                    # debug
                    # if n_trials == trial_interval:
                    #     n_trials = 0
                    #     self.collect(i=idx_best_w)
                    #     self.distribute()
                    # else:
                    #     n_trials += 1

                    # debug


                    # update basis of basis_dim
                    self.update_basis(basis_dim)
                    # # debug
                    # self.update_trials(basis_dim)

                    # highest score among all patterns
                    score = np.max(self.scores)
                    # # debug
                    # score = np.max(self.scores_trials, axis=0)
                    # idx_best_w = np.argmax(self.scores_trials, axis=0)

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
                    self.update_factors(k, u=self.basis_list[0][best_idx, :].T, v=self.basis_list[1][best_idx, :].T)
                    self.predict_X()
                    if self.verbose:
                        self.show_matrix(colorbar=True, clim=[0, 1], title=f'k: {k}, w: {self.w_now}')
                    self.evaluate(df_name='updates', head_info={'k': k, 'iter': n_iter, 'factor': 1-basis_dim, 'index': best_idx}, train_info={'score': best_score})
                    self.update_factors(k, u=0, v=0)

                    # debug
                    if self.verbose:
                        self.show_matrix([(self.basis_list[0], [0, 0], f'k = {k}'), (self.basis_list[1], [0, 1], f'k = {k}')])

                    # record the scores of all patterns
                    record(df_dict=self.logs, df_name='scores', columns=np.arange(self.n_basis).tolist(), records=self.scores.tolist(), verbose=self.verbose)

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                self.update_factors(k, u=self.basis_list[0][best_idx, :].T, v=self.basis_list[1][best_idx, :].T)
            
            self.evaluate(df_name='results', head_info={'k': k, 'iter': n_iter, 'index': best_idx}, train_info={'score': best_score})


    def update_basis(self, basis_dim):
        '''Use the basis of `basis_dim` to update its counterpart's basis.

        Parameters
        ----------
        basis_dim : int
        '''
        self.predict_X()
        target_dim = 1 - basis_dim
        cover_before = cover(gt=self.X_train, pd=self.X_pd, w=self.w_now, axis=basis_dim)

        for i in range(self.n_basis):
            self.scores[i], self.basis_list[target_dim][i] = Asso.get_vector(
                X_gt=self.X_train, 
                X_old=self.X_pd, 
                s_old=cover_before, 
                basis=self.basis_list[basis_dim][i], 
                basis_dim=basis_dim, 
                w=self.w_now)
            

    # def update_trials(self, basis_dim):
    #     self.predict_X()
    #     target_dim = 1 - basis_dim

    #     for j, w in enumerate(self.w_now):
    #         cover_before = cover(gt=self.X_train, pd=self.X_pd, w=w, axis=basis_dim)

    #         for i in range(self.n_basis):
    #             self.scores_trials[j][i], self.basis_trials[j][target_dim][i] = Asso.get_vector(
    #                 X_gt=self.X_train, 
    #                 X_old=self.X_pd, 
    #                 s_old=cover_before, 
    #                 basis=self.basis_trials[j][basis_dim][i], 
    #                 basis_dim=basis_dim, 
    #                 w=w)
                