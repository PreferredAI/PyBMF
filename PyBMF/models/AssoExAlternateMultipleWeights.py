import numpy as np
from ..utils import matmul, add, to_sparse, cover, show_matrix
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
import pandas as pd
from ..utils import record, isnum


class AssoExAlternateMultipleWeights(Asso):
    '''The Asso algorithm with alternative update between the two factors (experimental).
    '''
    def __init__(self, k, w, init_method='asso', tau=None, p=None, n_basis=None, seed=None):
        '''
        Parameters
        ----------
        k : int
            The rank.
        w : float, or list of float in [0, 1]
            The overcoverage (FP) penalty weight in the objective function during factorization. 
            It's also the lower bound of true positive ratio at each iteration of factorization. 
            If there's more than 1 `w`, we will find matching patterns with all these values and find the best among them.
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
        '''
        self.check_params(k=k, w=w, init_method=init_method, tau=tau, p=p, n_basis=n_basis, seed=seed)
        

    def check_params(self, **kwargs):
        super(Asso, self).check_params(**kwargs)

        # check init_method
        assert self.init_method in ['asso', 'random_rows', 'random_bits']
        if self.init_method == 'asso':
            assert self.tau is not None
        if self.init_method == 'random_bits':
            assert self.p is not None
        # check weights
        if isnum(self.w):
            self.w = [self.w]


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(Asso, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        self._init_factors()
        self._init_logs()

        if self.init_method == 'asso':
            if self.n_basis is None:
                self.n_basis = self.n
                print("[I] n_basis updated to : {}".format(self.n_basis))

            # n_basis should be no greater than the dimension of association matrix
            assert self.n_basis <= self.n

            # real-valued association matrix
            self.assoc = self.build_assoc(X=self.X_train, dim=1)
            # binary-valued basis candidates
            self.basis = self.build_basis(assoc=self.assoc, tau=self.tau)

            if self.n_basis < self.n:
                # down-sampling
                idx = np.random.choice(self.n, size=self.n_basis, replace=False)
                self.basis = self.basis[idx, :]

        elif self.init_method == 'random_rows':
            if self.n_basis is None:
                self.n_basis = min(self.m, self.n)
                print("[I] n_basis updated to : {}".format(self.n_basis))

            # n_basis should be no greater than the number of rows of X_train
            assert self.n_basis <= self.m

            # down-sampling
            idx = np.random.choice(self.m, size=self.n_basis, replace=False)
            self.basis = self.X_train[idx, :]

        elif self.init_method == 'random_bits':
            if self.n_basis is None:
                self.n_basis = min(self.m, self.n)
                print("[I] n_basis updated to : {}".format(self.n_basis))

            # sampling
            self.basis = np.random.choice([0, 1], size=(self.n_basis, self.n), p=[1 - self.p, self.p])
            self.basis = lil_matrix(self.basis, dtype=np.float64)

        # debug
        # if self.verbose:
        settings = [(self.basis, [0, 0], 'basis')]
        self.show_matrix(settings, colorbar=True, clim=[0, 1], title=f'tau: {self.tau}')


    def distribute(self):
        n_weights = len(self.w)

        self.basis_trials = []
        self.scores_trials = np.tile(self.scores, (n_weights, 1))

        for _ in self.w:
            self.basis_trials.append(self.basis.copy())


    def collect(self, i):
        '''
        Let i-th weigth be the main track.
        '''
        self.basis = self.basis_trials[i]
        self.scores = self.scores_trials[i]


    def _fit(self):
        # self.n_basis = self.basis.shape[0]
        self.basis = [lil_matrix((self.n_basis, self.m)), self.basis]
        self.scores = np.zeros(self.n_basis)

        # backup initialized factors
        self.basis_backup = self.basis.copy()
        
        for k in tqdm(range(self.k), position=0):
            print("[I] k   : {}".format(k))

            # restore initialized factors
            self.basis = self.basis_backup.copy()
            # TODO: re-initialize

            # index and score of the best basis
            best_idx = None
            best_score = 0 if k == 0 else best_score

            n_iter = 0
            n_stop = 0

            # n_trials = 0
            # trial_interval = 3

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
                    
                    # varying w
                    # self.w = 0.3 + 0.02 * n_iter

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
                        print("[I] iter: {}, dim: {}, score: {:.2f} -> {:.2f}".format(n_iter, basis_dim, best_score, score))
                    else:
                        # break if it stops improving in the last 2 updates
                        n_stop += 1
                        print("[I] iter: {}, dim: {}, stop: {}".format(n_iter, basis_dim, n_stop))
                        if n_stop == 2:
                            is_improving = False
                            break
                        else:
                            continue

                    # index and score of the best basis
                    best_score = score
                    best_idx = np.argmax(self.scores)

                    # evaluate
                    self.set_factors(k, u=self.basis[0][best_idx, :].T, v=self.basis[1][best_idx, :].T)
                    self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
                    self.show_matrix(colorbar=True, clim=[0, 1], title=f'k: {k}, w: {self.w}')
                    self.evaluate(df_name='updates', head_info={'k': k, 'iter': n_iter, 'factor': 1-basis_dim, 'index': best_idx}, train_info={'score': best_score})
                    self.set_factors(k, u=0, v=0)

                    # debug
                    if self.verbose:
                        self.show_matrix([(self.basis[0], [0, 0], f'k = {k}'), (self.basis[1], [0, 1], f'k = {k}')])

                    # record the scores of all patterns
                    record(df_dict=self.logs, df_name='scores', columns=np.arange(self.n_basis).tolist(), records=self.scores.tolist(), verbose=self.verbose)

            # update factors
            if best_idx is None:
                print("[W] Score stops improving at k: {}".format(k))
            else:
                self.set_factors(k, u=self.basis[0][best_idx, :].T, v=self.basis[1][best_idx, :].T)
            
            self.evaluate(df_name='results', head_info={'k': k, 'iter': n_iter, 'index': best_idx}, train_info={'score': best_score})


    def update_basis(self, basis_dim):
        '''Use the basis of `basis_dim` to update its counterpart's basis.

        Parameters
        ----------
        basis_dim : int
        '''
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
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
            

    def update_trials(self, basis_dim):
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        target_dim = 1 - basis_dim

        for j, w in enumerate(self.w):
            cover_before = cover(gt=self.X_train, pd=self.X_pd, w=w, axis=basis_dim)

            for i in range(self.n_basis):
                self.scores_trials[j][i], self.basis_trials[j][target_dim][i] = Asso.get_vector(
                    X_gt=self.X_train, 
                    X_old=self.X_pd, 
                    s_old=cover_before, 
                    basis=self.basis_trials[j][basis_dim][i], 
                    basis_dim=basis_dim, 
                    w=w)
                