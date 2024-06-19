from .ContinuousModel import ContinuousModel
from .NMFSklearn import NMFSklearn
import numpy as np
from utils import binarize, matmul, to_dense, to_interval, ismat, get_prediction, multiply, sigmoid, ignore_warnings
from solvers import line_search
from scipy.sparse import csr_matrix


class FastStep(ContinuousModel):
    '''The FastStep algorithm.

    - solver: projected line search
    '''
    def __init__(self, k, U=None, V=None, W='full', tau=20, solver='line-search', tol=0, min_diff=1e-2, max_iter=30, init_method='uniform', seed=None):
        self.check_params(k=k, U=U, V=V, W=W, tau=tau, solver=solver, tol=tol, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)

    
    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        assert self.solver in ['line-search']
        assert self.init_method in ['uniform']
        assert ismat(self.W) or self.W in ['mask', 'full']


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)
    
        self._fit()

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=False)
        self.X_pd = binarize(self.X_pd, self.tau)
        self.finish()


    def init_model(self):
        '''Initialize factors and logging variables.
        '''
        super().init_model()

        self.init_UV()
        
        # self.normalize_UV(method="balance")
        self.normalize_UV(method="normalize")

        # transform into dense float matrices
        self._to_float()
        self._to_dense()

        # debug: normalize to [1e-5, 0.01] as in the paper
        self.U = to_interval(self.U, 1e-5, 0.01)
        self.V = to_interval(self.V, 1e-5, 0.01)


    def _fit(self):
        '''The gradient descent of the k factors.
        '''
        n_round = 0
        is_factorizing = True

        while is_factorizing:
            # update n_round
            n_round += 1

            for k in range(self.k):
                # gradient descent of k-th factor
                n_iter = 0
                is_improving = True

                # initial point for factor k
                u, v = to_dense(self.U[:, k], squeeze=True), to_dense(self.V[:, k], squeeze=True)
                x_last = np.concatenate([u, v])
                p_last = -self.dF(x_last, k=k) # descent direction

                error_before_last = 0
                error_last = 0

                while is_improving:
                    # update n_iter
                    n_iter += 1

                    # set xk, pk
                    xk = x_last # starting point
                    pk = p_last # searching direction

                    alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, args=(), kwargs={'k': k}, maxiter=50)

                    if alpha is None:
                        print("[W] Search direction is not a descent direction.")
                        break

                    # get x_last, p_last
                    x_last = xk + alpha * pk
                    p_last = -new_slope # descent direction

                    # debug: projection
                    eps = 1e-5
                    # eps = np.finfo(np.float64).eps
                    x_last[x_last < eps] = eps
                    p_last = -self.dF(x_last, k=k)
                    new_fval = self.F(x_last, k=k)

                    # update U, V
                    self.U[:, k], self.V[:, k] = x_last[:self.m], x_last[self.m:]

                    # measurement
                    error_before_last = error_last
                    error_last = old_fval
                    error = new_fval
                    diff = np.abs(error - error_last)
                    diff_2 = np.abs(error - error_before_last) # due to projection, error might oscillate

                    # self.print_msg("    Wolfe line search iter       : {}".format(n_iter))
                    # self.print_msg("    num of function evals        : {}".format(fc))
                    # self.print_msg("    num of gradient evals        : {}".format(gc))
                    # self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))

                    # evaluate
                    self.X_pd = binarize(self.U @ self.V.T, self.tau)
                    self.evaluate(
                        df_name='updates', 
                        head_info={
                            'round': n_round, 
                            'k': k, 
                            'iter': n_iter, 
                            'original_F': new_fval, 
                            'projected_F': error, 
                        }, 
                    )

                    # early stop detection (on diff and diff_2)
                    if diff < self.min_diff or diff_2 < self.min_diff:
                        is_improving = False
                        print("[I] round: {}, k: {}, iter: {}, error: {:.3f}, diff: {:.3f}".format(n_round, k, n_iter, error, diff))

                        if k < self.k - 1:
                            self.print_msg("----------------------------------------------------------")
                        else:
                            self.print_msg("==========================================================")

                        # display
                        if self.verbose and self.display:
                            self.X_pd = binarize(self.U @ self.V.T, self.tau)
                            self.show_matrix(u=self.U[:, k], v=self.V[:, k], title=f"iter {n_iter}")
                    else:
                        self.print_msg("round: {}, k: {}, iter: {}, error: {:.3f}, diff: {:.3f}".format(n_round, k, n_iter, error, diff))

            # early stop detection (on n_round and error)
            is_factorizing = self.early_stop(n_iter=n_round+1, error=error)


    @ignore_warnings
    def F(self, params, k):
        '''The objective function.

        Parameters
        ----------
        params : (m + n, ) array
        '''
        # factor being updated
        u, v = params[:self.m], params[self.m:]

        # factor matrices with updated factors
        U, V = self.U.copy(), self.V.copy()
        U[:, k], V[:, k] = u, v

        # transformed ground truth matrix
        M = self.X_train.copy()
        M[M == 0] = -1

        S = U @ V.T
        X = multiply( - M, S - self.tau)

        X = multiply(self.W, X) # masking
        X = np.log(1 + np.exp(X))

        F = np.sum(X)
        return F


    @ignore_warnings
    def dF(self, params, k):
        '''The gradient of the objective function on the k-th factor.

        Parameters
        ----------
        params : (m + n, ) array
        '''
        # factor being updated
        u, v = params[:self.m], params[self.m:]

        # factor matrices with updated factors
        U, V = self.U.copy(), self.V.copy()
        U[:, k], V[:, k] = u, v

        # transformed ground truth matrix
        M = self.X_train.copy()
        M[M == 0] = -1

        S = U @ V.T
        X = multiply(M, S - self.tau)

        # denom = 1 + np.exp(X)
        # X = multiply( - M, 1 / denom)
        X_sigmoid = sigmoid(-X)
        X = multiply( - M, X_sigmoid)

        X = multiply(self.W, X) # masking

        du = X @ np.reshape(v, (-1, 1))
        du = to_dense(du, squeeze=True)

        dv = X.T @ np.reshape(u, (-1, 1))
        dv = to_dense(dv, squeeze=True)

        dF = np.concatenate([du, dv])
        return dF