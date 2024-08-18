from .ContinuousModel import ContinuousModel
from .NMFSklearn import NMFSklearn
import numpy as np
from ..utils import binarize, matmul, to_dense, to_interval, ismat, get_prediction, multiply, sigmoid, ignore_warnings, show_factor_distribution
from ..solvers import line_search
from scipy.sparse import csr_matrix
from tqdm import tqdm 


class FastStep(ContinuousModel):
    '''The FastStep algorithm. The projected line search is applied.

    .. topic:: Reference

        FastStep: Scalable Boolean matrix decomposition.
    '''
    def __init__(self, k, U=None, V=None, W='full', tau=20, solver='line-search', tol=0, min_diff=1e-2, max_round=30, max_iter=50, init_method='uniform', normalize_method=None, seed=None):
        self.check_params(k=k, U=U, V=V, W=W, tau=tau, solver=solver, tol=tol, min_diff=min_diff, max_round=max_round, max_iter=max_iter, init_method=init_method, normalize_method=normalize_method, seed=seed)

    
    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        assert self.solver in ['line-search']
        assert self.init_method in ['uniform']
        assert self.normalize_method in ['balance', 'matrixwise-normalize', 'columnwise-normalize', 'matrixwise-mapping', 'columnwise-mapping', None]
        assert ismat(self.W) or self.W in ['mask', 'full']


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        # debug: normalize to [1e-5, 0.01] as in the paper
        self.U = to_interval(self.U, 1e-5, 0.01)
        self.V = to_interval(self.V, 1e-5, 0.01)
    
        self._fit()

        self.X_pd = binarize(self.U @ self.V.T, self.tau)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


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
                new_fval = self.F(x_last, k=k) # initial value

                desc = "[I] round: {}, k: {}, iter: {}, error: {:.3f}".format(n_round, k, n_iter, new_fval)
                pbar = tqdm(total=self.max_iter, position=0, desc=desc, leave=False)
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
                    error_last = old_fval
                    error = new_fval # due to projection, error might oscillate
                    diff = np.abs(error - error_last)

                    self.print_msg("    Wolfe line search iter       : {}".format(n_iter))
                    self.print_msg("    num of function evals        : {}".format(fc))
                    self.print_msg("    num of gradient evals        : {}".format(gc))
                    self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))

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

                    # early stop detection
                    is_improving = self.early_stop(n_iter=n_iter, diff=diff, error=error, verbose=False)

                    # update pbar
                    pbar.update(1)
                    desc = "[I] round: {}, k: {}, iter: {}, error: {:.3f}".format(n_round, k, n_iter, error)
                    pbar.set_description(desc)

            # early stop detection (on n_round and error)
            is_factorizing = self.early_stop(n_round=n_round, error=error)

            # display
            if self.verbose and self.display:
                self.X_pd = binarize(self.U @ self.V.T, self.tau)
                self.show_matrix(settings=[(self.X_pd, [0, 0], 'pd')], title=f"iter {n_iter}")
                show_factor_distribution(U=self.U, V=self.V)


    def early_stop(self, error=None, diff=None, n_round=None, n_iter=None, n_factor=None, msg=None, k=None, verbose=True):
        '''Early stop detection wrapper.
        '''
        is_improving = super().early_stop(error=error, diff=diff, n_iter=n_iter, n_factor=n_factor, msg=msg, k=k, verbose=verbose)
        if n_round is not None and hasattr(self, 'max_round') and n_round > self.max_round:
            self._early_stop(msg="Reach maximum round", k=k, verbose=verbose)
            is_improving = False

        return is_improving
    

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