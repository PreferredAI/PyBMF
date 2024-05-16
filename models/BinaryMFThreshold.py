from .ContinuousModel import ContinuousModel
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract
import numpy as np
from solvers import line_search

class BinaryMFThreshold(ContinuousModel):
    '''Binary matrix factorization, thresholding algorithm.
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k, U, V, W='mask', u=0.5, v=0.5, lamda=100, min_diff=1e-3, max_iter=100, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        u, v : float
            Initial threshold for `U` and `V`.
        lamda : float
            The 'lambda' in sigmoid function.
        '''
        self.check_params(k=k, U=U, V=V, W=W, u=u, v=v, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        # self.set_params(['u', 'v', 'lamda'], **kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def init_model(self):
        '''Initialize factors and logging variables.
        '''
        super().init_model()

        self.normalize_UV(method="normalize")

        self._to_dense()
        self._to_float()


    def _fit(self):
        '''The gradient descent method.
        '''
        n_iter = 0
        is_improving = True

        x_last = np.array([self.u, self.v]) # initial point
        p_last = -self.dF(x_last) # descent direction
        new_fval = self.F(x_last) # initial value

        # initial evaluation
        self.predict_X(u=self.u, v=self.v, boolean=True)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'u': self.u, 'v': self.v, 'F': new_fval})
        
        while is_improving:
            n_iter += 1
            xk = x_last # starting point
            pk = p_last # searching direction
            # pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}, start: [{:.3f}, {:.3f}], direction: [{:.3f}, {:.3f}]".format(n_iter, *xk, *pk))

            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50)

            if alpha is None:
                self._early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            self.u, self.v = x_last

            diff = np.sqrt(np.sum((alpha * pk) ** 2))
            
            self.print_msg("[I] Wolfe line search for iter   : {}".format(n_iter))
            self.print_msg("    num of function evals made   : {}".format(fc))
            self.print_msg("    num of gradient evals made   : {}".format(gc))
            self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))
            self.print_msg("    threshold update             : [{:.3f}, {:.3f}] -> [{:.3f}, {:.3f}]".format(*xk, *x_last))
            self.print_msg("    threshold update direction   : [{:.3f}, {:.3f}]".format(*(alpha * pk)))
            self.print_msg("    threshold difference         : {:.6f}".format(diff))

            # evaluate
            self.predict_X(u=self.u, v=self.v, boolean=True)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'u': self.u, 'v': self.v, 'F': new_fval})

            # display
            if self.verbose and self.display and n_iter % 10 == 0:
                self.show_matrix(u=self.u, v=self.v, title=f"iter {n_iter}")

            # early stop detection
            is_improving = self.early_stop(diff=diff)



    

    def F(self, params):
        '''
        Parameters
        ----------
        params : [u, v]

        Returns
        -------
        F : F(u, v)
        '''
        u, v = params

        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)

        diff = self.X_train - U @ V.T
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u, v]

        Returns
        -------
        dF : [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        
        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)
        
        X_gt = self.X_train
        X_pd = U @ V.T

        dFdX = multiply(self.W, X_gt - X_pd)

        dFdU = dFdX @ V
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu) # (m, k)

        dFdV = U.T @ dFdX
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv.T) # (k, n)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    def dXdx(self, X, x):
        '''The fractional term in the gradient.

                      dU*     dV*     dW*     dH*
        This computes --- and --- (or --- and --- as in the paper).
                      du      dv      dw      dh
        
        Parameters
        ----------
        X : X*, sigmoid(X - x) in the paper
        '''
        diff = subtract(X, x)
        num = np.exp(-self.lamda * subtract(X, x)) * self.lamda
        denom_inv = sigmoid(diff * self.lamda) ** 2
        return num * denom_inv
