from .BaseContinuousModel import BaseContinuousModel
from .BaseModel import BaseModel
from .NMF import NMF
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract
import numpy as np
from scipy.sparse import spmatrix


class BinaryMFThreshold(BaseContinuousModel):
    '''Binary matrix factorization, Thresholding algorithm
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k, U, V, W='mask', u=0.5, v=0.5, lamda=100, min_diff=1e-6, max_iter=100, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        u : float
            Initial threshold for `U`.
        v : float
            Initial threshold for `V`.
        lamda : float
            The 'lambda' in sigmoid function.
        model : BaseModel
        '''
        self.check_params(k=k, U=U, V=V, W=W, u=u, v=v, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        self.set_params(['u', 'v', 'lamda'], **kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def _fit(self):
        '''The gradient descent method.
        '''
        x_last = np.array([self.u, self.v]) # initial threshold u, v
        p_last = -self.dF(x_last) # initial gradient dF(u, v)

        n_iter = 0
        
        is_improving = True
        while is_improving:
            xk = x_last # starting point
            pk = p_last # searching direction
            pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}, start: [{:.3f}, {:.3f}], direction: [{:.3f}, {:.3f}]".format(n_iter, *xk, *pk))

            alpha, fc, gc, new_fval, old_fval, new_slope = self.line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50)

            if alpha is None:
                self._early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            self.u, self.v = x_last

            diff = np.sqrt(np.sum((alpha * pk) ** 2))
            
            print("[I] Wolfe line search for iter   : {}".format(n_iter))
            print("    num of function evals made   : {}".format(fc))
            print("    num of gradient evals made   : {}".format(gc))
            print("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))
            print("    threshold update             : [{:.3f}, {:.3f}] -> [{:.3f}, {:.3f}]".format(*xk, *x_last))
            print("    threshold difference         : {:.6f}".format(diff))

            # evaluate
            self.predict_X(u=self.u, v=self.v, boolean=True)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'u': self.u, 'v': self.v, 'F': new_fval})

            # display
            if self.display and n_iter % 10 == 0:
                self.show_matrix(u=self.u, v=self.v, title=f"iter {n_iter}")

            is_improving = self.early_stop(diff=diff)
            n_iter += 1

        # self.show_matrix(u=self.u, v=self.v, title="result")


    def line_search(self, f, myfprime, xk, pk, maxiter=1000, c1=0.1, c2=0.4):
        '''Re-implementation of SciPy's Wolfe line search.

        It's compatible with `scipy.optimize.line_search`.
        >>> from scipy.optimize import line_search
        >>> line_search(f=f, myfprime=myfprime, xk=xk, pk=pk, maxiter=maxiter, c1=c1, c2=c2)
        '''
        alpha = 2
        a, b = 0, 10
        n_iter = 0
        fc, gc = 0, 0

        fk = f(xk)
        gk = myfprime(xk)
        fc, gc = fc + 1, gc + 1

        while n_iter <= maxiter:
            n_iter = n_iter + 1
            x = xk + alpha * pk

            armojo_cond = f(x) - fk <= alpha * c1 * dot(gk, pk)
            fc += 1

            if armojo_cond: # Armijo (Sufficient Decrease) Condition

                curvature_cond = dot(myfprime(x), pk) >= c2 * dot(gk, pk)
                gc += 1

                if curvature_cond: # Curvature Condition
                    break
                else:
                    if b < 10:
                        a = alpha
                        alpha = (a + b) / 2
                    else:
                        alpha = alpha * 1.2
            else:
                b = alpha
                alpha = (a + b) / 2

        new_fval, old_fval, new_slope = f(x), fk, myfprime(x)
        fc, gc = fc + 1, gc + 1

        return alpha, fc, gc, new_fval, old_fval, new_slope
    

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
        
        X_gt = multiply(self.W, self.X_train)
        X_pd = multiply(self.W, U @ V.T)

        dFdU = X_gt @ V - X_pd @ V
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu)

        dFdV = U.T @ X_gt - U.T @ X_pd
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv.T)

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
