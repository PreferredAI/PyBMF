from .bMF import bMF
from utils import multiply, step_function, sigmoid_function
import numpy as np
from scipy.optimize import line_search


class bMFThreshold(bMF):
    '''Binary matrix factorization (bMF, binaryMF)
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k, lamda=None, u=None, v=None, eps=None, max_iter=None):
        self.check_params(k=k, lamda=lamda, u=u, v=v, eps=eps, max_iter=max_iter, algorithm='threshold')
        

    def _fit(self):
        self.initialize()
        self.normalize()
        self.threshold_algorithm()


    def threshold_algorithm(self):
        '''A gradient descent method minimizing F(u, v), or 'F(w, h)' in the paper.
        '''
        self.V = self.V.T

        x_last = np.array([0.8, 0.2]) # or [self.u, self.v]
        p_last = - self.dF(x_last) # initial gradient

        us, vs, Fs, ds = [], [], [], []
        n_iter = 0
        while True:
            xk = x_last # start point
            pk = p_last # search direction
            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk)
            if alpha is None:
                print("[W] Search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = - new_slope # descent direction
            self.u, self.v = x_last
            us.append(self.u)
            vs.append(self.v)
            Fs.append(new_fval)
            diff = np.sum((alpha * pk) ** 2)
            ds.append(diff)
            
            print("[I] Wolfe line search for iter   : ", n_iter)
            print("    num of function evals made   : ", fc)
            print("    num of gradient evals made   : ", gc)
            print("    function value update        : ", old_fval, " -> ", new_fval)
            print("    threshold update             : ", xk, " -> ", x_last)
            print("    threshold difference         : ", alpha * pk, "(", diff, ")")

            n_iter += 1

            if n_iter > self.max_iter:
                self.early_stop("Reached maximum iteration count")
                break
            if diff < self.eps:
                self.early_stop("Difference lower than threshold")
                break

        self.U = step_function(self.U, self.u)
        self.V = step_function(self.V, self.v)
        self.V = self.V.T
        self.show_matrix(title="after thresholding algorithm")
    

    def F(self, params):
        '''
        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        U = sigmoid_function(self.U - u, self.lamda)
        V = sigmoid_function(self.V - v, self.lamda)

        # # debug
        # print(type(self.U), self.U.shape, type(self.V), self.V.shape)
        # print(type(U), U.shape, type(V), V.shape)

        rec = U @ V
        F = 0.5 * np.sum((self.X_train - rec) ** 2)
        return F
    

    def dF(self, params):
        '''
        params = [u, v]
        return = [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        sigmoid_U = sigmoid_function(self.U - u, self.lamda)
        sigmoid_V = sigmoid_function(self.V - v, self.lamda)

        dFdU = self.X_train @ sigmoid_V.T - sigmoid_U @ (sigmoid_V @ sigmoid_V.T)
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu)

        dFdV = sigmoid_U.T @ self.X_train - (sigmoid_U.T @ sigmoid_U) @ sigmoid_V
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    def dXdx(self, X, x):
        '''The fractional term in the gradient.
                      dU*     dV*     dW*     dH*
        This computes --- and ---, or --- and --- as noted in the paper
                      du      dv      dw      dh
        '''
        diff = X - x # compute X* - x, in which X* = sigmoid(X - x)
        numerator = np.exp(-self.lamda * diff) * self.lamda
        denominator_inv = sigmoid_function(diff, self.lamda) ** 2
        return multiply(numerator, denominator_inv)