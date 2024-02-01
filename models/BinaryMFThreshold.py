from .BinaryMF import BinaryMF
from utils import multiply, step, sigmoid
import numpy as np
from scipy.optimize import line_search


class BinaryMFThreshold(BinaryMF):
    '''Binary matrix factorization, Thresholding algorithm
    
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
        x_last = np.array([self.u, self.v]) # initial threshold u, v
        p_last = -self.dF(x_last) # initial gradient dF(u, v)

        us, vs, Fs, ds = [], [], [], []
        n_iter = 0
        while True:
            xk = x_last # starting point
            pk = p_last # searching direction

            pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}, start from [{:.3f}, {:.3f}], search direction [{:.3f}, {:.3f}]".format(n_iter, *xk, *pk))

            alpha, fc, gc, new_fval, old_fval, new_slope = self.line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk)
            if alpha is None:
                self.early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            self.u, self.v = x_last
            us.append(self.u)
            vs.append(self.v)
            Fs.append(new_fval)
            diff = np.sqrt(np.sum((alpha * pk) ** 2))
            ds.append(diff)
            
            self.print_msg("[I] Wolfe line search for iter   : {}".format(n_iter))
            self.print_msg("    num of function evals made   : {}".format(fc))
            self.print_msg("    num of gradient evals made   : {}".format(gc))
            self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))
            self.print_msg("    threshold update             : [{:.3f}, {:.3f}] -> [{:.3f}, {:.3f}]".format(*xk, *x_last))
            self.print_msg("    threshold difference         : {:.3f}".format(diff))

            n_iter += 1

            if n_iter > self.max_iter:
                self.early_stop("Reached maximum iteration count")
                break
            if diff < self.eps:
                self.early_stop("Difference lower than threshold")
                break

        self.U = step(self.U, self.u)
        self.V = step(self.V, self.v)
        self.show_matrix(title="after thresholding algorithm")


    def line_search(self, f, myfprime, xk, pk, maxiter=1000, c1=0.1, c2=0.4):
        line_search(f=f, myfprime=myfprime, xk=xk, pk=pk, maxiter=maxiter, c1=c1, c2=c2)
    

    def F(self, params):
        '''
        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        U = sigmoid((self.U - u) * self.lamda)
        V = sigmoid((self.V - v) * self.lamda)

        diff = self.X_train - U @ V.T
        F = 0.5 * np.sum(diff ** 2)       
        return F
    

    def dF(self, params):
        '''
        params = [u, v]
        return = [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        sigmoid_U = sigmoid((self.U - u) * self.lamda)
        sigmoid_V = sigmoid((self.V - v) * self.lamda)

        dFdU = self.X_train @ sigmoid_V - sigmoid_U @ (sigmoid_V.T @ sigmoid_V)
        dUdu = self.dXdx(sigmoid_U, u) # original paper
        # dUdu = self.dXdx(self.U, u) # authors' implementation
        dFdu = multiply(dFdU, dUdu)

        dFdV = sigmoid_U.T @ self.X_train - (sigmoid_U.T @ sigmoid_U) @ sigmoid_V.T
        dVdv = self.dXdx(sigmoid_V, v) # original paper
        # dVdv = self.dXdx(self.V, v) # authors' implementation
        dFdv = multiply(dFdV, dVdv.T)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    def dXdx(self, X, x):
        '''The fractional term in the gradient.
                      dU*     dV*     dW*     dH*
        This computes --- and ---, or --- and --- as noted in the paper
                      du      dv      dw      dh
        '''
        diff = X - x # compute X* - x, in which X* = sigmoid(X - x) in original paper
        numerator = np.exp(-self.lamda * diff) * self.lamda
        denominator_inv = sigmoid(diff * self.lamda) ** 2
        return multiply(numerator, denominator_inv)