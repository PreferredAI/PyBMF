from .BaseModel import BaseModel
from .NMF import NMF
import numpy as np
from scipy.optimize import line_search
from utils import matmul, check_sparse


class binaryMFdense(BaseModel):
    '''Binary matrix factorization
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, X, k, X_missing=None, U_idx=None, V_idx=None, algorithm='threshold', 
                 reg=None, reg_growth=None, # for 'penalty' only
                 lamda=None, u=None, v=None, # for 'threshold' only
                 eps=None, max_iter=None, # shared params
                 display_flag=False, seed=None) -> None:
        self.check_params(X=X, k=k, X_missing=X_missing, U_idx=U_idx, V_idx=V_idx, algorithm=algorithm, 
                          reg=reg, reg_growth=reg_growth, 
                          lamda=lamda, u=u, v=v, 
                          eps=eps, max_iter=max_iter, 
                          display_flag=display_flag, seed=seed)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        self.X = check_sparse(self.X, sparse=False) # to dense
        if 'algorithm' in kwargs:
            self.algorithm = kwargs.get('algorithm')
            print("[I] algorithm    :", self.algorithm)
        if self.algorithm == 'penalty':
            # check reg
            self.reg = kwargs.get('reg') # 'lambda' in paper
            if self.reg is None:
                self.reg = 10
            print("[I] reg          :", self.reg)
            # check reg_growth
            self.reg_growth = kwargs.get('reg_growth') # reg growing rate
            if self.reg_growth is None:
                self.reg_growth = 10
            print("[I] reg_growth   :", self.reg_growth)
            # check eps
            self.eps = kwargs.get('eps') # error tolerance 'epsilon' in paper
            if self.eps is None:
                self.eps = 0.01
            print("[I] eps          :", self.eps)
            # check max_iter
            self.max_iter = kwargs.get('max_iter') # max iteration
            if self.max_iter is None:
                self.max_iter = 100
            print("[I] max_iter     :", self.max_iter)
        elif self.algorithm == 'threshold':
            # initial threshold u and v
            self.u = kwargs.get("u")
            self.v = kwargs.get("v")
            self.u = 0.8 if self.u is None else self.u
            self.v = 0.8 if self.v is None else self.v
            print("[I] initial u, v :", [self.u, self.v])
            # check lamda
            self.lamda = kwargs.get('lamda') # 'lambda' for sigmoid function
            if self.lamda is None:
                self.lamda = 10
            print("[I] lamda        :", self.lamda)
            # check eps
            self.eps = kwargs.get('eps') # step size threshold
            if self.eps is None:
                self.eps = 1e-6
            print("[I] eps          :", self.eps)
            # check max_iter
            self.max_iter = kwargs.get('max_iter') # max iteration
            if self.max_iter is None:
                self.max_iter = 100
            print("[I] max_iter     :", self.max_iter)
        else:
            print("[E] binaryMF algorithm can only be 'penalty' or 'threshold'.")
    

    def solve(self):
        self.initialize()
        self.normalize()
        if self.algorithm == 'penalty':
            self.penalty_algorithm()
        elif self.algorithm == 'threshold':
            self.threshold_algorithm()
        

    def initialize(self):
        nmf = NMF(X=self.X, k=self.k, init='random', max_iter=2000, seed=None)
        nmf.solve()
        self.U = nmf.U
        self.V = nmf.V
        print("[D] max U:", np.max(self.U), "max V:", np.max(self.V))
        if self.display_flag:
            X = matmul(self.U, self.V, sparse=False, boolean=False) # dense matrix
            settings = [(self.U, [1, 0], "U"),
                        (self.V, [0, 1], "V"),
                        (X, [1, 1], "X"),
                        (self.X, [1, 2], "ground truth")]
            super().show_matrix(settings=settings, title="initialization")


    def normalize(self):
        diag_U = np.sqrt(np.max(self.U, axis=0))
        diag_V = np.sqrt(np.max(self.V, axis=1))
        for i in range(self.k):
            self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
            self.V[i, :] = self.V[i, :] * diag_U[i] / diag_V[i]
        print("[D] max U:", np.max(self.U), "max V:", np.max(self.V))
        if self.display_flag:
            X = matmul(self.U, self.V, sparse=False, boolean=False) # dense matrix
            settings = [(self.U, [1, 0], "U"),
                        (self.V, [0, 1], "V"),
                        (X, [1, 1], "X"),
                        (self.X, [1, 2], "ground truth")]
            super().show_matrix(settings=settings, title="normalization")


    def penalty_algorithm(self):
        '''An alternative minimization algorithm minimizing J(U, V), or 'J(W, H)' in the paper.
        '''
        errors = []
        n_iter = 0
        while True:
            error = self.penalty_error()
            errors.append(error)
            print("[I] n_iter: {}, reg: {}, error: {}".format(n_iter, self.reg, error))

            n_iter += 1

            if error < self.eps or n_iter > self.max_iter:
                break
            
            self.penalty_update_V()
            self.penalty_update_U()
            self.reg *= self.reg_growth

            if n_iter % 50 == 0:
                self.show_matrix(title="penalty iter {}".format(n_iter))


    def penalty_update_U(self):
        '''Multiplicative update of U.
        '''
        numerator = np.matmul(self.X, self.V.T) + 3 * self.reg * np.power(self.U, 2)
        denominator = np.matmul(self.U, np.matmul(self.V, self.V.T)) + 2 * self.reg * np.power(self.U, 3) + self.reg * self.U
        if np.any(denominator == 0.0):
            print("[W] Zero(s) detected in denominator.")
            denominator[denominator == 0.0] = np.min(denominator[denominator > 0.0]) # set a non-zero value
        self.U = np.multiply(self.U, numerator)
        self.U = np.divide(self.U, denominator)


    def penalty_update_V(self):
        '''Multiplicative update of V.
        '''
        numerator = np.matmul(self.U.T, self.X) + 3 * self.reg * np.power(self.V, 2)
        denominator = np.matmul(np.matmul(self.U.T, self.U), self.V) + 2 * self.reg * np.power(self.V, 3) + self.reg * self.V
        if np.any(denominator == 0.0):
            print("[W] Zero(s) detected in denominator.")
            denominator[denominator == 0.0] = np.min(denominator[denominator > 0.0]) # set a non-zero value
        self.V = np.multiply(self.V, numerator)
        self.V = np.divide(self.V, denominator)


    def penalty_error(self):
        '''Error for penalty function algorithm.
        '''
        error = 0
        # reconstrunction error term
        rec_error = np.sum((self.X - self.U @ self.V) ** 2)
        # regularization term
        reg_error = np.sum((self.U ** 2 - self.U) ** 2)
        reg_error += np.sum((self.V ** 2 - self.V) ** 2)

        error = 0 * rec_error + 1 * reg_error # choose the error(s) in count
        print("[I] rec_error: {}, reg_error: {}".format(rec_error, reg_error))
        return error
    

    def threshold_algorithm(self):
        '''A gradient descent method minimizing F(u, v), or 'F(w, h)' in the paper.
        '''
        # x0 = np.array([self.u, self.v]) # initial params
        x_last = np.array([0.8, 0.2])
        p_last = - self.dF(x_last) # initial gradient

        us = []
        vs = []
        Fs = []
        ds = []
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
                print("[W] Reached max iteration count.")
                break
            if diff < self.eps:
                print("[W] Reached min step size.")
                break

        self.U = self.step_function(self.U, self.u)
        self.V = self.step_function(self.V, self.v)
    

    def F(self, params):
        '''
        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        rec = self.sigmoid_function(self.U - u) @ self.sigmoid_function(self.V - v)
        F = 0.5 * np.sum((self.X - rec) ** 2)
        return F
    

    def dF(self, params):
        '''
        params = [u, v]
        return = [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        sigmoid_U = self.sigmoid_function(self.U - u)
        sigmoid_V = self.sigmoid_function(self.V - v)

        dFdU = self.X @ sigmoid_V.T - sigmoid_U @ (sigmoid_V @ sigmoid_V.T)
        dUdu = self.dXdx(self.U, u)
        dFdu = np.multiply(dFdU, dUdu)

        dFdV = sigmoid_U.T @ self.X - (sigmoid_U.T @ sigmoid_U) @ sigmoid_V
        dVdv = self.dXdx(self.V, v)
        dFdv = np.multiply(dFdV, dVdv)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF
    

    def step_function(self, X, threshold):
        '''Heaviside step function, 'theta' function in the paper.
        '''
        X[X >= threshold] = 1.0
        X[X < threshold] = 0.0
        return X
    

    def sigmoid_function(self, X):
        '''Sigmoid function. 'phi' function in the paper.
        '''
        X = 1 / (1 + np.exp(-self.lamda * X))
        return X


    def dXdx(self, X, x):
        '''The fractional term in the gradient.
                      dU*     dV*     dW*     dH*
        This computes --- and ---, or --- and --- as noted in the paper. 
                      du      dv      dw      dh
        '''
        diff = X - x # compute X* - x, in which X* = sigmoid(X - x)
        numerator = np.exp(-self.lamda * diff) * self.lamda
        denominator_inv = self.sigmoid_function(diff) ** 2
        return np.multiply(numerator, denominator_inv)


    def show_matrix(self, title=None):
        X = matmul(self.U, self.V, sparse=False, boolean=False) # dense matrix
        settings = [(self.U, [1, 0], "U"),
                    (self.V, [0, 1], "V"),
                    (X, [1, 1], "X"),
                    (self.X, [1, 2], "ground truth")]
        super().show_matrix(settings=settings, scaling=1, pixels=5, title=title)
