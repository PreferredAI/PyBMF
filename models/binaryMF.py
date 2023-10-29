from .BaseModel import BaseModel
from .NMF import NMF
import numpy as np
from scipy.optimize import line_search
from utils import multiply, matmul, check_sparse, to_sparse
from scipy.linalg import inv
# from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix


class binaryMF(BaseModel):
    '''Binary matrix factorization
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''        
    def __init__(self, k, algorithm, 
                 reg=None, reg_growth=None,     # for 'penalty' only
                 lamda=None, u=None, v=None,    # for 'threshold' only
                 eps=None, max_iter=None,       # shared params
                ) -> None:
        self.check_params(k=k, algorithm=algorithm, 
                          reg=reg, reg_growth=reg_growth, 
                          lamda=lamda, u=u, v=v, 
                          eps=eps, max_iter=max_iter)
        
    def fit(self, train_set, val_set=None, display=False, seed=None):
        self.check_params(display=display, seed=seed)
        super().check_dataset(train_set=train_set, val_set=val_set)

        self.initialize()
        self.normalize()
        if self.algorithm == 'penalty':
            self.penalty_algorithm()
        elif self.algorithm == 'threshold':
            self.threshold_algorithm()


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
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
        

    def initialize(self, nmf_max_iter=2000, nmf_seed=None):
        nmf = NMF(k=self.k, init='random', max_iter=nmf_max_iter, seed=nmf_seed)
        nmf.fit(train_set=self.train_set)

        self.U = to_sparse(nmf.U, type='csr')
        self.V = to_sparse(nmf.V, type='csr')
        self.X_train = to_sparse(self.X_train, type='csr')

        print("[I] After initialization: max U: {:.3f}, max V: {:.3f}".format(np.max(self.U), np.max(self.V)))
        self.show_matrix(title="after initialization", colorbar=True)


    def normalize(self):
        diag_U = np.sqrt(np.max(self.U, axis=0))
        diag_V = np.sqrt(np.max(self.V, axis=1))
        diag_U = check_sparse(diag_U, sparse=False)
        diag_V = check_sparse(diag_V, sparse=False)
        
        for i in range(self.k):
            self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
            self.V[i, :] = self.V[i, :] * diag_U[i] / diag_V[i]
        print("[I] After normalization: max U: {:.3f}, max V: {:.3f}".format(np.max(self.U), np.max(self.V)))
        self.show_matrix(title="after normalization", colorbar=True)


    def penalty_algorithm(self):
        '''An alternative minimization algorithm minimizing J(U, V), or 'J(W, H)' in the paper.
        '''
        errors = []
        n_iter = 0
        while True:
            error, rec_error, reg_error = self.penalty_error()
            errors.append(error)
            print("[I] iter: {}, reg: {:.3f}, err: {:.3f}, rec_err: {:.3f}, reg_err: {:.3f}".format(n_iter, self.reg, error, rec_error, reg_error))

            n_iter += 1

            if error < self.eps or n_iter > self.max_iter:
                break
            
            self.penalty_update_V()
            self.penalty_update_U()
            self.reg *= self.reg_growth

            if n_iter % 10 == 0:
                self.show_matrix(title=f"penalty iter {n_iter}")

        self.show_matrix(title="after penalty function algorithm")

        self.U = self.step_function(self.U, 0.5)
        self.V = self.step_function(self.V, 0.5)


    def penalty_update_U(self):
        '''Multiplicative update of U.
        '''
        numerator = self.X_train @ self.V.T + 3 * self.reg * self.U.power(2)
        denominator = self.U @ (self.V @ self.V.T) + 2 * self.reg * self.U.power(3) + self.reg * self.U
        denominator_size = denominator.shape[0] * denominator.shape[1]
        if denominator_size != denominator.nnz: # np.any(denominator == 0.0)
            print("[W] Zero(s) detected in denominator.")
            denominator[denominator == 0.0] = np.min(denominator[denominator > 0.0]) # set a non-zero value
        self.U = multiply(self.U, numerator)
        self.U = multiply(self.U, denominator.power(-1))


    def penalty_update_V(self):
        '''Multiplicative update of V.
        '''
        numerator = self.U.T @ self.X_train + 3 * self.reg * self.V.power(2)
        denominator = (self.U.T @ self.U) @ self.V + 2 * self.reg * self.V.power(3) + self.reg * self.V
        denominator_size = denominator.shape[0] * denominator.shape[1]
        if denominator_size != denominator.nnz: # np.any(denominator == 0.0)
            print("[W] Zero(s) detected in denominator.")
            denominator[denominator == 0.0] = np.min(denominator[denominator > 0.0]) # set a non-zero value
        self.V = multiply(self.V, numerator)
        self.V = multiply(self.V, denominator.power(-1))


    def penalty_error(self):
        '''Error for penalty function algorithm.
        '''
        error = 0
        # reconstrunction error term
        rec_error = np.sum((self.X_train - self.U @ self.V).power(2))
        # regularization term
        reg_error = np.sum((self.U.power(2) - self.U).power(2))
        reg_error += np.sum((self.V.power(2) - self.V).power(2))

        error = 0 * rec_error + 1 * reg_error # choose the error(s) in count
        return error, rec_error, reg_error
    

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
        self.show_matrix(title="after thresholding algorithm")
    

    def F(self, params):
        '''
        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        rec = self.sigmoid_function(self.U - u) @ self.sigmoid_function(self.V - v)
        F = 0.5 * np.sum((self.X_train - rec) ** 2)
        return F
    

    def dF(self, params):
        '''
        params = [u, v]
        return = [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        sigmoid_U = self.sigmoid_function(self.U - u)
        sigmoid_V = self.sigmoid_function(self.V - v)

        dFdU = self.X_train @ sigmoid_V.T - sigmoid_U @ (sigmoid_V @ sigmoid_V.T)
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu)

        dFdV = sigmoid_U.T @ self.X_train - (sigmoid_U.T @ sigmoid_U) @ sigmoid_V
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv)

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
        return multiply(numerator, denominator_inv)


    def show_matrix(self, title=None, scaling=1.0, pixels=5, colorbar=False):
        if self.display:
        
            # X = matmul(self.U, self.V, sparse=False, boolean=False)
            if self.algorithm == "penalty":
                u, v = 0.5, 0.5
            elif self.algorithm == "threshold":
                u, v = self.u, self.v

            U = self.U.copy()
            V = self.V.copy()
            U = self.step_function(X=U, threshold=u)
            V = self.step_function(X=V, threshold=v)
            X = matmul(U, V, sparse=False, boolean=True)

            U = check_sparse(U, sparse=False)
            V = check_sparse(V, sparse=False)
            X_train = check_sparse(self.X_train, sparse=False)

            settings = [# (U, [1, 0], "U"),
                        # (V, [0, 1], "V"),
                        (X, [1, 1], "X"),
                        (X_train, [1, 2], "ground truth")]
            super().show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)
