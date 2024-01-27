from .BinaryMF import BinaryMF
from utils import multiply, step
import numpy as np


class BinaryMFPenalty(BinaryMF):
    '''Binary matrix factorization, Penalty algorithm
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k, reg=None, reg_growth=None, eps=None, max_iter=None):
        self.check_params(k=k, reg=reg, reg_growth=reg_growth, eps=eps, max_iter=max_iter, algorithm='penalty')
        

    def _fit(self):
        self.initialize()
        self.normalize()
        self.penalty_algorithm()


    def penalty_algorithm(self):
        '''An alternative minimization algorithm minimizing J(U, V), or 'J(W, H)' as in BinaryMF paper
        '''
        errors = []
        n_iter = 0
        while True:
            error, rec_error, reg_error = self.penalty_error()
            errors.append(error)
            print("[I] iter: {}, reg: {:.1e}, err: {:.3f}, rec_err: {:.3f}, reg_err: {:.3f}".format(n_iter, self.reg, error, rec_error, reg_error))

            n_iter += 1

            if error < self.eps:
                self.early_stop("Error lower than threshold")
                break

            if n_iter > self.max_iter:
                self.early_stop("Reached maximum iteration")
                break
            
            self.V = self.V.T
            self.penalty_update_V()
            self.penalty_update_U()
            self.V = self.V.T

            self.reg *= self.reg_growth

            if n_iter % 10 == 0:
                self.show_matrix(title=f"penalty iter {n_iter}")

        self.show_matrix(title="after penalty function algorithm")

        # self.U = step(self.U, 0.5)
        # self.V = step(self.V, 0.5)


    def penalty_update_U(self):
        '''Multiplicative update of U

        U: (m, k)
        V: (k, m)
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
        '''Multiplicative update of V

        U: (m, k)
        V: (k, m)
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
        rec_error = np.sum((self.X_train - self.U @ self.V.T).power(2))
        # regularization term
        reg_error = np.sum((self.U.power(2) - self.U).power(2))
        reg_error += np.sum((self.V.power(2) - self.V).power(2))

        error = 0 * rec_error + 1 * reg_error # choose the error(s) in count
        return error, rec_error, reg_error