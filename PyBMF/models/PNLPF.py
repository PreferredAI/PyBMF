from .BinaryMFPenalty import BinaryMFPenalty, error, rec_error, reg_error
from ..utils import sigmoid, d_sigmoid, power, multiply, ignore_warnings, subtract, get_prediction
from .ContinuousModel import ContinuousModel
from ..utils import binarize, to_dense, power, multiply, ignore_warnings, get_prediction, get_prediction_with_threshold, show_factor_distribution
import numpy as np
from scipy.sparse import spmatrix
from tqdm import tqdm


class PNLPF(BinaryMFPenalty):
    '''PNLPF, binary matrix factorization's penalty function algorithm with sigmmoid link function.

    .. topic:: Reference

        Boolean decomposition of binary matrices using a post-nonlinear mixture approach.

        Solving the problem with multiplicative update:

        min 1/2 ||X - sigmoid(U @ V.T - 1/2)||_F^2 + 1/2 * reg * ||U^2 - U||_F^2 + 1/2 * reg * ||V^2 - V||_F^2

    Parameters
    ----------
    reg : float
        The regularization weight 'lambda' in the paper.
    reg_growth : float
        The growing rate of regularization weight.
    max_reg : float
        The upper bound of regularization weight.
    tol : float
        The error tolerance 'epsilon' in the paper.
    '''
    def __init__(self, k, U=None, V=None, W='full', reg=2.0, beta_loss="frobenius", solver="mu", link_lamda=10, reg_growth=3, max_reg=1e10, tol=0.01, min_diff=0.0, max_iter=100, init_method='custom', normalize_method='balance', seed=None):
        self.check_params(k=k, U=U, V=V, W=W, reg=reg, beta_loss=beta_loss, solver=solver, link_lamda=link_lamda, reg_growth=reg_growth, max_reg=max_reg, tol=tol, min_diff=min_diff, max_iter=max_iter, init_method=init_method, normalize_method=normalize_method, seed=seed)


    @ignore_warnings
    def update_U(self):
        '''Multiplicative update of U.
        '''
        self.U = update_U(X=self.X_train, W=self.W, U=self.U, V=self.V, reg=self.reg, link_lamda=self.link_lamda)


    @ignore_warnings
    def update_V(self):
        '''Multiplicative update of V.
        '''
        self.V = update_V(X=self.X_train, W=self.W, U=self.U, V=self.V, reg=self.reg, link_lamda=self.link_lamda)


    def get_prediction(self):
        return get_prediction_with_sigmoid(U=self.U, V=self.V, link_lamda=self.link_lamda)


def get_prediction_with_sigmoid(U, V, link_lamda):
    '''Get prediction with sigmoid link function.
    '''
    S = subtract(U @ V.T, 1/2) * link_lamda
    return sigmoid(S)


def update_U(X, W, U, V, reg, link_lamda, solver='mu', beta_loss='frobenius'):
    S = subtract(U @ V.T, 1/2) * link_lamda
    sig = sigmoid(S)
    d_sig = d_sigmoid(S)

    num = link_lamda * multiply(W, multiply(X, d_sig)) @ V
    num += 3 * reg * power(U, 2)

    denom = link_lamda * multiply(W, multiply(sig, d_sig)) @ V
    denom += 2 * reg * power(U, 3) + reg * U
    denom[denom == 0] = np.finfo(np.float64).eps

    U_next = multiply(U, num / denom)
    U_next[U_next == 0] = np.finfo(float).eps
    return U_next


def update_V(X, W, U, V, reg, link_lamda, solver='mu', beta_loss='frobenius'):
    S = subtract(U @ V.T, 1/2) * link_lamda
    sig = sigmoid(S)
    d_sig = d_sigmoid(S)

    num = link_lamda * multiply(W, multiply(X, d_sig)).T @ U
    num += 3 * reg * power(V, 2)

    denom = link_lamda * multiply(W, multiply(sig, d_sig)).T @ U
    denom += 2 * reg * power(V, 3) + reg * V
    denom[denom == 0] = np.finfo(np.float64).eps

    V_next = multiply(V, num / denom)
    V_next[V_next == 0] = np.finfo(float).eps
    return V_next