# Copyright the ELBMF Authors.
# https://github.com/renfujiwara/ELBMF
# ============================================================================
import numpy as np
import sys
from ..utils import to_dense, matmul, show_matrix, binarize
from .ContinuousModel import ContinuousModel
import numpy as np
import sys
from tqdm import tqdm


class ELBMF(ContinuousModel):
    def __init__(
            self,
            org_A,
            A,
            ncomponents,
            l1reg,
            l2reg,
            c, # = t -> c^t 
            maxiter,
            tolerance,
            random_seed          = 19,
            beta                 = 0.0, # inertial disabled by default
            batchsize            = None,
            with_rounding        = True,
            callback             = None
        ):
        self.org_A = org_A.copy()
        self.A = A.copy()
        self.n, self.m = np.shape(A)
        self.k = ncomponents
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.c = c
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.beta = beta
        if(batchsize is None):
            self.batchsize  = len(A)
        else:
            self.batchsize = batchsize
        self.with_rounding = with_rounding
        self.callback = callback
        self.U, self.V = self.init_factorization(random_seed)
        U_init = _rounding(self.U)
        V_init = _rounding(self.V)
        print('init_loss')
        _print_loss(org_A, U_init, V_init)

        # debug: create PyBMF variables
        self.X_train = self.A.copy()
        # self.X_val = self.A.copy()
        # self.X_test = self.A.copy()
        self.X_val = None
        self.X_test = None

        self.task = 'reconstruction'
        self.logs = {}

        # debug: initial evaluation of the real-valued loss
        F = np.linalg.norm(self.A - (self.U @ self.V)) ** 2
        ell = np.linalg.norm(A - _product(self.U, self.V), ord=2)
        self.X_pd = matmul(self.U, self.V, boolean=False, sparse=False)
        self.evaluate(df_name='updates', head_info={'iter': 0, 'F': F, 'ell': ell}, metrics=['RMSE', 'MAE'])

        # debug: initial evaluation of the binary error
        self.X_pd = matmul(U_init, V_init, boolean=True, sparse=True)
        self.evaluate(df_name='boolean', head_info={'iter': 0}, metrics=['Recall', 'Precision', 'Accuracy', 'F1'])

    
    def init_factorization(self, seed):
        np.random.seed(seed)
        return np.random.rand(self.n,self.k), np.random.rand(self.k,self.m)
    

    def factorize(self):
        if(self.batchsize >= len(self.A)):
            if(self.beta == 0):
                self.U, self.V = _factorize_palm(self)
            else:
                self.U, self.V = _factorize_ipalm(self, self.beta)
        else:
            self.U, self.V = _batched_factorize_ipalm(self)
            
        self.U = _rounding(self.U)
        self.V = _rounding(self.V)
        print('result_loss')
        self.print_loss()


    def print_loss(self):
        _print_loss(self.org_A, self.U, self.V)
            
# def _regularization_rate(c, t):
#     return pow(c, t)
def _product(X, Y):
    return np.clip(np.dot(X, Y), 0, 1)

def _print_loss(A, U, V):
    print(f'recall : {_recall(A, U, V)}, similarity : {_similarity(A, U, V)}, relative loss : {_relative_loss(A, U, V)}', flush=True)
        
def _recall(A, U, V):
    return np.sum(A * _product(U, V))/np.sum(A)

def _similarity(A, U, V):
    return np.sum(np.abs(A - _product(U, V)))/A.size

def _relative_loss(A, U, V):
    return np.sum(np.abs(A - _product(U, V)))*np.sum(A)

def _rounding(X, k = 0.5, l = 1.e+20):
    X = _prox(X, k, l)
    return np.round(np.clip(X, 0, 1))

def _proxel_1(X, k, l):
    tmp = np.zeros_like(X)
    tmp[X <=0.5] = (X[X <=0.5] - k * np.sign(X[X <=0.5])) / (1 + l)
    tmp[X >0.5] = (X[X > 0.5] - k * np.sign(X[X > 0.5] - 1) + l) / (1 + l)
    return tmp

def _prox(U, k, l):
    prox_U = _proxel_1(U, k, l)
    prox_U[prox_U<0] = 0
    return prox_U

def  _reducemf_impl(A, U, V, l1reg, l2reg):
    VVt =  np.dot(V,V.T)
    grad_u = np.dot(U, VVt) - np.dot(A, V.T)
    L = max(np.linalg.norm(VVt, ord=2), 1e-4)
    step_size = 1 / (1.1 * L)
    U = U - grad_u * step_size
    U = _prox(U, l1reg*step_size, l2reg*step_size)
    
    return U

def  _reducemf_impl_b(A, U, V, l1reg, l2reg, U_, beta):
    U__ = U.copy()
    U = U + (U - U_) * beta 
    VVt =  np.dot(V,V.T)
    grad_u = np.dot(U, VVt) - np.dot(A, V.T)
    L = max(np.linalg.norm(VVt, ord=2), 1e-4)
    step_size = 2 * (1 - beta) / (1 + 2 * beta) / L
    U = U - grad_u * step_size
    U = _prox(U, l1reg*step_size, l2reg*step_size)
    
    return U, U__
    
                   
def _factorize_palm(elbmf):
    U = elbmf.U
    V = elbmf.V
    l2reg_init = elbmf.l2reg
    l1reg = elbmf.l1reg
    c = elbmf.c
    A = elbmf.A
    tol = elbmf.tolerance
    ell0 = sys.float_info.max
    ell = 0
    for iter in tqdm(range(elbmf.maxiter)):
        l2reg = l2reg_init * pow(c, iter)
        U = _reducemf_impl(A, U, V, l1reg, l2reg)
        V = _reducemf_impl(A.T, V.T, U.T, l1reg, l2reg).T
        ell = np.linalg.norm(A - _product(U,V), ord=2)
        if(abs(ell - ell0) < tol): break
        ell0 = ell

        # debug: initial evaluation of the real-valued loss
        F = np.linalg.norm(elbmf.A - (U @ V)) ** 2
        elbmf.X_pd = matmul(U, V, boolean=False, sparse=False)
        elbmf.evaluate(df_name='updates', head_info={'iter': iter, 'F': F, 'ell': ell}, metrics=['RMSE', 'MAE'])

        # debug: initial evaluation of the binary error
        U_init = _rounding(U)
        V_init = _rounding(V)
        elbmf.X_pd = matmul(U_init, V_init, boolean=True, sparse=True)
        elbmf.evaluate(df_name='boolean', head_info={'iter': iter}, metrics=['Recall', 'Precision', 'Accuracy', 'F1'])
        
    return U, V

def _factorize_ipalm(elbmf, beta):
    U = elbmf.U
    V = elbmf.V
    l2reg_init = elbmf.l2reg
    l1reg = elbmf.l1reg
    c = elbmf.c
    A = elbmf.A
    tol = elbmf.tolerance
    ell0 = sys.float_info.max
    ell = 0
    U_    = U.copy()
    Vt_   = V.T.copy()
    for iter in tqdm(range(elbmf.maxiter)):
        l2reg = l2reg_init * pow(c, iter)
        U, U_ = _reducemf_impl_b(A, U, V, l1reg, l2reg, U_, beta)
        Vt, Vt_ = _reducemf_impl_b(A.T, V.T, U.T, l1reg, l2reg, Vt_, beta)
        V = Vt.T.copy()
        ell = np.linalg.norm(A - _product(U,V), ord=2)
        if(abs(ell - ell0) < tol): break
        ell0 = ell

        # debug: initial evaluation of the real-valued loss
        F = np.linalg.norm(elbmf.A - (U @ V)) ** 2
        elbmf.X_pd = matmul(U, V, boolean=False, sparse=False)
        elbmf.evaluate(df_name='updates', head_info={'iter': iter, 'F': F, 'ell': ell}, metrics=['RMSE', 'MAE'])

        # debug: initial evaluation of the binary error
        U_init = _rounding(U)
        V_init = _rounding(V)
        elbmf.X_pd = matmul(U_init, V_init, boolean=True, sparse=True)
        elbmf.evaluate(df_name='boolean', head_info={'iter': iter}, metrics=['Recall', 'Precision', 'Accuracy', 'F1'])

    return U, V

def _batched_factorize_ipalm(elbmf):
    # do this later
    U = elbmf.U.copy()
    V = elbmf.V.copy()
    
    return U, V