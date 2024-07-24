# import torch
from .ContinuousModel import ContinuousModel
import numpy as np
from ..utils import binarize, matmul, to_dense, to_sparse, ismat, get_prediction_with_threshold, show_matrix, multiply
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

try:
    import torch
except ImportError:
    print('[E] Missing package: torch. Please install it first.')
    pass


class ELBMF(ContinuousModel):
    def __init__(
            self, 
            k,
            U=None,
            V=None, 
            W='full', 
            init_method='custom', 
            reg_l1=0.01, 
            reg_l2=0.02, 
            reg_growth=1.02, 
            rounding=False, 
            beta=0.0, 
            tol=0.0, 
            max_iter=1000, 
            min_diff=1e-8, 
            seed=None):
        '''
        
        Parameters
        ----------
        k : int
            Number of latent factors.
        U, V : numpy.ndarray, spmatrix
            Need to be prepared if `init_method` is 'custom'.
        W : numpy.ndarray, spmatrix or str in ['full', 'mask']
            Weight matrix.
        init_method : str in ['custom', 'normal', 'uniform']
        reg_l1, reg_l2, reg_growth : float
        rounding : bool
            Whether to round the factors to {0, 1}.
            Disable this for diagnosis purposes.
        beta : float
            Inertial coefficient of iPALM.
        '''
        self.check_params(k=k, U=U, V=V, W=W, init_method=init_method, reg_l1=reg_l1, reg_l2=reg_l2, reg_growth=reg_growth, rounding=rounding, beta=beta, tol=tol, max_iter=max_iter, min_diff=min_diff, seed=seed)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        '''Initialize factors and logging variables.
        '''
        super().init_model()

        # initialize U, V
        self.init_UV()

        # normalize U, V
        # self.normalize_UV(method="balance")
        self.normalize_UV(method="normalize")

        self._to_dense()

        # replace zeros in U, V
        self.U[self.U == 0] = np.finfo(float).eps
        self.V[self.V == 0] = np.finfo(float).eps


    def _fit(self):

        # elbmf-python

        # X = np.array(self.X_train.toarray(), dtype=np.float64)
        # X = torch.from_numpy(X).float()

        # self.U, self.V = elbmf(
        #     X                   = X,
        #     U                   = self.U,
        #     V                   = self.V, 
        #     n_components        = self.k, 
        #     l1reg               = self.reg_l1,
        #     l2reg               = self.reg_l2,
        #     regularization_rate = lambda t: self.reg_growth ** t,
        #     maxiter             = self.max_iter,
        #     tolerance           = self.min_diff,
        #     beta                = self.beta,
        #     callback            = None,
        #     with_rounding       = self.rounding, 
        #     seed                = self.seed
        # )

        # self.U, self.V = csr_matrix(self.U), csr_matrix(self.V).T
        # self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)
        # self.evaluate(df_name='boolean')

        # re-implementation

        self.iPALM()



    def iPALM(self):
        U_last, V_last = self.U.copy(), self.V.copy()

        err, gap = np.inf, np.inf

        pbar = tqdm(total=self.max_iter, desc="[I] f: -, U: [-, -], V: [-, -]")

        n_iter = 0
        is_improving = True
        while is_improving:
            reg_l1, reg_l2 = self.reg_l1, self.reg_l2 * (self.reg_growth ** n_iter)

            U, U_last = update_U(self.X_train, self.U, self.V, self.W, reg_l1, reg_l2, self.beta, U_last)
            V, V_last = update_U(self.X_train.T, self.V, self.U, self.W.T, reg_l1, reg_l2, self.beta, V_last)

            # compute err
            err, err_last = np.power(self.X_train - U @ V.T, 2).sum(), err

            # compute gap
            U_gap, V_gap = get_integrality_gap(U, reg_l1, reg_l2), get_integrality_gap(V, reg_l1, reg_l2)
            gap, gap_last = U_gap + V_gap, gap

            self.U, self.V = U, V

            # update pbar
            pbar.update(1)
            desc = f'[I] n_iter: {n_iter}, err: {err:.3f}, gap: {gap:.3f}, U: [{U.min():.4f}, {U.max():.4f}], V: [{V.min():.4f}, {V.max():.4f}]'
            pbar.set_description(desc)
            # print(desc)

            self.X_pd = get_prediction_with_threshold(U=U, V=V, u=0.5, v=0.5, sparse=True)
            self.evaluate(
                df_name='updates', 
                head_info={
                    'iter': n_iter,
                    'reg_l1': reg_l1,
                    'reg_l2': reg_l2,
                    'gap': gap, 
                    'U_gap': U_gap,
                    'V_gap': V_gap, 
                    'error': err,
                }, 
                metrics=['ERR', 'Accuracy', 'Recall', 'Precision', 'F1'],
            )
            # if n_iter % 20 == 0:
            #     show_matrix([(self.X_pd, [0, 0], 'X_pd')], scaling=0.3, colorbar=True)

            # measure error and diff
            # diff = abs(err - err_last)
            diff = abs(gap - gap_last)
            is_improving = self.early_stop(error=gap, diff=diff, n_iter=n_iter)
            n_iter += 1


def get_integrality_gap(U, reg_l1, reg_l2):
    dist = np.where(
        U < 0.5, 
        np.abs(U), 
        np.abs(U - 1)
    )
    gap = reg_l1 * dist + reg_l2 * dist ** 2

    return gap.sum()


def update_U(X, U, V, W, reg_l1, reg_l2, beta, U_last):
    '''A step of Gauss-Seidel optimization for U, applies to V as well.
    '''
    # U{t}, U_{t-1}, U_{t-2}
    U, U_last, U_before_last = None, U, U_last

    # step size eta in terms of the Lipschitz constant
    L = max(np.linalg.norm(V.T @ V, ord=2), 1e-4)
    eta = 1 / (1.1 * L) if beta == 0 else 2 * (1 - beta) / (1 + 2 * beta) / L
    
    # gradient descent
    U = U_last + beta * (U_last - U_before_last)
    dFdU = multiply(W, U @ V.T - X) @ V
    U = U - eta * dFdU

    # proximal operator
    U = prox(U, reg_l1 * eta, reg_l2 * eta)

    return U, U_last
        

def prox(U, kai, lamda):
    '''Proximal operator of the elastic net penalty.
    '''
    U_prox = np.where(
        U <= 0.5, 
        U - kai * np.sign(U), 
        U - kai * np.sign(U - 1) + lamda
    )
    U_prox = U_prox / (1 + lamda)
    U_prox[U_prox < 0] = 0

    return U_prox



# def proxelbmf(x, k, l): 
#     return torch.where(x <= 0.5, x - k * torch.sign(x), x - k * torch.sign(x - 1) + l) / (1 + l)

# def proxelbmf_(x, k, l, L):
#     return torch.where(x <= 0.5, x - k * torch.sign(x), x - k * torch.sign(x - 1) + l) / (1 + l)

# def proxelbmfbox(x, k, l): 
#     return torch.clamp(proxelbmf(x, k, l), 0, 1) 


# def proxelbmfnn(x, k, l): 
#     return torch.max(proxelbmf(x, k, l), torch.zeros_like(x))

# def proxelbmfnn_(x, k, l, L): 
#     return torch.max(proxelbmf_(x, k, l, L), torch.zeros_like(x))

# def integrality_gap_elastic(e, l1reg, l2reg): 
#      return torch.min((l1reg * e.abs() + l2reg * (e)**2), l1reg * (e - 1).abs() + l2reg * (e - 1)**2).sum()


# @torch.no_grad()
# def elbmf_step_ipalm(X, U, V, Uold, l1reg, l2reg, tau, beta):
#     VVt, XVt = V@V.T, X@V.T
#     L = max(VVt.norm().item(), 1e-4)
    
#     if beta != 0:
#         U += beta * (U - Uold)
#         Uold = U
#         step_size = 2 * (1 - beta) / (1 + 2 * beta) / L
#     else:
#         step_size = 1 / (1.1 * L)
#         # step_size = 1 / L

#     U -= (U @ VVt - XVt) * step_size
#     U = proxelbmfnn(U, l1reg * step_size, l2reg * tau * step_size)
#     U = proxelbmfnn_(U, l1reg, l2reg * tau, L)
#     # debug
#     U[U == 0] = np.finfo(float).eps
#     return U


# @torch.no_grad()
# def elbmf_ipalm(
#         X,
#         U,
#         V,
#         l1reg,
#         l2reg,
#         regularization_rate,
#         maxiter,
#         tolerance,
#         beta,
#         callback
#     ):
#         if beta != 0:
#             Uold, Vold = U.clone(), V.T.clone()
#         else:
#             Uold, Vold = None, None

#         fn = torch.inf

#         pbar = tqdm(total=maxiter, desc="[I] error: -, U: -, V: -")
#         for t in range(maxiter):
            
#             tau = regularization_rate(t)
            
#             U = elbmf_step_ipalm(X, U, V, Uold, l1reg, l2reg, tau, beta)
#             V = elbmf_step_ipalm(X.T, V.T, U.T, Vold, l1reg, l2reg, tau, beta).T
            
#             fn0, fn = fn, (X - (U @ V)).norm() ** 2

#             # update pbar
#             pbar.update(1)
#             desc = f'[I] error: {fn:.4f}, U: [{U.min():.4f}, {U.max():.4f}], V: [{V.min():.4f}, {V.max():.4f}]'
#             pbar.set_description(desc)
            
#             if callback != None: 
#                  callback(t, U, V, fn)
#             # if (abs(fn - fn0) < tolerance):
#             #      print("[I] Converged")
#             #      break
#         return U, V


# @torch.no_grad()
# def elbmf(
#         X,
#         n_components,
#         U                   = None, 
#         V                   = None,
#         l1reg               = 0.01,
#         l2reg               = 0.02,
#         regularization_rate = lambda t: 1.02 ** t,
#         maxiter             = 3000,
#         tolerance           = 1e-8,
#         beta                = 0.0001,
#         callback            = None,
#         with_rounding       = True,
#         seed                = None
#     ):
#     """
#     This function implements the algorithm described in the paper

#     Sebastian Dalleiger and Jilles Vreeken. “Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent”. 
#     In: Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS). 2022

#     Arguments:
#     X                       a Boolean n*m matrix  
#     n_components            number of components
#     l1reg                   l1 coefficient
#     l2reg                   l2 coefficient
#     regularization_rate     monotonically increasing regularization-rate function
#     maxiter                 maximum number of iterations
#     tolerance               the threshold to the absolute difference between the current and previous losses determines the convergence
#     beta                    inertial coefficient of iPALM
#     callback                e.g. lambda t, U, V, fn: print(t, fn)
#     with_rounding           rounds U and V in case of early stopping

#     Returns:
#     U                       n*k factor matrix
#     V                       k*m factor matrix 
#     """
#     if seed is not None:
#         torch.manual_seed(seed)
#     if U is None or V is None:
#         U, V = torch.rand(X.shape[0], n_components, dtype=X.dtype), torch.rand(n_components, X.shape[1], dtype=X.dtype)
#     else:
#         # imported U, V
#         U = np.array(U.toarray(), dtype=np.float64)
#         U = torch.from_numpy(U).float()

#         V = np.array(V.toarray(), dtype=np.float64).T
#         V = torch.from_numpy(V).float()


#     U, V = elbmf_ipalm(X, U, V, l1reg, l2reg, regularization_rate, maxiter, tolerance, beta, callback)
#     if with_rounding:
#         with torch.no_grad():
#             U = proxelbmfnn(U, 0.5, l2reg * 1e12)
#             V = proxelbmfnn(V, 0.5, l2reg * 1e12)
#             return U.round(), V.round()
#     else:
#         return U, V
        

