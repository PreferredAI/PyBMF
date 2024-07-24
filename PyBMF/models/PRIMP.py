import torch
from .ContinuousModel import ContinuousModel
import numpy as np
from ..utils import binarize, matmul, to_dense, to_sparse, ismat
from scipy.sparse import lil_matrix
from tqdm import tqdm


class PRIMP(ContinuousModel):
    def __init__(self, k, reg=0.01, reg_growth=1.02, max_iter=1000, min_diff=1e-8, beta=1e-4, seed=None):
        self.check_params(k=k, reg=reg, reg_growth=reg_growth, max_iter=max_iter, min_diff=min_diff, beta=beta, seed=seed)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        # modify X
        X = np.array(self.X_train.toarray(), dtype=np.float64)
        X = torch.from_numpy(X).float()

        self.U, self.V = primp(
            X                   = X,
            n_components        = self.k, 
            l1reg               = self.reg,
            l2reg               = 0,
            regularization_rate = lambda t: self.reg_growth ** t,
            maxiter             = self.max_iter,
            tolerance           = self.min_diff,
            beta                = self.beta,
            callback            = None,
            with_rounding       = True, 
            seed                = self.seed
        )

        self.U, self.V = lil_matrix(self.U), lil_matrix(self.V.T)
        self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)
        self.evaluate(df_name='boolean')


def _proxelbmfnn(x, k, l): # PRIMP
    return torch.min(proxelbmf(x, k, l), torch.ones_like(x))


def proxelbmf(x, k, l): 
    return torch.where(x <= 0.5, x - k * torch.sign(x), x - k* torch.sign(x - 1) + l) / (1 + l)


def proxelbmfbox(x, k, l): 
    return torch.clamp(proxelbmf(x, k, l), 0, 1) 


def proxelbmfnn(x, k, l): 
    return torch.max(proxelbmf(x, k, l), torch.zeros_like(x))


def integrality_gap_elastic(e, l1reg, l2reg): 
     return torch.min((l1reg * e.abs() + l2reg * (e)**2), l1reg * (e - 1).abs() + l2reg * (e - 1)**2).sum()


@torch.no_grad()
def elbmf_step_ipalm(X, U, V, Uold, l1reg, l2reg, tau, beta):
    VVt, XVt = V@V.T, X@V.T
    L = max(VVt.norm().item(), 1e-4)
    
    if beta != 0:
        U += beta * (U - Uold)
        Uold = U
        step_size = 2 * (1 - beta) / (1 + 2 * beta) / L
    else:
        step_size = 1 / (1.1 * L)

    U -= (U @ VVt - XVt) * step_size
    U = proxelbmfnn(U, l1reg * step_size, l2reg * tau * step_size)
    # PRIMP
    U = _proxelbmfnn(U, l1reg * step_size, l2reg * tau * step_size)
    return U


@torch.no_grad()
def elbmf_ipalm(
        X,
        U,
        V,
        l1reg,
        l2reg,
        regularization_rate,
        maxiter,
        tolerance,
        beta,
        callback
    ):
        if beta != 0:
            Uold, Vold = U.clone(), V.T.clone()
        else:
            Uold, Vold = None, None

        fn = torch.inf

        pbar = tqdm(total=maxiter, desc="[I] error: -, U: -, V: -")
        for t in range(maxiter):
            
            tau = regularization_rate(t)
            
            U = elbmf_step_ipalm(X, U, V, Uold, l1reg, l2reg, tau, beta)
            V = elbmf_step_ipalm(X.T, V.T, U.T, Vold, l1reg, l2reg, tau, beta).T
            
            fn0, fn = fn, (X - (U @ V)).norm() ** 2

            # update pbar
            pbar.update(1)
            desc = f'[I] error: {fn:.4f}, U: [{U.min():.4f}, {U.max():.4f}], V: [{V.min():.4f}, {V.max():.4f}]'
            pbar.set_description(desc)
            
            if callback != None: 
                 callback(t, U, V, fn)
            if (abs(fn - fn0) < tolerance): 
                 break
        return U, V


@torch.no_grad()
def primp(
        X,
        n_components,
        l1reg               = 0.01,
        l2reg               = 0, # always 0
        regularization_rate = lambda t: 1.02 ** t,
        maxiter             = 3000,
        tolerance           = 1e-8,
        beta                = 0.0001,
        callback            = None,
        with_rounding       = True,
        seed                = None
    ):
    if seed is not None:
        torch.manual_seed(seed)
    U, V = torch.rand(X.shape[0], n_components, dtype=X.dtype), torch.rand(n_components, X.shape[1], dtype=X.dtype)

    U, V = elbmf_ipalm(X, U, V, l1reg, l2reg, regularization_rate, maxiter, tolerance, beta, callback)
    if with_rounding:
        with torch.no_grad():
            U = proxelbmfnn(U, 0.5, l2reg * 1e12)
            V = proxelbmfnn(V, 0.5, l2reg * 1e12)
            return U.round(), V.round()
    else:
        return U, V
        

