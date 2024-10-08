# Copyright the elbmf-python Authors.
# source: https://github.com/sdall/elbmf-python
# ============================================================================
import sys
from .ContinuousModel import ContinuousModel
import numpy as np
from ..utils import binarize, matmul, to_dense, to_sparse, ismat
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

try:
    import torch
except ImportError:
    print('[E] Missing package: torch. Please install it first.')


class ELBMF(ContinuousModel):
    def __init__(self, k, U=None, V=None, init_method='custom', reg_l1=0.01, reg_l2=0.02, reg_growth=1.02, max_iter=3000, min_diff=1e-8, beta=1e-4, seed=None):
        self.check_params(k=k, U=U, V=V, init_method=init_method, reg_l1=reg_l1, reg_l2=reg_l2, reg_growth=reg_growth, max_iter=max_iter, min_diff=min_diff, beta=beta, seed=seed)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        # convert X to torch tensor
        X = np.array(self.X_train, dtype=np.float64)
        X = torch.from_numpy(X).float()

        # call elbmf
        U, V = self.elbmf(
            X                   = X,
            U                   = self.U,
            V                   = self.V, 
            n_components        = self.k, 
            l1reg               = self.reg_l1,
            l2reg               = self.reg_l2,
            regularization_rate = lambda t: self.reg_growth ** t,
            maxiter             = self.max_iter,
            tolerance           = self.min_diff,
            beta                = self.beta,
            callback            = None,
            with_rounding       = False, 
            seed                = self.seed
        )

        # the final result is the non-thresholded values
        self.U, self.V = csr_matrix(U), csr_matrix(V).T


    @torch.no_grad()
    def elbmf(
        self,
        X,
        n_components,
        U                   = None, 
        V                   = None,
        l1reg               = 0.01,
        l2reg               = 0.02,
        regularization_rate = lambda t: 1.02 ** t,
        maxiter             = 3000,
        tolerance           = 1e-8,
        beta                = 0.0001,
        callback            = None,
        with_rounding       = False,
        seed                = None):
        """
        This function implements the algorithm described in the paper

        Sebastian Dalleiger and Jilles Vreeken. “Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent”. 
        In: Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS). 2022

        Arguments:
        X                       a Boolean n*m matrix  
        n_components            number of components
        l1reg                   l1 coefficient
        l2reg                   l2 coefficient
        regularization_rate     monotonically increasing regularization-rate function
        maxiter                 maximum number of iterations
        tolerance               the threshold to the absolute difference between the current and previous losses determines the convergence
        beta                    inertial coefficient of iPALM
        callback                e.g. lambda t, U, V, fn: print(t, fn)
        with_rounding           rounds U and V in case of early stopping

        Returns:
        U                       n*k factor matrix
        V                       k*m factor matrix 
        """
        # debug: inspect parameters
        print("\n============= parameters of elbmf() ===============")
        print("X: ", X.shape, type(X))
        print("n_components: ", n_components)
        print("U: ", U.shape if U is not None else None, type(U))
        print("V: ", V.shape if V is not None else None, type(V))
        print("l1reg: ", l1reg)
        print("l2reg: ", l2reg)
        print("regularization_rate: ", regularization_rate)
        print("maxiter: ", maxiter)
        print("tolerance: ", tolerance)
        print("beta: ", beta)
        print("callback: ", callback)
        print("with_rounding: ", with_rounding)
        print("seed: ", seed)
        print("=====================================================\n")

        # debug: accept parameters random seed
        if seed is not None:
            torch.manual_seed(seed)

        if U is None or V is None:
            U, V = torch.rand(X.shape[0], n_components, dtype=X.dtype), torch.rand(n_components, X.shape[1], dtype=X.dtype)
        else:
            # debug: accept initial factors U and V

            # U = np.array(U.toarray(), dtype=np.float64)
            U = torch.from_numpy(U).float()
            # V = np.array(V.toarray(), dtype=np.float64).T
            V = torch.from_numpy(V).float().T

        U, V = self.elbmf_ipalm(X, U, V, l1reg, l2reg, regularization_rate, maxiter, tolerance, beta, callback)

        if with_rounding:
            with torch.no_grad():
                U = proxelbmfnn(U, 0.5, l2reg * 1e12)
                V = proxelbmfnn(V, 0.5, l2reg * 1e12)
                return U.round(), V.round()
        else:
            return U, V


    @torch.no_grad()
    def elbmf_ipalm(
        self,
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

            # debug: evaluate the real-valued loss
            self.U, self.V = csr_matrix(U), csr_matrix(V).T
            self.X_pd = matmul(self.U, self.V.T, boolean=False, sparse=False)
            self.evaluate(df_name='updates', head_info={'iter': t, 'F': fn.item()}, metrics=['RMSE', 'MAE'])

            # debug: evaluate the binary error
            self.U, self.V = binarize(self.U, 0.5), binarize(self.V, 0.5)
            self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)
            self.evaluate(df_name='boolean', head_info={'iter': t}, metrics=['Recall', 'Precision', 'Accuracy', 'F1'])

            # debug: update pbar and display the range of U and V
            pbar.update(1)
            desc = f'[I] error: {fn:.4f}, U: [{U.min():.4f}, {U.max():.4f}], V: [{V.min():.4f}, {V.max():.4f}]'
            pbar.set_description(desc)
            
            if callback != None: 
                callback(t, U, V, fn)
            if (abs(fn - fn0) < tolerance):
                print("[I] Converged")
                break
        return U, V
            

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

    return U




    

