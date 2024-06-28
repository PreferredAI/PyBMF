import torch


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

    U -= (U@VVt - XVt) * step_size
    U = proxelbmfnn(U, l1reg * step_size, l2reg * tau * step_size)
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

        for t in range(maxiter):
            
            tau = regularization_rate(t)
            
            U = elbmf_step_ipalm(X, U, V, Uold, l1reg, l2reg, tau, beta)
            V = elbmf_step_ipalm(X.T, V.T, U.T, Vold, l1reg, l2reg, tau, beta).T
            
            fn0, fn = fn, (X - (U@V)).norm()**2 
            
            if callback != None: 
                 callback(t, U, V, fn)
            if (abs(fn - fn0) < tolerance): 
                 break
        return U, V


@torch.no_grad()
def ELBMF(
        X,
        n_components,
        l1reg               = 0.01,
        l2reg               = 0.02,
        regularization_rate = lambda t: 1.02**t,
        maxiter             = 3000,
        tolerance           = 1e-8,
        beta                = 0.0001,
        callback            = None,
        with_rounding       = True
    ):
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
    U, V = torch.rand(X.shape[0], n_components, dtype=X.dtype), torch.rand(n_components, X.shape[1], dtype=X.dtype)

    U, V = elbmf_ipalm(X, U, V, l1reg, l2reg, regularization_rate, maxiter, tolerance, beta, callback)
    if with_rounding:
        with torch.no_grad():
            U = proxelbmfnn(U, 0.5, l2reg * 1e12)
            V = proxelbmfnn(V, 0.5, l2reg * 1e12)
            return U.round(), V.round()
    else:
        return U, V
        

