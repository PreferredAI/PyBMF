from ..utils import dot


def line_search(f, myfprime, xk, pk, args=(), kwargs={}, maxiter=1000, c1=0.1, c2=0.4):
        '''Re-implementation of SciPy's Wolfe line search.

        It's compatible with `scipy.optimize.line_search`:
        >>> from scipy.optimize import line_search
        >>> line_search(f=f, myfprime=myfprime, xk=xk, pk=pk, maxiter=maxiter, c1=c1, c2=c2)

        Parameters
        ----------
        f : function
            The objective function to be minimized.
        myfprime : function
            The gradient of the objective function to be minimized.
        xk : array_like
            The starting point for the line search.
        pk : array_like
            The search direction for the line search.
        args : tuple
            Additional arguments passed to `f` and `myfprime`.
        kwargs : dict
            Additional keyword arguments passed to `f` and `myfprime`.
        maxiter : int
            The maximum number of iterations.
        c1 : float
        c2 : float
        '''
        alpha = 2
        a, b = 0, 10
        n_iter = 0
        fc, gc = 0, 0

        fk = f(xk, *args, **kwargs)
        gk = myfprime(xk, *args, **kwargs)
        fc, gc = fc + 1, gc + 1

        while n_iter <= maxiter:
            n_iter = n_iter + 1

            x = xk + alpha * pk

            armojo_cond = f(x, *args, **kwargs) - fk <= alpha * c1 * dot(gk, pk)
            fc += 1

            if armojo_cond: # Armijo (Sufficient Decrease) Condition

                curvature_cond = dot(myfprime(x, *args, **kwargs), pk) >= c2 * dot(gk, pk)
                gc += 1

                if curvature_cond: # Curvature Condition
                    break
                else:
                    if b < 10:
                        a = alpha
                        alpha = (a + b) / 2
                    else:
                        alpha = alpha * 1.2
            else:
                b = alpha
                alpha = (a + b) / 2

        new_fval, old_fval, new_slope = f(x, *args, **kwargs), fk, myfprime(x, *args, **kwargs)
        fc, gc = fc + 1, gc + 1

        return alpha, fc, gc, new_fval, old_fval, new_slope


def limit_step_size(x_min, x_max, x_last, xk, alpha, pk):
    '''Limit the step size by limiting ``x_last`` by ``x_min`` and ``x_max``.
    
    This is similar to setting ``amax`` or ``extra_condition`` in ``scipy.optimize.line_search``.

    Parameters
    ----------
    x_min : array_like
    x_max : array_like
    x_last : array_like
    xk : array_like
    alpha : float
    pk : array_like

    Returns
    -------
    x_last : array_like
    alpha_new : float
    '''
    if (x_last <= x_max).all() and (x_last >= x_min).all():
        return x_last, alpha

    alpha_new = alpha

    for i in range(len(x_last)):
        alpha_tmp = alpha_new
        if x_last[i] > x_max[i]:
            alpha_tmp = (x_max[i] - xk[i]) / pk[i]
        if x_last[i] < x_min[i]:
            alpha_tmp = (x_min[i] - xk[i]) / pk[i]
        if alpha_tmp < alpha_new:
            alpha_new = alpha_tmp

    x_last = xk + alpha_new * pk

    return x_last, alpha_new