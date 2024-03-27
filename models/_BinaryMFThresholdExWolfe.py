from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, binarize, sigmoid, dot
import numpy as np


class BinaryMFThresholdExWolfe(BinaryMFThreshold):
    '''Binary matrix factorization, Thresholding algorithm (experimental)
    
    With self implemented Wolfe line search algorithm.
    '''
    def line_search(self, f, myfprime, xk, pk, maxiter=1000, c1=0.1, c2=0.4):
        alpha = 2
        a, b = 0, 10
        n_iter = 0
        fc, gc = 0, 0

        fk = f(xk)
        gk = myfprime(xk)
        fc, gc = fc + 1, gc + 1

        while n_iter <= maxiter:
            n_iter = n_iter + 1
            x = xk + alpha * pk

            armojo_cond = f(x) - fk <= alpha * c1 * dot(gk, pk)
            fc += 1

            if armojo_cond: # Armijo (Sufficient Decrease) Condition

                curvature_cond = dot(myfprime(x), pk) >= c2 * dot(gk, pk)
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

        new_fval, old_fval, new_slope = f(x), fk, myfprime(x)
        fc, gc = fc + 1, gc + 1

        return alpha, fc, gc, new_fval, old_fval, new_slope
