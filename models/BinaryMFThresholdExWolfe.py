from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, binarize, sigmoid, dot
import numpy as np


class BinaryMFThresholdExWolfe(BinaryMFThreshold):
    '''Binary matrix factorization, Thresholding algorithm (experimental)
    
    With self implemented Wolfe line search algorithm.
    '''
    def __init__(self, k, lamda=None, u=None, v=None, eps=None, max_iter=None):
        self.check_params(k=k, lamda=lamda, u=u, v=v, eps=eps, max_iter=max_iter, algorithm='threshold')
        

    def _fit(self):
        self.initialize()
        self.normalize()
        self.threshold_algorithm()


    def threshold_algorithm(self):
        '''A gradient descent method minimizing F(u, v), or 'F(w, h)' in the paper.
        '''
        x_last = np.array([self.u, self.v]) # initial threshold u, v
        p_last = -self.dF(x_last) # initial searching direction

        us, vs, Fs, ds = [], [], [], []
        n_iter = 0
        while True:
            xk = x_last # starting point
            pk = p_last # searching direction

            # pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}, start from [{:.3f}, {:.3f}], search direction [{:.3f}, {:.3f}]".format(n_iter, *xk, *pk))

            alpha, fc, gc, new_fval, old_fval, new_slope = self.line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk)
            # if alpha is None:
            #     self.early_stop("search direction is not a descent direction.")
            #     break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            self.u, self.v = x_last
            us.append(self.u)
            vs.append(self.v)
            Fs.append(new_fval)
            diff = np.sqrt(np.sum((alpha * pk) ** 2))
            ds.append(diff)

            # debug
            print(alpha, diff)
            
            self.print_msg("[I] Wolfe line search for iter   : {}".format(n_iter))
            self.print_msg("    num of function evals made   : {}".format(fc))
            self.print_msg("    num of gradient evals made   : {}".format(gc))
            self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))
            self.print_msg("    threshold update             : [{:.3f}, {:.3f}] -> [{:.3f}, {:.3f}]".format(*xk, *x_last))
            self.print_msg("    threshold difference         : {:.3f}".format(diff))

            n_iter += 1

            if n_iter > self.max_iter:
                self.early_stop("Reached maximum iteration count")
                break
            if diff < self.eps:
                self.early_stop("Difference lower than threshold")
                break

        self.U = binarize(self.U, self.u) # debug: iteratively update U, V?
        self.V = binarize(self.V, self.v)
        self.show_matrix(title="after thresholding algorithm")


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
            
