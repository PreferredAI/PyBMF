from .BaseModel import BaseModel
from .GreConD import get_concept
from scipy.sparse import lil_matrix, hstack
from ..utils import multiply, bool_to_index, ERR, get_prediction, show_matrix, get_residual, matmul, add, coverage_score, invert, ignore_warnings
import numpy as np


class GreConDPlus(BaseModel):
    '''The GreConD+ algorithm for approximate Boolean decomposition.
    
    .. topic:: Reference

        Discovery of optimal factors in binary data via a novel method of matrix decomposition.
        A new algorithm for boolean matrix factorization which admits overcovering.

    Parameters
    ----------
    k : int, optional
        The target rank.
        If ``None``, it will factorize until the error is smaller than `tol`, or when other stopping criteria is met.
    tol : float, default: 0
        The error tolerance.
    w_fp : float
        The penalty weights for false positives (FP).
    w_fn : float
        The penalty weights for false negatives (FN).
    '''
    def __init__(self, k=None, tol=0, w_fp=0.5, w_fn=None):
        self.check_params(k=k, tol=tol, w_fp=w_fp, w_fn=w_fn)
        
        
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model.
        '''
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    @ignore_warnings
    def _fit(self):
        '''The main process if fitting.
        '''
        k = 0
        is_factorizing = True
        self.X_rs = lil_matrix(self.X_train.copy())
        while is_factorizing:
            score, u, v = get_concept(self.X_train, self.X_rs)
            if score == 0:
                # the score for exact decomposition
                is_factorizing = self.early_stop(msg="No pattern found", k=k)
                break

            # expansion
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)
            u_exp, v_exp = expansion(
                X_gt=self.X_train, 
                X_old=self.X_rs, 
                u=u, v=v, 
                w_fp=self.w_fp, 
                w_fn=self.w_fn, 
            )

            u = add(u, u_exp, sparse=True, boolean=True)
            v = add(v, v_exp, sparse=True, boolean=True)

            # update factors and extensions
            self.set_factors(k, u=u, v=v)
            self.set_extensions(k, u_exp=u_exp, v_exp=v_exp)

            # remove those fully covered patterns covered by k-th pattern
            self.remove_covered(k)
            # remove overlapped rows and columns of the extension area of patterns
            self.remove_overlapped()

            # evaluate
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

            self.evaluate(
                df_name='updates', 
                head_info={
                    'k': k, 
                    'score': score, 
                    'shape': [u.sum(), v.sum()], 
                }, 
            )

            # early stop detection
            error = ERR(gt=self.X_train, pd=self.X_pd)
            print("[I] k: {}, score: {}, error: {:.3f}, shape: [{}, {}]".format(k, score, error, u.sum(), v.sum()))
            is_factorizing = self.early_stop(error=error, n_factor=k+1, k=k)

            # update k
            # since pattern can be removed in remove_covered(), k may not be constantly increasing
            k = self.U.shape[1]


    def set_extensions(self, k, u_exp, v_exp):
        '''Add the extension part of each factor (k = 0, 1, ...).

        Parameters
        ----------
        k : int
            The number of factor to be added.
        u_exp : spmatrix
            The extension ``u_exp`` to be added to ``U_exp``.
        v_exp : spmatrix
            The extension ``v_exp`` to be added to ``V_exp``.
        '''
        if not hasattr(self, 'U_exp'):
            self.U_exp = lil_matrix((self.m, 1))
            self.V_exp = lil_matrix((self.n, 1))

        if self.U_exp.shape[1] < k + 1:
            # self.extend_factors(k + 1)
            self.U_exp = hstack([
                self.U_exp, 
                lil_matrix((self.m, (k + 1) - self.U_exp.shape[1]))
            ]).tolil()
            self.V_exp = hstack([
                self.V_exp, 
                lil_matrix((self.n, (k + 1) - self.V_exp.shape[1]))
            ]).tolil()
            
        self.U_exp[:, k] = u_exp
        self.V_exp[:, k] = v_exp


    def remove_covered(self, k):
        '''Remove fully covered patterns by k-th pattern (k = 0, 1, ...).
        
        Parameters
        ----------
        k : int
            The index of pattern to check with.
        '''
        u = self.U[:, k]
        v = self.V[:, k]

        idx = [] # indices of patterns to be reserved
        for i in range(self.U.shape[1]):
            # keep the k-th pattern itself
            if i == k:
                idx.append(i)
                continue

            # detect if the pattern is fully covered
            u_covered = add(u, self.U[:, i], boolean=True, sparse=True).sum() == u.sum()
            v_covered = add(v, self.V[:, i], boolean=True, sparse=True).sum() == v.sum()

            if u_covered and v_covered:
                # remove the i-th pattern
                continue
            else:
                # keep the i-th pattern
                idx.append(i)

        n_removed = self.U.shape[1] - len(idx)

        print("[I]     remove_covered() finished with {} patterns removed.".format(n_removed))

        if n_removed != 0:

            self.U = self.U[:, idx]
            self.V = self.V[:, idx]

            self.U_exp = self.U_exp[:, idx]
            self.V_exp = self.V_exp[:, idx]


    def remove_overlapped(self):
        '''Remove overlapped columns and rows.
        '''
        # count the number of patterns that cover a cell
        coverage = matmul(self.U, self.V.T, boolean=False, sparse=True)

        for k in range(self.U.shape[1]):

            i_idx = bool_to_index(self.U[:, k])
            j_idx = bool_to_index(self.V[:, k])

            for i in range(self.m):
                if self.U_exp[i, k] == 1:
                    # check if it's covered by any other patterns
                    is_covered = multiply(coverage[i][:, j_idx], self.X_train[i][:, j_idx], boolean=False).min() >= 2
                    if is_covered:
                        # remove the i-th row in k-th pattern's expansion
                        self.U[i, k] = 0
                        self.U_exp[i, k] = 0
                        coverage[i, j_idx] -= 1
                        print("[I]     remove_overlapped() removed {}-th row from {}-th pattern.".format(i, k))

            for j in range(self.n):
                if self.V_exp[j, k] == 1:
                    # check if it's covered by any other patterns
                    is_covered = multiply(coverage[i_idx][:, j], self.X_train[i_idx][:, j], boolean=False).min() >= 2
                    if is_covered:
                        # remove the j-th column in k-th pattern's expansion
                        self.V[j, k] = 0
                        self.V_exp[j, k] = 0
                        coverage[i_idx, j] -= 1
                        print("[I]     remove_overlapped() removed {}-th column from {}-th pattern.".format(j, k))


def expansion(X_gt, X_old, u, v, w_fp, w_fn):
    '''Row-wise or column-wise expansion of a pattern given u and v.

    In GreConD+, factors are initially the dense cores.
    Factor ``u`` and ``v`` will grow in each iteration.
    The expansion part is recorded in ``u_exp`` and ``v_exp``.

    Parameters
    ----------
    X_gt : spmatrix
        The ground-truth matrix.
    X_old : spmatrix
        The residual matrix before the join of u and v.
    u : (m, 1) spmatrix
        The factors.
    v : (n, 1) spmatrix
        The factors.
    w_fp : float
        The penalty weights for false positives (FP).
    w_fn : float
        The penalty weights for false negatives (FN).

    Returns
    -------
    u, : (m, 1) spmatrix
        The expansion part in ``u``.
    v, : (n, 1) spmatrix
        The expansion part in ``v``.
    '''
    m, n = X_gt.shape
    u, v = u.copy(), v.copy()
    u_exp = lil_matrix((m, 1))
    v_exp = lil_matrix((n, 1))

    n_iter = 0
    is_improving = True
    while is_improving:

        # row-wise expansion
        r_score, r_index = _expansion(X_gt, X_old, u, v, w_fp, w_fn, axis=1)

        # column-wise expansion
        c_score, c_index = _expansion(X_gt, X_old, u, v, w_fp, w_fn, axis=0)

        # update the expansion
        if r_score > c_score and r_score > 0:
            u[r_index] = 1
            u_exp[r_index] = 1
        elif c_score > r_score and c_score > 0:
            v[c_index] = 1
            v_exp[c_index] = 1
        else:
            is_improving = False
        n_iter += 1

    print(f"[I]     expansion() finished after {n_iter} iterations.")

    return u_exp, v_exp


def _expansion(X_gt, X_old, u, v, w_fp, w_fn, axis):
    '''Row-wise or column-wise expansion of a pattern given u and v.

    Parameters
    ----------
    axis : int, [0, 1]
        `axis` stands for the dimension of factor that remain unchanged.
        `1` for row-wise expansion and `0` for column-wise expansion.
    '''
    s_old = coverage_score(
        gt=X_gt, 
        pd=X_old, 
        w_fp=w_fp, 
        w_fn=w_fn, 
        axis=axis
    )

    if axis == 1:
        # i_idx = bool_to_index(invert(u))
        # j_idx = bool_to_index(v)
        _u, _v = invert(u), v
    elif axis == 0:
        # i_idx = bool_to_index(u)
        # j_idx = bool_to_index(invert(v))
        _u, _v = u, invert(v)

    pattern = matmul(_u, _v.T, sparse=True, boolean=True)
    X_new = add(X_old, pattern, sparse=True, boolean=True)

    s_new = coverage_score(
        gt=X_gt, 
        pd=X_new, 
        w_fp=w_fp, 
        w_fn=w_fn, 
        axis=axis
    )
    d_scores = s_new - s_old

    score = d_scores.max()
    index = d_scores.argmax()

    return score, index