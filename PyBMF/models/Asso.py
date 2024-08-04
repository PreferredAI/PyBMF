import numpy as np
from ..utils import matmul, add, to_sparse, to_dense, binarize, binarize
from ..utils import invert, coverage_score, ERR, description_length
from ..utils import get_prediction
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm


class Asso(BaseModel):
    '''The Asso algorithm.

    .. topic:: Reference

        The discrete basis problem. Zhang et al. 2007.
    
    Parameters
    ----------
    k : int, optional
        The target rank.
        If ``None``, it will factorize until the error is smaller than ``tol``, or when other stopping criteria is met.
    tol : float, default: 0
        The error tolerance.
    tau : float
        The binarization threshold when building basis.
        Can be determined via model selection techniques.
    w_fp : float
        The penalty weights for FP. 
    w_fn : float, optional, default: None
        The penalty weights for FN. 
        If ``w_fn`` is ``None``, it will be treated as ``1 - w_fp``.
    '''
    def __init__(self, tau, k=None, tol=0, w_fp=0.5, w_fn=None):
        self.check_params(tau=tau, k=k, tol=tol, w_fp=w_fp, w_fn=w_fn)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model.
        '''
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        '''Initialize the model.
        '''
        super().init_model()
        
        # real-valued association matrix
        self.assoc = build_assoc(X=self.X_train, dim=1)
        # binary-valued basis candidates
        self.basis = build_basis(assoc=self.assoc, tau=self.tau)

        settings = [(self.assoc, [0, 0], 'assoc'), (self.basis, [0, 1], 'basis')]
        self.show_matrix(settings, colorbar=True, clim=[0, 1], title=f'tau: {self.tau}')


    def _fit(self):
        '''The main procedure of fitting.
        '''
        k = 0
        is_improving = True
        pbar = tqdm(total=self.k, position=0)
        while is_improving:

            best_row, best_col, best_idx = None, None, None
            best_score = 0 if k == 0 else best_score # best coverage score inherited from previous factors
            n_basis = self.basis.shape[0] # number of basis candidates

            # early stop detection
            if n_basis == 0:
                is_improving = self.early_stop(msg="Candidate list is empty", k=k)
                break

            # row-wise score of previous factors
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            s_old = coverage_score(gt=self.X_train, pd=self.X_pd, w_fp=self.w_fp, w_fn=self.w_fn, axis=1)

            for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k}"):
                row = self.basis[i]
                score, col = get_vector(
                    X_gt=self.X_train, 
                    X_old=self.X_pd, 
                    s_old=s_old, 
                    basis=row, 
                    basis_dim=1, 
                    w_fp=self.w_fp,
                    w_fn=self.w_fn,
                )
                if score > best_score:
                    best_score, best_row, best_col, best_idx = score, row, col, i

            # early stop detection
            if best_idx is None:
                is_improving = self.early_stop(msg="No pattern found.", k=k)
                break

            # update factors
            self.set_factors(k, best_col.T, best_row.T)

            # remove this basis (unnecessary)
            idx = np.array([j for j in range(n_basis) if j != best_idx])
            self.basis = self.basis[idx]

            # evaluation
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            # show matrix at every step
            if self.verbose and self.display:
                self.show_matrix(title=f"k: {k}, tau: {self.tau}, w: {[self.w_fp, self.w_fn]}")

            # original
            # self.evaluate(df_name='updates', head_info={'k': k}, train_info={'score': best_score}, verbose=self.verbose)

            # ex01: score and description length
            score = coverage_score(gt=self.X_train, pd=self.X_pd, w_fp=0.5, axis=None)
            desc_len = description_length(gt=self.X_train, pd=self.X_pd, U=self.U, V=self.V, w_model=1, w_fp=1, w_fn=1)
            self.evaluate(
                df_name='updates', 
                head_info={'k': k}, 
                train_info={
                    'score': best_score, 
                    'score_0.5': score, 
                    'desc_len': desc_len,
                    'shape': [best_col.sum(), best_row.sum()], 
                }, 
                metrics=['TP', 'TPR', 'FP', 'FPR', 'FN', 'FNR', 'ERR', 'ACC', 'Recall', 'Precision', 'F1'], 
                verbose=self.verbose, 
            )

            # early stop detection
            is_improving = self.early_stop(error=ERR(gt=self.X_train, pd=self.X_pd), k=k)
            is_improving = self.early_stop(n_factor=k+1)
            
            # update pbar and k
            pbar.update(1)
            k += 1



def get_vector(X_gt, X_old, s_old, basis, basis_dim, w_fp, w_fn):
    '''Return the optimal column/row vector given a row/column basis candidate.

    Parameters
    ----------
    X_gt : spmatrix
        The ground-truth matrix.
    X_old : spmatrix
        The prediction matrix before adding the current pattern.
    s_old : array
        The column/row-wise coverage scores of previous prediction `X_pd`.
    basis : (1, n) spmatrix
        The basis vector.
    basis_dim : int
        The dimension to which `basis` belongs.
        If `basis_dim == 0`, a pattern is considered `basis.T * vector`.
        Otherwise, it's considered `vector.T * basis`.
        Note that both `basis` and `vector` are row vectors.
    w_fp, w_fn : float
        The penalty weights for false positives and false negatives.

    Returns
    -------
    score : float
        The coverage score.
    vector : (1, n) spmatrix
        The matched vector.
    '''
    vector_dim = 1 - basis_dim
    vector = lil_matrix(np.ones((1, X_gt.shape[vector_dim])))
    pattern = matmul(basis.T, vector, sparse=True, boolean=True)
    pattern = pattern if basis_dim == 0 else pattern.T

    # new X and coverage score vector
    X_new = add(X_old, pattern, sparse=True, boolean=True)
    s_new = coverage_score(gt=X_gt, pd=X_new, w_fp=w_fp, w_fn=w_fn, axis=basis_dim)

    vector = lil_matrix(np.array(s_new > s_old, dtype=int))

    # scores from old and new entries
    s_old = s_old[to_dense(invert(vector), squeeze=True).astype(bool)]
    s_new = s_new[to_dense(vector, squeeze=True).astype(bool)]
    score = s_old.sum() + s_new.sum()

    return score, vector
        

def build_assoc(X, dim):
    '''Build the real-valued association matrix.

    Parameters
    ----------
    X : ndarray, spmatrix
        The data matrix.
    dim : int
        The dimension which ``basis`` belongs to.
        If ``dim`` == 0, ``basis`` is treated as a column vector and ``vector`` as a row vector.

    Returns
    -------
    assoc : spmatrix
        The association matrix.
    '''
    assoc = X @ X.T if dim == 0 else X.T @ X
    assoc = to_sparse(assoc, 'lil').astype(float)
    s = X.sum(axis=1-dim)
    s = to_dense(s, squeeze=True)
    for i in range(X.shape[dim]):
        assoc[i, :] = (assoc[i, :] / s[i]) if s[i] > 0 else 0
    return assoc
    

def build_basis(assoc, tau):
    '''Get the binary-valued basis candidates.

    Parameters
    ----------
    assoc : spmatrix
        The association matrix.
    tau : float
        The threshold for the association matrix.

    Returns
    -------
    basis : spmatrix
        The binary-valued basis candidates.
    '''
    basis = binarize(assoc, tau)
    basis = to_sparse(basis, 'lil').astype(int)
    nonzero_idx = np.array(basis.sum(axis=1) != 0).squeeze()
    basis = basis[nonzero_idx]
    return basis