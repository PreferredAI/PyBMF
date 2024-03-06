import numpy as np
from utils import matmul, add, to_sparse, to_dense, binarize, binarize
from utils import invert, cover, eval, record, header
from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from tqdm import tqdm


class Asso(BaseModel):
    '''The Asso algorithm.
    
    Reference
    ---------
    The discrete basis problem. Zhang et al. 2007.
    '''
    def __init__(self, k, tau=None, w=None):
        """
        Parameters
        ----------
        k : int
            The rank.
        tau : float
            The binarization threshold when building basis.
        w : float in [0, 1]
            The ratio of true positives.
        """
        self.check_params(k=k, tau=tau, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "tau" in kwargs:
            tau = kwargs.get("tau")
            assert tau is not None, "Missing tau."
            self.tau = tau
            print("[I] tau          :", self.tau)
        if "w" in kwargs:
            w = kwargs.get("w")
            assert w is not None, "Missing w."
            self.w = w
            print("[I] weights      :", self.w)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val)
        self.init_model()

        # real-valued association matrix
        self.assoc = self.build_assoc(X=self.X_train, dim=1)
        # binary-valued basis candidates
        self.basis = self.build_basis(assoc=self.assoc, tau=self.tau)
        if self.verbose:
            self.show_matrix([(self.assoc, [0, 0], 'assoc'), (self.basis, [0, 1], 'basis')], colorbar=True, clim=[0, 1], title=f'tau: {self.tau}')

        self._fit()

        display(self.logs['updates'])
        self.show_matrix(colorbar=True, discrete=True, clim=[0, 1], title="result")


    @staticmethod
    def build_assoc(X, dim):
        '''Build the real-valued association matrix.

        Parameters
        ----------
        X : ndarray, spmatrix
        dim : int
            The dimension which `basis` belongs to.
            If `dim` == 0, `basis` is treated as a column vector and `vector` as a row vector.
        '''
        assoc = X @ X.T if dim == 0 else X.T @ X
        assoc = to_sparse(assoc, 'lil').astype(float)
        s = X.sum(axis=1-dim)
        s = to_dense(s, squeeze=True)
        for i in range(X.shape[dim]):
            assoc[i, :] = (assoc[i, :] / s[i]) if s[i] > 0 else 0
        return assoc
        

    @staticmethod
    def build_basis(assoc, tau):
        '''Get the binary-valued basis candidates.

        Parameters
        ----------
        basis : spmatrix
            Each row of `basis` is a candidate basis.
        '''
        basis = binarize(assoc, tau)
        basis = to_sparse(basis, 'lil').astype(int)
        return basis


    def _fit(self):
        for k in tqdm(range(self.k), position=0):
            best_row, best_col, best_idx = None, None, None
            # best coverage score inherited from previous factors
            best_score = 0 if k == 0 else best_score
            # number of basis candidates
            n_basis = self.basis.shape[0]
            # early stop detection
            if n_basis == 0:
                self.early_stop(msg="No basis left.", k=k)
                break

            self.predict_X()
            s_old = cover(gt=self.X_train, pd=self.X_pd, w=self.w, axis=1)
            for i in tqdm(range(n_basis), leave=False, position=0, desc=f"[I] k = {k}"):
                row = self.basis[i]
                score, col = self.get_vector(
                    X_gt=self.X_train, X_old=self.X_pd, s_old=s_old, 
                    basis=row, basis_dim=1, w=self.w)
                if score > best_score:
                    best_score, best_row, best_col, best_idx = score, row, col, i
            # early stop detection
            if best_idx is None:
                self.early_stop(msg="Coverage stops improving.", k=k)
                break
            # update factors
            self.U[:, k], self.V[:, k] = best_col.T, best_row.T
            # remove this basis
            idx = np.array([j for j in range(n_basis) if j != best_idx])
            self.basis = self.basis[idx]
            # show matrix at every step
            if self.verbose and self.display:
                self.show_matrix(title=f"k: {k}, tau: {self.tau}, w: {self.w}")
                
            self.evaluate(names=['k', 'score'], values=[k, best_score], df_name='updates')


    @staticmethod
    def get_vector(X_gt, X_old, s_old, basis, basis_dim, w):
        '''Return the optimal column/row vector given a row/column basis candidate.

        Parameters
        ----------
        X_gt : spmatrix
            The ground-truth matrix.
        X_old : spmatrix
            The prediction matrix before adding the current pattern.
        s_old : array
            The column/row-wise cover scores of previous prediction `X_pd`.
        basis : (1, n) spmatrix
            The basis vector.
        basis_dim : int
            The dimension which `basis` belongs to.
            If `basis_dim == 0`, a pattern is considered `basis.T * vector`. Otherwise, it's considered `vector.T * basis`. Note that both `basis` and `vector` are row vectors.
        w : float in [0, 1]

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
        X_new = add(X_old, pattern)
        s_new = cover(gt=X_gt, pd=X_new, w=w, axis=basis_dim)

        vector = lil_matrix(np.array(s_new > s_old, dtype=int))
        s_old = s_old[to_dense(invert(vector), squeeze=True).astype(bool)]
        s_new = s_new[to_dense(vector, squeeze=True).astype(bool)]
        score = s_old.sum() + s_new.sum()

        return score, vector
        