from ..utils import check_sparse, safe_indexing, to_sparse, sum, ignore_warnings
from scipy.sparse import csr_matrix, spmatrix
import time
import numpy as np
from typing import Union

class BaseSplit:
    '''Base class for data splitting and negative sampling methods ``NoSplit``, ``RatioSplit`` and ``CrossValidation``.

    .. note::

        Attributes of ``BaseSplit``.

        X_train : spmatrix
            The training data matrix.
        X_val : spmatrix
            The validation data matrix.
        X_test : spmatrix
            The test data matrix.

    Parameters
    ----------
    X : ndarray, spmatrix
        The data matrix.
    '''
    def __init__(self, X: Union[np.ndarray, spmatrix]):

        # input
        self.X = to_sparse(X, 'csr')
        
        # output
        self.X_train = None
        self.X_val = None
        self.X_test = None


    def negative_sample(self):
        '''Negative sampling.

        .. note::
        
            We can only add 0's using csr/csc_matrix, and validate negative samples using ``coo_matrix`` or triplet.

            ``coo_matrix`` does not support value assignment;
            ``lil_matrix`` has no effect when adding 0's onto it.

            Any arithmetic operation or ``csr_matrix.eliminate_zeros()`` will cause a sparse matrix to lose the negative samples.
        '''
        raise NotImplementedError("Missing negative_sample method.")


    def check_params(self, **kwargs):
        '''Check patameters.
        
        Checking the random seed.
        '''
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'): # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I]   seed         :", self.seed)
            elif seed is not None: # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I]   seed         :", self.seed)
            else: # self.rng remains unchanged
                pass


    def load_pos_data(self, train_idx, val_idx, test_idx):
        '''Load positive data.

        Used in ``RatioSplit`` and ``CrossValidation``.

        Leave ``X_val``, ``X_test`` empty if ``val_idx``/``test_idx`` length is 0 for negative sampling.

        Parameters
        ----------
        train_idx : ndarray
            The indices of training data.
        val_idx : ndarray
            The indices of validation data.
        test_idx : ndarray
            The indices of test data.
        '''
        self.X_train = safe_indexing(self.X, train_idx)
        self.X_val = safe_indexing(self.X, val_idx) if len(val_idx) > 0 else csr_matrix(self.X.shape)
        self.X_test = safe_indexing(self.X, test_idx) if len(test_idx) > 0 else csr_matrix(self.X.shape)
        
        self.pos_train_size = len(train_idx)
        self.pos_val_size = len(val_idx)
        self.pos_test_size = len(test_idx)


    def get_neg_indices(self, n_negatives, type):
        '''Generate negative indices.

        Used in ``RatioSplit.negative_sample`` and ``CrossValidation.negative_sample``.

        This is fast but intractable for large dataset. Use trial-and-error for large dataset.

        Parameters
        ----------
        n_negatives : int
            Number of negative samples.
        type : str
            Negative sampling type.
        '''
        if n_negatives == 0:
            return np.array([]), np.array([])
        
        assert type in ['uniform', 'popularity'], "Unsupported negative sampling option: {}".format(type)
        
        m, n = self.X.shape
        if type == "uniform":
            p = np.ones((m, n))
        elif type == "popularity":
            p = np.zeros((m, n))
            pu, pv = sum(self.X)
            pu = pu / self.X.nnz
            pv = pv / self.X.nnz
            for r in range(m):
                p[r] = pu[r] * pv
        else:
            raise ValueError("[E] Unsupported negative sampling option: {}".format(type))
        
        p[self.X.toarray() == 1] = 0
        p = p.flatten()
        p = p / p.sum()
        indices = self.rng.choice(a=m*n, size=n_negatives, replace=False, p=p)
        
        U_neg = (indices / n).astype(int)
        V_neg = (indices % n).astype(int)

        return U_neg, V_neg
    

    @ignore_warnings
    def load_neg_data(self, train_idx, val_idx, test_idx, U_neg, V_neg):
        '''Load negative data.

        Used in ``RatioSplit.negative_sample`` and ``CrossValidation.negative_sample``.
        '''
        self.X_train = to_sparse(self.X_train, type='csr')
        self.X_val = to_sparse(self.X_val, type='csr')
        self.X_test = to_sparse(self.X_test, type='csr')

        self.X_train.eliminate_zeros()
        self.X_val.eliminate_zeros()
        self.X_test.eliminate_zeros()

        # SparseEfficiencyWarning
        self.X_train[U_neg[train_idx], V_neg[train_idx]] = 0
        if len(val_idx) > 0:
            self.X_val[U_neg[val_idx], V_neg[val_idx]] = 0
        if len(test_idx) > 0:
            self.X_test[U_neg[test_idx], V_neg[test_idx]] = 0

        self.neg_train_size = len(train_idx)
        self.neg_val_size = len(val_idx)
        self.neg_test_size = len(test_idx)