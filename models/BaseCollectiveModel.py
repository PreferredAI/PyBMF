from .BaseModel import BaseModel
from typing import Tuple, List, Union
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from utils import split_factor_list, get_factor_list, get_matrices, get_settings, get_factor_dims, concat_Xs_into_X, get_factor_starts, get_dummy_factor_info
from utils import to_sparse, reverse_index, matmul, show_matrix


class BaseCollectiveModel(BaseModel):
    def __init__(self) -> None:
        raise NotImplementedError("Missing init method.")
    

    def fit(self, Xs_train, factors, Xs_val=None, **kwargs):
        """Fit the model to observations, with validation if necessary.

        Please implement your own fit method.
        
        X_train : ndarray, spmatrix
            Data for matrix factorization.
        X_val : ndarray, spmatrix
            Data for model selection.
        kwargs :
            Common parameters that are checked and set in `BaseModel.check_params()` and the model-specific parameters included in `self.check_params()`.
        """
        raise NotImplementedError("Missing fit method.")
        
        
    def load_dataset(self, Xs_train, factors, Xs_val=None):
        """Load train and val data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not modified, csr or csc is preferred.

        Parameters
        ----------
        Xs_train : list of np.ndarray or spmatrix
            List of Boolean matrices for training.
        factors : list of int list
            List of factor id pairs, indicating the row and column factors of each matrix.
        Xs_val : list of np.ndarray, spmatrix and None, optional
            List of Boolean matrices for validation. It should have the same length of Xs_train. In the ``list``, `None` can be used as placeholders of those matrices that are not being validated. When `Xs_val` is `None`, the matrix with factor id [0, 1] is used as the only matrix being validated.
        """
        if Xs_train is None:
            raise TypeError("Missing training data.")
        if factors is None:
            raise TypeError("Missing factors.")
        if Xs_val is None:
            print("[W] Missing validation data.")

        self.Xs_train = [to_sparse(X, 'csr') for X in Xs_train]
        self.factors = factors
        self.Xs_val = None if Xs_val is None else [to_sparse(X, 'csr') for X in Xs_val]

        self.X_train = concat_Xs_into_X(Xs_train, factors)
        self.matrices = get_matrices(factors)
        self.factor_list = get_factor_list(factors)
        self.factor_dims = get_factor_dims(Xs_train, factors)
        self.row_factors, self.col_factors = split_factor_list(factors)
        self.row_starts, self.col_starts = get_factor_starts(Xs_train, factors)

        self.n_factors = len(self.factor_list)
        self.n_matrices = len(Xs_train)
        

    def init_model(self):
        """Initialize factors and logging variables.

        Us : list of spmatrix
        logs : dict
            The dict containing dataframes, arrays and lists.
        """
        self.Us = []
        for dim in self.factor_dims:
            U = lil_matrix(np.zeros((dim, self.k)))
            self.Us.append(U)
        self.logs = {}


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        """The show_matrix() wrapper for CMF models.

        If `settings` is None, show the factors and their boolean product.
        """
        if not self.display:
            return
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            self.update_Xs()
            settings = get_settings(Xs=self.Xs, factors=self.factors, Us=self.Us)

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def update_Xs(self):
        if not hasattr(self, 'Xs'):
            self.Xs = [None] * self.n_matrices
        for i, factors in enumerate(self.factors):
            a, b = factors
            X = matmul(U=self.Us[a], V=self.Us[b].T, boolean=True, sparse=True)
            self.Xs[i] = X