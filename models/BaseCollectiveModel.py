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
        factor_list : list of int
        factor_dims : list of int
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


    def show_matrix(self, settings: List[Tuple]=None, 
                    matrix: Union[np.ndarray, spmatrix]=None, 
                    factors: List[List[int]]=None,
                    factor_info: List[Tuple]=None,
                    scaling=None, pixels=None, title=None, colorbar=True, **kwargs):
        """The show_matrix() wrapper for CMF models.

        If both settings and matrix are None, show the factors and their boolean product.

        settings : list of tuple
            to show matrices given their positions and labels.
        matrix : np.ndarray, spmatrix
            show a single matrix even when settings are provided.
        factor_info : list of tuples
            the info tuples of each factor.
        factors : list of int lists
            the factors of a matrix.
        """
        if not self.display:
            return
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        factors = self.factors if factors is None else factors
        factor_info = get_dummy_factor_info(self.Xs_train, factors) if factor_info is None else factor_info
        all_factors = self.factor_list # must be full factors

        if settings is None and matrix is None:
            Xs = []
            for a, b in factors:
                X = matmul(U=self.Us[a], V=self.Us[b].T, boolean=True, sparse=True)
                Xs.append(X)
            settings = get_settings(Xs, factors, factor_info, self.Us, all_factors)

        elif matrix is not None:
            settings = [(matrix, [0, 0], title)]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar, **kwargs)
