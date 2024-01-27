from .BaseModel import BaseModel
from typing import Tuple, List, Union
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from utils import get_dummy_factor_info, get_factor_list, get_matrices, get_settings
from utils import to_sparse, reverse_index, matmul, show_matrix


class BaseCollectiveModel(BaseModel):
    def __init__(self) -> None:
        # model parameters
        self.Us = None
    

    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.check_collective_dataset(X_train, X_val)
        self._fit()
        
        
    def check_collective_dataset(self, 
                                 Xs_train: List[np.ndarray, spmatrix], 
                                 Xs_train_factors: List[List[int]], 
                                 Xs_train_factor_info: List[Tuple]=None,
                                 Xs_val: List[np.ndarray, spmatrix]=None, 
                                 Xs_val_factors: List[List[int]]=None, 
                                 Xs_val_factor_info: List[Tuple]=None):
        """Load train and val data

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not modified, csr or csc is preferred.
        """
        if Xs_train is None:
            raise TypeError("Missing training data.")
        else:
            if Xs_train_factors is None:
                raise TypeError("Missing training factors, using dummy info.")
            if Xs_train_factor_info is None:
                print("[W] Missing training factor info.")
                Xs_train_factor_info = get_dummy_factor_info(Xs=Xs_train, factors=Xs_train_factors)

        if Xs_val is None:
            print("[W] Missing validation data.")
        else:
            if Xs_val_factors is None:
                raise TypeError("Missing validation factors.")
            if Xs_val_factor_info is None:
                print("[W] Missing validation factor info, using dummy info.")
                Xs_val_factor_info = get_dummy_factor_info(Xs=Xs_val, factors=Xs_val_factors)

        self.Xs_train = [to_sparse(X, 'csr') for X in Xs_train]
        self.Xs_train_factors = Xs_train_factors
        self.Xs_train_matrices = get_matrices(Xs_train_factors)
        self.Xs_train_factor_info = Xs_train_factor_info
        self.Xs_train_factor_list = get_factor_list(Xs_train_factors)

        self.Xs_val = None if Xs_val is None else [to_sparse(X, 'csr') for X in Xs_val]
        self.Xs_val_factors = None if Xs_val is None else Xs_val_factors
        self.Xs_val_matrices = None if Xs_val is None else get_matrices(Xs_val_factors)
        self.Xs_val_factor_info = None if Xs_val is None else Xs_val_factor_info
        self.Xs_val_factor_list = None if Xs_val is None else get_factor_list(Xs_val_factors)

        self.n_factors = len(self.Xs_train_factor_list)
        
        self.Us = []
        for i in range(self.n_factors):
            dim = len(self.Xs_train_factor_info[i][0])
            U = lil_matrix(np.zeros((dim, self.k)))
            self.Us.append(U)

    def show_matrix(self, settings: List[Tuple]=None, 
                    matrix: Union[np.ndarray, spmatrix]=None, 
                    factors: List[List[int]]=None,
                    factor_info: List[Tuple]=None,
                    scaling=None, pixels=None, title=None, colorbar=True, **kwargs):
        """The show_matrix() wrapper for CMF models, enabled by self.display

        If both settings and matrix are None, show the factors and their boolean product.

        settings : list of tuples
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

        factors = self.Xs_train_factors if factors is None else factors
        factor_info = self.Xs_train_factor_info if factor_info is None else factor_info
        all_factors = self.Xs_train_factors # must be full factors

        if settings is None and matrix is None:
            Xs = []
            for a, b in factors:
                X = matmul(U=self.Us[a], V=self.Us[b].T, boolean=True, sparse=True)
                Xs.append(X)
            settings = get_settings(Xs, factors, factor_info, self.Us, all_factors)

        elif matrix is not None:
            settings = [(matrix, [0, 0], title)]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar, **kwargs)
