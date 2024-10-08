import pickle
import os
from ..utils import sample, split_factor_list, concat_Xs_into_X, concat_factor_info, get_factor_starts, show_matrix, get_settings, get_cache_path
import numpy as np


class BaseData:
    '''Base class for built-in datasets.

    .. note:: 
    
        Attributes of ``BaseData`` for a single-matrix dataset.

        X : spmatrix
            The data matrix, which can be passed to ``NoSplit``, ``RatioSplit`` or ``CrossValidation`` or be used for factorization directly.
        factor_info : list of 2 tuples
            The list of factor info. For example, [``user_info``, ``item_info``].
            More specifically, the list may look like [(``u_order``, ``u_idmap``, ``u_alias``), (``i_order``, ``i_idmap``, ``i_alias``)].

    .. note::
    
        Attributes of ``BaseData`` for a multi-matrix dataset.

        Xs : list of spmatrix
            E.g., [``X_ratings``, ``X_genres``, ``X_cast``]
        factors : list of lists of 2 ints
            The list of factor id pairs.
            For example, [[0, 1], [2, 1], [3, 1]] if the 3 datasets are user-movie, genre-movie and cast-movie.
        factor_info : list of tuples
            The list of factor info. For example, [``user_info``, ``movie_info``, ``genre_info``, ``cast_info``].
    '''
    def __init__(self):

        self.X = None
        self.Xs = None
        self.factors = None
        self.factor_info = None

        self.is_single = None
        self.name = None


    def load(self, overwrite_cache=False):
        '''Load data.

        If pickle exists, load from cache directory.
        If not, read from data directory. Dump to pickle when ``overwrite_cache`` is True.

        Parameters
        ----------
        overwrite_cache : bool, default: False
            If True, overwrite the cache.
        '''
        if self.pickle_exists and not overwrite_cache:
            self.read_pickle()
        else:
            self.read_data()
            self.load_data()
            self.dump_pickle()
    

    @property
    def pickle_exists(self):
        '''If pickle exists.
        '''
        return os.path.exists(self.pickle_path)

    
    @property
    def pickle_path(self):
        '''The path of pickle file.
        '''
        if self.name is None:
            raise ValueError("name is not set.")
        else:
            pickle_path, _ = get_cache_path(relative_path="pickles/" + self.name + ".pickle", cache_dir=None)
            return pickle_path
    

    def read_data(self):
        '''Read data.
        '''
        raise NotImplementedError("Missing read data method.")
        

    def load_data(self):
        '''Load data.
        '''
        raise NotImplementedError("Missing load data method.")


    def read_pickle(self):
        '''Read pickle from cache directory.
        '''
        with open(self.pickle_path, 'rb') as handle:
            data = pickle.load(handle)
        if len(data) == 2:
            self.X = data['X']
            self.factor_info = data['factor_info']
        elif len(data) == 3:
            self.Xs = data['Xs']
            self.factors = data['factors']
            self.factor_info = data['factor_info']


    def dump_pickle(self):
        '''Dump pickle to cache directory.
        '''
        data = {'X': self.X, 'factor_info': self.factor_info} if self.is_single else {'Xs': self.Xs, 'factors': self.factors, 'factor_info': self.factor_info}

        with open(self.pickle_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def sample(self, factor_id, idx=None, n_samples=None, seed=None):
        '''Sample the whole dataset with given ``factor_id`` and ``idx``.

        Parameters
        ----------
        factor_id : int
            For single-matrix dataset, ``factor_id`` is the axis to sample, i.e., 0 and 1 for rows and columns.

            For multi-matrix dataset, ``factor_id`` is the index of the factor to sample.

        idx : np.ndarray
            The given indices to sample with.
        n_samples : int
            Randomly down-sample to this length.
        seed : int
            Random seed for down-sampling.

        Returns
        -------
        idx : np.ndarray
            The sampled indices.
        '''
        if self.is_single:
            idx, self.factor_info, self.X = sample(X=self.X, factor_info=self.factor_info, axis=factor_id, idx=idx, n_samples=n_samples, seed=seed)
        else:
            matrix_ids = [i for i in range(len(self.factors)) if factor_id in self.factors[i]] # which matrix to sample
            matrix_axis = [f.index(factor_id) for f in self.factors if factor_id in f] # which axis to sample
            for i, mat_id in enumerate(matrix_ids):
                if i == 0: # first time sampling
                    idx, self.factor_info[factor_id], self.Xs[mat_id] = sample(X=self.Xs[mat_id], factor_info=self.factor_info[factor_id], axis=matrix_axis[i], idx=idx, n_samples=n_samples, seed=seed)
                else: # the rest of matrices
                    _, _, self.Xs[mat_id] = sample(X=self.Xs[mat_id], axis=matrix_axis[i], idx=idx, n_samples=n_samples, seed=seed)
        return idx
    

    def to_single(self):
        '''Concatenate ``Xs`` to form a single ``X``.
        '''
        if self.is_single:
            print("[I] Being single matrix data already.")
            return
        else:
            self.X = concat_Xs_into_X(Xs=self.Xs, factors=self.factors)
            # self.factor_info = concat_factor_info(factor_info=self.factor_info, factors=self.factors)
            self.is_single = True


    def show_matrix(
            self, 
            scaling=1.0, pixels=5, 
            colorbar=True, 
            discrete=True, 
            center=True, 
            clim=[0, 1], 
            keep_nan=True, 
            **kwargs):
        '''The ``show_matrix`` wrapper for Boolean datasets.
        '''
        if self.is_single:
            settings = [(self.X, [0, 0], "X")]
        else:
            settings = get_settings(self.Xs, factors=self.factors)

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, 
                colorbar=colorbar, discrete=discrete, center=center, clim=clim, keep_nan=keep_nan, **kwargs)