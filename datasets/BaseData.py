import pickle
import os
from utils import sample, split_factor_list, concat_Xs_into_X, concat_factor_info, get_factor_starts
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


class BaseData:
    def __init__(self, path=None):
        """Built-in datasets

        for single-matrix dataset:
            X:
                spmatrix, which will be passed to NoSplit, RatioSplit or CrossValidation later
            factor_info:
                list of tuples
                e.g., [(u_order, u_idmap, u_alias), (i_order, i_idmap, i_alias)]

        for multi-matrix dataset:
            Xs:
                list of single dataset
                e.g., [ratings, genres, cast]
            factors:
                list of factor id pairs
                e.g., [[0, 1], [2, 1], [3, 1]] if the 3 datasets are user-movie, genre-movie and cast-movie
            factor_info:
                list of factor info
                e.g., [user_info, movie_info, genre_info, cast_info]
        """
        self.X, self.Xs, self.factors, self.factor_info = None, None, None, None
        self.is_single, self.name = None, None

        import platform
        plt = platform.system()

        self.root = os.path.abspath("D:/OneDrive - Singapore Management University/datasets" if plt == 'Windows' else "~/datasets")
        self.cache_path = os.path.abspath('D:/Dropbox/PyBMF/cache') # os.path.abspath("../cache")
        self.pickle_path = path


    def load(self, overwrite_cache=False):
        self.pickle_path = os.path.join(self.cache_path, self.name + '.pickle') if self.pickle_path is None else self.pickle_path
        if self.has_pickle and not overwrite_cache:
            self.read_pickle()
        else:
            self.read_data()
            self.load_data()
            self.dump_pickle()
    

    @property
    def has_pickle(self):
        return os.path.exists(self.pickle_path)
    

    def read_data(self):
        raise NotImplementedError("Missing read data method.")
        

    def load_data(self):
        raise NotImplementedError("Missing load data method.")


    def read_pickle(self):
        '''Read pickle from cache directory
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


    def dump_pickle(self, name=None):
        '''Dump pickle to cache directory

        single:
            single or collaborative dataset.
        name:
            name of pickle file.
        path:
            path of pickle file, will overwrite name.
        '''
        data = {'X': self.X, 'factor_info': self.factor_info} if self.is_single else {'Xs': self.Xs, 'factors': self.factors, 'factor_info': self.factor_info}

        path = self.pickle_path if name is None else os.path.join(self.cache_path, name + '.pickle')

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def sample(self, factor_id, idx=None, n_samples=None, seed=None):
        '''Sample the whole dataset with given factor_id and idx

        factor_id:
            for single-matrix dataset, factor_id is the axis to sample, i.e., 0 and 1 for rows and columns.
            for multi-matrix dataset, factor_id is the index of the factor to sample.
        idx:
            sample with given indices.
        n_samples:
            randomly down-sample to this length.
        seed:
            seed for down-sampling.
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
        '''Concatenate Xs to form a single X
        '''
        if self.is_single:
            print("[I] Being single matrix data already.")
            return
        else:
            self.X = concat_Xs_into_X(Xs=self.Xs, factors=self.factors)
            # self.factor_info = concat_factor_info(factor_info=self.factor_info, factors=self.factors)
            self.is_single = True
