import os
import pandas as pd
from scipy.sparse import csr_matrix
from ..utils import binarize, cache, parse_list
from .NetflixData import NetflixData
from itertools import chain
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np


class NetflixGenreCastData(NetflixData):
    '''Load Netflix dataset with genre and cast information.
     
    Genre and cast information comes from Netflix-Prize-IMDB-TMDB-Joint-Dataset on GitHub: 
    https://github.com/felixnie/Netflix-Prize-IMDB-TMDB-Joint-Dataset

    Parameters
    ----------
    path : str
        Path to the cached dataset.
    size : str in {'small', 'full'}
        Netflix data 'small' version, size 15MB, users ~10k, items 4945, ratings ~608k.
        Netflix data 'full' version, size 2.43GB, users ~480k, items 17770, ratings ~100M.
    source : str in {'imdb', 'tmdb'}
        Source should be 'imdb' or 'tmdb'.
    '''
    def __init__(self, size='small', source='imdb'):
        super().__init__(size=size)
        self.is_single = False
        assert source in ['imdb', 'tmdb'], "Source should be 'imdb' or 'tmdb'."
        self.name = self.name + '_genre_cast_' + source
        self.source = source


    def read_data(self):
        '''Read data.
        '''
        # ratings and titles
        super().read_data()

        # genres and cast
        path = cache("https://github.com/felixnie/Netflix-Prize-IMDB-TMDB-Joint-Dataset/raw/main/netflix_all.csv", relative_path="data/netflix/netflix_all.csv", unzip=False)
        self.df_info = pd.read_csv(path)
        parse_list(df=self.df_info, columns=['imdb_genres', 'tmdb_genres', 'imdb_cast', 'tmdb_cast'])


    def load_data(self):
        '''Load data.
        '''
        # X and factor_info from ratings
        super().load_data()
        X = self.X
        user_info, movie_info = self.factor_info

        Y, genre_alias = self.get_attribute_info(self.source + '_genres')
        Z, cast_alias = self.get_attribute_info(self.source + '_cast')
        
        genre_order = np.arange(len(genre_alias))
        genre_idmap = np.arange(len(genre_alias))
        genre_info = [genre_order, genre_idmap, genre_alias]

        cast_order = np.arange(len(cast_alias))
        cast_idmap = cast_alias.astype(int)
        cast_info = [cast_order, cast_idmap, cast_alias]

        # align the 3 matrices
        movie_ids = self.df_info['netflix_id'].values
        movie_idmap = movie_info[1]
        idx = [i for i in range(len(movie_ids)) if movie_ids[i] in movie_idmap]
        Y = Y[:, idx]
        Z = Z[:, idx]
        
        self.Xs = [X, Y, Z]
        self.factors = [[0, 1], [2, 1], [3, 1]]
        self.factor_info = [user_info, movie_info, genre_info, cast_info]


    def get_attribute_info(self, attribute):
        '''Get attribute information.

        Parameters
        ----------
        attribute : str
            The name of columns in ``df_info``.
        '''
        df = self.df_info.dropna(subset=[attribute])

        attr_list = sorted(list(set(chain.from_iterable(df[attribute]))))
        attr_list = np.array(attr_list).astype(str)

        attr_dict = {}
        for i, key in enumerate(attr_list):
            attr_dict[key] = i

        rows = list(chain.from_iterable(df[attribute]))
        rows = [attr_dict[str(x)] for x in rows]

        cols = []
        for i in df.index:
            for _ in range(len(df[attribute][i])):
                cols.append(i)

        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        values = np.ones(len(rows))

        m, n = len(attr_list), len(self.df_info)
        attr_mat = csr_matrix((values, (rows, cols)), shape=(m, n))
        return attr_mat, attr_list
