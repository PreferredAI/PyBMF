import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, hstack
from utils import binarize
from .MovieLensData import MovieLensData
from itertools import chain


class MovieLensGenreCastData(MovieLensData):
    '''Load MovieLens dataset with IMDB genre and cast information

    size:
        100k
        1m
    '''
    def __init__(self, path=None, size='1m'):
        super().__init__(path=path, size=size)
        self.is_single = False
        self.name = self.name + '_genre_cast'


    def read_data(self):
        # ratings and titles
        super().read_data()

        # genres and cast
        path = os.path.join(self.root, "MovieLens-IMDB-Dataset", "ml_" + self.size + "_imdb.pickle")
        self.df_info = pd.read_pickle(path)

        # preprocessing: generate 'cast' column
        def merge_list(row, columns=['director', 'actor', 'actress']):
            merged = []
            for c in columns:
                if isinstance(row[c], list):
                    merged = merged + row[c]
            merged = list(set(merged))
            return merged
        
        self.df_info['imdb_cast'] = self.df_info.apply(lambda x: merge_list(x), axis=1)


    def load_data(self):
        super().load_data()
        X = self.X
        user_info, movie_info = self.factor_info

        Y, genre_alias = self.get_attribute_info('imdb_genres')
        Z, cast_alias = self.get_attribute_info('imdb_cast')
        
        genre_order = np.arange(len(genre_alias))
        genre_idmap = np.arange(len(genre_alias))
        genre_info = (genre_order, genre_idmap, genre_alias)

        cast_order = np.arange(len(cast_alias))
        cast_idmap = np.array([int(id[2:]) for id in cast_alias])
        cast_info = (cast_order, cast_idmap, cast_alias)

        # align the 3 matrices
        movie_ids = self.df_info['ml_id'].values
        movie_idmap = movie_info[1]
        idx = [i for i in range(len(movie_ids)) if movie_ids[i] in movie_idmap]
        Y = Y[:, idx]
        Z = Z[:, idx]
        
        self.Xs = [X, Y, Z]
        self.factors = [[0, 1], [2, 1], [3, 1]]
        self.factor_info = [user_info, movie_info, genre_info, cast_info]


    def get_attribute_info(self, attribute):
        '''
        attribute:
            the name of column in df_info
        sttr_mat:
            spmatrix, with axis 0 being genres or cast
        sttr_list:
            list of attribute alias
        '''
        df = self.df_info.dropna(subset=[attribute])

        attr_list = sorted(list(set(chain.from_iterable(df[attribute]))))
        attr_list = np.array(attr_list).astype(str)

        attr_dict = {}
        for i, key in enumerate(attr_list):
            attr_dict[key] = i

        rows = list(chain.from_iterable(df[attribute]))
        rows = [attr_dict[x] for x in rows]

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
