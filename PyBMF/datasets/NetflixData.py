import os
import pandas as pd
from scipy.sparse import csr_matrix
from ..utils import binarize
from .BaseData import BaseData


class NetflixData(BaseData):
    '''Load Netflix dataset.

    Parameters
    ----------
    path : str
        Path to the cached dataset.
    size : str in {'small', 'full'}
        Netflix data 'small' version, size 15MB, users ~10k, items 4945, ratings ~608k.
        Netflix data 'full' version, size 2.43GB, users ~480k, items 17770, ratings ~100M.
    '''
    def __init__(self, path=None, size='small'):
        super().__init__(path=path)
        self.is_single = True
        assert size in ['small', 'full'], "Size is unavailable."
        self.name = 'netflix_' + size
        self.size = size


    def read_data(self):
        '''Read data.
        '''
        # ratings dataset
        path = os.path.join(self.root, "netflix-cornac", "data.csv" if self.size == 'full' else "data_small.csv")
        self.df_ratings = pd.read_csv(path, header=None, names=['uid','iid','rating','date'])
        self.df_ratings['date'] = pd.to_datetime(self.df_ratings['date'])

        # titles dataset in item-year-title, check sc01_fix_netflix_titles.ipynb
        path = os.path.join(self.root, "netflix-cornac", "movie_titles.csv")
        self.df_titles = pd.read_csv(path, header=None, names=['iid', 'year', 'title'])
        self.df_titles['year'] = pd.to_datetime(self.df_titles['year'])


    def load_data(self):
        '''Load data.
        '''
        # generate row_idx and col_idx
        self.df_ratings['row_idx'], _ = pd.factorize(self.df_ratings['uid'])
        self.df_ratings['col_idx'], _ = pd.factorize(self.df_ratings['iid'])

        # row_idx - uid
        df_user = self.df_ratings.drop_duplicates(subset=['uid', 'row_idx'])[['uid', 'row_idx']]

        # col_idx - iid - title
        df_item = self.df_ratings.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']]
        df_item = self.df_titles.merge(df_item[['iid', 'col_idx']], on='iid', how='right')

        # row_idx - col_idx - rating
        df_triplet = self.df_ratings[['row_idx', 'col_idx', 'rating']]

        # ratings factor_info
        u_order = df_user['row_idx'].values.astype(int)
        u_idmap = df_user['uid'].values.astype(int)
        u_alias = df_user['uid'].values.astype(str)
        u_info = [u_order, u_idmap, u_alias]

        i_order = df_item['col_idx'].values.astype(int)
        i_idmap = df_item['iid'].values.astype(int)
        i_alias = df_item['title'].values.astype(str)
        i_info = [i_order, i_idmap, i_alias]

        # ratings matrix
        rows = df_triplet['row_idx'].values.astype(int)
        cols = df_triplet['col_idx'].values.astype(int)
        values = df_triplet['rating'].values.astype(int)
        values = binarize(values, threshold=0.5)

        self.X = csr_matrix((values, (rows, cols)))
        self.factor_info = [u_info, i_info]
