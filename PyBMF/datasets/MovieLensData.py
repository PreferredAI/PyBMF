import os
import pandas as pd
from scipy.sparse import csr_matrix
from ..utils import binarize, reverse_index, safe_indexing
from .BaseData import BaseData


class MovieLensData(BaseData):
    '''Load MovieLens dataset.

    Parameters
    ----------
    path : str
        Path to the cached dataset.
    size : str in {'100k', '1m'}
        MovieLens dataset size.
    '''
    def __init__(self, path=None, size="1m"):
        super().__init__(path=path)
        self.is_single = True
        assert size in ['100k', '1m'], "Size not available."
        self.name = 'ml_' + size
        self.size = size


    def read_data(self):
        '''Read data.
        '''
        # ratings
        if self.size == '100k':
            path = os.path.join(self.root, "ml-100k", "u.data")
            sep, engine = '\t', 'c'
        elif self.size == '1m':
            path = os.path.join(self.root, "ml-1m", "ratings.dat")
            sep, engine = '::', 'python'

        self.df_ratings = pd.read_table(path, delimiter=sep, engine=engine, header=None, names=['uid', 'iid', 'rating', 'timestamp'])
        self.df_ratings[['uid', 'iid']] = self.df_ratings[['uid', 'iid']].astype(int)
        self.df_ratings["timestamp"] = pd.to_datetime(self.df_ratings["timestamp"], unit='s')

        # titles
        if self.size == '100k':
            path = os.path.join(self.root, "ml-100k", "u.item")
            sep, engine = '|', 'c'
        elif self.size == '1m':
            path = os.path.join(self.root, "ml-1m", "movies.dat")
            sep, engine = '::', 'python'

        self.df_titles = pd.read_table(path, delimiter=sep, engine=engine, header=None, encoding="latin1", usecols=[0, 1], names=['iid', 'title'])
        self.df_titles['iid'] = self.df_titles['iid'].astype(int)


    def load_data(self):
        '''Load data.
        '''
        # generate row_idx and col_idx
        self.df_ratings['row_idx'], _ = pd.factorize(self.df_ratings['uid'], sort=True)
        self.df_ratings['col_idx'], _ = pd.factorize(self.df_ratings['iid'], sort=True)

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

        self.X, u_info = self.sort_factor(self.X, dim=0, factor_info=u_info)
        self.X, i_info = self.sort_factor(self.X, dim=1, factor_info=i_info)
        self.factor_info = [u_info, i_info]


    def sort_factor(self, X, dim, factor_info):
        '''Sort the matrix and factor_info by factor order.

        Parameters
        ----------
        X : csr_matrix
            The matrix to be sorted.
        dim : int
            If ``dim`` is 0, sort rows.
            If ``dim`` is 1, sort columns.
        factor_info : list of 2 tuples
            The list of factor info. For example, [``u_order``, ``u_idmap``, ``u_alias``].

        Returns
        -------
        X : csr_matrix
            The sorted matrix.
        factor_info : list of 2 tuples
            The list of factor info. For example, [``u_order``, ``u_idmap``, ``u_alias``].
        '''
        f_order, f_idmap, f_alias = factor_info
        idx = reverse_index(f_order)
        f_order = safe_indexing(f_order, idx)
        f_idmap = safe_indexing(f_idmap, idx)
        f_alias = safe_indexing(f_alias, idx)
        X = X[idx, :] if dim == 0 else X[:, idx]
        factor_info = [f_order, f_idmap, f_alias]
        return X, factor_info