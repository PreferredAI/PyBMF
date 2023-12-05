from .NetflixData import NetflixData
from utils import binarize
from scipy.sparse import csr_matrix


class NetflixGenreData(NetflixData):
    def __init__(self, path=None, small=True):
        super().__init__(path, small)
        self.set_data()


    def set_data(self):
        U_order = self.df_genres_U_info['row_idx'].values.astype(int)
        U_idmap = self.df_genres_U_info['row_idx'].values.astype(int)
        U_alias = self.df_genres_U_info['genre'].values.astype(str)
        U_info = (U_order, U_idmap, U_alias)

        V_order = self.df_genres_V_info['col_idx'].values.astype(int)
        V_idmap = self.df_genres_V_info['iid'].values.astype(int)
        V_alias = self.df_genres_V_info['title'].values.astype(str)
        V_info = (V_order, V_idmap, V_alias)

        rows = self.df_genres_triplet['row_idx'].values.astype(int)
        cols = self.df_genres_triplet['col_idx'].values.astype(int)
        values = self.df_genres_triplet['value'].values.astype(int)
        values = binarize(values, threshold=0.5)
        self.X = csr_matrix((values, (rows, cols)))
        self.factor_info = [U_info, V_info]