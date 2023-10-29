from .data import Data
import os
import pandas as pd
from scipy.sparse import csr_matrix
from utils import binarize


class NetflixData(Data):
    def __init__(self, small=True):
        super().__init__()
        # prepare netflix data with genre information
        self.set_paths(small)
        self.read()
        self.calibrate_genres()
        self.remove_genres(threshold=0.01)
        self.calibrate_movies()
        self.set_ratings_data()
        self.set_genres_data()
        self.set_data()
        

    def calibrate_genres(self):
        # filter out movies in df_genres that are not in df_ratings (for netflix_small)
        self.df_genres = self.df_genres[self.df_genres['iid'].isin(self.df_ratings['iid'])]


    def calibrate_movies(self):
        # filter out movies in df_ratings that are not in df_genres
        self.df_ratings = self.df_ratings[self.df_ratings['iid'].isin(self.df_genres['iid'])]


    def set_ratings_data(self):
        # generate row_idx and col_idx
        self.df_ratings['row_idx'] = pd.factorize(self.df_ratings['uid'])[0]
        self.df_ratings['col_idx'] = pd.factorize(self.df_ratings['iid'])[0]
        # row_idx - uid
        self.df_U_info = self.df_ratings.drop_duplicates(subset=['uid', 'row_idx'])[['uid', 'row_idx']]
        # col_idx - iid
        self.df_V_info = self.df_ratings.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']]
        # row_idx - col_idx - rating
        self.df_ratings_triplet = self.df_ratings[['row_idx', 'col_idx', 'rating']]


    def set_genres_data(self):
        # one-hot to triplet
        self.df_genres = self.df_genres.set_index('iid').stack().reset_index()
        self.df_genres.columns = ['iid', 'genre', 'value']
        # remove zeros
        idx_list = self.df_genres[self.df_genres['value'] == 0].index
        self.df_genres = self.df_genres.drop(index=idx_list)
        # generate row_idx and col_idx
        self.df_genres['row_idx'] = pd.factorize(self.df_genres['genre'])[0]
        self.df_genres = self.df_genres.merge(self.df_V_info[['iid', 'col_idx']], on='iid', how='left') # col_idx is merged from df_ratings

        # row_idx - genre
        self.df_W_info = self.df_genres.drop_duplicates(subset=['genre', 'row_idx'])[['genre', 'row_idx']]
        # col_idx - iid - title
        self.df_M_info = self.df_genres.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']] # this is the same as df_V_idmap from df_ratings
        self.df_M_info = self.df_titles.merge(self.df_M_info[['iid', 'col_idx']], on='iid', how='right')
        self.df_M_info = self.df_M_info[['iid', 'title', 'col_idx']]
        # row_idx - col_idx - value
        self.df_genres_triplet = self.df_genres[['row_idx', 'col_idx', 'value']]


    def set_data(self):
        # prepare to load into dataset: build factors
        U_order = self.df_U_info['row_idx'].values.astype(int)
        U_idmap = self.df_U_info['uid'].values.astype(int)
        U_alias = self.df_U_info['uid'].values.astype(str)

        V_order = self.df_V_info['col_idx'].values.astype(int)
        V_idmap = self.df_V_info['iid'].values.astype(int)
        V_alias = self.df_M_info['title'].values.astype(str) # borrowed alias from M

        rows = self.df_ratings_triplet['row_idx'].values.astype(int)
        cols = self.df_ratings_triplet['col_idx'].values.astype(int)
        values = self.df_ratings_triplet['rating'].values.astype(int)
        values = binarize(values, threshold=0.5)
        X_matrix = csr_matrix((values, (rows, cols)))
        
        self.X.update(matrix=X_matrix, name='X')
        self.U.update(order=U_order, idmap=U_idmap, alias=U_alias, name='U')
        self.V.update(order=V_order, idmap=V_idmap, alias=V_alias, name='V')


    def remove_genres(self, threshold=0.01):
        # remove genres that cover less then 1% of the movies
        threshold *= len(self.df_genres)
        stat = self.df_genres.sum()
        genre_list = stat[stat < threshold].index.tolist()
        idx_list = self.df_genres[self.df_genres[genre_list].eq(1).any(axis=1)].index
        self.df_genres.drop(idx_list, inplace=True)
        self.df_genres.drop(columns=genre_list, axis=1, inplace=True)


    def read(self):
        # read netflix ratings
        self.df_ratings = pd.read_csv(self.ratings_path, header=None, names=['uid','iid','rating','date'])
        self.df_ratings['date'] = pd.to_datetime(self.df_ratings['date'])

        # read movie titles
        self.df_titles = pd.read_csv(self.titles_path, header=None, names=['iid', 'year', 'title'])
        self.df_titles['year'] = pd.to_datetime(self.df_titles['year'])

        # read movie genres
        df = pd.read_csv(self.genres_path)
        df = df.rename(columns={'movieId': 'iid'})
        genres = df['genres'].str.get_dummies(sep='|')
        self.df_genres = pd.concat([df['iid'], genres], axis=1)


    def set_paths(self, small):
        dropbox_path = os.path.abspath("D:/Dropbox/datasets")
        onedrive_path = os.path.abspath("D:/OneDrive - Singapore Management University/datasets")

        if small:
            # 15MB processed UIRT dataset from cornac: U10k, I4945, R608k
            self.ratings_path = os.path.join(onedrive_path, "netflix-cornac", "data_small.csv") 
        else:
            # 2.43GB processed UIRT dataset from cornac: U480k, I17770, R100M
            self.ratings_path = os.path.join(onedrive_path, "netflix-cornac", "data.csv") 

        # processed title dataset: item-year-title
        self.titles_path = os.path.join(onedrive_path, "netflix-cornac", "movie_titles.csv")

        # processed genres from GitHub (not verified)
        self.genres_path = os.path.join(onedrive_path, "netflix-genres", "netflix_genres.csv")



class NetflixGenreData(NetflixData):
    def __init__(self, small=True):
        super().__init__(small)
        self.set_data()


    def set_data(self):
        M_order = self.df_M_info['col_idx'].values.astype(int) # same as above
        M_idmap = self.df_M_info['iid'].values.astype(int)
        M_alias = self.df_M_info['title'].values.astype(str)

        W_order = self.df_W_info['row_idx'].values.astype(int)
        W_idmap = self.df_W_info['row_idx'].values.astype(int)
        W_alias = self.df_W_info['genre'].values.astype(str)
        
        rows = self.df_genres_triplet['row_idx'].values.astype(int)
        cols = self.df_genres_triplet['col_idx'].values.astype(int)
        values = self.df_genres_triplet['value'].values.astype(int)
        values = binarize(values, threshold=0.5)
        Z_matrix = csr_matrix((values, (rows, cols)))

        self.X.update(matrix=Z_matrix, name='Z')
        self.U.update(order=W_order, idmap=W_idmap, alias=W_alias, name='W')
        self.V.update(order=M_order, idmap=M_idmap, alias=M_alias, name='V')