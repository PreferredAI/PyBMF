import os
import pandas as pd
from scipy.sparse import csr_matrix
from utils import binarize
from .BaseData import BaseData


class NetflixData(BaseData):
    '''Load Netflix dataset with preprocessing
    '''
    def __init__(self, path=None, small=True):
        self.X = None # csr_matrix, to be split
        self.factor_info = None # tuple, (order, idmap, alias)

        self.set_paths(path, small)
        self.read()
        self.calibrate_genres()
        self.remove_genres(threshold=0.01)
        self.calibrate_movies()
        self.set_ratings_data()
        self.set_genres_data()
        self.set_data()


    def set_paths(self, path, small):
        if path is not None:
            self.ratings_path = path
        else:
            dropbox_path = os.path.abspath("D:/Dropbox/datasets")
            onedrive_path = os.path.abspath("D:/OneDrive - Singapore Management University/datasets")

            if small:
                # preprocessed UIRT dataset from cornac: size 15MB, users ~10k, items 4945, ratings ~608k
                self.ratings_path = os.path.join(onedrive_path, "netflix-cornac", "data_small.csv") 
            else:
                # preprocessed UIRT dataset from cornac: size 2.43GB, users ~480k, items 17770, ratings ~100M
                self.ratings_path = os.path.join(onedrive_path, "netflix-cornac", "data.csv")

        # preprocessed titles dataset: item-year-title
        # check sc01_fix_netflix_titles.ipynb
        self.titles_path = os.path.join(onedrive_path, "netflix-cornac", "movie_titles.csv")

        # preprocessed genres dataset from GitHub (not verified)
        self.genres_path = os.path.join(onedrive_path, "netflix-genres", "netflix_genres.csv")


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
        

    def calibrate_genres(self):
        # filter out movies in df_genres that are not in df_ratings (for netflix_small)
        self.df_genres = self.df_genres[self.df_genres['iid'].isin(self.df_ratings['iid'])]


    def remove_genres(self, threshold=0.01):
        # remove genres that cover less then 1% of the movies
        threshold *= len(self.df_genres)
        stat = self.df_genres.sum()
        genre_list = stat[stat < threshold].index.tolist()
        idx_list = self.df_genres[self.df_genres[genre_list].eq(1).any(axis=1)].index
        self.df_genres.drop(idx_list, inplace=True)
        self.df_genres.drop(columns=genre_list, axis=1, inplace=True)


    def calibrate_movies(self):
        # filter out movies in df_ratings that are not in df_genres
        self.df_ratings = self.df_ratings[self.df_ratings['iid'].isin(self.df_genres['iid'])]


    def set_ratings_data(self):
        # generate row_idx and col_idx
        self.df_ratings['row_idx'] = pd.factorize(self.df_ratings['uid'])[0]
        self.df_ratings['col_idx'] = pd.factorize(self.df_ratings['iid'])[0]
        # row_idx - uid
        self.df_ratings_U_info = self.df_ratings.drop_duplicates(subset=['uid', 'row_idx'])[['uid', 'row_idx']]
        # col_idx - iid
        self.df_ratings_V_info = self.df_ratings.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']]
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
        self.df_genres = self.df_genres.merge(self.df_ratings_V_info[['iid', 'col_idx']], on='iid', how='left') # col_idx is merged from df_ratings

        # row_idx - genre
        self.df_genres_U_info = self.df_genres.drop_duplicates(subset=['genre', 'row_idx'])[['genre', 'row_idx']]
        # col_idx - iid - title
        self.df_genres_V_info = self.df_genres.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']] # this is the same as df_V_idmap from df_ratings
        self.df_genres_V_info = self.df_titles.merge(self.df_genres_V_info[['iid', 'col_idx']], on='iid', how='right')
        self.df_genres_V_info = self.df_genres_V_info[['iid', 'title', 'col_idx']]
        # row_idx - col_idx - value
        self.df_genres_triplet = self.df_genres[['row_idx', 'col_idx', 'value']]


    def set_data(self):
        U_order = self.df_ratings_U_info['row_idx'].values.astype(int)
        U_idmap = self.df_ratings_U_info['uid'].values.astype(int)
        U_alias = self.df_ratings_U_info['uid'].values.astype(str)
        U_info = (U_order, U_idmap, U_alias)

        V_order = self.df_ratings_V_info['col_idx'].values.astype(int)
        V_idmap = self.df_ratings_V_info['iid'].values.astype(int)
        V_alias = self.df_genres_V_info['title'].values.astype(str)
        V_info = (V_order, V_idmap, V_alias)

        rows = self.df_ratings_triplet['row_idx'].values.astype(int)
        cols = self.df_ratings_triplet['col_idx'].values.astype(int)
        values = self.df_ratings_triplet['rating'].values.astype(int)
        values = binarize(values, threshold=0.5)
        self.X = csr_matrix((values, (rows, cols)))
        self.factor_info = [U_info, V_info]