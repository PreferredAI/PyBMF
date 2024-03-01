import os
import pandas as pd
from scipy.sparse import csr_matrix
from utils import binarize
from .NetflixData import NetflixData


class NetflixGenreData(NetflixData):
    '''Load Netflix dataset with genre information
     
    Genre information comes from netflix_genres.csv.
    Genres with low ratio are dropped.

    full:
        False, size 15MB, users ~10k, items 4945, ratings ~608k
        True, size 2.43GB, users ~480k, items 17770, ratings ~100M
    '''
    def __init__(self, path=None, full=False):
        super().__init__(path=path, full=full)
        self.is_single = False
        self.name = 'netflix_genre_data'


    def read_data(self):
        super().read_data()

        genres_path = os.path.join(self.root, "netflix-genres/netflix_genres.csv")
        df_genres = pd.read_csv(genres_path)
        df_genres = df_genres.rename(columns={'movieId': 'iid'})
        df_dummies = df_genres['genres'].str.get_dummies(sep='|')
        self.df_genres = pd.concat([df_genres['iid'], df_dummies], axis=1)
        

    def calibrate_genres(self):
        # filter out movies in df_genres that are not in df_ratings (for full=False)
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


    def set_genres_data(self):
        # one-hot to triplet
        self.df_genres = self.df_genres.set_index('iid').stack().reset_index()
        self.df_genres.columns = ['iid', 'genre', 'value']
        # remove zeros
        idx_list = self.df_genres[self.df_genres['value'] == 0].index
        self.df_genres = self.df_genres.drop(index=idx_list)
        # generate row_idx and col_idx
        self.df_genres['row_idx'] = pd.factorize(self.df_genres['genre'])[0]
        self.df_genres = self.df_genres.merge(self.df_ratings_item[['iid', 'col_idx']], on='iid', how='left') # col_idx is from df_ratings

        # row_idx - genre
        self.df_genres_genre = self.df_genres.drop_duplicates(subset=['genre', 'row_idx'])[['genre', 'row_idx']]
        # col_idx - iid - title
        self.df_genres_movie = self.df_genres.drop_duplicates(subset=['iid', 'col_idx'])[['iid', 'col_idx']] # this is the same as item id from df_ratings
        self.df_genres_movie = self.df_titles.merge(self.df_genres_movie[['iid', 'col_idx']], on='iid', how='right')
        self.df_genres_movie = self.df_genres_movie[['iid', 'title', 'col_idx']]
        # row_idx - col_idx - value
        self.df_genres_triplet = self.df_genres[['row_idx', 'col_idx', 'value']]


    def load_data(self):

        self.calibrate_genres()
        self.remove_genres(threshold=0.01)
        self.calibrate_movies()
        self.set_ratings_data()
        self.set_genres_data()
        self.set_data()

        super().set_data()

        # genres
        genre_order = self.df_genres_genre['row_idx'].values.astype(int)
        genre_idmap = self.df_genres_genre['row_idx'].values.astype(int)
        genre_alias = self.df_genres_genre['genre'].values.astype(str)
        genre_info = [genre_order, genre_idmap, genre_alias]

        movie_order = self.df_genres_movie['col_idx'].values.astype(int)
        movie_idmap = self.df_genres_movie['iid'].values.astype(int)
        movie_alias = self.df_genres_movie['title'].values.astype(str)
        movie_info = [movie_order, movie_idmap, movie_alias]

        rows = self.df_genres_triplet['row_idx'].values.astype(int)
        cols = self.df_genres_triplet['col_idx'].values.astype(int)
        values = self.df_genres_triplet['value'].values.astype(int)
        values = binarize(values, threshold=0.5)

        user_info, _ = self.factor_info

        X = self.X
        Y = csr_matrix((values, (rows, cols)))
        
        self.Xs = [X, Y]
        self.factors = [[0, 1], [2, 1]]
        self.factor_info = [user_info, movie_info, genre_info]