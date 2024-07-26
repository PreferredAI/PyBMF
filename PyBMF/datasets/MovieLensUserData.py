import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, hstack
from ..utils import binarize
from .MovieLensData import MovieLensData


class MovieLensUserData(MovieLensData):
    '''Load MovieLens dataset with user profiles

    Parameters
    ----------
    path : str
        Path to the cached dataset.
    size : str in {'100k', '1m'}
        MovieLens dataset size.
    '''
    def __init__(self, path=None, size='1m'):
        super().__init__(path=path, size=size)
        self.is_single = False
        self.name = self.name + '_user'


    def read_data(self):
        '''Read data.
        '''
        # ratings and titles
        super().read_data()

        # profiles
        if self.size == '100k':
            path = os.path.join(self.root, "ml-100k", "u.user")
            sep, engine, names = '|', 'c', ['uid', 'age', 'gender', 'occupation', 'zip']
        elif self.size == '1m':
            path = os.path.join(self.root, "ml-1m", "users.dat")
            sep, engine, names = '::', 'python', ['uid', 'gender', 'age', 'occupation', 'zip']

        self.df_profiles = pd.read_table(path, delimiter=sep, engine=engine, header=None, names=names)

        # occupations
        path = os.path.join(self.root, "ml-100k", "u.occupation")
        self.df_occupations = pd.read_table(path, delimiter='|', header=None, names=['occupation'])

        # preprocessing
        self.df_profiles['gender'] = self.df_profiles['gender'].apply(lambda x: 0 if x == 'F' else 1)
        self.df_profiles['age'] = self.df_profiles['age'].apply(lambda x: int(x / 15))
        self.df_profiles['occupation'] = self.df_profiles['occupation'].apply(lambda x: self.df_occupations[self.df_occupations['occupation'] == x].index[0] if isinstance(x, str) else x)
        
        from uszipcode import SearchEngine
        engine = SearchEngine()
        self.df_profiles['zip'] = self.df_profiles['zip'].apply(lambda x: engine.by_zipcode(x).state if engine.by_zipcode(x) is not None else 'NA')


    def load_data(self):
        '''Load data.
        '''
        super().load_data()
        X = self.X
        user_info, movie_info = self.factor_info

        Y, profile_alias = self.get_user_profile()

        profile_order = np.arange(len(profile_alias))
        profile_idmap = np.arange(len(profile_alias))
        profile_info = [profile_order, profile_idmap, profile_alias]

        self.Xs = [X, Y]
        self.factors = [[0, 1], [0, 2]]
        self.factor_info = [user_info, movie_info, profile_info]


    def get_user_profile(self):
        '''Get user profile.
        '''
        attributes = ['age', 'occupation', 'zip']

        # genger
        attr_list = np.array(['gender'])
        Y = csr_matrix(self.df_profiles['gender'].values).T
        for attr in attributes:
            # attribute values
            attr_vals = sorted(self.df_profiles[attr].unique())
            
            # new sub-matrix
            Z = lil_matrix((Y.shape[0], len(attr_vals)))

            for col, val in enumerate(attr_vals):
                rows = self.df_profiles.index[self.df_profiles[attr] == val]
                Z[rows, col] = 1

            if attr == 'age':
                attr_vals = ['age_' + str(s) for s in attr_vals]
            if attr == 'occupation':
                attr_vals = self.df_occupations['occupation'].values

            Y = hstack((Y, Z), format='csr')
            attr_list = np.append(attr_list, attr_vals)
            
        return Y, attr_list
