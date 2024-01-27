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
        self.X = None
        self.factor_info = None
