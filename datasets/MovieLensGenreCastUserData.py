from .MovieLensData import MovieLensData
from .MovieLensUserData import MovieLensUserData
from .MovieLensGenreCastData import MovieLensGenreCastData


class MovieLensGenreCastUserData(MovieLensData):
    '''Load MovieLens dataset with user profiles

    size:
        100k
        1m
    '''
    def __init__(self, path=None, size='1m'):
        super().__init__(path=path, size=size)
        self.is_single = False
        self.name = self.name + '_genre_cast_user'
        self.path = path


    def read_data(self):
        pass


    def load_data(self):
        ml_user = MovieLensUserData(path=self.path, size=self.size)
        ml_user.load()

        ml_imdb = MovieLensGenreCastData(path=self.path, size=self.size)
        ml_imdb.load()

        self.Xs = [ml_imdb.Xs[0], ml_user.Xs[1], ml_imdb.Xs[1], ml_imdb.Xs[2]]
        self.factors = [[0, 1], [0, 2], [3, 1], [4, 1]]
        self.factor_info = ml_user.factor_info + ml_imdb.factor_info[2:]
