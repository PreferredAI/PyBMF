from .MovieLensData import MovieLensData
from .MovieLensUserData import MovieLensUserData
from .MovieLensGenreCastData import MovieLensGenreCastData


class MovieLensGenreCastUserData(MovieLensData):
    '''Load MovieLens dataset with user profiles.

    Parameters
    ----------
    path : str
        Path to the cached dataset.
    size : str in {'100k', '1m'}
        MovieLens dataset size.
    '''
    def __init__(self, size='1m'):
        super().__init__(size=size)
        self.is_single = False
        self.name = self.name + '_genre_cast_user'


    def read_data(self):
        '''Read data.
        '''
        pass


    def load_data(self):
        '''Load data.
        '''
        ml_user = MovieLensUserData(size=self.size)
        ml_user.load()

        ml_imdb = MovieLensGenreCastData(size=self.size)
        ml_imdb.load()

        self.Xs = [ml_imdb.Xs[0], ml_user.Xs[1], ml_imdb.Xs[1], ml_imdb.Xs[2]]
        self.factors = [[0, 1], [0, 2], [3, 1], [4, 1]]
        self.factor_info = ml_user.factor_info + ml_imdb.factor_info[2:]
