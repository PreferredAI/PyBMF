from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, step, sigmoid
import numpy as np
from tqdm import tqdm


class BinaryMFThresholdExSimple(BinaryMFThreshold):
    '''Binary matrix factorization, simple Thresholding algorithm (experimental)
     
    Grid search over [u, v] given interval, or u_grid and v_grid.

    u_grid, v_grid : list or 1-d array
    '''
    def __init__(self, k, interval=None, u_grid=None, v_grid=None):
        self.check_params(k=k, interval=interval, u_grid=u_grid, v_grid=v_grid, algorithm='threshold')


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        # check interval
        if 'interval' in kwargs:
            self.interval = kwargs.get('interval')
            if self.interval is None:
                self.interval = 0.01 # default grid search interval
            print("[I] interval     :", self.interval)
        # check grid
        if 'u_grid' in kwargs and 'v_grid' in kwargs:
            u_grid, v_grid = kwargs.get('u_grid'), kwargs.get('v_grid')
            if u_grid is not None and v_grid is not None:
                self.u_grid, self.v_grid = u_grid, v_grid
                print("[I] using external grid")
            else:
                self.u_grid, self.v_grid = None, None

        
    def _fit(self):
        self.initialize()
        self.normalize()
        self.threshold_algorithm()


    def threshold_algorithm(self):
        '''A gradient descent method minimizing F(u, v), or 'F(w, h)' in the paper.
        '''
        if self.u_grid is not None and self.v_grid is not None:
            self.u_grid = np.arange(start=0, step=self.interval, stop=self.U.max())
            self.v_grid = np.arange(start=0, step=self.interval, stop=self.V.max())
        grid = np.array(np.meshgrid(self.u_grid, self.v_grid)).reshape(2, -1)

        best_F = self.F([0, 0])
        for u, v in tqdm(zip(grid[0], grid[1]), total=len(self.u_grid)*len(self.v_grid)):
            F = self.F([u, v])
            if F < best_F:
                best_F, self.u, self.v = F, u, v

        print("[I] u = {}, v = {}, min(F) = {}".format(self.u, self.v, best_F))


    def F(self, params):
        '''Same F, but with step function rather than sigmoid

        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        U = step(self.U, u)
        V = step(self.V, v)

        diff = self.X_train - U @ V.T
        F = 0.5 * np.sum(diff ** 2)       
        return F
