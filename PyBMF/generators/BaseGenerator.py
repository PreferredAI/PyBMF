import numpy as np
import time
from ..utils import matmul, shuffle_by_dim, add_noise, to_sparse, reverse_index, to_dense, show_matrix, isnum


class BaseGenerator:
    '''Base class of Boolean matrix generator.

    .. note::

        Attributes of ``BaseGenerator``.

        X : ndarray, spmatrix
            A ``m``-by-``n`` data matrix. :math:`X = U * V`
        U : ndarray, spmatrix
            A ``m``-by-``k`` factor matrix.
        V : ndarray, spmatrix
            A ``n``-by-``k`` factor matrix.
        factor_info: list
            [``U_info``, ``V_info``]
    '''
    def __init__(self):
        self.X = None
        self.U = None
        self.V = None
        self.factor_info = None


    def check_params(self, **kwargs):
        '''Check parameters.
        
        Parameters
        ----------
        overlap_flag : bool
            Whether overlap is allowed or not.
        size_range : list of 2 or 4 floats
            The lower and upper bounds of factor rectangle size (height_low, height_high, width_low, width_high), or just upper bounds (height_high, width_high).

            The real size limit is the bounds times size m, n divided by k.

            E.g., if `k` = 5 and the image height `m` = 1000, the lower and upper bounds are [0.2, 2.0] * 1000 / 5.
        seed : int
            Random seed.
        '''
        self.set_params(**kwargs)
        self.set_config(**kwargs)

        # check overlap_flag
        if "overlap_flag" in kwargs:
            overlap_flag = kwargs.get("overlap_flag")
            if overlap_flag is None:
                overlap_flag = False # no overlap
            self.overlap_flag = overlap_flag
            print("[I] overlap_flag :", self.overlap_flag)

        # check size_range
        if "size_range" in kwargs:
            size_range = kwargs.get("size_range")
            if size_range is None:
                size_range = [0.2, 2.0, 0.2, 2.0] # height_low, height_high, width_low, width_high
            elif isinstance(size_range, list) and len(size_range) == 4:
                pass
            else:
                print("[W] size_range should hold the format (height_low, height_high, width_low, width_high)")

            # check hight and width bounds
            size_range_u_ok = size_range[1] > size_range[0]
            size_range_v_ok = size_range[3] > size_range[2]
            if size_range_u_ok and size_range_v_ok:
                self.size_range = np.array(size_range)
                print("[I] size_range   :", self.size_range)
            else:
                print("[W] Upper bounds should be higher than lower bounds")

    def set_params(self, **kwargs):
        kwconfigs = ['seed']
        for param in kwargs:
            if param in kwconfigs:
                continue

            value = kwargs.get(param)
            setattr(self, param, value)

            # display
            if isinstance(value, list):
                value = len(value)

            print("[I] {:<12} : {}".format(param, value))


    def set_config(self, **kwargs):
        # check seed
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'):
                # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            elif seed is not None:
                # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            else:
                # self.rng remains unchanged
                pass


    def generate(self):
        '''Generate a matrix.
        '''
        raise NotImplementedError("Missing generate method.")
    

    def generate_factors(self):
        '''Generate factors.
        '''
        raise NotImplementedError("Missing generate_factors method.")
    

    def generate_factor(self):
        '''Generate a factor.
        '''
        raise NotImplementedError("Missing generate_factor method.")


    def measure(self):
        '''Measure a matrix.

        Returns
        -------
        measured_density
            Percentage on the number of 1's.
        measured_overlap
            Percentage on the number of overlapped 1's.
        '''
        self.measured_density = self.measure_density()
        self.measured_overlap = self.measure_overlap()
        print("[I] Density of X :", self.measured_density)
        print("[I] Overlap of X :", self.measured_overlap)
        return self.measured_density, self.measured_overlap
    
    
    def measure_density(self):
        '''Measure the density of a matrix.
        '''
        return np.sum(self.X) / (self.m * self.n)
    
    
    def measure_overlap(self):
        '''Measure the overlap of a matrix.
        '''
        return np.sum(matmul(self.U, self.V.T, boolean=True) > 1) / (self.m * self.n)

        
    def shuffle(self, seed=None):
        '''Shuffle a matrix together with its factors.

        Parameters
        ----------
        seed : int
            Random seed.
        '''
        self.check_params(seed=seed)
        self.U_order, self.U, self.rng = shuffle_by_dim(X=self.U, dim=0, rng=self.rng)
        self.V_order, self.V, self.rng = shuffle_by_dim(X=self.V, dim=0, rng=self.rng)
        self.X = matmul(self.U, self.V.T, boolean=True)
        

    def shuffle_factors(self, seed=None):
        '''Shuffle the factors of a matrix to re-arrange the bi-clusters.
        '''
        self.check_params(seed=seed)
        _, self.U, self.rng = shuffle_by_dim(X=self.U, dim=1, rng=self.rng)
        _, self.V, self.rng = shuffle_by_dim(X=self.V, dim=1, rng=self.rng)
        self.X = matmul(self.U, self.V.T, boolean=True)


    def sortout(self, method=None):
        '''Sort out a matrix.
        '''
        pass


    def sorted_index(self):
        '''Make index sorted for a sorted matrix.
        '''
        self.U_order = np.array([i for i in range(self.m)])
        self.V_order = np.array([i for i in range(self.n)])


    def set_factor_info(self):
        '''Set factor_info.
        '''
        U_info = [self.U_order, self.U_order, self.U_order.astype(str)]
        V_info = [self.V_order, self.V_order, self.V_order.astype(str)]
        self.factor_info = [U_info, V_info]
        

    def add_noise(self, noise=[0.0, 0.0], seed=None):
        '''Add noise to a matrix.
        
        Parameters
        ----------
        noise : list of 2 float in [0, 1]
            Probabilities for false negative (p_pos) and false positive (p_neg).
        seed : optional
            Random seed.
        '''
        X = self.X
        self.check_params(noise=noise, seed=seed)
        self.X, self.rng = add_noise(X=self.X, noise=self.noise, rng=self.rng)
        self.to_sparse() # debug

    
    def boolean_matmul(self):
        '''Boolean matrix multiplication.
        '''
        self.X = matmul(self.U, self.V.T, boolean=True)


    def to_sparse(self, type='csr'):
        '''Convert U, V, X to sparse matrices.
        '''
        self.U = to_sparse(self.U, type=type)
        self.V = to_sparse(self.V, type=type)
        self.X = to_sparse(self.X, type=type)


    def to_dense(self):
        '''Convert U, V, X to dense matrices.
        '''
        self.U = to_dense(self.U)
        self.V = to_dense(self.V)
        self.X = to_dense(self.X)


    def show_matrix(
            self, 
            scaling=1.0, pixels=5, 
            colorbar=True, 
            discrete=True, 
            center=True, 
            clim=[0, 1], 
            keep_nan=True, 
            **kwargs):
        '''The `show_matrix` wrapper for Boolean matrix generators.
        '''
        # U_inv = reverse_index(idx=self.U_order)
        # V_inv = reverse_index(idx=self.V_order)

        # U, V = self.U[U_inv], self.V[V_inv]

        # X = self.X[U_inv, :]
        # X = X[:, V_inv]

        X, U, V = self.X, self.U, self.V

        settings = [(X, [0, 0], "X"), (U, [0, 1], "U"), (V.T, [1, 0], "V")]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, 
                    colorbar=colorbar, discrete=discrete, center=center, clim=clim, keep_nan=keep_nan, **kwargs)