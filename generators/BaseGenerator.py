import numpy as np
import time
from utils import matmul, shuffle_by_dim, add_noise, to_sparse, reverse_index, to_dense, show_matrix


class BaseGenerator:
    def __init__(self):
        """Base Boolean matrix class

        X = U * V

        X, U, V: np.ndarray | sparse.spmatrix
            X is an m-by-n data matrix.
            U is an m-by-k factor matrix.
            V is a n-by-k factor matrix.

        factor_info: list, [U_info, V_info]
        """
        self.X = None
        self.U = None
        self.V = None
        self.factor_info = None


    def check_params(self, **kwargs):
        """All tunable parameters
        
        Parameters
        ----------
        m : int
            Number of rows in X
        n : int, optional
            Number of columns in X
        k : int, optional
            Rank
        noise : float, list(float) -> np.array
            Accept a value between [0, 1] or a 2-element list
            When it's a list, it represents probabilities for false
            negative (p_pos) and false positive (p_neg)
            Typically, noise lies between [0.0, 0.25]
        density : float, list(float) -> np.array
            Accept a value between [0, 1] or a 2-element list
            When it's a list, it represents densities for factor U
            (density of columns of X) and factor V (density of rows of X)
            Typically, density lies between [0.1, 0.3]
        overlap : float, list(float) -> np.array
            Accept a value between [0, 1], a 2-element list or a 4-element list
            When it's a list, it represents overlap ratio for factor 
            U (overlap among columns) and factor V (overlap among rows)
            Randomness is introduced by defining span
            A complete setting of overlap should hold the format (overlap_u, span_u, overlap_v, span_v)
        overlap_flag : bool
            Whether overlap is allowed or not
        size_range : float, list(float) -> np.array
            Accept a value, a 2-element list or a 4-element list
            When it's a list, it represents the lower and upper bounds of factor rectangle size 
            (height_low, height_high, width_low, width_high) or just upper bounds (height_high, width_high)
            The real size limit is the bounds times size m, n divided by k, e.g., [0.2, 2.0] * 1000 / 5
            Depending on the selection of k, a typical upper size limit is 2.0
            A complete setting of overlap should hold the format (height_low, height_high, width_low, width_high)
        seed : int
            Decide the current state of self.rng
        """
        # check dimensions
        if "m" in kwargs:
            self.m = kwargs.get("m")
            print("[I] m            :", self.m)
        if "n" in kwargs:
            self.n = kwargs.get("n")
            print("[I] n            :", self.n)
        if "k" in kwargs:
            self.k = kwargs.get("k")
            print("[I] k            :", self.k)

        # check noise
        if "noise" in kwargs:
            noise = kwargs.get("noise")
            if noise is None:
                noise = [0.0, 0.0] # no noise
            elif isinstance(noise, (int, float)):
                noise = [noise, noise] # p_pos and p_neg
            self.noise = np.array(noise)
            print("[I] noise        :", self.noise)

        # check density
        if "density" in kwargs:
            density = kwargs.get("density")
            if density is None:
                density = [0.2, 0.2]
            elif isinstance(density, (int, float)):
                density = [density, density] # density_u and density_v
            self.density = np.array(density)
            print("[I] density      :", self.density)

        # check overlap
        if "overlap" in kwargs:
            overlap = kwargs.get("overlap")
            if overlap is None:
                overlap = [0.0, 0.0, 0.0, 0.0] # no overlap
            elif isinstance(overlap, list) and len(overlap) == 4:
                pass
            else:
                print("[W] overlap should hold the format (overlap_u, span_u, overlap_v, span_v)")
            
            # check overlap and span
            overlap_u_ok = abs(overlap[0]) + overlap[1] <= 1.0 and abs(overlap[0]) - overlap[1] >= 0.0
            overlap_v_ok = abs(overlap[2]) + overlap[3] <= 1.0 and abs(overlap[2]) - overlap[3] >= 0.0
            if overlap_u_ok and overlap_v_ok:
                self.overlap = np.array(overlap)
                print("[I] overlap      :", self.overlap)
            else:
                print("[W] overlap_u and overlap_v should be in [-1, 1]")
                print("[W] span_u and span_v should not make the sum with abs(overlap_*) exceed [0, 1]")

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

        # check seed
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'): # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            elif seed is not None: # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            else: # self.rng remains unchanged
                pass


    def generate(self):
        raise NotImplementedError("Missing generate method.")
    

    def generate_factors(self):
        raise NotImplementedError("Missing generate_factors method.")
    

    def generate_factor(self):
        raise NotImplementedError("Missing generate_factor method.")


    def measure(self):
        """Measure a matrix

        true_density
            percentage on the number of 1's
        true_overlap
            percentage on the number of overlapped 1's
        """
        self.measured_density = self.measure_density()
        self.measured_overlap = self.measure_overlap()
        print("[I] Density of X :", self.measured_density)
        print("[I] Overlap of X :", self.measured_overlap)
        return (self.measured_density, self.measured_overlap)
    
    
    def measure_density(self):
        return np.sum(self.X) / (self.m * self.n)
    
    
    def measure_overlap(self):
        return np.sum(matmul(self.U, self.V.T, boolean=True) > 1) / (self.m * self.n)

        
    def shuffle(self, seed=None):
        """Shuffle a matrix together with its factors
        """        
        self.check_params(seed=seed)
        self.U_order, self.U, self.rng = shuffle_by_dim(X=self.U, dim=0, rng=self.rng)
        self.V_order, self.V, self.rng = shuffle_by_dim(X=self.V, dim=0, rng=self.rng)
        self.X = matmul(self.U, self.V.T, boolean=True)
        

    def shuffle_factors(self, seed=None):
        """Shuffle the factors of a matrix to re-arrange the bi-clusters
        """
        self.check_params(seed=seed)
        _, self.U, self.rng = shuffle_by_dim(X=self.U, dim=1, rng=self.rng)
        _, self.V, self.rng = shuffle_by_dim(X=self.V, dim=1, rng=self.rng)
        self.X = matmul(self.U, self.V.T, boolean=True)


    def sortout(self, method=None):
        """Sort out a matrix
        """
        pass


    def sorted_index(self):
        """Make index sorted for a sorted matrix
        """
        self.U_order = np.array([i for i in range(self.m)])
        self.V_order = np.array([i for i in range(self.n)])


    def set_factor_info(self):
        """Set factor_info
        """
        U_info = (self.U_order, self.U_order, self.U_order.astype(str))
        V_info = (self.V_order, self.V_order, self.V_order.astype(str))
        self.factor_info = [U_info, V_info]
        

    def add_noise(self, noise=None, seed=None):
        self.check_params(noise=noise, seed=seed)
        self.X, self.rng = add_noise(X=self.X, noise=self.noise, rng=self.rng)

    
    def boolean_matmul(self):
        self.X = matmul(self.U, self.V.T, boolean=True)


    def to_sparse(self, type='csr'):
        '''Convert U, V, X to sparse matrices
        '''
        self.U = to_sparse(self.U, type=type)
        self.V = to_sparse(self.V, type=type)
        self.X = to_sparse(self.X, type=type)


    def to_dense(self):
        '''Convert U, V, X to dense matrices
        '''
        self.U = to_dense(self.U)
        self.V = to_dense(self.V)
        self.X = to_dense(self.X)


    def show_matrix(self, scaling=1.0, pixels=5, title=None, colorbar=False):
        U_inv = reverse_index(idx=self.U_order)
        V_inv = reverse_index(idx=self.V_order)
        U, V = self.U[U_inv], self.V[V_inv]
        X = self.X[U_inv, :]
        X = X[:, V_inv]
        U, V, X = to_dense(U), to_dense(V), to_dense(X)
        settings = [(U, [0, 1], "U"), (V.T, [1, 0], "V"), (X, [0, 0], "X")]
        show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)