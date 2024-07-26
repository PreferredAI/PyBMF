import numpy as np
from .BaseGenerator import BaseGenerator
    

class BlockMatrixGenerator(BaseGenerator):
    '''The block Boolean matrix generator.
    
    This generation procedure produces factor matrices U and V with C1P (contiguous-1 property).
    The factors form a arbitrarily placed block matrix with or without overlapping.
    The matrix is sorted by nature upon generation.

    Parameters
    ----------
    m : int
        The number of rows in X.
    n : int, optional
        The number of columns in X.
    k : int, optional
        The rank.
    overlap_flag : bool
        Whether overlap is allowed or not.
    size_range : list of 2 or 4 floats
        The lower and upper bounds of factor rectangle size (height_low, height_high, width_low, width_high), or just upper bounds (height_high, width_high).

        The real size limit is the bounds times size m, n divided by k.

        E.g., if `k` = 5 and the image height `m` = 1000, the lower and upper bounds are [0.2, 2.0] * 1000 / 5.
    '''
    def __init__(self, m, n, k, overlap_flag=None, size_range=None):
        super().__init__()
        self.check_params(m=m, n=n, k=k, overlap_flag=overlap_flag, size_range=size_range)


    def generate(self, seed=None):
        '''Generate a matrix.

        Parameters
        ----------
        seed : int
            Random seed.
        '''
        self.check_params(seed=seed)
        self.generate_factors()
        self.boolean_matmul()
        # self.sorted_index()
        # self.set_factor_info()
        self.to_sparse(type='csr')


    def generate_factors(self):
        '''Generate factors.
        '''
        # trials using two point sequences with proper overlapping
        while True:
            points_start_u, points_end_u = self.generate_factor_points(n=self.m, k=self.k, low=self.size_range[0], high=self.size_range[1])
            points_start_v, points_end_v = self.generate_factor_points(n=self.n, k=self.k, low=self.size_range[2], high=self.size_range[3])

            trials = 0
            if self.check_overlap(self.k, points_start_u, points_end_u, points_start_v, points_end_v) == True:
                trials += 1
                print("[I]   check overlap trials: ", trials)
                self.U = self.generate_factor(n=self.m, k=self.k, points_start=points_start_u, points_end=points_end_u)
                self.V = self.generate_factor(n=self.n, k=self.k, points_start=points_start_v, points_end=points_end_v)
                break # try until a qualified config is found
        

    def generate_factor_points(self, n, k, low, high):
        '''Generate a factor point sequence.

        Parameters
        ----------
        n : int
            The number of points.
        k : int
            The rank.
        low : float
            The lower bound of the interval.
        high : float
            The upper bound of the interval.

        Returns
        -------
        points_start : list of ints
            The start points.
        points_end : list of ints
            The end points.
        '''
        # trials for a point sequence with proper intervals and do not exceed that dimension
        avg = n / k # average size
        points_end = np.zeros(k)
        trials = 0
        while True:
            trials += 1
            print("[I]   generate factor trials: ", trials)
            points_start = self.rng.randint(0, n, size=k)
            for i in range(k):
                a = np.floor(points_start[i] + avg * low)
                b = np.ceil(points_start[i] + avg * high)
                b = b + 1 if a == b else b
                points_end[i] = self.rng.randint(low=a, high=b)
            if all([e <= n for e in points_end]):
                return points_start.astype(int), points_end.astype(int)
        

    def check_overlap(self, k, points_start_u, points_end_u, points_start_v, points_end_v):
        '''Check overlap.

        Parameters
        ----------
        k : int
            The rank.
        points_start_u : list of ints
            The start points of U.
        points_end_u : list of ints
            The end points of U.
        points_start_v : list of ints
            The start points of V.
        points_end_v : list of ints
            The end points of V.

        Returns
        -------
        is_overlapped : bool
        '''
        # build a list of rectangles
        rectangles = [[points_start_u[i], points_end_u[i], points_start_v[i], points_end_v[i]] for i in range(k)]
        for i in range(k):
            for j in range(k):
                if j == i:
                    continue
                is_overlapped = self.is_overlapped(A=rectangles[i], B=rectangles[j])
                if is_overlapped == 2:
                    return False # fully overlap (included) is not allowed
                if is_overlapped == 1 and self.overlap_flag == False:
                    return False # partial overlap is not allowed if overlap_flag is False
        return True
    

    def is_overlapped(self, A, B):
        '''Check if two rectangles are overlapped.

        Parameters
        ----------
        A : list of 4 ints
            The first rectangle.
        B : list of 4 ints
            The second rectangle.

        Returns
        -------
        is_overlapped : int
        '''
        u_overlapped = min(A[1], B[1]) > max(A[0], B[0])
        v_overlapped = min(A[3], B[3]) > max(A[2], B[2])
        u_included = A[1] <= B[1] and A[0] >= B[0]
        v_included = A[3] <= B[3] and A[2] >= B[2]
        if u_included and v_included:
            return 2
        if u_overlapped and v_overlapped:
            return 1
        else:
            return 0


    def generate_factor(self, n, k, points_start, points_end):
        '''Generate a factor.
        
        Parameters
        ----------
        n : int
            The number of rows in X.
        k : int
            The rank.
        points_start : list of ints
            The start points of X.
        points_end : list of ints
            The end points of X.
        '''
        # build the C1P factor matrix
        X = np.zeros([n, k])
        for c in range(k):
            X[points_start[c]:points_end[c], c] = 1
        return X
