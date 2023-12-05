import numpy as np
from .BaseBooleanMatrix import BaseBooleanMatrix


class DiagBooleanMatrix(BaseBooleanMatrix):
    """Block diagonal Boolean matrix 

    This generation procedure produces factor matrices U and V with C1P (contiguous-1 property)
    The factors form a block diagonal matrix with overlap configuration (when overlap < 0, there's no overlap)
    The matrix is sorted by nature upon generation
    """
    def __init__(self, m=None, n=None, k=None, overlap=None):
        super().__init__()
        self.check_params(m=m, n=n, k=k, overlap=overlap)

    def generate(self, seed=None):
        self.check_params(seed=seed)
        self.generate_factors()
        self.boolean_matmul()
        self.sorted_index()
        self.set_factor_info()
        self.to_sparse()

    def generate_factors(self):
        self.U = self.generate_factor(self.m, self.k, overlap=self.overlap[0], span=self.overlap[1])
        self.V = self.generate_factor(self.n, self.k, overlap=self.overlap[2], span=self.overlap[3])

    def generate_factor(self, n, k, overlap, span):
        # trials for a point sequence with proper intervals
        min_gap = np.ceil(n / k / 2)
        while True:
            points_all = self.rng.randint(1, n, size=k-1)
            points_all = np.sort(points_all)
            points_all = np.append(0, points_all)
            points_all = np.append(points_all, n)
            points_gap = np.diff(points_all)
            if all([g >= min_gap for g in points_gap]):
                break
            
        points_start = np.copy(points_all[:-1])
        points_end = np.copy(points_all[1:])
        
        # adjust end points
        for i in range(k-1):
            # overlapped or seperated
            gap = points_gap[i+1] if overlap >= 0 else points_gap[i]
            a = np.floor(points_end[i] + gap * (overlap - span))
            b = np.ceil(points_end[i] + gap * (overlap + span))
            b = b + 1 if a == b else b
            points_end[i] = self.rng.randint(low=a, high=b)
        
        # build the C1P factor matrix
        X = np.zeros([n, k])
        for c in range(k):
            X[points_start[c]:points_end[c], c] = 1
        return X