from .Asso import Asso
import numpy as np
import time
from utils import matmul, ERR
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool, cpu_count


class AssoOpt(Asso):
    '''The Asso algorithm using exhaustive search
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)

    
    def fit(self, train_set, val_set=None, display=False):
        super().fit(train_set=train_set, val_set=val_set, display=display)
        self.exhaustive_search()


    def exhaustive_search(self):
        '''Using exhaustive search to refine U
        '''
        self.Uis = self.generate_candidates(bits=self.n, dim=1) # all candidates for a row in U
        self.Xis = matmul(U=self.Uis, V=self.V, sparse=True, boolean=True) # corresponding rows in X

        start_time = time.perf_counter()
        with Pool() as pool:
            result = pool.map(self.find_optimal_Ui, range(self.m))
        finish_time = time.perf_counter()
        print("[I] Exhaustive search finished in {} seconds with parallelism.".format(finish_time-start_time))

        # debug
        # print(result)
        self.U = self.Uis[result] # refine U


    def get_row_candidate(self, bits, dim):
        '''Return all possible choices for a column (dim=0) or a row (dim=1) in a matrix

        E.g.

        When dim == 0 and m = 2, the output will be
            0 0 1 1
            0 1 0 1

        When dim == 1 and n = 2, the output will be
            0 0
            0 1
            1 0
            1 1
        '''
        # rows, cols = bits, 2 ** bits
        # candidates = np.zeros((rows, cols), dtype=int)
        # for i in range(cols):
        #     binary = bin(i)[2:].zfill(rows)  # convert i to binary representation
        #     for j in range(rows):
        #         candidates[i, j] = int(binary[j])  # fill each cell with binary digit
        # return candidates if dim == 0 else candidates.T
        bits = self.m if dim == 0 else self.n
        rows, cols = bits, 2 ** bits
        candidates = lil_matrix(shape=(rows, cols), dtype=np.uint8)
        for i in range(cols):
            binary = bin(i)[2:].zfill(rows)  # convert i to binary representation
            for j in range(rows):
                candidates[i, j] = np.uint8(binary[j])  # fill each cell with binary digit
        return candidates if dim == 0 else candidates.T


    def find_optimal_Ui(self, i):
        trials = 2 ** self.k
        scores = np.zeros(trials)
        current_row_in_X = self.X[i].astype(int)
        for t in range(trials):
            scores[t] = self.vec_cover(current_row_in_X, self.Xis[t])
        return np.argmax(scores)


    def vec_cover(self, x, y):
        '''Vector version for cover function
        x: ground truth
        y: reconstruction
        '''
        covered = np.sum(np.bitwise_and(x, y))
        overcovered = np.sum(np.maximum(y-x, 0))
        return self.w[0] * covered - self.w[1] * overcovered
    

    # def boolean_vecmatmul(vec, mat):
    #     '''Boolean multiplication between a length-k vector and a k-by-n matrix
    #     '''
    #     n = mat.shape[1]
    #     x = [np.any(np.bitwise_and(vec, mat[:, c])) * 1 for c in range(n)]
    #     x = np.array(x)
    #     return x
    

    def error(self, X=None, Y=None, axis=None) -> Union[float, np.ndarray]:
        '''Measure the error between X and Y
        '''
        if X is None:
            X = self.X_train
        if Y is None:
            Y = matmul(self.U, self.V, sparse=True, boolean=True)
        error = 