from .ASSO import ASSO
import numpy as np
from multiprocessing import Pool
import time
from utils import boolean_matmul, generate_candidates


class ASSOOPT(ASSO):
    '''ASSOOPT algorithm using exhaustive search
    '''
    def __init__(self, X, k, tau, w=[0.5, 0.5], U_idx=None, V_idx=None, display_flag=False):
        super().__init__(X=X, k=k, tau=tau, w=w, U_idx=U_idx, V_idx=V_idx, display_flag=display_flag)

    
    def solve(self):
        super().solve()
        self.exhaustive_search()


    def exhaustive_search(self):
        '''Using exhaustive search to refine U
        '''
        self.Uis = generate_candidates(bits=self.n, dim=1) # all candidates for a row in U
        self.Xis = boolean_matmul(self.Uis, self.V) # corresponding rows in X

        start_time = time.perf_counter()
        with Pool() as pool:
            result = pool.map(self.find_optimal_Ui, range(self.m))
        finish_time = time.perf_counter()
        print("[I] Exhaustive search finished in {} seconds with parallelism.".format(finish_time-start_time))

        # debug
        # print(result)
        self.U = self.Uis[result] # refine U


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
