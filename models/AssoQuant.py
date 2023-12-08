import numpy as np
from utils import matmul, add, to_sparse, TP, FP
from .BaseModel import BaseModel
from scipy.sparse import issparse, lil_matrix
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool, cpu_count
from .Asso import Asso
from scipy.stats import rankdata


class AssoQuant(Asso):
    '''The quantified ASSO algorithm
    
    Modified from the paper 'The discrete basis problem'.
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)

                
    def build_basis(self):
        '''Get the binary-valued basis candidates

        Note: rankdata doesn't have sparsity support.

        The highest value will be assigned index 1.
        '''
        self.basis = rankdata(-self.assoc.toarray(), method='ordinal', axis=1)
        # self.basis = to_sparse(self.basis)


    # def _fit(self):
    #     for k in range(self.k):
    #         # ver 1: original
    #         # if k == 0:
    #         #     # record the highest cover score of each candidate basis
    #         #     scores = np.zeros(self.n)
    #         # else:
    #         #     # new basis should surpass the cover score of previous factors
    #         #     scores = self.cover() * np.ones(self.n)
    #         # 
    #         # # to store the optimal column for each of the n basis vectors
    #         # col_candidates = csc_matrix((self.m, self.n))
    #         #
    #         # for c in tqdm(range(self.n)):
    #         #     self.V[k, :] = self.basis[c]
    #         #     self.U[:, k] = 0 # reset current column in U
    #         #     for r in range(self.m):
    #         #         self.U[r, k] = 1
    #         #         score = self.cover()
    #         #         if score > scores[c]:
    #         #             col_candidates[r, c] = 1 # act the same as current column in U
    #         #             scores[c] = score
    #         #         else:
    #         #             self.U[r, k] = 0 # reset current cell in U
    #         # 
    #         # c = np.argmax(scores) # get the id of the basis with the highest score
    #         # self.V[k, :] = self.basis[c] # load the best basis
    #         # self.U[:, k] = col_candidates[:, c] # load the corresponding column
    #         # self.basis[c] = 0 # remove this basis
    #         # 
    #         # self.show_matrix(title="tau: {}, w: {}, step: {}".format(self.tau, self.w, k+1))
        

    #         # ver 2: vectorized
    #         best_basis = lil_matrix(np.zeros((1, self.n)))
    #         best_column = lil_matrix(np.zeros((self.m, 1)))

    #         best_score = 0 if k == 0 else best_score
    #         for c in tqdm(range(self.n)):
    #             score, column = self._get_col_candidate(c, return_col=True)
    #             if score > best_score:
    #                 best_score = score
    #                 best_basis = self.basis[c]
    #                 best_column = column

    #         self.V[k, :] = best_basis
    #         self.U[:, k] = best_column
    #         self.basis[c] = 0 # remove this basis

    #         # self.show_matrix(title="tau: {}, w: {}, step: {}".format(self.tau, self.w, k+1))


    #         # ver 3: parallel
    #         # best_basis = lil_matrix(np.zeros((1, self.n)))
    #         # best_column = lil_matrix(np.zeros((self.m, 1)))

    #         # best_score = 0 if k == 0 else best_score
    #         # with Pool(cpu_count()) as p:
    #         #     scores = list(tqdm(p.imap(self._get_col_candidate, range(self.n)), total=self.n))
    #         #     # scores = p.map(self._get_col_candidate, np.arange(self.n))

    #         # if max(scores) > best_score:
    #         #     best_score = max(scores)
    #         #     c = scores.index(best_score)
    #         #     best_basis = self.basis[c]
    #         #     _, best_column = self._get_col_candidate(c, return_col=True)

    #         # self.V[k, :] = best_basis
    #         # self.U[:, k] = best_column
    #         # self.basis[c] = 0 # remove this basis

    #         # self.show_matrix(title="tau: {}, w: {}, step: {}".format(self.tau, self.w, k+1))


    # def _get_col_candidate(self, c, return_col=False):
    #     before = matmul(self.U, self.V, sparse=True, boolean=True)
    #     V = self.basis[c]
    #     U = lil_matrix(np.ones([self.m, 1]))
    #     after = matmul(U, V, sparse=True, boolean=True)
    #     after = add(before, after)

    #     before_cover = self.cover(Y=before, axis=1)
    #     after_cover = self.cover(Y=after, axis=1)
    #     col_candidate = (after_cover > before_cover) * 1

    #     U = lil_matrix(col_candidate).transpose()

    #     after = matmul(U, V, sparse=True, boolean=True)
    #     after = add(before, after)
    #     cover = self.cover(Y=after)

    #     if return_col:
    #         return cover, U
    #     else:
    #         return cover
    
        
    # def cover(self, X=None, Y=None, w=None, axis=None) -> Union[float, np.ndarray]:
    #     '''Measure the coverage for X using Y
    #     '''
    #     if X is None:
    #         X = self.X_train
    #     if Y is None:
    #         Y = matmul(self.U, self.V, sparse=True, boolean=True)
    #     covered = TP(X, Y, axis=axis)
    #     overcovered = FP(X, Y, axis=axis)
    #     w = self.w if w is None else w
    #     return w[0] * covered - w[1] * overcovered
    