from .ContinuousModel import ContinuousModel
from .NMFSklearn import NMFSklearn
import numpy as np
from utils import binarize, matmul, to_dense, to_sparse
from scipy.sparse import csr_matrix


class BinaryMF(ContinuousModel):
    '''Binary MF template class.

    Instantiate BinaryMFPenalty or BinaryMFThreshold instead.
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    





