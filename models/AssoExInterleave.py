import numpy as np
from utils import matmul, add, to_sparse
from .Asso import Asso
from scipy.sparse import lil_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import pickle


class AssoExInterleave(Asso):
    '''The Asso algorithm with iterative update among the factors (experimental)
    '''
    pass