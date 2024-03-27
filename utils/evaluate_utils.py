from .boolean_utils import multiply, matmul, dot
from .sparse_utils import to_dense, to_triplet, to_sparse
from .metrics import get_metrics
from scipy.sparse import spmatrix, issparse, csr_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import t_imap


def eval(metrics, task, X_gt, X_pd=None, U=None, V=None):
    """Evaluate with given metrics.

    X_gt : array or spmatrix
    X_pd : array or spmatrix, optional
    U : spmatrix, optional
    V : spmatrix, optional
    metrics : list of str
        List of metric names.
    task : str in {'prediction', 'reconstruction'}
        If `task` == 'prediction', it ignores the missing values and only use the triplet from the ``spmatrix``. The triplet may contain zeros, depending on whether negative sampling has been used.
        If `task` == 'reconstruction', it uses the whole matrix, which considers all missing values as zeros in ``spmatrix``.
    """
    using_matrix = X_pd is not None
    using_factors = U is not None and V is not None
    assert using_matrix or using_factors, "[E] User should provide either `U`, `V` or `X_pd`."

    if task == 'prediction':
        U_idx, V_idx, gt_data = to_triplet(X_gt)
        if using_factors and len(gt_data) < 5000: # faster only if the amount of samples is small
            pd_data = np.zeros(len(gt_data), dtype=int)
            for i in tqdm(range(len(gt_data)), leave=False, position=1, desc="[I] Making predictions"):
                pd_data[i] = dot(U[U_idx], V[V_idx], boolean=True)
        else:
            if not using_matrix:
                X_pd = matmul(U=U, V=V.T, sparse=True, boolean=True)
            pd_data = np.zeros(len(gt_data), dtype=float)
            # for i in tqdm(range(len(gt_data)), leave=False, position=1, desc="[I] Making predictions"):
            for i in range(len(gt_data)):
                pd_data[i] = X_pd[U_idx[i], V_idx[i]]

    elif task == 'reconstruction':
        gt_data = to_sparse(X_gt, type='csr')
        if not using_matrix:
            pd_data = matmul(U=U, V=V.T, sparse=True, boolean=True)
        else:
            pd_data = to_sparse(X_pd, type='csr')
    
    results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
    return results


def record(df_dict, df_name, columns, records, verbose=False):
    '''Create and add records to a dataframe in a logs dict.

    Parameters
    ----------
    df_dict : dict
    df_name : str
    columns : list of str or str tuple
    records : list
    verbose : bool, default: False
    caption : str, optional
    '''
    if not df_name in df_dict: # create dataframe in logs dict
        if isinstance(columns[0], tuple): # using multi-level headers
            time = header(['time'], levels=len(columns[0]))
            columns = pd.MultiIndex.from_tuples(time + columns)
        else:
            columns = ['time'] + columns
        df_dict[df_name] = pd.DataFrame(columns=columns)

    ts = [pd.Timestamp.now().strftime("%d/%m/%y %I:%M:%S")]
    records = ts + records # add timestamp
    df_dict[df_name].loc[len(df_dict[df_name].index)] = records # add record

    if verbose: # print the last 5 lines
        display(df_dict[df_name].tail())


def header(names, levels, depth=None):
    '''Create multi-level headers.

    >>> header(['time', 'k', 'score'], levels=3, depth=2)
    >>> [('', 'time', ''), ('', 'k', ''), ('', 'score', '')]
    '''
    if depth is None:
        depth = levels
    output = []
    for name in names:
        list = [''] * levels
        list[depth-1] = name
        output.append(tuple(list))
    return output