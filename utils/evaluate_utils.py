from .boolean_utils import multiply, matmul, dot
from .sparse_utils import to_dense, to_triplet, to_sparse
from scipy.sparse import spmatrix, issparse, csr_matrix
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from tqdm import tqdm


def get_metrics(gt: Union[np.ndarray, spmatrix], pd: Union[np.ndarray, spmatrix], metrics: List[str], axis=None):
    """Get results of the metrics all at once.

    Metrics from sklearn.metrics are included as sanity check. Their input must be binary ``array``, which makes them slow and less flexible.

    Parameters
    ----------
    gt : array, spmatrix
        Ground truth, can be 1d array, 2d dense or sparse matrix.
    pd : array, spmatrix
        Prediction, can be 1d array, 2d dense or sparse matrix.
        When the input are matrices, row and column-wise measurement can be conducted by defining `axis`.
    metrics : list of str
        The name of metrics.
    axis : int in {0, 1}
        When `axis` == 0, The `result` containing the column-wise measurement has the same length as columns.

    Returns
    -------
    results : list
    """
    if np.isnan(to_dense(pd, squeeze=True)).any():
        raise TypeError("NaN is found in prediction.")

    functions = {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR,
        'PPV': PPV, 'ACC': ACC, 'ERR': ERR, 'F1': F1,
        'Recall': TPR, 'Precision': PPV, 'Accuracy': ACC, 'Error': ERR, # alias
    }
    sklearn_metrics = { 
        'recall_score': recall_score, 'precision_score': precision_score, 
        'accuracy_score': accuracy_score, 'f1_score': f1_score,
    }
    results = []
    for m in metrics:
        if m in functions:
            results.append(functions[m](gt, pd, axis))
        elif m in sklearn_metrics: # must be binary arrays
            gt = to_dense(gt).flatten()
            pd = to_dense(pd).flatten()
            results.append(sklearn_metrics[m](gt, pd))
        else:
            results.append(None)
    return results


def TP(gt, pd, axis=None):
    s = multiply(gt, pd, boolean=True).sum(axis=axis)
    return np.array(s).squeeze()


def FP(gt, pd, axis=None):
    diff = pd - gt
    if issparse(gt):
        s = diff.maximum(0).sum(axis=axis)
        return np.array(s).squeeze()
    else:
        s = np.maximum(diff, 0).sum(axis=axis)
        return s


def TN(gt, pd, axis=None):
    return TP(gt=invert(gt), pd=invert(pd), axis=axis)


def FN(gt, pd, axis=None):
    return FP(gt=pd, pd=gt, axis=axis)


def TPR(gt, pd, axis=None):
    """sensitivity, recall, hit rate, or true positive rate
    """
    denom = gt.sum(axis=axis)
    return TP(gt, pd, axis=axis) / denom if denom > 0 else 0


def TNR(gt, pd, axis=None):
    """specificity, selectivity or true negative rate
    """
    denom = invert(gt).sum(axis=axis)
    return TN(gt, pd, axis=axis) / denom if denom > 0 else 0


def FPR(gt, pd, axis=None):
    """fall-out or false positive rate
    """
    return 1 - TNR(gt, pd, axis=axis)


def FNR(gt, pd, axis=None):
    """miss rate or false negative rate
    """
    return 1 - TPR(gt, pd, axis=axis)


def PPV(gt, pd, axis=None):
    """precision or positive predictive value
    """
    denom = pd.sum(axis=axis)
    return TP(gt, pd, axis=axis) / denom if denom > 0 else 0


def ACC(gt, pd, axis=None):
    """accuracy
    """
    if len(pd.shape) == 2:
        n = pd.shape[0] * pd.shape[1] if axis is None else pd.shape[axis]
    else:
        n = len(pd)
    return (TP(gt, pd, axis) + TN(gt, pd, axis)) / n


def ERR(gt, pd, axis=None):
    """error
    """
    return 1 - ACC(gt, pd, axis)


def F1(gt, pd, axis=None):
    """F1 score

    tp = TP(gt, pd, axis)
    fp = FP(gt, pd, axis)
    fn = FN(gt, pd, axis)
    return 2 * tp / (2 * tp + fp + fn)
    """
    precision = PPV(gt, pd, axis)
    recall = TPR(gt, pd, axis)
    return 2 * precision * recall / (precision + recall)


def cover(gt, pd, w, axis=None):
    '''Measure the coverage of X using Y.

    Parameters
    ----------
    w : float in [0, 1], optional
        The weights [1 - `w`, `w`] are the reward for coverage and the penalty for over-coverage. It can also be considered as the lower-bound of true positive ratio when `cover` is used as a factorization criteria.
    axis : int in {0, 1}, default: None
        The dimension of the basis.
        When `axis` is None, return the overall coverage score. When `axis` is 0, the basis is at dimension 0, thus return the column-wise coverage scores.

    Returns
    -------
    score : float, array
        The overall or the column/row-wise coverage score.
    '''
    covered = TP(gt, pd, axis=axis)
    overcovered = FP(gt, pd, axis=axis)
    score = (1 - w) * covered - w * overcovered
    return score


def invert(X):
    if issparse(X):
        X = csr_matrix(np.ones(X.shape)) - X
    elif isinstance(X, np.ndarray):
        X = 1 - X
    else:
        raise TypeError
    return X


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
        if using_factors and len(gt_data) < 5000: # faster only for a small amount of samples
            pd_data = np.zeros(len(gt_data), dtype=int)
            for i in tqdm(range(len(gt_data)), leave=False, position=0, desc="[I] Making predictions"):
                pd_data[i] = dot(U[U_idx], V[V_idx], boolean=True)
        else:
            if not using_matrix:
                X_pd = matmul(U=U, V=V.T, sparse=True, boolean=True)
            pd_data = np.zeros(len(gt_data), dtype=int)
            for i in tqdm(range(len(gt_data)), leave=False, position=0, desc="[I] Making predictions"):
                pd_data[i] = X_pd[U_idx[i], V_idx[i]]
    elif task == 'reconstruction':
        gt_data = to_sparse(X_gt, type='csr')
        if not using_matrix:
            pd_data = matmul(U=U, V=V.T, sparse=True, boolean=True)
        else:
            pd_data = to_sparse(X_pd, type='csr')
        
    results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
    return results


def record(df_dict, df_name, columns, records, verbose=False, caption=None):
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
    if not df_name in df_dict: # create a dataframe in a logs dict
        if isinstance(columns[0], tuple): # using multi-level headers
            time = header('time', levels=len(columns[0]))
            columns = pd.MultiIndex.from_tuples(time + columns)
        else:
            columns = ['time'] + columns
        df_dict[df_name] = pd.DataFrame(columns=columns)

    records = [pd.Timestamp.now().strftime("%d/%m/%y %I:%M:%S")] + records # add timestamp
    df_dict[df_name].loc[len(df_dict[df_name].index)] = records # add a line of records

    if verbose: # print the last 5 lines
        if caption is None:
            display(df_dict[df_name].tail())
        else:
            styles = [dict(selector="caption", props=[("font-size", "100%"), ("font-weight", "bold")])]
            display(df_dict[df_name].tail().style.set_caption(caption).set_table_styles(styles))


def header(name, levels=2, depth=None):
    if isinstance(name, str):
        name = [name]
    if depth is None:
        depth = levels
    output = []
    for n in name:
        list = [''] * levels
        list[depth-1] = n
        output.append(tuple(list))
    return output