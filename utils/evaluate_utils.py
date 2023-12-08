from .boolean_utils import multiply
from .sparse_utils import to_dense
from scipy.sparse import spmatrix, issparse, csr_matrix
import numpy as np
from typing import Union, List
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


def get_metrics(gt: Union[np.ndarray, spmatrix], pd: Union[np.ndarray, spmatrix], metrics: List[str], axis=None):
    """Get results of metrics all at once

    Metrics from sklearn.metrics are included as sanity check.
    The input must be binary array, which makes them slow and less flexible.

    gt, pd: 
        ground truth and prediction.
        these can be 1d array, 2d dense or sparse matrix.
        when the input are matrices, row and column-wise measurement can be conducted by defining axis.
    metrics:
        string list containing matric names.
    axis:
        0 for row-wise measurement. a result array is returned which has the same length as rows.
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


def invert(X):
    if issparse(X):
        X = csr_matrix(np.ones(X.shape)) - X
    elif isinstance(X, np.ndarray):
        X = 1 - X
    else:
        raise TypeError
    return X


def add_log(df, line, verbose=False):
    df.loc[len(df.index)] = line
    if verbose: # print last 5 records upon update
        display(df.tail())