from .boolean_utils import multiply, matmul, dot, power
from .sparse_utils import to_dense, to_triplet, to_sparse
from scipy.sparse import spmatrix, issparse, csr_matrix
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


def get_metrics(gt, pd, metrics, axis=None):
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
        'RMSE': RMSE, # real distances
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
    """Accuracy.
    """
    if len(pd.shape) == 2:
        n = pd.shape[0] * pd.shape[1] if axis is None else pd.shape[axis]
    else:
        n = len(pd)
    return (TP(gt, pd, axis) + TN(gt, pd, axis)) / n


def ERR(gt, pd, axis=None):
    """Error rate.
    """
    return 1 - ACC(gt, pd, axis)


def F1(gt, pd, axis=None):
    """F1 score.

    tp = TP(gt, pd, axis)
    fp = FP(gt, pd, axis)
    fn = FN(gt, pd, axis)
    return 2 * tp / (2 * tp + fp + fn)
    """
    precision = PPV(gt, pd, axis)
    recall = TPR(gt, pd, axis)
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0


def RMSE(gt, pd, axis=None):
    rmse = np.sqrt(power(gt - pd, 2).sum(axis) / gt.sum(axis))
    return rmse


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
    # debug: adaptive w
    c_all = covered + overcovered
    if w >= 1 and axis is not None:
        to_be_covered = np.asarray(gt.sum(axis=axis)).squeeze()
        n_all = gt.shape[axis]
        score = covered * n_all - w * multiply(to_be_covered, c_all)
    elif w >= 1 and axis is None:
        to_be_covered = gt.sum()
        n_all = gt.shape[0] * gt.shape[1]
        score = covered * n_all - w * to_be_covered * c_all
    else:
        score = (1 - w) * covered - w * overcovered
    # print(score.shape if isinstance(score, np.ndarray) else score)
    return score


def invert(X):
    if issparse(X):
        X = csr_matrix(np.ones(X.shape)) - X
    elif isinstance(X, np.ndarray):
        X = 1 - X
    else:
        raise TypeError
    return X
