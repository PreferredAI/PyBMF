from .boolean_utils import multiply
from scipy.sparse import spmatrix, issparse, csr_matrix
import numpy as np
from typing import Union, List


def get_metrics(gt: Union[np.ndarray, spmatrix], pd: Union[np.ndarray, spmatrix], metrics: List[str], axis=None):
    functions = {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR,
        'PPV': PPV, 'ACC': ACC, 'ERR': ERR, 'F1': F1,
        'Recall': TPR, 'Precsion': PPV, 'Accuracy': ACC, 'Error': ERR, # alias
    }
    results = []
    for m in metrics:
        if m in functions:
            results.append(functions[m](gt, pd, axis))
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
    return TP(gt=csr_matrix(np.ones(gt.shape))-gt, pd=csr_matrix(np.ones(gt.shape))-pd, axis=axis)


def FN(gt, pd, axis=None):
    return FP(gt=pd, pd=gt, axis=axis)


def TPR(gt, pd, axis=None):
    """sensitivity, recall, hit rate, or true positive rate
    """
    return TP(gt, pd, axis=axis) / gt.sum(axis=axis)


def TNR(gt, pd, axis=None):
    """specificity, selectivity or true negative rate
    """
    return TN(gt, pd, axis=axis) / (csr_matrix(np.ones(gt.shape))-gt).sum(axis=axis)


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
    return TP(gt, pd, axis=axis) / pd.sum(axis=axis)


def ACC(gt, pd, axis=None):
    """accuracy
    """
    if hasattr(pd, "shape"):
        if axis is None:
            n = pd.shape[0] * pd.shape[1]
        else:
            n = pd.shape[axis]
    else:
        n = len(pd)
    return (TP(gt, pd, axis) + TN(gt, pd, axis)) / n


def ERR(gt, pd, axis=None):
    """error
    """
    return 1 - ACC(gt, pd, axis)


def F1(gt, pd, axis=None):
    """F1 score
    """
    # tp = TP(gt, pd, axis)
    # fp = FP(gt, pd, axis)
    # fn = FN(gt, pd, axis)
    # return 2 * tp / (2 * tp + fp + fn)
    precision = PPV(gt, pd, axis)
    recall = TPR(gt, pd, axis)
    return 2 * precision * recall / (precision + recall)