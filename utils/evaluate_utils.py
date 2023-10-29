from .boolean_utils import multiply
from scipy.sparse import issparse
import numpy as np


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
    return TP(gt=1-gt, pd=1-pd, axis=axis)


def FN(gt, pd, axis=None):
    return FP(gt=pd, pd=gt, axis=axis)


def TPR(gt, pd, axis=None):
    """sensitivity, recall, hit rate, or true positive rate
    """
    return TP(gt, pd, axis=axis) / gt.sum(axis=axis)


def TNR(gt, pd, axis=None):
    """specificity, selectivity or true negative rate
    """
    return TN(gt, pd, axis=axis) / (1-gt).sum(axis=axis)


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