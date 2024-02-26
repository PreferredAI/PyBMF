from .evaluate_utils import cover
import numpy as np
from scipy.sparse import vstack, hstack


def collective_cover(gt, pd, w, axis, starts=None):
    '''

    Parameters
    ----------
    axis : int in {0, 1}, default: None
        The dimension of the basis.

    Returns
    -------
    scores : (n_submat, n_basis) array
    '''
    n_submat = len(w)
    assert len(starts) == n_submat + 1, "[E] Starting points and the number of sub-matrices don't match."

    scores = np.zeros((n_submat, gt.shape[1-axis]))
    for i in range(n_submat):
        a, b = starts[i], starts[i+1]
        s = cover(gt=gt[:, a:b] if axis else gt[a:b, :], 
                  pd=pd[:, a:b] if axis else pd[a:b, :], 
                  w=w[i], axis=axis)
        scores[i] = s

    return scores
