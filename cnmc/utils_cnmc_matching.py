import torch as pt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _marginalise(Q):
    reduced = tuple(range(Q.ndim-1, 1, -1))
    Q = Q.sum(dim=reduced)
    Q /= Q.sum(axis=0)
    return Q

def trivial_assign(x1, x2, *args):
    j1 = pt.arange(x1.shape[0])
    j2 = pt.arange(x2.shape[0])
    return j1, j2

def scipy_assign(x1, x2, *args):
    # Build cost matrix
    M = cdist(x1.numpy(), x2.numpy())
    M /= M.max()
    j1, j2 = linear_sum_assignment(M)
    return pt.from_numpy(j1), pt.from_numpy(j2)
