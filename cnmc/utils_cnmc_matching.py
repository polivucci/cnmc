import torch as pt
from ot import emd, dist, unif
from ot.gromov import gromov_wasserstein, entropic_gromov_wasserstein
from ot.gromov import fused_gromov_wasserstein, entropic_fused_gromov_wasserstein
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
    M = dist(x1.numpy(), x2.numpy())
    M /= M.max()
    j1, j2 = linear_sum_assignment(M)
    return pt.from_numpy(j1), pt.from_numpy(j2)

def ot_assign(x1, x2, *args):
    # Build cost matrix
    M = dist(x1, x2)
    M /= M.max()
    # Solve OT
    G = emd(unif(x1.shape[0], type_as=x1), unif(x2.shape[0], type_as=x1), M)
    return pt.nonzero(G, as_tuple=True)

def ot_emd(x1, x2, p1, p2, *args):
    # Build cost matrix
    M = dist(x1, x2)
    M /= M.max()
    # Solve OT
    G = emd(p1, p2, M)
    return pt.nonzero(G, as_tuple=True)

def ot_gw(x1, x2, C1, C2, *args):
    # Densify
    if C1.is_sparse: C1=C1.to_dense()
    if C2.is_sparse: C2=C2.to_dense()
    # Marginalise higher order! The algorithm only works only with dim 2 matrices
    if C1.ndim>2: C1=_marginalise(C1)
    if C2.ndim>2: C2=_marginalise(C2)
    # G = entropic_gromov_wasserstein(C1, C2, symmetric=False, solver='PPA', epsilon=1e-3)
    G = gromov_wasserstein(C1, C2, symmetric=False, loss_fun='square_loss')
    # print(G)
    return pt.nonzero(G, as_tuple=True)

def ot_fusedgw(x1, x2, C1, C2, *args):
    # Densify
    if C1.is_sparse: C1=C1.to_dense()
    if C2.is_sparse: C2=C2.to_dense()
    # Marginalise higher order! Works only with dim 2 matrices
    if C1.ndim>2: C1=_marginalise(C1)
    if C2.ndim>2: C2=_marginalise(C2)
    M = dist(x1, x2)
    M /= M.max()
    G = fused_gromov_wasserstein(M, C1, C2, symmetric=False, loss_fun='square_loss')
    return pt.nonzero(G, as_tuple=True)

# def ot_fusedgw(x1, x2, C1, C2, *args):
#     pass
