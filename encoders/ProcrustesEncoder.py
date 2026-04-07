import torch as pt
pt.set_default_dtype(pt.float64)

from flowtorch.rom.base import Encoder

# tensor convention (hard coded)
ax_data = 1
ax_dims = 0


class ProcrustesEncoder(Encoder):
    def __init__(self, reference_state: pt.Tensor = None, reg=None):
        super().__init__()
        '''Inputs are pt.Tensor of shape (ndata, ndim)
        '''
        
        # reference point cloud 
        # if reference_state is not None:
        #     self.reference_state = reference_state
        self.vt_ref = None
        self._state_size = None
        
        # transformation parameters learned during train()
        self.mean_ = None
        self.scale_ = None
        self.rotation_ = None
        
        self.reg = reg        # SVD regularisation

        if reference_state is not None:
            reference_state = reference_state.detach()
            self._state_size = reference_state.shape[ax_dims]
            self.ref_shape = reference_state.shape

            # learn empirical frame of reference_state (centre -> scale -> principal components)
            ref_scaled, _, _ = centre_and_standardize(reference_state)
            self.ref_scaled = ref_scaled
            u_ref, S_ref, _ = pt.linalg.svd(ref_scaled, full_matrices=False)
            self.vt_ref = u_ref @ pt.diag(S_ref/S_ref[0])

    def train(self, state: pt.Tensor):
        """
        Learns 
        1) the principal frame of 'reference_state' and 
        2) the affine transformation required to map 'state' to the principal frame
        If state is identical to reference_state, it just scales and centres.
        """

        state = state.detach()

        # learn empirical frame of target state (centre -> scale -> principal components)
        scaled, self.mean_, self.scale_ = centre_and_standardize(state)
        self.scaled = scaled
        u_curr, S_curr, _ = pt.linalg.svd(scaled, full_matrices=False)
        self.vt_curr = u_curr @ pt.diag(S_curr/S_curr[0])

        # align signs of paired principal components
        self.vt_curr = align_signs(self.vt_curr, self.vt_ref) 
        # if reference data, leave as is
        if state.shape==self.ref_shape:
            if pt.all(self.vt_ref==self.vt_curr):
                self.rotation_ = pt.eye(self._state_size)
                self.trained = True
                return self
        
        # else learn frame alignment (Kabsch)
        self.rotation_ = kabsch(self.vt_curr.T, self.vt_ref.T, reg=self.reg).T # transpose as kabsch works on N,D tensors
        self.trained = True
        return self

    def encode(self, state: pt.Tensor) -> pt.Tensor:
        """
        Applies the learned affine transformation.
        """

        if not self.trained:
            raise RuntimeError("Encoder must be trained before encoding.")

        self._check_state_shape(state.shape)

        state = state.detach()

        # Apply transformation sequence
        centered = state - self.mean_
        scaled = centered / self.scale_
        aligned = pt.matmul(self.rotation_, scaled)
        
        return aligned

    def decode(self, reduced_state: pt.Tensor) -> pt.Tensor:
        """
        Applies the inverse affine transformation.
        """
        if not self.trained:
            raise RuntimeError("Encoder must be trained before decoding.")

        self._check_reduced_state_size(reduced_state.shape)

        reduced_state = reduced_state.detach()

        # Reverse Rotation (using transpose/inv)
        unrotated_np = pt.matmul(self.rotation_.T, reduced_state)
        
        # Reverse Scaling and Centering
        unscaled = (unrotated_np * self.scale_) + self.mean_
        
        return unscaled

    @property
    def state_shape(self) -> pt.Size:
        """Get the size of the full state.

        :return: size of the full state
        :rtype: pt.Size
        """
        return pt.Size((self._state_size,))

    @property
    def reduced_state_size(self) -> int:
        """Get the size of the reduced state.

        :return: size of the reduced state.
        :rtype: int
        """
        return self._state_size

def centre_and_standardize(state: pt.Tensor):
    mean = pt.mean(state, axis=ax_data, keepdims=True)
    centered = state - mean
    scale = pt.sqrt(pt.mean(pt.sum(centered**2, axis=ax_dims, keepdims=True), axis=ax_data))
    scaled = centered / scale
    return scaled, mean, scale

def kabsch(P: pt.Tensor, Q: pt.Tensor, reg=None) -> pt.Tensor:
    """

    Kabsch algorithm on two sets of paired points P and Q, centered around the centroid. 
    Adapted from: https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    R : matrix
        Rotation matrix (D,D)
    """

    # compute the covariance matrix
    C = pt.matmul(P.t(), Q)

    # compute the optimal rotation matrix
    if reg is not None: C += reg * pt.eye(C.shape[0])
    U, S, Vt = pt.linalg.svd(C, full_matrices=False)

    # correct rotation matrix to ensure right-handed frame
    d = (pt.linalg.det(U) * pt.linalg.det(Vt)) < 0.0
    if d:
        print('det<0')
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]

    # rotation matrix R
    R = pt.matmul(U, Vt)

    return R

def align_signs(V1: pt.Tensor, V2: pt.Tensor) -> pt.Tensor:
    """
    For each column pair (v1, v2), flip the sign of v1 if it makes the sign of
    the dot product with v2 positive (i.e., if dot(v1, v2) < 0).

    Args:
        V1: pt.Tensor of shape (n, m)
        V2: pt.Tensor of shape (n, m)

    Returns:
        A copy of V1 with column signs adjusted to maximise dot products with V2.
    """
    # dot products for each column pair: shape (m,)
    dot_products = (V1 * V2).sum(dim=0)

    # mask is -1 where flipping improves alignment, +1 otherwise
    signs = pt.where(dot_products < 0, pt.Tensor([-1.0]), pt.Tensor([1.0]))

    # apply sign correction (broadcasting over rows)
    return V1 * signs.unsqueeze(0)

def rmsd(P: pt.Tensor, Q: pt.Tensor) -> float:
    """
    Calculate Root-mean-square deviation from two sets of vectors P and Q.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = P - Q
    return pt.sqrt((diff * diff).sum() / P.shape[0])
