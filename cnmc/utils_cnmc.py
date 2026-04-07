from typing import Dict, Tuple
from collections import defaultdict
from copy import deepcopy

# third party packages
import numpy as np
import torch as pt
pt.set_default_dtype(pt.float64)

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

# flowtorch packages
from ..crom import CNM


def reshape_columns(tensor: pt.Tensor, shape: Tuple[int, ...]) -> list[pt.Tensor]:
    """Reshapes columns of a 2D tensor and stores them in a list.

    Args:
        tensor (pt.Tensor): (m,n) tensor.
        shape (Tuple[int, ...]): shape to reshape each column into.

    Returns:
        list[pt.Tensor]: list of n tensors containing the reshaped columns.
    """
    
    result: list[pt.Tensor] = []
    for i in range(tensor.shape[1]):
        reshaped_column = tensor[:, i].reshape(shape)
        result.append(reshaped_column)
    
    return result
    
def dict_to_sparse_coo(data_dict: Dict[str, np.ndarray], shape: Tuple[int, ...]) -> pt.Tensor:
    indices = []
    values = []
    
    for key, value_array in data_dict.items():
        # Extract indices from the key (from last to second)
        key_indices = [int(idx) for idx in key.split(',')]
        key_indices.reverse()

        indices.append(key_indices)
        values.append(float(value_array))

    # Convert to tensor
    indices = pt.tensor(indices, dtype=pt.long).t()
    values = pt.tensor(values)#, dtype=pt.float)
    
    # Create and return the sparse COO tensor with the given shape
    return pt.sparse_coo_tensor(indices, values, size=shape).coalesce()

def custom_dict_to_sparse_coo(data_dict: Dict[str, np.ndarray], shape: Tuple[int, ...]) -> pt.Tensor:
    indices = []
    values = []
    
    for key, value_array in data_dict.items():
        # Extract indices from the key (from last to second)
        key_indices = [int(idx) for idx in key.split(',')]
        key_indices.reverse()
        # Iterate through each row in the value array
        for first_idx, element_value in value_array:
            # Combine all indices
            full_indices = [int(first_idx)] + key_indices
            indices.append(full_indices)
            values.append(element_value)

    # Convert to tensor
    indices = pt.tensor(indices, dtype=pt.long).t()
    values = pt.tensor(values)

    # Create and return the sparse COO tensor with the given shape
    return pt.sparse_coo_tensor(indices, values, size=shape).coalesce()

def sparse_coo_to_dict(sparse_tensor: pt.Tensor) -> Dict[str, np.ndarray]:
    # Ensure the tensor is in sparse COO format
    if not sparse_tensor.is_sparse:
        sparse_tensor=sparse_tensor.to_sparse_coo()

    # Ensure the sparse tensor is coalesced
    if not sparse_tensor.is_coalesced():
        sparse_tensor = sparse_tensor.coalesce()
    
    # Get indices and values
    indices = sparse_tensor.indices().t().numpy()
    values = sparse_tensor.values().numpy()
    
    # Create the dictionary
    result_dict = defaultdict(list)
    
    for idx, value in zip(indices, values): 
        # Create the key from the indices
        key = ','.join(map(str, idx[::-1]))
        
        # Append to the list for this key
        result_dict[key] = value
    
    # Convert lists to values
    return result_dict

def sparse_coo_to_custom_dict(sparse_tensor: pt.Tensor) -> Dict[str, np.ndarray]:
    # Ensure the tensor is in sparse COO format
    if not sparse_tensor.is_sparse:
        sparse_tensor=sparse_tensor.to_sparse_coo()

    # Ensure the sparse tensor is in COO format
    if not sparse_tensor.is_coalesced():
        sparse_tensor = sparse_tensor.coalesce()
    
    # Get indices and values
    indices = sparse_tensor.indices().t().numpy()
    values = sparse_tensor.values().numpy()
    
    # Create the dictionary
    result_dict = defaultdict(list)
    
    for idx, value in zip(indices, values):
        # Extract the first index and the rest
        first_idx, *rest_indices = idx
        
        # Create the key from the rest of the indices (from last to second)
        key = ','.join(map(str, rest_indices[::-1]))
        
        # Append to the list for this key
        result_dict[key].append([first_idx, value])

    # Convert lists to numpy arrays
    for k, v in result_dict.items():
        result_dict[k] = np.array(v)

    return result_dict

def get_CNM(centroids, Q, T, dt, spline_order=3, encoder=None) -> CNM:
    """Build an instance of CNM with the given properties.
    """
    # Extract cluster number and model order from transition matrix
    n_clusters = Q.shape[0];
    model_order=len(Q.shape)-1

    # Initialise CNM instance without data
    cnm = CNM(encoder=encoder, 
              dt=dt, 
              n_clusters=n_clusters, 
              model_order=model_order,
              spline_order = spline_order,
              )
    
    # Convert Q and T to the dictionary format used by CNM
    Q = sparse_coo_to_custom_dict(Q)
    T = sparse_coo_to_dict(T)
    cnm._transition_prob = Q
    cnm._transition_time = T
    cnm.cluster_centers = centroids
    cnm._cluster._n_threads = _openmp_effective_n_threads()

    return cnm

def sequential_rearrange_tensors(tensor_list, matching_pairs):
    result = [tensor_list[0]]  # First tensor remains unchanged
    matched_indices = [matching_pairs[0][0]]
    j2_prev = matching_pairs[0][0] # initialise previous indices
    for l, (current_tensor, (j1, j2)) in enumerate(zip(tensor_list[1:], matching_pairs)):
        # Reorder the current tensor based on current reordering and previous index
        reordered_tensor = current_tensor[j2[j2_prev]]
        matched_indices.append(j2[j2_prev])
        result.append(reordered_tensor)
        j2_prev = j2[j2_prev]
    
    return result, matched_indices

def model_batch_predict(models, ocs, t):
    """Temporary interface to make predictions across multiple OCs.
    """

    num_batches, num_coords = len(models), 1

    nocs = pt.atleast_2d(pt.Tensor(ocs))
    ocsmin = pt.min(nocs, dim=0, keepdim=True).values
    ocsmax = pt.max(nocs, dim=0, keepdim=True).values
    t = pt.atleast_2d(pt.Tensor(t))
    t = (t - ocsmin) / (ocsmax-ocsmin)
    num_t = t.shape[0]


    # Initialize the output tensorm
    interpolated = pt.zeros((num_t, num_batches, num_coords))
    # for batch, spline in enumerate(models):
    #     for coord, spline_i in enumerate(spline):
    #         interpolated[:, batch, coord] = pt.Tensor(spline_i.predict(t.numpy()))
    #         print('predict y', interpolated[:, batch, coord])

    for batch, spline in enumerate(models):
        interpolated[:, batch, :] = pt.Tensor(np.atleast_2d(spline[0].predict(t.numpy())).T)
    

    return interpolated

from sklearn.linear_model import LinearRegression

def fit_model_P(Ps: list[pt.Tensor], ocs: list, base_model=None, base_model_options={}): 
    """Fits regression model of a transition matrix's nonzero elements (Q or T).

    Args:
        Ps (list): list of sparse COO transition property matrices (Q or T) of length n_ocs
        ocs (list): list of OCs of length n_ocs

    Returns:
        tuple: fitted model, nnz indices, nnz size
    """

    # Normalise ocs 0 to 1 on the grid:
    nocs = pt.atleast_2d(pt.Tensor(ocs))
    ocsmin = pt.min(nocs, dim=0, keepdim=True).values
    ocsmax = pt.max(nocs, dim=0, keepdim=True).values
    nocs = (nocs - ocsmin) / (ocsmax-ocsmin)

    # Extract elements of Q,T which are non zero in at least one of the OCs:
    Ps_nnz = unify_sparse_tensors(Ps)

    # Fit model on nnz values:

    # Initialise model :
    if base_model is None:
        base_model = LinearRegression

    # Perform regression for each batch and coordinate
    Ps_nnz_dense = unified_sparse_to_dense_flat(Ps_nnz)[0].numpy()

    V_regressor = []
    for batch in range(Ps_nnz_dense.shape[-1]):
        model_i = deepcopy(base_model)
        model_i.fit(np.array(nocs), Ps_nnz_dense[:, batch])
        V_regressor.append([model_i])
    
    return V_regressor, Ps_nnz[0].indices(), Ps_nnz[0].size()

def zero_adjacent_equal_indices_dense(tensor):
    """Sets to zero the elements of the transition tensor which have an equal pair of adjacent
    indices. This excludes self transitions.
    """ 
    # Create a mask of the same shape as the input tensor
    mask = pt.ones_like(tensor, dtype=pt.bool)
    
    # Check equality for each adjacent pair of dimensions
    for dim in range(tensor.ndim - 1):
        equal_indices = pt.eq(pt.arange(tensor.shape[dim], device=tensor.device).view(-1, *[1]*(tensor.ndim-dim-1)),
                              pt.arange(tensor.shape[dim+1], device=tensor.device).view(*[1]*dim, -1, *[1]*(tensor.ndim-dim-2)))
        mask &= ~equal_indices
    
    # Apply the mask to the tensor
    return tensor * mask

def zero_adjacent_equal_indices_sparse(sparse_tensor):
    # Get the indices and values of the sparse tensor
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # Create a mask, initially all True
    mask = pt.ones(indices.shape[1], dtype=pt.bool, device=indices.device)
    
    # Check equality for each adjacent pair of dimensions
    for dim in range(indices.shape[0] - 1):
        equal_indices = pt.eq(indices[dim], indices[dim + 1])
        mask &= ~equal_indices
    
    # Apply the mask to the values and indices
    new_values = values[mask]
    new_indices = indices[:, mask]
    
    # Create a new sparse tensor with the updated values and indices
    return pt.sparse_coo_tensor(new_indices, new_values, sparse_tensor.size())

def rearrange_tensor_sparse(Q, idx):
    """Sparse version of `rearrange_tensor` written by Claude
    """

    # Ensure Q is in COO format
    Q = Q.coalesce()

    # Get the indices and values of the sparse tensor
    indices = Q.indices()
    values = Q.values()
    
    # Convert idx to a tensor if it's not already
    if not isinstance(idx, pt.Tensor):
        idx = pt.tensor(idx, device=Q.device)

    # Check if the length of idx matches the size of each dimension
    assert idx.size(0) == Q.size(0), "Length of idx must match the size of each dimension"

    new_indices = indices.clone()
    new_values = values.clone()

    # Create the index map
    index_map = {old: new for new, old in enumerate(idx.tolist())}

    # Rearrange all dimensions
    for dim in range(Q.dim()):
        # Create mask for valid indices
        mask = pt.isin(new_indices[dim], idx)
        
        # Apply mask to indices and values
        new_indices = new_indices[:, mask]
        new_values = new_values[mask]

        # Map old indices to new positions
        new_indices[dim] = pt.tensor([index_map[i.item()] for i in new_indices[dim]], 
                                     device=Q.device)
    
    # Create the new sparse tensor
    size = [len(idx)] * Q.dim()
    
    return pt.sparse_coo_tensor(new_indices, new_values, size=size)

def unify_sparse_tensors(sparse_tensor_list):
    """Unifies a list of sparse COO tensors to have the same number of explicitly declared elements.
    All tensors in the returned list will have the same number of explicitly declared elements,
    which will be zero if they were missing in the corresponding input tensor but nonzero in
    another.  
    
    Args:
        sparse_tensor_list (list): A list of sparse COO tensors.
    
    Returns:
        list: A new list of coalesced sparse COO tensors with the same number of explicitly declared
        elements. 
    """

    # Detect the unique indices of all nonzero elements across all tensors
    all_indices = []
    for tensor in sparse_tensor_list:
         for idx in tensor._indices().t():
            tidx = tuple(idx)
            if tidx not in all_indices: all_indices.append(tidx)

    # Create a new list of tensors with the same number of explicitly declared elements
    unified_tensors = []
    for tensor in sparse_tensor_list:
        new_indices = pt.tensor([list(idx) for idx in all_indices], dtype=pt.long, device=tensor.device)
        new_values = pt.zeros(len(all_indices), dtype=tensor.dtype, device=tensor.device)
        
        orig_indices = tensor._indices().t()
        orig_values = tensor._values()
        for i, idx in enumerate(new_indices):
            if tuple(idx) in map(tuple, orig_indices):
                new_values[i] = orig_values[list(map(tuple, orig_indices)).index(tuple(idx))]
    
        
        #  must be coalesced otherwise ordering wont be consistent
        unified_tensors.append(pt.sparse_coo_tensor(new_indices.t(), new_values, tensor.shape).coalesce())
    
    return unified_tensors

def unified_sparse_to_dense_flat(unified_tensor_list):
    """
    Converts a list of unified sparse COO tensors into a dense 2D tensor.
    
    Args:
        unified_tensor_list (list): A list of unified sparse COO tensors.
        
    Returns:
        pt.Tensor: A dense tensor of shape (len(unified_tensor_list), nnz).
    """
    assert unified_tensor_list[0].is_coalesced() # must be coalesced otherwise ordering wont be consistent
    
    nnz = unified_tensor_list[0]._nnz()
    num_tensors = len(unified_tensor_list)
    
    # Create the output dense tensor
    dense_output = pt.zeros((num_tensors, nnz), dtype=unified_tensor_list[0].dtype, device=unified_tensor_list[0].device)
    
    # Fill the dense tensor with values from each unified sparse tensor
    for i, tensor in enumerate(unified_tensor_list):
        dense_output[i,:] = tensor._values()
    
    index_output = unified_tensor_list[0]._indices().T
    
    return dense_output, index_output

def normalize_sparse_columns(sparse_tensor):
    """
    Normalize a sparse tensor so that the first dimension sums to 1 for each slice.
    
    Args:
        sparse_tensor (torch.Tensor): A sparse tensor
    
    Returns:
        torch.Tensor: Normalized sparse tensor
    """
    # Ensure the input is a sparse tensor
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Compute column sums using sparse reduction
    column_sums = pt.sparse.sum(sparse_tensor, dim=0)

    # Prepare indices and values for the input sparse tensor
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Coalesce the sums tensor
    column_sums = column_sums.coalesce()
    sums_indices = column_sums._indices()
    sums_values = column_sums._values()

    # Prepare indices for the remaining dimensions
    remaining_indices = indices[1:]
    
    # Create normalized values
    normalized_values = values.clone()


    # Iterate through non-zero elements
    kill_zerosums = pt.zeros(len(values), dtype=bool)
    # TODO: do this without for loops!!
    for i in range(len(values)):
        # Find the matching sum by comparing indices of remaining dimensions
        match_mask = pt.all(sums_indices == indices[1:, i:i+1], dim=0)
        if sums_values[match_mask][0]==0.0:
            kill_zerosums[i] = True
        else:
            if match_mask.any():
                # Normalize the value
                normalized_values[i] /= sums_values[match_mask][0]
        # if match_mask.any():
        #     # Normalize the value
        #     normalized_values[i] /= sums_values[match_mask][0]
    normalized_values = normalized_values.nan_to_num(0.0)
    

    normalized_tensor = pt.sparse_coo_tensor(
        indices[:, ~kill_zerosums], 
        normalized_values[~kill_zerosums], 
        size=sparse_tensor.size()
    )

    # Create and return the normalized sparse tensor
    return normalized_tensor, kill_zerosums

def stack_sparse_tensors_to_hybrid(sparse_tensors):
    """
    Stacks a list of PyTorch sparse COO tensors along a new last dimension
    to create a hybrid sparse tensor with a dense last dimension.

    Args:
        sparse_tensors (list of torch.sparse_coo_tensor): List of sparse COO tensors to stack.
        
    Returns:
        torch.sparse_coo_tensor: A hybrid sparse tensor with a dense last dimension.
    """
    if len(sparse_tensors)==0:
        raise ValueError("The list of sparse tensors is empty.")
    
    # Get the shape of the sparse tensors
    base_shape = sparse_tensors[0].shape
    dense_dim = len(sparse_tensors)
    
    # Check if all tensors have the same shape
    for tensor in sparse_tensors:
        if tensor.shape != base_shape:
            raise ValueError("All sparse tensors must have the same shape.")
    
    # Initialize lists for combined indices and values
    all_indices = []
    all_values = []
    
    for i, tensor in enumerate(sparse_tensors):
        # Extract indices and values
        indices = tensor._indices()
        values = tensor._values()
        
        # Add a new dimension for stacking
        stacked_indices = pt.cat([indices, pt.full((1, indices.size(1)), i, dtype=pt.long)], dim=0)
        all_indices.append(stacked_indices)
        all_values.append(values)
    
    # Concatenate all indices and values
    combined_indices = pt.cat(all_indices, dim=1)
    combined_values = pt.cat(all_values, dim=0)
    
    # Define the new size
    new_size = base_shape + (dense_dim,)
    
    # Create the stacked sparse tensor
    hybrid_tensor = pt.sparse_coo_tensor(combined_indices, combined_values, size=new_size)
    
    return hybrid_tensor