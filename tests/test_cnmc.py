"""Unit tests for cnmc submodules
"""
# third party packages
import torch as pt
import numpy as np
import pandas as pd
# flowtorch packages
from flowtorch.rom import CNM
from tqdm import tqdm

from flowtorch.rom import CNM
from flowtorch.rom import ROMList, CNMcPierzynaButt22
from cnmc.utils_cnmc import *
from cnmc.cnmc import fit_model_P
from sklearn.cluster import KMeans

pt.manual_seed(4224)
pt.set_default_dtype(pt.float64)

def test_zero_adjacent_equal_indices_dense():
    Q = pt.rand((3,3,3))
    print('Original tensor')
    print(Q)
    Q = zero_adjacent_equal_indices_dense(Q)
    print('Zeroed tensor')
    print(Q)
    sum = pt.sum(Q, dim=(0,), keepdim=True)
    print(' ')
    print(sum)
    Q /= sum
    Q = Q.nan_to_num(0.0)
    print(Q)
    print(pt.sum(Q, dim=(0,), keepdim=True))

def test_normalise_probabilities_dense():
    """Probabilities are normalised according to:
    $Q_ijk...n = p(i|j,k,...,n)$
    $\sum_i p(i|j,k,...,n) = 1.0 \foreach {j,k,...,n}$
    $\sum_i Q_ijk...n = 1.0$
    """
    Q = pt.rand((3,3,3))
    # print(pt.sum(Q, dim=(0,), keepdim=True).shape)
    # print(pt.sum(Q, dim=(0,), keepdim=True))
    Q /= pt.sum(Q, dim=(0,), keepdim=True) # conservation of probability
    for i,j in zip(range(Q.shape[1]),range(Q.shape[2])):
        print(Q[:,i,j].sum())

def test_sparse_formats_custom():
    shape = (2,2,2)
    Q = pt.rand(shape).to_sparse_coo()
    print(Q)
    print(Q.to_dense())
    Q_dict = sparse_coo_to_custom_dict(Q)
    print(Q_dict)
    Q_sparse = custom_dict_to_sparse_coo(Q_dict, shape)
    print(Q_sparse)
    print(sparse_coo_to_custom_dict(Q.to_dense()/pt.sum(Q.to_dense(), dim=(0,), keepdim=True)))
    assert pt.all(Q.to_dense()==Q_sparse.to_dense())

def test_sparse_formats():
    shape = (2,2,2)
    Q = pt.rand(shape).to_sparse_coo()
    Q_dict = sparse_coo_to_dict(Q)
    Q_sparse = dict_to_sparse_coo(Q_dict, shape)
    assert pt.all(Q.to_dense()==Q_sparse.to_dense())

def test_CNM_setter():
    K=10; 
    Q = pt.rand((K,K,K))
    Q = zero_adjacent_equal_indices_dense(Q) 
    Q /= pt.sum(Q, dim=(0,), keepdim=True)
    T = pt.rand((K,K,K))
    # print(Q)
    centroids = pt.rand((K,3)).numpy()
    cnm = get_CNM(centroids, Q, T, 0.1)
    pred_cnm = cnm.predict(pt.rand((1,3)), 1.0, 0.1)

def test_CNM_setter_from_cnm(test_cnm):
    centroids = test_cnm.cluster_centers
    K = centroids.shape[0]
    Q = custom_dict_to_sparse_coo(test_cnm.Q, shape=(K,K,K)) 
    T = dict_to_sparse_coo(test_cnm.T, shape=(K,K,K))
    print(Q.shape)
    print(T.shape)
    cnm = get_CNM(centroids, Q, T, test_cnm.dt)
    assert np.all(cnm.cluster_centers==test_cnm.cluster_centers)
    assert cnm.dt==test_cnm.dt
    for jk in cnm.Q.keys():
        assert np.all(test_cnm.Q[jk]==cnm.Q[jk])
    for jk in test_cnm.Q.keys():
        assert np.all(test_cnm.Q[jk]==cnm.Q[jk])
    for jk in cnm.T.keys():
        assert np.all(test_cnm.T[jk]==cnm.T[jk])
    for jk in test_cnm.T.keys():
        assert np.all(test_cnm.T[jk]==cnm.T[jk])
    x0 = pt.from_numpy(centroids[3,:])
    # reproducibility
    np.random.default_rng(4224)
    pred_cnm_set = cnm.predict(x0, 1.0, cnm.dt)
    pred_cnm_tes = test_cnm.predict(x0, 1.0, test_cnm.dt)
    print(pred_cnm_set.size(), pred_cnm_tes.size())
    assert pt.all(pred_cnm_set==pred_cnm_tes)

def test_reorder_tensors():
    tensors = [pt.randint(0,9,(3,2)), pt.randint(0,9,(3,2))]
    j1, j2 = pt.IntTensor([0,1,2]), pt.IntTensor([1,0,2])
    pair_list = [(j1, j2), ]
    reordered_tensors, reordered_indices = sequential_rearrange_tensors(tensors, pair_list)
    print(tensors)
    print(reordered_tensors)
    matched_centroids = pt.stack(reordered_tensors)
    matched_centroids=pt.transpose(matched_centroids,0,1)
    print(matched_centroids)
    print(matched_centroids.shape)
    print(matched_centroids[1,:,:])
    for cs in matched_centroids:
        print(cs.shape)
        print(cs[:,0],cs[:,1])
    Q = pt.randint(0,9,(3,3,3))
    print('Q', Q)
    print('reordered_indices', reordered_indices)
    idx = reordered_indices[1]
    Q = Q[idx, ...]
    Q = Q[:,idx, ...]
    print('Q_reordered', Q)

def test_unify_sparse_tensors():
    # Example usage
    tensor1 = pt.sparse_coo_tensor([[0, 1], [2, 3]], [1.0, 2.0], (4, 5))
    # tensor2 = pt.sparse_coo_tensor([[1, 2], [3, 4]], [3.0, 4.0], (4, 5))
    tensor3 = pt.sparse_coo_tensor([[0, 1], [2, 2]], [5.0, 6.0], (4, 5))

    sparse_tensor_list = [tensor1, tensor3]
    # sparse_tensor_list = [tensor1, tensor2, tensor3]
    unified_tensors = unify_sparse_tensors(sparse_tensor_list)

    for tensor, unified_tensor in zip(sparse_tensor_list, unified_tensors):
        print('input tensor:')
        print(tensor)
        print(' ')
        print('unified tensor:')
        print(unified_tensor)

    dense_tensors = unified_sparse_to_dense_flat(unified_tensors)
    print(' ')
    print('densified tensors:')
    print(dense_tensors)

def test_rearrange_sparse_tensor():
    # idx = pt.IntTensor([2, 0, 1])
    idx = pt.IntTensor([1,0,2])
    Q = pt.randint(0,5,(3,3,3))
    # Q1 = rearrange_tensor(Q,idx)
    Q2 = rearrange_tensor_sparse(Q.to_sparse_coo(),idx)
    print(Q)
    # print(Q1)
    print(Q2.to_dense())
    # assert pt.all(Q1==Q2.to_dense())

def test_P_model_givenQ(Q1, Q3):
    print('col sum Q1', pt.sum(Q1, dim=(0,)))
    print('col sum Q3', pt.sum(Q3, dim=(0,)))

    print('Q1', Q1)
    print('Q3', Q3)
    print('Q1 elements check', Q1[0,0,:], Q1[1,1,:], Q1[2,2,:], Q1[:,0,0], Q1[:,1,1], Q1[:,2,2])
    print('Q3 elements check', Q3[0,0,:], Q3[1,1,:], Q3[2,2,:], Q3[:,0,0], Q3[:,1,1], Q3[:,2,2])

    Q2 = 0.5*(Q1+Q3)
    sum2 = pt.sum(Q2, dim=(0,), keepdim=True)
    # Q2 = zero_adjacent_equal_indices_dense(Q2) # set non trivial transitions to zero # no SVD
    Q2 /= sum2
    Q2 = Q2.nan_to_num(0.0)
    print('col sum Q2 avg.', pt.sum(Q2, dim=(0,)))
    print('Q2 elements check', Q2[0,0,:], Q2[1,1,:], Q2[2,2,:], Q2[:,0,0], Q2[:,1,1], Q2[:,2,2])

    Qs = [Q1.to_sparse_coo(), Q3.to_sparse_coo()]
    ocs = [2.0, 4.0]
    oc = pt.Tensor([[3.0,]])
    model, nnz_indices, size = fit_model_P(Qs, ocs)
    values = model_batch_predict(model, np.array(ocs)[:,None], oc.numpy())[:,:,0].T # for use with linear and sidny
    Q = pt.sparse_coo_tensor(indices=nnz_indices, values=values, size=size+(values.shape[1],))
    Q = Q.to_dense()[...,0]
    # Q = zero_adjacent_equal_indices_dense(Q) # set non trivial transitions to zero # no SVD
    Q[(Q < 0.0)]  = 0.0; Q[(Q > 1.0)]  = 1.0 # no SVD
    sum = pt.sum(Q, dim=(0,), keepdim=True)
    Q /= sum
    Q = Q.nan_to_num(0.0)
    print('col sum Q2 mod.', pt.sum(Q, dim=(0,)))

    print('Q2 linear interp:')
    print(Q)

    print('Q2 averaging:')
    print(Q2)
    
    print('Diff:')
    print((Q-Q2))
    print('Norm of Diff:')
    print(pt.linalg.vector_norm((Q-Q2).flatten()))
    
    return Q2

def test_P_model():
    K=3
    Q1, Q3 = pt.rand((K,K,K)), pt.rand((K,K,K))
    Q1 = zero_adjacent_equal_indices_dense(Q1) # cancel out self transitions
    Q3 = zero_adjacent_equal_indices_dense(Q3) # cancel out self transitions
    Q1[:,2,1] = pt.zeros((K,)) # arbitrary impossible transitions
    Q3[:,1,2] = pt.zeros((K,)) # arbitrary impossible transitions
    sum1 = pt.sum(Q1, dim=(0,), keepdim=True)
    sum3 = pt.sum(Q3, dim=(0,), keepdim=True)
    Q1 /= sum1
    Q3 /= sum3
    Q1 = Q1.nan_to_num(0.0)
    Q3 = Q3.nan_to_num(0.0)
    # print(Q1)
    # print(Q3)
    test_P_model_givenQ(Q1, Q3)

def test_normalize_sparse_columns():
    
    shape = (3,3,3)
    Q = pt.rand(shape)
    Q /= Q.max()
    Q[(Q<0.6)] = 0.0 

    print("original dense Q")
    print(Q)
    print()
    print("normalised dense Q")
    normalized_Q_dense = Q/Q.sum(dim=(0,), keepdim=True)
    normalized_Q_dense=normalized_Q_dense.nan_to_num(0.0)
    print(normalized_Q_dense)
    print(normalized_Q_dense.sum(dim=(0,)))
    print()

    Q = Q.to_sparse_coo()
    normalized_Q = normalize_sparse_columns(Q)

    print("Original sparse Q:")
    print(Q)
    print()
    print("Normalized sparse Q:")
    print(normalized_Q)
    print()
    print("sum Q")
    print(pt.sparse.sum(Q, dim=0))
    print()
    print('sum normalised Q')
    print(pt.sparse.sum(normalized_Q, dim=0))
    print()

    print(normalized_Q_dense.sum(dim=(0,)))
    print(normalized_Q.to_dense().sum(dim=(0,)))
    assert(pt.all(normalized_Q.to_dense()==normalized_Q_dense))

def test_stack_sparse_tensors_to_hybrid():
    # Create some example sparse tensors
    sparse1 = pt.sparse_coo_tensor(
        indices=pt.tensor([[0, 1], [1, 2]]),
        values=pt.tensor([1.0, 2.0]),
        size=(3, 3)
    )
    
    sparse2 = pt.sparse_coo_tensor(
        indices=pt.tensor([[0, 2], [0, 1]]),
        values=pt.tensor([3.0, 4.0]),
        size=(3, 3)
    )
    
    # Convert to hybrid tensor
    hybrid_tensor = stack_sparse_tensors_to_hybrid([sparse1, sparse2])
    
    print("Hybrid Sparse Tensor:")
    print(hybrid_tensor)
    print("\nTensor Size:", hybrid_tensor.size())
    print("\nNonzero Elements:", hybrid_tensor._nnz())


if __name__=="__main__":

    ################################################################################################
    # Unit tests on basic components (tensor manipulation ...):

    # Test sparse conversion from dense and vice versa
    # test_sparse_formats()
    # test_sparse_formats_custom()

    # Test CNM setter
    # test_CNM_setter()

    # Test tensors reordering
    # test_reorder_tensors()

    # Test probability conservation
    # test_normalise_probabilities_dense()

    # Test utils for QT interpolation
    # test_unify_sparse_tensors()

    # Test rearrange sparse tensor
    # test_rearrange_sparse_tensor()

    # Test removing trivial transitions
    # test_zero_adjacent_equal_indices_dense()

    # Test P interpolator on synthetic transition tensors
    # test_P_model()

    # Test columns sum to one for sparse tensors
    # test_normalize_sparse_columns()

    test_stack_sparse_tensors_to_hybrid()

    ################################################################################################
    # Unit tests on dynamical data (elements of CNMc proper):

    def test_CNM_getter():
        # Setup data:
        DATAPATH = '/home/paolo/Desktop/semaan/semaan-project/cnm/hands-on/flowtorch/lorenz/lorenz/data/'
        FILENAMES = ['Lorenz_b29.000.csv',]
        PARAMETERS = [29.0,]
        data=[]
        for filename in FILENAMES:
            d = pd.read_csv(DATAPATH+filename, header=None, index_col=0)[[1,2,3,4]].to_numpy()
            dt = d[1,0]-d[0,0]
            d = pt.from_numpy(d.T)
            print(d.shape)
            data.append(d[1:4,-10_000:])

        # Setup and fit n CNMs:
        encoder = None
        K=10; L=2
        cnm = CNM(reduced_state=data[0], 
                encoder=encoder, 
                dt=dt, 
                n_clusters=K, 
                model_order=L,
                cluster_config={'algorithm': KMeans, 'n_init': 25, 'init': 'k-means++',},
                )

        test_CNM_setter_from_cnm(cnm)

    # test_CNM_getter()

    def test_CNMc_loop():
        # Setup data:
        DATAPATH = '/home/paolo/Desktop/semaan/semaan-project/cnm/hands-on/flowtorch/lorenz/lorenz/data/'
        FILENAMES = ['Lorenz_b30.000.csv', 'Lorenz_b70.000.csv']
        PARAMETERS = [30.0, 70.0]
        data=[]
        for filename in FILENAMES:
            d = pd.read_csv(DATAPATH+filename, header=None, index_col=0)[[1,2,3,4]].to_numpy()
            dt = d[1,0]-d[0,0]
            d = pt.from_numpy(d.T)
            print(d.shape)
            data.append(d[1:4,-10_000:])

        # Setup and fit n CNMs:
        encoder = None
        K=3; L=2
        cnmss = []
        for m, X in enumerate(tqdm(data)):
            # print('Fitting CNM on parameter '+str(m))
            cnm = CNM(reduced_state=X, 
                    encoder=encoder, 
                    dt=dt, 
                    n_clusters=K, 
                    model_order=L,
                    cluster_config={'algorithm': KMeans, 'n_init': 25, 'init': 'k-means++',},
                    )
            cnmss.append(cnm)

        # Feed models into CNMc.
        cnms = [{"oc": oc, "rom": cnm} for oc, cnm in zip(PARAMETERS, cnmss)]
        cnmlist = ROMList(cnms)
        cnmc = CNMcPierzynaButt22(cnmlist, centroid_interpolate='Spline')
        # cnmc = CNMcPierzynaButt22(cnmlist, centroid_interpolate='PiecewiseLinear')
        # print(cnmc.centroids)

        print()

        Q1 = custom_dict_to_sparse_coo(cnmss[0].Q, (K,)*(L+1)).to_dense()
        Q3 = custom_dict_to_sparse_coo(cnmss[1].Q, (K,)*(L+1))
        print(cnmc.matched_indices)
        Q3 = rearrange_tensor_sparse(Q3, cnmc.matched_indices[1]).to_dense()
        Q2 = test_P_model_givenQ(Q1, Q3)

        oc=pt.Tensor([50.0,]).unsqueeze(1)
        Qs, Ts = cnmc._predict_QT(oc)
        print('Q2 cnm predicted', Qs[...,0])
        print('Q2 cnm diff', Qs[...,0]-Q2)
        print('Norm of Diff cnmc:')
        print(pt.linalg.vector_norm((Qs[...,0]-Q2).flatten()))

        # # Test matching
        # cnmc._match_centroids()
        # # print(cnmc.centroids)
        # # print(cnmc.matching)
        # # print(cnmc.matched_indices)
        # # for cs in cnmc.matched_centroids:
        # #     print(cs)

        # # Test centroid modelling
        # cnmc._fit_model_centroids()

        # # Test transition modelling
        # cnmc._fit_models_QT()

        # ## Test Predictions:##
        # oc=pt.Tensor([40.0,60.0,]).unsqueeze(1) # batch of 1 OCs
        # # oc=pt.Tensor([35.0, 40.0,]).unsqueeze(1) # batch of 2 OCs

        # # # Test centroid prediction
        # # centroids, interpolator = cnmc.centroid_interpolate(oc.numpy())
        # # centroids = pt.from_numpy(cnmc.centroid_interpolate(oc.numpy()))

        # # Test transition prediction
        # cnmc._predict_QT(oc)

        # # # Test CNM predictions
        # cnm_test = cnmc.predict_model(oc)

        # # # Test CNM predictions
        # # x0 = pt.Tensor(cnmss[1].cluster_centers[3])
        # # cnm_test[0].predict(x0,1,dt)

    # test_CNMc_loop()