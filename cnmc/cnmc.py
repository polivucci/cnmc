"""Implementation of control-oriented cluster-based network modeling (CNMc).

0: Pick K
1: Fit CNM_OCi (K, L) models for each OCi

Centroid tracking
3: Solve N_OC-1 Procrustes problems to determine centroid labels
4: Interpolate in space using SINDy
Transition modelling
5: Apply scaling to Q, T
6: Train a RF model on Q, T

Prediction at OC'
7: Get centroids from SINDy(OC')
8: Get Q, T from RF(OC')
9: Query CNM_OC with an initial condition
"""
import sys
PATH_FLOWTORCH = '/home/paolo/Desktop/semaan/semaan-project/cnm/hands-on/flowtorch/lorenz/'
sys.path.append(PATH_FLOWTORCH)

# standard library packages
from abc import ABC #, abstractmethod, abstractproperty
# from typing import Dict, Tuple

# third party packages
import numpy as np
import torch as pt
pt.set_default_dtype(pt.float64)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, MinMaxScaler, minmax_scale
from sklearn.cluster import KMeans

from pysindy.pysindy import SINDy

# Flowtorch packages
from flowtorch.rom.base import ROM, Encoder

from .utils_cnmc import (get_CNM, 
                         sequential_rearrange_tensors, 
                         custom_dict_to_sparse_coo, 
                         dict_to_sparse_coo, 
                         barycentric_batch_interpolate, 
                         spline_batch_interpolate,
                         fit_model_P,
                         sindy_batch_regression,
                         spline_batch_predict,
                         model_batch_predict,
                         zero_adjacent_equal_indices_sparse,
                         normalize_sparse_columns,
                         stack_sparse_tensors_to_hybrid,
                         rearrange_tensor_sparse,
                         unify_sparse_tensors,
                         unified_sparse_to_dense_flat
                         ) 

from .utils_cnmc_matching import trivial_assign

class ROMList(ABC):
    """Generic container for a list of ROMs.
    """
    def __init__(self, roms: list[dict]) -> None:
        self.idx = [j for j, _ in enumerate(roms)]
        self.ocs = [rom["oc"] for rom in roms]
        self.roms = [rom["rom"] for rom in roms]

class CNMc(ABC):
    """Control-oriented CNM as in manuscript.
    """

    def __init__(self, 
                 roms: ROMList, 
                 encoders: list[Encoder], 
                 encoder_model: None, 
                 clustering, 
                 transition_model: None,
                 **kwargs) -> None:
        """

        Args:
            roms (ROMList): M instances of a CROM model, one for each operating condition (OC) in the data.
            encoders (list[Encoder]): M instances of encoder.
            encoder_model (None): supervised model for the parametric transformation.
            clustering (_type_): instance of a sklearn-style clustering algorithm.
            transition_model (None): instance of a sklearn-style supervised model for a single transition property.

        Methods:
            train: list of tensors and train the CNMc model.
        """

        super().__init__()

        self.roms = roms.roms
        self.ocs = roms.ocs
        self.encoders = encoders
        self.encoder_model = encoder_model
        self.clustering = clustering

        self.Ks = [rom.n_clusters for rom in self.roms]
        self.Ls = [rom.model_order for rom in self.roms]
        self.spline_order = self.roms[0].spline_order
        self.shapes = [(K,)*(L+1) for K,L in zip(self.Ks, self.Ls)]
        self.dt = self.roms[0].dt

        # Restricted to equal K,L for all OCs
        assert all(self.Ks[0] == x for x in self.Ks)
        assert all(self.Ls[0] == x for x in self.Ls)
        # Restricted to equal dt for all OCs
        assert all(rom.dt == self.dt for rom in self.roms)

        # Set up CNMc elements:

        # Set matching algorithm:
        if 'alignment_algorithm' in kwargs.keys(): 
            self.alignment_algorithm = kwargs['alignment_algorithm']
        else:
            self.alignment_algorithm = trivial_assign

        # Set transition property model:
        self.transition_model = transition_model
        self.transition_model_options = {}
        self.transition_model_options_t = {}

    def train(self, data_train):
        data_encoded = self._transform_data(data_train)
        self._fit_clusters(data_encoded)
        self._fit_croms(data_encoded)
        self._match_centroids()
        self._fit_model_encoder()
        self._fit_models_QT()
        return self

    def _transform_data(self, data_train):
        from .transformation_clustering import train_encoders
        self.encoders = train_encoders(data_train, self.encoders)
        return {oc: self.encoders[oc].encode(data_train[oc]) for oc in data_train.keys()}

    def _fit_clusters(self, data_train):
        from .transformation_clustering import train_clusters
        if 'cluster_centers_' in self.clustering['algorithm'].__dict__.keys():  # precomputed clustering
            print('Using precomputed clusters')
            return None
        else: 
            print('Computing clustering...')
            return train_clusters(self.clustering, data_train, output_csv='clusters.csv')

    def _fit_croms(self, data_train):
        from .transformation_clustering import train_croms
        train_croms(self.roms, data_train, self.encoders, self.clustering)
        self.centroids = [pt.from_numpy(rom._cluster.cluster_centers_) for rom in self.roms]

    def _match_centroids(self) -> None:
        """Solve N_OC-1 linear assignement problems to match centroids across OCs.
        """
        # c_pairs = zip(self.centroids[:-1], self.centroids[1:])
        rom_pairs = zip(self.roms[:-1], self.roms[1:])
        matching = []
        # Solve matching for each centroid pair
        for l, (m1, m2) in enumerate(rom_pairs):
            c1, Q1, p1 = m1._cluster.cluster_centers_, m1.Q, m1._histogram
            c2, Q2, p2 = m2._cluster.cluster_centers_, m2.Q, m2._histogram
            c1, c2 = pt.Tensor(c1), pt.Tensor(c2)
            p1, p2 = pt.Tensor(p1), pt.Tensor(p2)
            Q1, Q2 = custom_dict_to_sparse_coo(Q1, self.shapes[l]), custom_dict_to_sparse_coo(Q1, self.shapes[l+1])
            j1, j2 = self.alignment_algorithm(c1, c2, Q1, Q2)
            matching.append((j1, j2))
            
        self.matching = matching
        reordered_centroids, self.matched_indices = sequential_rearrange_tensors(self.centroids, 
                                                                                 self.matching)
        self.matched_centroids = pt.stack(reordered_centroids)
        # self.matched_centroids = pt.transpose(pt.stack(reordered_centroids),0,1)

    def _fit_model_encoder(self):
        """Train parametric encoder.
        """
        if len(self.ocs) < 2:
            raise ValueError("At least two OCs must be available to interpolate")
        self.encoder_model.train(self.encoders, self.ocs)

    def _fit_models_QT(self):
        """Gets and fits the transition property models. 
        """

        # fit Q model 
        # TODO: do this without making a copy of Q
        Qs = [custom_dict_to_sparse_coo(rom.Q, shp) for rom, shp in zip(self.roms,self.shapes)]
        # reorder centroids and transition matrices
        Qs = [rearrange_tensor_sparse(Q, idxs) for Q, idxs in zip(Qs,self.matched_indices)]
        self.model_Q =  fit_model_P(Qs, self.ocs, 
                                    base_model=self.transition_model, 
                                    base_model_options=self.transition_model_options) 

        # fit T model
        Ts = [dict_to_sparse_coo(rom.T, shp) for rom, shp in zip(self.roms,self.shapes)]
        # reorder centroids and transition matrices
        Ts = [rearrange_tensor_sparse(T, idxs) for T, idxs in zip(Ts,self.matched_indices)]
        self.model_T = fit_model_P(Ts, self.ocs, 
                                   base_model=self.transition_model, 
                                   base_model_options=self.transition_model_options_t) 

    # TODO: order OCs and the rest of the lists
    # TODO: add scaling step in predict!

    # TODO: check if this trans matrix interp can be done once on P then P -> Q,T
    # TODO: check exact constraint on markow matrices: 0 to 1, row-sum constraints, eigenvalues

    # TODO: instead of storing into lists for each OC, store into tensors with a batch dimension
    # TODO: make sense of where dense and sparse tensors are used. SVD is dense only. use also

    def _predict_transition_property(self, model, oc):
        '''Evaluate supervised model of Q or T at unseen OC.
        '''

        V_regressor, nnz_indices, size = model
        values = model_batch_predict(V_regressor, 
                                     np.atleast_2d(np.array(self.ocs)),  
                                     oc.numpy()
                                     )[:,:,0].T # for use with linear and 
        # values = spline_batch_predict(V_regressor, 
        #                               np.array(self.ocs)[:,None], 
        #                               oc.numpy()
        #                               )[:,:,0] # for use with spline and sindy

        # This is an hybrid tensor where last dimension is dense and equal to number of predicted OCs
        Q = pt.sparse_coo_tensor(indices=nnz_indices, values=values, size=size+(values.shape[1],))

        return Q

    def _predict_T(self, oc):
        """Predicts transition time matrix T at an unseen OC.
        """
        
        # get raw prediction
        Ts = self._predict_transition_property(self.model_T, oc).coalesce() 

        assert(Ts.shape[-1]==oc.shape[0]) 

        # kill transitions below time resolution:
        min_dt = self.dt 
        kill_negatives = (Ts.values()<min_dt)
        Ts.values()[kill_negatives] = 0.0

        return Ts.coalesce()

    def _predict_Q(self, oc, Ts):
        """Predicts transition matrix Q at an unseen OC.
        """

        # get raw prediction
        Qs = self._predict_transition_property(self.model_Q, oc).coalesce() 

        assert(Qs.shape[-1]==oc.shape[0]) 

        # apply constraints on stochastic matrix Q
        kill_lt0 = (Qs.values()<0.0)           # kill entries <0
        kill_gt1 = (Qs.values()>1.0)           # kill entries >1
        # kill_tiny = (Qs.values()<1e-4)         # kill transitions whose prob<.01%
        kill_nonpositives = (Ts.values()<=0)   # kill entries w non positive holding time
        Qs.values()[kill_lt0] = 0.0
        Qs.values()[kill_gt1] = 1.0
        # Qs.values()[kill_tiny] = 0.0
        Qs.values()[kill_nonpositives] = 0.0

        sparse_Qs=[]
        sparse_Ts=[]
        for n in range(Qs.shape[-1]): 
            Q = Qs[...,n].coalesce()
            T = Ts[...,n].coalesce()

            # find isolated transitions "loners"
            Q_indices = Q.indices()
            Q_values = Q.values()
            _, inverse, count_unique = pt.unique(Q_indices[1:,:], 
                                                 return_counts=True, return_inverse=True, dim=1)
            loners_unique = (count_unique==1)   # loners among unique indices
            loners = loners_unique[inverse]       # loners among all indices
            naughts = (Q_values<0.5)                 # loners under p=0.5
            kill_loners_naughts = loners * naughts          # mask of loners to be dropped

            # build Q
            Q = pt.sparse_coo_tensor(indices=Q_indices[:,~kill_loners_naughts], 
                                    values=Q_values[~kill_loners_naughts], 
                                    size=Q.size()).coalesce()
            # set self transitions (diagonal) to zero; may or may not be redundant:
            Q = zero_adjacent_equal_indices_sparse(Q.coalesce()) 
            # normalise Q column sum to 1
            Q, kill_zerosums = normalize_sparse_columns(Q)
            
            # build T 
            kill_entries = pt.logical_or(kill_loners_naughts, kill_zerosums)  # drop if either loner or zerosum
            T = pt.sparse_coo_tensor(indices=T.indices()[:,~kill_entries], 
                        values=T.values()[~kill_entries], 
                        size=T.size()).coalesce()

            sparse_Qs.append(Q.coalesce())
            sparse_Ts.append(T.coalesce())
        
        Qs = sparse_Qs
        Ts = sparse_Ts
        # Qs = stack_sparse_tensors_to_hybrid(sparse_Qs)

        return Qs, Ts

    def _predict_QT(self, oc):
        Ts = self._predict_T(oc)
        Qs, Ts = self._predict_Q(oc, Ts)
        return Qs, Ts

    def predict_model(self, oc) -> ROM:
        """Main predictive method that returns CNM model at the queried OCs.

        Args:
            oc (_type_): batch of Operating Conditions.

        Returns:
            ROM: _description_
        """
        
        # predict parametric transformation
        pred_encoder = self.encoder_model.eval(oc)

        # predict Q, T
        Qs, Ts = self._predict_QT(oc)

        # return list of CNM models 
        return [get_CNM(self.matched_centroids[n].numpy(), 
                        Qs[n], Ts[n], 
                        spline_order=self.spline_order, 
                        dt=self.roms[0].dt,
                        encoder=pred_encoder) 
                        for n in range(oc.shape[0])] 

    def predict(self, oc, initial_state: pt.Tensor,
                end_time: float, step_size: float) -> pt.Tensor:
        """Advance initial_state in time at the queried OC.
        """
        cnm = self.predict_model(oc)
        return cnm.predict(initial_state, end_time, step_size)