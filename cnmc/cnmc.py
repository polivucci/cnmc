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
from typing import Dict, Tuple
from collections import defaultdict, deque
from itertools import groupby

# third party packages
import numpy as np
import torch as pt
pt.set_default_dtype(pt.float64)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, MinMaxScaler, minmax_scale

from pysindy.pysindy import SINDy

# Flowtorch packages
from flowtorch.rom.base import ROM

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

    def __init__(self, roms: ROMList, **kwargs) -> None:
        """Inits and fits CNMc to the given observed OC values.

        Args:
            roms (ROMList): contains pre-fitted ROMs at corresponding observed parameters.
        """
        super().__init__()

        self.roms = roms.roms
        self.ocs = roms.ocs
        self.centroids = [pt.from_numpy(rom._cluster.cluster_centers_) for rom in self.roms]
        self.mean_states = [rom.mean_state for rom in self.roms]
        self.Ks = [centroids.shape[0] for centroids in self.centroids]
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

        # Set centroid model:
        self.centroid_model = spline_batch_interpolate
        if 'centroid_model' in kwargs.keys(): 
            if kwargs['centroid_model']=='PiecewiseLinear':
                self.centroid_model = barycentric_batch_interpolate
            elif kwargs['centroid_model']=='Spline':
                self.centroid_model = spline_batch_interpolate
            elif kwargs['centroid_model']=='SINDy':
                self.centroid_model = sindy_batch_regression
            else:
                raise NotImplementedError("Must be one of: PiecewiseLinear, Spline, SINDy")
        self.centroid_model_kwargs = {}
        if 'centroid_model_options' in kwargs.keys(): 
            self.centroid_model_kwargs = kwargs['centroid_model_options']

        # Set transformation model:
        # self.transformation_model = 

        # Set transition property model:
        self.transition_model = None
        self.transition_model_options = {}
        if 'transition_model' in kwargs.keys(): 
            if kwargs['transition_model']=='Linear':
                self.transition_model = LinearRegression
                self.transition_model_options = {}
                self.transition_model_options_t = {}
            elif kwargs['transition_model']=='RF':
                self.transition_model = RandomForestRegressor
                self.transition_model_options = {"n_estimators" : int(1e+2),
                                                 "criterion" : "absolute_error",
                                                 "max_features" : "log2",
                                                 "bootstrap" : True,
                                                 "oob_score" : False,
                                                 "n_jobs" : -1,
                                                 "random_state":0,
                                                 "warm_start" : True,}
            elif kwargs['transition_model']=='Polynomial':
                ############################ 
                ### working settings as in manuscript, 
                ### with centroid_model='PiecewiseLinear' and linear trajectory interpolation in CNM 
                knots = minmax_scale(np.array(self.ocs))
                self.transition_model_options = [MinMaxScaler(), PolynomialFeatures(degree=3), Lasso(1e-2)]  
                # self.transition_model_options = [MinMaxScaler(), SplineTransformer(knots=knots, degree=1), Ridge(1e-5)]
                # self.transition_model_options = [MinMaxScaler(), PolynomialFeatures(degree=3), Ridge(1e-2)]     
                ############################
                #### other experiments:
                # self.transition_model_options = [PolynomialFeatures(degree=5), Lasso(1e-1)]
                # self.transition_model_options = [PolynomialFeatures(degree=len(self.ocs)-1), LinearRegression()]
                # self.transition_model_options = [MinMaxScaler(), SplineTransformer(n_knots=len(self.ocs), degree=3), Ridge(1e-3)]
                self.transition_model_options_t = [MinMaxScaler(), PolynomialFeatures(degree=3), Lasso(1e-2)]  
                # self.transition_model_options_t = [MinMaxScaler(), SplineTransformer(knots=knots, degree=1), Ridge(1e-3)]
                # self.transition_model_options = [SplineTransformer(knots=knots, degree=1), LinearRegression()]
                self.transition_model = make_pipeline
            else:
                raise NotImplementedError("Must be one of: Linear, RF, Polynomial")

        # Fit model
        self.train()

    def train(self):
        self._match_centroids()
        # self._transform_data()
        # self._fit_croms()
        self._fit_model_centroids()
        self._fit_models_QT()
        return self

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
        self.matched_centroids = pt.transpose(pt.stack(reordered_centroids),0,1)

    def _fit_model_centroids(self):
        """Interpolate centroids in phase space.
        """
        if len(self.ocs) < 2:
            raise ValueError("At least two OCs must be available " +
                            "to interpolate")
        pass
        # state_size = self.centroids[0].shape[-1]
        # for centroids in self.matched_centroids:
        #     for dim in range(state_size):
        #         interpolator = InterpolatedUnivariateSpline(
        #             self.ocs, 
        #             centroids[:, dim],
        #             k=1 #min(3, len(self.times)-1)
        #         )
        # self.centroid_interpolator = interpolator

    def _fit_croms(self):

        self.roms = ROMList([{"oc": oc, "rom": cnm} for oc, cnm in models_cnm_train.items()])

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
        kill_tiny = (Qs.values()<1e-4)         # kill transitions whose prob<.01%
        kill_nonpositives = (Ts.values()<=0)   # kill entries w non positive holding time
        Qs.values()[kill_lt0] = 0.0
        Qs.values()[kill_gt1] = 1.0
        Qs.values()[kill_tiny] = 0.0
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
            kill_entries = kill_loners_naughts or kill_zerosums # drop if either loner or zerosum
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
        
        # TODO: centralise ocs normalisation and unify methods

        centroidss, self.centroid_interpolator = self.centroid_model(self.matched_centroids, 
                                                                     self.ocs, 
                                                                     oc, 
                                                                     **self.centroid_model_kwargs)

        # predict Q, T
        Qs, Ts = self._predict_QT(oc)
        
        # return list of CNM models 
        return [get_CNM(centroidss[n].numpy(), Qs[n], Ts[n], 
                        spline_order=self.spline_order, 
                        dt=self.roms[0].dt) 
                        for n in range(oc.shape[0])] 

    def predict(self, oc, initial_state: pt.Tensor,
                end_time: float, step_size: float) -> pt.Tensor:
        """Advance initial_state in time at the queried OC.
        """
        cnm = self.predict_model(oc)
        return cnm.predict(initial_state, end_time, step_size)

    # @property
    # def ocs(self):
    #     return self.ocs

    # @property
    # def cluster_centers(self):
    #     return self.centroids