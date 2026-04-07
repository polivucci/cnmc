from typing import Union
from collections import defaultdict, deque

import numpy as np
import torch as pt

from sklearn.cluster import KMeans
from scipy.interpolate import InterpolatedUnivariateSpline

from flowtorch.rom import CNM
from flowtorch.rom.base import Encoder
from flowtorch.rom.utils import (remove_sequential_duplicates, 
                                 )

class CNM(CNM):
    """Modified Cluster-based network modeling implementation.

    Separates train from init and adds functionality to support CNMc.
    
    Argument reference as in the base class.
    """

    def __init__(self, 
                 encoder: Union[Encoder, None] = None,
                 n_clusters: int = 10, 
                 model_order: int = 1, 
                 dt: float = 1.0, 
                 spline_order: int = 3,
                 cluster_config: dict = {}):
        """Create a new CNM instance
        """

        # model hyperparams:
        self.dt = dt
        self.n_clusters = n_clusters
        self.model_order = model_order
        self.spline_order = spline_order
        self.cluster_config = cluster_config
        self.encoder = encoder

        # initialize params
        self._cluster = KMeans(n_clusters=self.n_clusters)
        self._dtype = None
        self._sequence = None
        self._transition_prob = None
        self._transition_counts = None
        self._transition_time = None

        # initialise prediction vars
        self._times = None
        self._visited_clusters = None
        self._tree = None


    def train(self, reduced_state):
            
            self._check_reduced_state(reduced_state)
            
            cluster = KMeans # default
            if 'algorithm' in self.cluster_config.keys():
                cluster = self.cluster_config.pop('algorithm')

            if 'cluster_centers_' in cluster.__dict__.keys(): # precomputed clustering
                print('Using precomputed clusters')
                cluster.labels_ = cluster.predict(reduced_state.T.numpy())
                self._cluster = cluster
            else:
                self._cluster = cluster(self.n_clusters, **self.cluster_config).fit(
                                        reduced_state.T.numpy()  # batch dimension comes first in Scikit-Learn
                                        )
            
            # self._cluster.labels_ = _nonperiodc_block_removal(self._cluster.labels_)
            self._sequence = remove_sequential_duplicates(self._cluster.labels_)
            self._histogram = self._compute_histogram()
            self._transition_prob, self._transition_counts = self._compute_transition_prob()
            self._transition_time = self._compute_transition_time()

            return self

    def _compute_histogram(self):
        """Computes the probabilities associated with each cluster i.e. the histogram of the
        invariant distribution.

        Returns:
            np.ndarray: array of shape (K,1) that sums to one.
        """
        _, p = np.unique(self._cluster.labels_, return_counts=True)
        m = self._cluster.labels_.shape[0]
        assert(p.sum()==m)
        p = p.astype(np.float64, copy=False) / m
        return p[:,None]
    
    def _compute_transition_prob(self) -> dict[str, np.ndarray]:
        """Compute the transition probabilities between clusters.

        :raises Exception: if the model order is shorter than or equal to the
            cluster sequence found in the initial dataset
        :return: cluster sequence of length `self.model_order` as key and
            transition probabilities for each potential next cluster as value;
            the probabilities are stored as 2D arrays, where the first column
            corresponds to the id of the next cluster and the second column
            contains the associated probability
        :rtype: Dict[str, np.ndarray]
        """

        if self.model_order >= self._sequence.size:
            raise Exception("Could not compute transition probabilities: " +
                            f"length of cluster sequence ({self._sequence.size}) must be higher " +
                            f"than chosen model order ({self.model_order})")

        visited_clusters = deque(
            self._sequence[:self.model_order], self.model_order)
        prob = defaultdict(list)
        _counts = defaultdict(list)
        for next_cluster in self._sequence[self.model_order:]:
            key = ",".join(map(str, visited_clusters))
            prob[key].append(next_cluster)
            _counts[key].append(next_cluster)
            visited_clusters.append(next_cluster)
        for key, next_clusters in prob.items():
            unique, counts = np.unique(next_clusters, return_counts=True)
            prob[key] = np.stack((unique, counts/counts.sum())).T
            float_sum = prob[key].sum(axis=0)[-1]
            err_round = 1.0-float_sum
            max_prob_id = prob[key][:,-1].argmax()
            prob[key][max_prob_id, -1] += err_round
            _counts[key] = np.stack((unique, counts)).T

        return prob, _counts
    
    def _interpolate_trajectory(self, step_size: float) -> pt.Tensor:
        """Add interpolated clusters to overall trajectory.

        :param step_size: time step size at which to place clusters
        :type step_size: float
        :raises ValueError: if the trajectory has fewer than two clusters
        :return: trajectory with interpolated clusters
        :rtype: pt.Tensor
        """
        if len(self.times) < 2:
            raise ValueError("At least two predictions must be available " +
                             "to interpolate a trajectory")
        times = np.arange(
            self.times[0], self.times[-1]+0.5*step_size, step_size)
        if self.encoder is None:
            state_size = self.cluster_centers.shape[-1]
        else:
            state_size = self.encoder.reduced_state_size
        prediction = pt.empty((state_size, times.size), dtype=self._dtype)
        for dim in range(state_size):
            spline = InterpolatedUnivariateSpline(
                self._times, 
                self.cluster_centers[self.visited_clusters][:, dim],
                k=min(self.spline_order, len(self.times)-1)
            )
            prediction[dim, :] = pt.from_numpy(spline(times))
        # # Create Catmull-Rom spline interpolator
        # x = self.cluster_centers[self.visited_clusters]
        # s = CatmullRom(x, alpha=0)
        # # Evaluate the spline on a fine grid
        # t_eval = np.linspace(s.grid[0], s.grid[-1], times.shape[0])
        # prediction = pt.tensor(s.evaluate(t_eval)).T  # shape (dim,len)
        return prediction
    
    @property
    def cluster_centers(self):
        return self._cluster.cluster_centers_

    @cluster_centers.setter
    def cluster_centers(self, value):
        assert len(value.shape)==2
        self._cluster.cluster_centers_ = value