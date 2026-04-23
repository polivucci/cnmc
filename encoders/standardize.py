import torch as pt
from flowtorch.rom.base import Encoder
from .procrustes import centre_and_standardize

class MaxStdEncoder(Encoder):

    def __init__(self):
        """Derived class constructor.

        :param rank: rank to truncate the SVD
        :type rank: int, optional
        """
        super(MaxStdEncoder, self).__init__()
        self._state_size = None
        self.mean_ = None
        self.scale_ = None

    def train(self, state: pt.Tensor) -> dict:
        self._state_size = state.shape[0]

        # self.mean_ = data.mean(dim=-1, keepdims=True)
        # self.scale_ = data.std(dim=-1).max().item()

        # learn empirical frame of target state (centre -> scale -> principal components)
        state = state.detach()
        _, self.mean_, self.scale_ = centre_and_standardize(state)

        self.trained = True
        return self

    def encode(self, full_state: pt.Tensor) -> pt.Tensor:
        """Scale with standard deviation.
        """
        self._check_state_shape(full_state.shape)
        if not self.trained:
            raise Exception("Encoding not possible: the encoder has not been trained")
        return (full_state - self.mean_) / self.scale_
        # return self.mean_ + (full_state - self.mean_) / self.scale_

    def decode(self, reduced_state: pt.Tensor) -> pt.Tensor:
        """Unscale with std.
        """
        self._check_reduced_state_size(reduced_state.shape)
        if not self.trained:
            raise Exception("Decoding not possible: the encoder has not been trained")
        # return self.mean_ + (reduced_state - self.mean_) * self.scale_
        return self.mean_ + reduced_state * self.scale_

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
