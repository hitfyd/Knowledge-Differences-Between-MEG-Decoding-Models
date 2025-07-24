# Feature Selection Method

import abc
import sys

import numpy as np

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class FSMethod(ABC):
    """
    FSMethod is the base class for Feature Selection Method (FSM).
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a FSMethod object.
        """

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """
        Fit a feature selection method on data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def computing_contribution(self, *argv, **kwargs):
        """
        Return the contribution of each feature
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, *argv, **kwargs):
        """
        Apply feature selection on the data based on the feature contribution and return the transformed data
        """
        raise NotImplementedError


class NoneFSM(FSMethod):
    def __init__(self):
        super(NoneFSM, self).__init__()
        self.contributions = None

    def fit(self, x: np.ndarray, *argv, **kwargs):
        assert len(x.shape) == 2
        n_samples, n_features = x.shape
        self.contributions = np.zeros(n_features)

    def computing_contribution(self, *argv, **kwargs):
        return self.contributions

    def transform(self, x, *argv, **kwargs):
        assert len(x.shape) == 2
        return x, np.array(range(x.shape[1]))

