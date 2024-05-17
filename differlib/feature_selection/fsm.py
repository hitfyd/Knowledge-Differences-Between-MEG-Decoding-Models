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
    def fit(self, *argv, **kwargs):
        pass

    def computing_contribution(self, *argv, **kwargs):
        pass

    def transform(self, x, *argv, **kwargs):
        return x


class RandFSM(FSMethod):
    def __init__(self):
        super(RandFSM, self).__init__()
        self.contributions = None

    def fit(self, x: np.ndarray, *argv, **kwargs):
        assert len(x.shape) == 2
        n_samples, n_features = x.shape
        self.contributions = np.random.rand(n_features)


    def computing_contribution(self, *argv, **kwargs):
        pass

    def transform(self, x, *argv, rate=0.1, **kwargs):
        assert len(x.shape) == 2
        kth = int(len(self.contributions) * rate)
        ind = np.argpartition(self.contributions, kth=-kth)[-kth:]
        threshold = np.min(self.contributions[ind])
        print(kth, threshold)
        return x[:, ind]
