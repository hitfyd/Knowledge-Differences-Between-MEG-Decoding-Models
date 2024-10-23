# Feature Attribution Explainer Baseclass

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class FAExplainer(ABC):
    """
    Feature Attribution Explainer Baseclass
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize an object.
        """

    @abc.abstractmethod
    def set_params(self, *argv, **kwargs):
        """
        Set parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def attribution(self, *argv, **kwargs):
        """
        计算一个预训练模型对一个特定样本的特征归因图
        """
        raise NotImplementedError

    @abc.abstractmethod
    def joint_attribution(self, *argv, **kwargs):
        """
        针对一个特定样本，同时计算两/多个预训练模型的特征归因图
        """
        raise NotImplementedError
