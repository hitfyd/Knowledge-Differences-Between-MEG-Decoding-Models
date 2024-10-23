# Feature Attribution Similarity Analyzer Baseclass

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class FASAnalyzer(ABC):
    """
    Feature Attribution Similarity Analyzer Baseclass
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
    def analysis(self, *argv, **kwargs):
        """
        分析单一样本的特征归因相似性
        """
        raise NotImplementedError

    @abc.abstractmethod
    def weighted_average_analysis(self, *argv, **kwargs):
        """
        针对一个样本子集，计算平均后的特征归因相似性
        """
        raise NotImplementedError
