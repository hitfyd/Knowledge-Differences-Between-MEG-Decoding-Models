from similarity.attribution.Shapley import Shapley


class Joint_Shapley(Shapley):
    """
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize an object.
        """
        super(Shapley, self).__init__()

    def joint_attribution(self, *argv, **kwargs):
        """
        针对一个特定样本，同时计算两/多个预训练模型的特征归因图
        """
        raise NotImplementedError