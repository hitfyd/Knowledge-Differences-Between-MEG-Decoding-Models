import numpy as np
import ray
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from .fsm import FSMethod
from ..engine.utils import predict


class DiffShapleyFS(FSMethod):
    def __init__(self, parallel=False, n_jobs=64):
        super(DiffShapleyFS, self).__init__()
        self.parallel = parallel
        self.n_jobs = n_jobs
        if self.parallel:
            if not ray.is_initialized():
                ray.init(num_gpus=0, num_cpus=self.n_jobs,  # 计算资源
                         local_mode=False,  # 是否启动串行模型，用于调试
                         ignore_reinit_error=True,  # 重复启动不视为错误
                         include_dashboard=False,  # 是否启动仪表盘
                         configure_logging=False,  # 不配置日志
                         log_to_driver=False,  # 日志记录不配置到driver
                         )
        self.method = None
        self.contributions = None
        self.window_length = 250
        self.M = 2

    def fit(self, x: np.ndarray, model1, model2, *args, channels=204, points=250, **kwargs):
        n_samples, _ = x.shape
        if self.parallel:
            feature = diff_shapley_parallel(x.reshape((n_samples, channels, points)), model1, model2,
                                            window_length=self.window_length, M=self.M, NUM_CLASSES=2)
        else:
            feature = diff_shapley(x.reshape((n_samples, channels, points)), model1, model2,
                                   window_length=self.window_length, M=self.M, NUM_CLASSES=2)
        self.contributions = feature.sum(axis=0) / n_samples
        self.contributions = np.abs(self.contributions[:, 0])
        self.contributions = np.repeat(self.contributions, self.window_length)
        print(self.contributions.shape)
        print(self.contributions)

    def computing_contribution(self, *argv, **kwargs):
        return self.contributions

    def transform(self, x: np.ndarray, rate=0.1, *args, **kwargs):
        assert len(x.shape) == 2
        kth = int(len(self.contributions) * rate)
        ind = np.argpartition(self.contributions, kth=-kth)[-kth:]
        threshold = np.min(self.contributions[ind])
        print(kth, threshold)
        return x[:, ind]


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def diff_shapley(data, model1, model2, window_length, M, NUM_CLASSES):
    n_samples, channels, points = data.shape
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)
    S2 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)

    for feature in range(features_num):
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # bool_类型不能改为int8类型
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
            for index in range(n_samples):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index  # 参考样本不能是样本本身
                reference_input = data[reference_index]
                S1[index, feature, m] = S2[index, feature, m] = feature_mark * data[index] + ~feature_mark * reference_input
                S1[index, feature, m][channel_list[feature],
                point_start_list[feature]:point_start_list[feature] + window_length] = \
                    data[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]

    # 计算S1和S2的预测差值
    S1 = S1.reshape(-1, channels, points)
    S2 = S2.reshape(-1, channels, points)
    S1_preds = predict(model1, S1, NUM_CLASSES, eval=True) - predict(model2, S1, NUM_CLASSES, eval=True)
    S2_preds = predict(model1, S2, NUM_CLASSES, eval=True) - predict(model2, S2, NUM_CLASSES, eval=True)
    features = (S1_preds.view(n_samples, features_num, M, -1) -
                S2_preds.view(n_samples, features_num, M, -1)).sum(axis=2) / M
    return features.cpu().detach().numpy()


def diff_shapley_parallel(data, model1, model2, window_length, M, NUM_CLASSES):
    n_samples, channels, points = data.shape
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)
    S2 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)

    @ray.remote
    def run(feature, data_r):
        S1_r = np.zeros((n_samples, M, channels, points), dtype=np.float16)
        S2_r = np.zeros((n_samples, M, channels, points), dtype=np.float16)
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)    # bool_类型不能改为int8类型
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
            for index in range(n_samples):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index # 参考样本不能是样本本身
                reference_input = data_r[reference_index]
                S1_r[index, m] = S2_r[index, m] = feature_mark * data_r[index] + ~feature_mark * reference_input
                S1_r[index, m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                    data_r[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        return feature, S1_r, S2_r

    data_ = ray.put(data)
    rs = [run.remote(feature, data_) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, S1_r, S2_r in rs_list:
        S1[:, feature] = S1_r
        S2[:, feature] = S2_r

    # 计算S1和S2的预测差值
    S1 = S1.reshape(-1, channels, points)
    S2 = S2.reshape(-1, channels, points)
    S1_preds = predict(model1, S1, NUM_CLASSES, eval=True) - predict(model2, S1, NUM_CLASSES, eval=True)
    S2_preds = predict(model1, S2, NUM_CLASSES, eval=True) - predict(model2, S2, NUM_CLASSES, eval=True)
    features = (S1_preds.view(n_samples, features_num, M, -1) -
                S2_preds.view(n_samples, features_num, M, -1)).sum(axis=2) / M
    return features.cpu().detach().numpy()
