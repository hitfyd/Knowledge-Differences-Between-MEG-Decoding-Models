import os
import time

import numpy as np
import ray
import torch
from scipy import stats
from tqdm import tqdm

from .fsm import FSMethod
from ..engine.utils import predict, save_checkpoint, load_checkpoint


def compute_all_sample_feature_maps(dataset: str, data: np.ndarray, model1: torch.nn, model2: torch.nn,
                                    n_classes, window_length, M,
                                    *args, parallel=True, num_gpus=1, num_cpus=8, **kwargs):
    save_path = "./feature_maps/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, f"{dataset}_{model1.__class__.__name__}_{model2.__class__.__name__}_{window_length}_{M}")
    save_file_ = os.path.join(save_path, f"{dataset}_{model2.__class__.__name__}_{model1.__class__.__name__}_{window_length}_{M}")
    log_file = os.path.join(save_path, f"{dataset}_{model1.__class__.__name__}_{model2.__class__.__name__}_{window_length}_{M}.log")

    if os.path.exists(save_file):
        all_sample_feature_maps = load_checkpoint(save_file)
        print("feature_maps has been loaded")
    elif os.path.exists(save_file_):
        all_sample_feature_maps = load_checkpoint(save_file_)
        # all_sample_feature_maps = -all_sample_feature_maps
        print("feature_maps has been loaded")
    else:
        time_start = time.perf_counter()
        if parallel:
            if not ray.is_initialized():
                ray.init(num_gpus=num_gpus, num_cpus=num_cpus,  # 计算资源
                         local_mode=False,  # 是否启动串行模型，用于调试
                         ignore_reinit_error=True,  # 重复启动不视为错误
                         include_dashboard=False,  # 是否启动仪表盘
                         configure_logging=False,  # 不配置日志
                         log_to_driver=False,  # 日志记录不配置到driver
                         )
            all_sample_feature_maps = diff_shapley_parallel(data, model1, model2, window_length, M, n_classes,
                                                            num_gpus=num_gpus/num_cpus, log_file=log_file)
        else:
            all_sample_feature_maps = diff_shapley(data, model1, model2, window_length, M, n_classes, log_file=log_file)
        save_checkpoint(all_sample_feature_maps, save_file)

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print("DiffShapley Computation Time ({} {} {}): {:.6f}s".format(dataset, {model1.__class__.__name__}, {model2.__class__.__name__}, run_time))
        with open(log_file, "a") as writer:
            writer.write("DiffShapley Computation Time ({} {} {}): {:.6f}s".format(dataset, {model1.__class__.__name__}, {model2.__class__.__name__}, run_time))
    return all_sample_feature_maps


class DiffShapleyFS(FSMethod):
    def __init__(self):
        super(DiffShapleyFS, self).__init__()
        self.method = None
        self.all_sample_feature_maps = None
        self.contributions = None
        self.logit_delta = None
        self.sample_weights = None
        self.threshold = 3  # 2/3

    def fit(self, x: np.ndarray, model1, model2, channels, points, n_classes, window_length, M, all_sample_feature_maps,
            *args, threshold=3, parallel=True, num_gpus=1, num_cpus=16, **kwargs):
        n_samples, channels, points = x.shape
        # x = x.reshape((n_samples, channels, points))
        assert points % window_length == 0
        self.logit_delta = predict(model1, x, n_classes, eval=True) - predict(model2, x, n_classes, eval=True)
        # self.logit_delta = self.logit_delta.cpu().detach().numpy()
        self.sample_weights = self.logit_delta[:, 0]
        # if parallel:
        #     if not ray.is_initialized():
        #         ray.init(num_gpus=num_gpus, num_cpus=num_cpus,  # 计算资源
        #                  local_mode=False,  # 是否启动串行模型，用于调试
        #                  ignore_reinit_error=True,  # 重复启动不视为错误
        #                  include_dashboard=False,  # 是否启动仪表盘
        #                  configure_logging=False,  # 不配置日志
        #                  log_to_driver=False,  # 日志记录不配置到driver
        #                  )
        #     self.all_sample_feature_maps = diff_shapley_parallel(x, model1, model2, window_length, M, n_classes,
        #                                                          num_gpus=num_gpus/num_cpus)
        # else:
        #     self.all_sample_feature_maps = diff_shapley(x, model1, model2, window_length, M, n_classes)
        self.all_sample_feature_maps = all_sample_feature_maps
        # self.contributions = self.all_sample_feature_maps.mean(axis=0)
        # self.contributions = self.all_sample_feature_maps.sum(axis=0) / n_samples
        self.contributions = np.average(self.all_sample_feature_maps, axis=0, weights=self.sample_weights)
        self.contributions = self.contributions[:, 0]
        self.contributions = np.repeat(self.contributions, window_length)
        # print(self.contributions.shape)
        # print(self.contributions)
        self.threshold = threshold

    def computing_contribution(self, *argv, **kwargs):
        return self.contributions

    def transform(self, x: np.ndarray, rate=0.1, *args, **kwargs):
        assert len(x.shape) == 2
        lmax = stats.yeojohnson_normmax(self.contributions)
        yj_contributions = stats.yeojohnson(self.contributions, lmbda=lmax)
        yj_contributions = self.contributions
        mean = yj_contributions.mean()
        std = yj_contributions.std()
        z_contributions = (yj_contributions) / std
        abs_contributions = np.abs(z_contributions)
        # condition = (z_contributions > self.threshold) | (z_contributions < -self.threshold)
        condition = (abs_contributions > self.threshold)
        indices = np.where(condition)[0]
        return x[:, indices], indices

        # abs_contributions = np.abs(self.contributions)
        # kth = int(len(abs_contributions) * rate)
        # ind = np.argpartition(abs_contributions, kth=-kth)[-kth:]
        # threshold = np.min(abs_contributions[ind])
        # print(kth, threshold)
        # return x[:, ind]


# def feature_segment(channels, points, window_length):
#     channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
#     features_num = channels * channel_windows_num
#     channel_list, point_start_list = [], []
#     for feature_id in range(features_num):
#         channel_list.append(int(feature_id / channel_windows_num))
#         point_start_list.append(int(feature_id % channel_windows_num * window_length))
#     return features_num, channel_list, point_start_list


def diff_shapley(data, model1, model2, window_length, M, NUM_CLASSES, log_file=None):
    n_samples, channels, points = data.shape
    features_num = (channels * points) // window_length
    data = data.reshape((n_samples, channels * points))
    all_sample_feature_maps = np.zeros((n_samples, features_num, NUM_CLASSES))
    with open(log_file, "a") as writer:
        writer.write("n_samples: {}\n".format(n_samples))

    for index in tqdm(range(n_samples)):
        time_start = time.perf_counter()
        S1 = np.zeros((features_num, M, channels * points), dtype=np.float16)
        S2 = np.zeros((features_num, M, channels * points), dtype=np.float16)
        for feature in range(features_num):
            for m in range(M):
                # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # bool_类型不能改为int8类型
                feature_mark[feature] = 0
                feature_mark = np.repeat(feature_mark, window_length)

                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index  # 参考样本不能是样本本身
                S1[feature, m] = S2[feature, m] = feature_mark * data[index] + ~feature_mark * data[reference_index]
                S1[feature, m][feature*window_length:(feature+1)*window_length] = \
                    data[index][feature*window_length:(feature+1)*window_length]

        # 计算S1和S2的预测差值
        S1 = S1.reshape(-1, channels, points)
        S2 = S2.reshape(-1, channels, points)
        S1_preds = predict(model1, S1, NUM_CLASSES, eval=True) - predict(model2, S1, NUM_CLASSES, eval=True)
        S2_preds = predict(model1, S2, NUM_CLASSES, eval=True) - predict(model2, S2, NUM_CLASSES, eval=True)
        sample_feature_maps = (S1_preds.reshape(features_num, M, -1) - S2_preds.reshape(features_num, M, -1)).sum(axis=1) / M

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        with open(log_file, "a") as writer:
            writer.write("{}\t{:.6f}s\n".format(index, run_time))

        all_sample_feature_maps[index] = sample_feature_maps
    return all_sample_feature_maps


def diff_shapley_feature(data, model1, model2, window_length, M, NUM_CLASSES):
    n_samples, n_features = data.shape

    S1 = np.zeros((n_samples, n_features, M, n_features), dtype=np.float16)
    S2 = np.zeros((n_samples, n_features, M, n_features), dtype=np.float16)

    for feature in range(n_features):
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, n_features, dtype=np.bool_)  # bool_类型不能改为int8类型
            feature_mark[feature] = 0
            for index in range(n_samples):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index  # 参考样本不能是样本本身
                reference_input = data[reference_index]
                S1[index, feature, m] = S2[index, feature, m] = feature_mark * data[index] + ~feature_mark * reference_input
                S1[index, feature, m, feature] = data[index, feature]

    # 计算S1和S2的预测差值
    # S1 = S1.reshape(-1, channels, points)
    # S2 = S2.reshape(-1, channels, points)
    # S1_preds = predict(model1, S1, NUM_CLASSES, eval=True) - predict(model2, S1, NUM_CLASSES, eval=True)
    # S2_preds = predict(model1, S2, NUM_CLASSES, eval=True) - predict(model2, S2, NUM_CLASSES, eval=True)
    # features = (S1_preds.view(n_samples, features_num, M, -1) -
    #             S2_preds.view(n_samples, features_num, M, -1)).sum(axis=2) / M
    # return features.cpu().detach().numpy()


# def diff_shapley_parallel(data, model1, model2, window_length, M, NUM_CLASSES):
#     n_samples, channels, points = data.shape
#     features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
#
#     S1 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)
#     S2 = np.zeros((n_samples, features_num, M, channels, points), dtype=np.float16)
#
#     @ray.remote
#     def run(feature, data_r):
#         S1_r = np.zeros((n_samples, M, channels, points), dtype=np.float16)
#         S2_r = np.zeros((n_samples, M, channels, points), dtype=np.float16)
#         for m in range(M):
#             # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
#             feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)    # bool_类型不能改为int8类型
#             feature_mark[feature] = 0
#             feature_mark = np.repeat(feature_mark, window_length)
#             feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
#             for index in range(n_samples):
#                 # 随机选择一个参考样本，用于替换不考虑的特征核
#                 reference_index = (index + np.random.randint(1, n_samples)) % n_samples
#                 assert index != reference_index # 参考样本不能是样本本身
#                 reference_input = data_r[reference_index]
#                 S1_r[index, m] = S2_r[index, m] = feature_mark * data_r[index] + ~feature_mark * reference_input
#                 S1_r[index, m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
#                     data_r[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
#         return feature, S1_r, S2_r
#
#     data_ = ray.put(data)
#     rs = [run.remote(feature, data_) for feature in range(features_num)]
#     rs_list = ray.get(rs)
#     for feature, S1_r, S2_r in rs_list:
#         S1[:, feature] = S1_r
#         S2[:, feature] = S2_r
#
#     # 计算S1和S2的预测差值
#     S1 = S1.reshape(-1, channels, points)
#     S2 = S2.reshape(-1, channels, points)
#     S1_preds = predict(model1, S1, NUM_CLASSES, eval=True) - predict(model2, S1, NUM_CLASSES, eval=True)
#     S2_preds = predict(model1, S2, NUM_CLASSES, eval=True) - predict(model2, S2, NUM_CLASSES, eval=True)
#     features = (S1_preds.view(n_samples, features_num, M, -1) -
#                 S2_preds.view(n_samples, features_num, M, -1)).sum(axis=2) / M
#     return features.cpu().detach().numpy()


def diff_shapley_parallel(data, model1, model2, window_length, M, NUM_CLASSES, num_gpus=0.125, log_file=None):
    n_samples, channels, points = data.shape
    features_num = (channels * points) // window_length
    data = data.reshape((n_samples, channels * points))
    all_sample_feature_maps = np.zeros((n_samples, features_num, NUM_CLASSES))
    with open(log_file, "a") as writer:
        writer.write("n_samples: {}\n".format(n_samples))

    @ray.remote(num_gpus=num_gpus)
    def run(index, data_r, model1_r, model2_r):
        time_start = time.perf_counter()
        S1 = np.zeros((features_num, M, channels * points), dtype=np.float16)
        S2 = np.zeros((features_num, M, channels * points), dtype=np.float16)
        for feature in range(features_num):
            for m in range(M):
                # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # bool_类型不能改为int8类型
                feature_mark[feature] = 0
                feature_mark = np.repeat(feature_mark, window_length)

                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index  # 参考样本不能是样本本身
                S1[feature, m] = S2[feature, m] = feature_mark * data_r[index] + ~feature_mark * data_r[reference_index]
                S1[feature, m][feature * window_length:(feature + 1) * window_length] = \
                    data_r[index][feature * window_length:(feature + 1) * window_length]

        # 计算S1和S2的预测差值
        S1 = S1.reshape(-1, channels, points)
        S2 = S2.reshape(-1, channels, points)
        S1_preds = predict(model1_r, S1, NUM_CLASSES, eval=True) - predict(model2_r, S1, NUM_CLASSES, eval=True)
        S2_preds = predict(model1_r, S2, NUM_CLASSES, eval=True) - predict(model2_r, S2, NUM_CLASSES, eval=True)
        feature_maps = (S1_preds.reshape((features_num, M, -1)) - S2_preds.reshape((features_num, M, -1))).sum(axis=1) / M

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        with open(log_file, "a") as writer:
            writer.write("{}\t{:.6f}s\n".format(index, run_time))

        return index, feature_maps


    data_ = ray.put(data)
    model1_ = ray.put(model1)
    model2_ = ray.put(model2)

    rs = [run.remote(index, data_, model1_, model2_) for index in range(n_samples)]
    for index, sample_feature_maps in tqdm(ray.get(rs), total=n_samples, desc="Processing"):
        all_sample_feature_maps[index] = sample_feature_maps

    return all_sample_feature_maps


def diff_shapley_parallel_features(data, model1, model2, window_length, M, NUM_CLASSES, num_gpus=0.125, log_file=None):
    n_samples, channels, points = data.shape
    features_num = (channels * points) // window_length
    data = data.reshape((n_samples, channels * points))
    all_sample_feature_maps = np.zeros((n_samples, features_num, NUM_CLASSES))
    with open(log_file, "a") as writer:
        writer.write("n_samples: {}\n".format(n_samples))

    data_ = ray.put(data)
    model1_ = ray.put(model1)
    model2_ = ray.put(model2)

    for index in tqdm(range(n_samples)):
        time_start = time.perf_counter()
        sample_feature_maps = np.zeros((features_num, NUM_CLASSES))

        @ray.remote(num_gpus=num_gpus)
        def run(feature, data_r, model1_r, model2_r):
            print(feature)
            S1 = np.zeros((M, channels * points), dtype=np.float16)
            S2 = np.zeros((M, channels * points), dtype=np.float16)
            for m in range(M):
                # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # bool_类型不能改为int8类型
                feature_mark[feature] = 0
                feature_mark = np.repeat(feature_mark, window_length)

                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, n_samples)) % n_samples
                assert index != reference_index  # 参考样本不能是样本本身
                S1[m] = S2[m] = feature_mark * data_r[index] + ~feature_mark * data_r[reference_index]
                S1[m][feature * window_length:(feature + 1) * window_length] = \
                    data_r[index][feature * window_length:(feature + 1) * window_length]

            # 计算S1和S2的预测差值
            S1_preds = predict(model1_r, S1, NUM_CLASSES, eval=True) - predict(model2_r, S1, NUM_CLASSES, eval=True)
            S2_preds = predict(model1_r, S2, NUM_CLASSES, eval=True) - predict(model2_r, S2, NUM_CLASSES, eval=True)
            feature_contribution = (S1_preds - S2_preds).sum(axis=0) / M

            return feature, feature_contribution

        rs = [run.remote(feature, data_, model1_, model2_) for feature in range(features_num)]
        for feature, feature_contribution in tqdm(ray.get(rs), total=n_samples, desc="Processing"):
            sample_feature_maps[feature] = feature_contribution

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        with open(log_file, "a") as writer:
            writer.write("{}\t{:.6f}s\n".format(index, run_time))

        all_sample_feature_maps[index] = sample_feature_maps
    return all_sample_feature_maps
