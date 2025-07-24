import os
import time

import numpy as np
import torch
from numpy import argmax
from tqdm import tqdm

from similarity.attribution.MEG_Shapley_Values import torch_predict
from .fsm import FSMethod
from ..engine.utils import predict, save_checkpoint, load_checkpoint


def compute_all_sample_feature_maps(dataset: str, data: np.ndarray, model1: torch.nn, model2: torch.nn,
                                    n_classes, window_length, M,
                                    *args, flag=None, device: torch.device = torch.device('cuda'), **kwargs):
    if flag is None:
        save_path = "./feature_maps/"
    else:
        save_path = f"./feature_maps_{flag}/"
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
        # 临时优化
        if model1.__class__.__name__ in ["ATCNet", "NewEEGNetv1"]:
            model1 = torch.compile(model1, mode="max-autotune")
        if model2.__class__.__name__ in ["ATCNet"]:
            model2 = torch.compile(model2, mode="max-autotune")
        if model2.__class__.__name__ in ["EEGNetv4"]:
            model2 = torch.compile(model2, mode="default")
        all_sample_feature_maps = diff_shapley(data, model1, model2, window_length, M, n_classes, device=device, log_file=log_file)
        if not isinstance(all_sample_feature_maps, np.ndarray):
            all_sample_feature_maps = all_sample_feature_maps.detach().cpu().numpy()
        save_checkpoint(all_sample_feature_maps, save_file)

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print("DiffShapley Computation Time ({} {} {}): {:.6f}s".format(dataset, {model1.__class__.__name__}, {model2.__class__.__name__}, run_time))
        # with open(log_file, "a") as writer:
        #     writer.write("DiffShapley Computation Time ({} {} {}): {:.6f}s".format(dataset, {model1.__class__.__name__}, {model2.__class__.__name__}, run_time))
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
            *args, threshold=3, device: torch.device = torch.device('cuda'), **kwargs):

        n_samples, channels, points = x.shape
        # x = x.reshape((n_samples, channels, points))
        assert points % window_length == 0
        self.logit_delta = predict(model1, x, n_classes, eval=True, device=device) - predict(model2, x, n_classes, eval=True, device=device)
        # self.logit_delta = self.logit_delta.cpu().detach().numpy()
        label_logit_delta = abs(self.logit_delta).sum(axis=0)
        if self.logit_delta.shape[1] == 2:
            self.sample_weights = self.logit_delta[:, 0]
        else:
            self.sample_weights = self.logit_delta[:, argmax(label_logit_delta)]

        assert isinstance(all_sample_feature_maps, np.ndarray) or isinstance(all_sample_feature_maps, torch.Tensor)
        self.all_sample_feature_maps = all_sample_feature_maps
        self.contributions = np.average(self.all_sample_feature_maps, axis=0, weights=self.sample_weights)   #
        if self.logit_delta.shape[1] == 2:
            self.contributions = self.contributions[:, 0]
        else:
            self.contributions = self.contributions[:, argmax(label_logit_delta)]
        self.contributions = np.repeat(self.contributions, window_length)
        self.threshold = threshold

    def computing_contribution(self, *argv, **kwargs):
        return self.contributions

    def transform(self, x: np.ndarray, rate=0.1, *args, **kwargs):
        assert len(x.shape) == 2
        mean = self.contributions.mean()
        std = self.contributions.std()
        # print("mean", mean, "std", std)
        z_contributions = (self.contributions-mean) / std   # (self.contributions) / std
        abs_contributions = np.abs(z_contributions)
        condition = (abs_contributions > self.threshold)
        indices = np.where(condition)[0]
        return x[:, indices], indices


def diff_shapley(data, model1, model2, window_length, M, NUM_CLASSES, reference_num=100, batch_size = 1024, device: torch.device = torch.device('cuda'), log_file=None):
    n_samples, channels, points = data.shape
    features_num = (channels * points) // window_length
    data = torch.from_numpy(data).to(device=device)
    all_sample_feature_maps = torch.zeros((n_samples, features_num, NUM_CLASSES), device=device)
    S1 = torch.zeros((features_num, channels, points), device=device, dtype=torch.float32)
    S2 = torch.zeros((features_num, channels, points), device=device, dtype=torch.float32)
    # with open(log_file, "a") as writer:
    #     writer.write("n_samples: {}\n".format(n_samples))

    for index in tqdm(range(n_samples)):
        time_start = time.perf_counter()

        rand_idx = torch.randint(len(data), (reference_num,), device=device)
        reference_dataset = data[rand_idx]
        # 特征归因图
        diff_attribution_maps_all = torch.zeros((M, features_num, NUM_CLASSES), device=device)    # (3, M, features_num, NUM_CLASSES)
        model_attribution_maps_all = torch.zeros((2, M, features_num, NUM_CLASSES), device=device)
        for m in range(M):  # M在外层
            # 批量生成随机参考样本 [features_num, ...]
            rand_idx = torch.randint(reference_num, (features_num,), device=device)
            reference_inputs = reference_dataset[rand_idx]  # [features_num, C, T]

            # 生成包含feature的特征集合 [features_num, features_num]
            feature_marks = torch.rand(features_num, features_num, device=device) > 0.5
            # 确保当前对角线位置始终为True
            feature_marks.fill_diagonal_(True)
            # 将特征标记扩展到时间维度 [features_num, features_num] -> [features_num, C, T]
            with_mask = feature_marks.unsqueeze(-1).repeat(1, 1, window_length).view(features_num, channels, points)

            # 生成不包含feature的特征集合 [features_num, features_num]
            feature_marks.fill_diagonal_(False)
            # 将特征标记扩展到时间维度 [features_num, features_num] -> [features_num, C, T]
            without_mask = feature_marks.unsqueeze(-1).repeat(1, 1, window_length).view(features_num, channels, points)

            # 批量生成S1和S2 [features_num, C, T]
            S1[:] = data[index] * with_mask + reference_inputs * ~with_mask
            S2[:] = data[index] * without_mask + reference_inputs * ~without_mask

            S1_preds_diff = torch_predict(model1, S1, batch_size=batch_size)[0] - torch_predict(model2, S1, batch_size=batch_size)[0]
            S2_preds_diff = torch_predict(model1, S2, batch_size=batch_size)[0] - torch_predict(model2, S2, batch_size=batch_size)[0]
            diff_attribution_maps_all[m] = S1_preds_diff.reshape(features_num, -1) - S2_preds_diff.reshape(features_num, -1)

        all_sample_feature_maps[index] = diff_attribution_maps_all.mean(dim=0)

        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # with open(log_file, "a") as writer:
        #     writer.write("{}\t{:.6f}s\n".format(index, run_time))

    return all_sample_feature_maps
