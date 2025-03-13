from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
import torch
from math import floor, ceil
from matplotlib import pyplot as plt, gridspec, colors, colorbar
from matplotlib.collections import LineCollection
from sklearn import metrics
from tqdm import tqdm
from matplotlib import pyplot as plt


class IterationLogger:
    def __init__(self):
        self.records = []

    def log(self, **kwargs):
        """动态记录变量"""
        self.records.append(kwargs)

    def get_df(self):
        """生成带类型推断的DataFrame"""
        return pd.DataFrame(self.records).convert_dtypes()

    def analyze(self):
        """自动分析统计量"""
        df = self.get_df()
        return df.agg(['mean', 'std', 'min', 'max']).T


@dataclass
class DatasetInfo(object):
    # dataset information
    dataset: str
    label_names: list
    channels: int
    points: int
    classes: int


@dataclass
class SampleInfo(object):
    # sample information
    sample_id: int
    origin_input: np.ndarray    # which shape is [channels, points]
    truth_label: int


@dataclass
class PredictionInfo(object):
    # model information and model predictions for the sample
    model_name: str
    predicted: np.ndarray   # which shape is [classes]
    predicted_label: int


@dataclass
class AttributionInfo(object):
    # feature attribution method
    attribution_method: str
    # feature attribution maps, which shape is [channels, points, classes]
    attribution_maps: np.ndarray
    # the run time of feature attribution method
    run_time: float = 0.0


class Record(object):
    def __init__(self, dataset_info: DatasetInfo, sample_info: SampleInfo,
                 prediction_info: PredictionInfo, attribution_info: AttributionInfo):
        self.dataset_info = dataset_info
        self.sample_info = sample_info
        self.prediction_info = prediction_info
        self.attribution_info = attribution_info
        self.record_id = "{}_{}_{}_{}".format(dataset_info.dataset, sample_info.sample_id,
                                              prediction_info.model_name, attribution_info.attribution_method)

class ShapleyValueExplainer(object):
    def __init__(self,
                 dataset_info: DatasetInfo,
                 model: torch.nn.Module or torch.nn.ModuleList,  # model or model list
                 reference_dataset: np.ndarray, # The original reference dataset. ‘reference_num’ reference samples are randomly selected when no reference dataset filter is used
                 reference_num: int = 100, window_length: int = 5, M: int = 256,  # The interpreter hyperparameters
                 reference_filter: bool = False, antithetic_variables: bool = False, # The parameters of the variant algorithm, used for comparative ablation experiments
                 device: torch.device = torch.device('cuda'),   # Hardware Information: 'cpu' or 'cuda', cuda(Nvidia GPU) usually provides tens of times speedup
                 ):
        # input check
        assert len(reference_dataset.shape) == 3
        assert len(reference_dataset) >= reference_num
        assert isinstance(model, torch.nn.Module) or isinstance(model, list)
        # dataset information
        self.dataset = dataset_info.dataset
        self.channels = dataset_info.channels
        self.points = dataset_info.points
        self.classes = dataset_info.classes
        # model information
        # 默认为torch.nn.ModuleList，支持同时计算多个模型在同一个样本上的特征归因
        if isinstance(model, torch.nn.ModuleList):
            self.model = model
        else:
            self.model = torch.nn.ModuleList([model])
        # The original reference dataset
        self.reference_dataset = reference_dataset  # which shape is [*, channels, points]
        # The interpreter hyperparameters
        self.reference_num = reference_num  # n
        self.window_length = window_length  # w
        self.M = M  # m
        # The parameters of the variant algorithm
        self.reference_filter = reference_filter
        self.antithetic_variables = antithetic_variables
        # Automatically generated attribution explainer name
        self.explainer_name = "attribution_{}_{}_{}_{}_{}".format(
            reference_num, window_length, M, reference_filter, antithetic_variables)
        # Hardware Information
        self.device = device

    def __call__(self, origin_input: np.ndarray):
        attribution_maps = feature_attribution_M(self.model, self.classes, origin_input,
                                                 self.reference_dataset,
                                                 self.reference_num, self.window_length, self.M,
                                                 self.reference_filter, self.antithetic_variables,
                                                 self.device)
        return attribution_maps


# 可以认为双层循环，外层循环为特征数量，内侧循环为采样次数（向量实现），由于采样次数往往远小于特征数量且采样数量较小较难达到硬件极限，
# 因此总计算时间与特征数量成正比，几乎与采样数量无关，同等特征数量和采样数量情况下计算时间长
def feature_attribution(model: torch.nn.ModuleList,
                        classes: int,
                        origin_input: np.ndarray,
                        reference_dataset: np.ndarray,
                        reference_num: int = 100,
                        window_length: int = 5,
                        M: int = 256,
                        reference_filter: bool = False,
                        antithetic_variables: bool = False,
                        device: torch.device = torch.device('cuda')):
    # 将输入数据转换为PyTorch张量并转移到GPU
    origin_input = torch.from_numpy(origin_input).float().to(device)
    reference_dataset = torch.from_numpy(reference_dataset).float().to(device)
    # 转移到GPU设备
    model = model.to(device).eval()
    model_num = len(model)

    # 参数校验（保持原有逻辑）
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0
    assert len(reference_dataset) >= reference_num

    # 特征分段（需要确保返回PyTorch张量）
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    channel_list = torch.from_numpy(np.array(channel_list)).to(device)
    point_start_list = torch.from_numpy(np.array(point_start_list)).to(device)

    # 参考数据集处理
    if reference_filter:
        reference_dataset = reference_dataset_filter(origin_input, reference_dataset, model[0], reference_num)
    else:
        rand_idx = torch.randint(len(reference_dataset), (reference_num,), device=device)
        reference_dataset = reference_dataset[rand_idx]

    # 预分配内存（使用PyTorch张量）
    final_M = M * 2 if antithetic_variables else M
    S1 = torch.zeros((final_M, channels, points), device=device)
    S2 = torch.zeros_like(S1)

    # 特征归因图
    attribution_maps = torch.zeros((model_num, channels, points, classes), device=device)

    for feature in tqdm(range(features_num)):
        # 设置当前特征的区域
        ch = channel_list[feature]
        st = point_start_list[feature]
        ed = st + window_length

        # 批量生成随机参考样本 [M, ...]
        rand_idx = torch.randint(reference_num, (M,), device=device)
        reference_inputs = reference_dataset[rand_idx]  # [M, C, T]

        # 生成包含feature的特征集合 [M, features_num]
        feature_marks = torch.rand(M, features_num, device=device) > 0.5
        # 确保当前特征位置始终为True
        feature_marks[:, feature] = True
        # 将特征标记扩展到时间维度 [M, features_num] -> [M, features_num, window_length]
        with_mask = feature_marks.unsqueeze(-1).repeat(1, 1, window_length).view(M, channels, points)

        # 生成不包含feature的特征集合
        # 确保当前特征位置始终为False，双mask方案
        feature_marks[:, feature] = False
        # 将特征标记扩展到时间维度 [M, features_num] -> [M, features_num, window_length]
        without_mask = feature_marks.unsqueeze(-1).repeat(1, 1, window_length).view(M, channels, points)

        # # 双mask变体（单mask克隆+访存修改）
        # without_mask = with_mask.clone()
        # without_mask[:, ch, st:ed] = False

        # 批量生成S1和S2 [M, C, T]，双mask略快于双mask变体（单mask克隆+访存修改）略快于单mask+S2访存修改
        S1 = origin_input * with_mask + reference_inputs * ~with_mask
        S2 = origin_input * without_mask + reference_inputs * ~without_mask # 双mask和双mask变体（单mask克隆+访存修改）方案适用

        # # 单mask+S2访存修改方案
        # S2 = S1.clone()
        # S2[:, ch, st:ed] = reference_inputs[:, ch, st:ed]

        # 处理对抗变量
        if antithetic_variables:
            # 使用反向mask生成对抗样本
            S1_anti = origin_input * ~without_mask + reference_inputs * without_mask
            S2_anti = origin_input * ~with_mask + reference_inputs * with_mask # 双mask和双mask变体（单mask克隆+访存修改）方案适用
            # S2_anti[:, ch, st:ed] = reference_inputs[:, ch, st:ed]    # 单mask+S2访存修改方案适用
            S1 = torch.cat([S1, S1_anti], dim=0)
            S2 = torch.cat([S2, S2_anti], dim=0)

        for model_id in range(model_num):
            model_ = model[model_id]
            with torch.no_grad():
                S1_preds, _ = torch_predict(model_, S1)  # [final_M, classes]
                S2_preds, _ = torch_predict(model_, S2)

            # 计算特征权重 [classes]
            feature_weight = (S1_preds - S2_preds).mean(axis=0)

            # 更新归因图
            attribution_maps[model_id, ch, st:ed] = feature_weight

    return attribution_maps.cpu().numpy()


# 可以认为双层循环，外层循环为采样次数，内侧循环为特征数量（向量实现），由于特征数量较大能够最大发挥硬件性能，
# 因此总计算时间与特征数量和采样数量均成正比，同等特征数量和采样数量情况下计算时间更短
def feature_attribution_M(model: torch.nn.ModuleList,
                          classes: int,
                          origin_input: np.ndarray,
                          reference_dataset: np.ndarray,
                          reference_num: int = 100,
                          window_length: int = 5,
                          M: int = 256,
                          reference_filter: bool = False,
                          antithetic_variables: bool = False,
                          device: torch.device = torch.device('cuda')):
    # 将输入数据转换为PyTorch张量并转移到GPU
    origin_input = torch.from_numpy(origin_input).float().to(device)
    reference_dataset = torch.from_numpy(reference_dataset).float().to(device)
    # 转移到GPU设备
    model = model.to(device).eval()
    model_num = len(model)

    # 参数校验（保持原有逻辑）
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0
    assert len(reference_dataset) >= reference_num

    # 特征分段（需要确保返回PyTorch张量）
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    channel_list = torch.from_numpy(np.array(channel_list)).to(device)
    point_start_list = torch.from_numpy(np.array(point_start_list)).to(device)

    # 参考数据集处理
    if reference_filter:
        reference_dataset = reference_dataset_filter(origin_input, reference_dataset, model[0], reference_num)
    else:
        rand_idx = torch.randint(len(reference_dataset), (reference_num,), device=device)
        reference_dataset = reference_dataset[rand_idx]

    # 预分配内存（使用PyTorch张量）
    final_M = M * 2 if antithetic_variables else M
    S1 = torch.zeros((reference_num, channels, points), device=device)
    S2 = torch.zeros_like(S1)

    # 特征归因图
    attribution_maps_all = torch.zeros((model_num, M, features_num, classes), device=device)

    for m in tqdm(range(M)):   # M在外层
        # 批量生成随机参考样本 [features_num, ...]
        rand_idx = torch.randint(reference_num, (features_num,), device=device)
        reference_inputs = reference_dataset[rand_idx]  # [features_num, C, T]

        # # 低差异序列: Sobol, Owen Scrambled Sobol，无明显改善但大幅增加计算时间
        # sobol = torch.quasirandom.SobolEngine(dimension=features_num, scramble=True)  # scramble=True, Owen Scrambled Sobol
        # feature_marks = sobol.draw(features_num).to(device) > 0.5

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
        S1 = origin_input * with_mask + reference_inputs * ~with_mask
        S2 = origin_input * without_mask + reference_inputs * ~without_mask

        # 处理对偶样本对
        if antithetic_variables:
            # 使用反向mask生成对抗样本
            S1_anti = origin_input * ~without_mask + reference_inputs * without_mask
            S2_anti = origin_input * ~with_mask + reference_inputs * with_mask
            S1 = torch.cat([S1, S1_anti], dim=0)
            S2 = torch.cat([S2, S2_anti], dim=0)

        for model_id in range(model_num):
            model_ = model[model_id]
            with torch.no_grad():
                S1_preds, _ = torch_predict(model_, S1)  # [features_num, classes]
                S2_preds, _ = torch_predict(model_, S2)

            # 计算特征权重 [classes]
            feature_weight = S1_preds - S2_preds
            if antithetic_variables:
                feature_weight = feature_weight.view(2, features_num, -1).mean(dim=0)

            # 更新归因图
            attribution_maps_all[model_id, m] = feature_weight

    attribution_maps = attribution_maps_all.mean(dim=1)
    attribution_maps = attribution_maps.unsqueeze(-2).repeat(1, 1, window_length, 1).view(model_num, channels, points, classes)

    attribution_maps_all = attribution_maps_all.unsqueeze(-2).repeat(1, 1, 1, window_length, 1).view(model_num, M, channels, points, classes)

    return attribution_maps.cpu().numpy(), attribution_maps_all.cpu().numpy()


# 分割特征，按照window_length在时间维度上分割
def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


# 只保留贡献前reserved_percent百分比的特征贡献，其余特征贡献置为0，避免干扰对比；变化趋势小于当前总和的1%？作为截断阈值条件
def contribution_smooth(attribution_maps: np.ndarray, reserved_percents: int=30, percents=1000):
    channels, points, classes = attribution_maps.shape
    print(attribution_maps.reshape(-1, classes).sum(axis=0), np.abs(attribution_maps.reshape(-1, classes)).sum(axis=0))
    attribution_maps_ = attribution_maps.reshape(-1)
    attribution_maps_abs = np.abs(attribution_maps_)
    reserved_maps_ = np.zeros_like(attribution_maps_)

    sort_index = attribution_maps_abs.argsort()[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)
    for reserved_percent in range(1, reserved_percents+1):
        threshold = int(index_num * reserved_percent / percents)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        reserved_points = sort_index[:threshold]
        reserved_maps_[reserved_points] = attribution_maps_[reserved_points]
        reserved_maps = reserved_maps_.reshape(attribution_maps.shape)
        print(reserved_percent, reserved_maps.reshape(-1, classes).sum(axis=0), np.abs(reserved_maps.reshape(-1, classes)).sum(axis=0))
    return reserved_maps


# 加性效率归一化，满足Shapley Values的Efficiency (有效性)：所有参与者的Shapley值之和等于总合作收益，总合作收益（$V_{\text{total}}$）是模型对某样本的预测值 $f(x)$ 与基线期望值 $E[f(x)]$ 之间的差值
def additive_efficient_normalization(predicted_values: np.ndarray, baseline_values: np.ndarray, attribution_maps: np.ndarray):
    classes = len(predicted_values)
    total_values = predicted_values - baseline_values   # 总合作收益
    current_values = attribution_maps.reshape([-1, classes]).sum(axis=0) # 当前的总合作收益
    norm_attribution_maps = attribution_maps * (total_values / current_values)
    return norm_attribution_maps


# predict函数，支持GPU批量处理，需保证model和inputs在同一个硬件上
def torch_predict(model: torch.nn.Module, inputs: torch.Tensor, batch_size=1024):
    device = next(model.parameters()).device
    inputs = inputs.float().to(device)
    """输入形状：[batch_size, channels, time_points]"""
    with torch.no_grad():
        if len(inputs) > batch_size:  # 分批次处理防止OOM
            outputs = []
            for x in torch.split(inputs, batch_size):  # 可调整分块大小
                outputs.append(model(x))
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs = model(inputs)

    # # 临时方案，保证预测概率之和为1
    # if model.__class__.__name__ == 'HGRN':
    #     outputs = torch.exp(outputs)
    # else:
    #     outputs = torch.softmax(outputs, dim=1)

    # 更通用的方案，动态输出类型检测，保证预测概率之和为1，略微增加计算时间
    # 方法1：数值范围检测（优先）
    is_log_space = (outputs < 0).any()  # 存在负值可能是log_softmax
    # 方法2：和值验证（二次验证）
    sum_exp = torch.exp(outputs).sum(dim=1).mean()
    sum_raw = outputs.sum(dim=1).mean()
    # 决策逻辑
    if (-1e-3 < sum_raw - 1.0 < 1e-3):  # 已经是概率
        probs = outputs
    elif (-1e-3 < sum_exp - 1.0 < 1e-3) and is_log_space:
        probs = torch.exp(outputs)
    else:
        probs = torch.softmax(outputs, dim=1)
    # 数值修正确保概率有效性
    probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
    outputs = probs / probs.sum(dim=1, keepdim=True)  # 归一化

    return outputs, outputs.max(1, keepdim=True)[1]


# 单个样本的predict函数，需保证model和one_input在同一个硬件上
def torch_individual_predict(model: torch.nn.Module, one_input: torch.Tensor):
    predictions, predicted_labels = torch_predict(model, one_input.unsqueeze(dim=0))
    return predictions[0], predicted_labels[0]


def reference_dataset_filter(origin_input: torch.Tensor, reference_dataset: torch.Tensor, model: torch.nn.Module,
                             reference_num: int):
    reference_dataset_num = len(reference_dataset)
    device = reference_dataset.device
    assert device == next(model.parameters()).device

    dist = torch.zeros(reference_dataset_num, dtype=torch.float32, device=device)
    origin_pred, origin_pred_label = torch_individual_predict(model, origin_input)

    channels, points = origin_input.shape
    point_slice, channel_slice = points // 2, channels // 2
    for i in range(reference_dataset_num):
        reference = reference_dataset[i]
        temp_dataset = torch.zeros((4, channels, points), dtype=torch.float32, device=device)
        temp_dataset[0, :, :point_slice] = origin_input[:, :point_slice]
        temp_dataset[0, :, point_slice:] = reference[:, point_slice:]
        temp_dataset[1, :, :point_slice] = reference[:, :point_slice]
        temp_dataset[1, :, point_slice:] = origin_input[:, point_slice:]
        temp_dataset[2, :channel_slice, :] = origin_input[:channel_slice, :]
        temp_dataset[2, channel_slice:, :] = reference[channel_slice:, :]
        temp_dataset[3, :channel_slice, :] = reference[:channel_slice, :]
        temp_dataset[3, channel_slice:, :] = origin_input[channel_slice:, :]
        predictions, predicted_labels = torch_predict(model, temp_dataset)
        dist[i] = torch.linalg.norm(predictions.sum(dim=0) - origin_pred)
    sort_index = dist.argsort()
    optimized_reference_dataset = reference_dataset[sort_index[:reference_num]]

    return optimized_reference_dataset


# 在某一预测标签上的删除测试，只关心正值；负值代表抑制，如果同时删除掉负值，则样本在该预测标签上的置信度可能并不会降低
def deletion_test(model: torch.nn.Module, origin_input: np.ndarray, attribution_maps: np.ndarray,
                  deletion_max: int = 30, deletion_step: int = 1, deletion_baseline: float or str = 0, percents=1000):
    device = next(model.parameters()).device
    _, pred_label = torch_individual_predict(model, torch.from_numpy(origin_input).to(device))
    pred_label = pred_label.cpu().numpy()

    attribution_map = attribution_maps[:, :, pred_label]

    if deletion_baseline == 'mean':
        deletion_baseline = origin_input.mean()

    sort_index = attribution_map.reshape(-1).argsort()[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)
    # 根据权重排序依次计算删除1%采样点后的预测置信度
    delete_batch, deletion_percent_list = [], []
    deletion_percent = 0
    while deletion_percent <= deletion_max:
        threshold = int(index_num * deletion_percent / percents)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        delete_points = sort_index[:threshold]
        delete_input = origin_input.copy().reshape(-1)
        delete_input[delete_points] = deletion_baseline
        delete_input = delete_input.reshape(origin_input.shape)

        delete_batch.append(delete_input)
        deletion_percent_list.append(deletion_percent / percents)

        deletion_percent += deletion_step

    # 将delete_batch处理为torch形式，并进行预测
    delete_batch = torch.from_numpy(np.array(delete_batch)).to(device)
    delete_predictions, delete_pred_labels = torch_predict(model, delete_batch)
    delete_predictions, delete_pred_labels = delete_predictions.cpu().numpy(), delete_pred_labels.cpu().numpy()

    auc = metrics.auc(deletion_percent_list, delete_predictions[:, pred_label].squeeze())

    print('Deletion_Test:\tmodel:{}\tpred_label:{}\tauc:{}'.format(model.__class__.__name__, pred_label, auc))

    return deletion_percent_list, delete_predictions, delete_pred_labels, auc


# 差异贡献插入测试
def insertion_test(model: torch.nn, origin_input: np.ndarray, attribution_maps: np.ndarray,
                   insertion_max: int = 30, insertion_step: int = 1, insertion_baseline: float or str = 0, percents=1000):
    device = next(model.parameters()).device
    _, pred_label = torch_individual_predict(model, torch.from_numpy(origin_input).to(device))
    pred_label = pred_label.cpu().numpy()

    attribution_map = attribution_maps[:, :, pred_label]

    if insertion_baseline == 'mean':
        insertion_baseline = origin_input.mean()
    baseline_input = np.full_like(origin_input, insertion_baseline)

    sort_index = attribution_map.reshape(-1).argsort()[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)
    # 根据权重排序依次计算删除1%采样点后的预测置信度
    insertion_batch, insertion_percent_list = [], []
    insertion_percent = 0
    while insertion_percent <= insertion_max:
        threshold = int(index_num * insertion_percent / percents)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        insertion_points = sort_index[:threshold]
        insertion_input = baseline_input.copy().reshape(-1)
        insertion_input[insertion_points] = origin_input.reshape(-1)[insertion_points]
        insertion_input = insertion_input.reshape(origin_input.shape)

        insertion_batch.append(insertion_input)
        insertion_percent_list.append(insertion_percent / percents)

        insertion_percent += insertion_step

    # 将delete_batch处理为torch形式，并进行预测
    insertion_batch = torch.from_numpy(np.array(insertion_batch)).to(device)
    insertion_predictions, insertion_pred_labels = torch_predict(model, insertion_batch)
    insertion_predictions, insertion_pred_labels = insertion_predictions.cpu().numpy(), insertion_pred_labels.cpu().numpy()

    auc = metrics.auc(insertion_percent_list, insertion_predictions[:, pred_label].squeeze())

    print('Insertion_Test:\tmodel:{}\tpred_label:{}\tauc:{}'.format(model.__class__.__name__, pred_label, auc))

    return insertion_percent_list, insertion_predictions, insertion_pred_labels, auc


# 全局归因结果对比，此时删除测试应该包含所有标签。预期差异删除应该导致预测差异减少，共识删除应该导致预测差异变大
def compare_deletion_test(model1: torch.nn, model2: torch.nn, origin_input: np.ndarray, attribution_maps: np.ndarray,
                          deletion_max: int = 30, deletion_step: int = 1, deletion_baseline: float or str = 0, consensus=False, percents=1000):
    device = next(model1.parameters()).device
    assert device == next(model2.parameters()).device

    attribution_map = np.abs(attribution_maps).sum(axis=-1)

    if deletion_baseline == 'mean':
        deletion_baseline = origin_input.mean()

    sort_index = attribution_map.reshape(-1).argsort()  # 差异升序排序，即共识在前
    if not consensus:
        sort_index = sort_index[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)

    # 根据权重排序依次计算删除1%采样点后的预测置信度
    delete_batch, deletion_percent_list = [], []
    deletion_percent = 0
    while deletion_percent <= deletion_max:
        threshold = int(index_num * deletion_percent / percents)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        delete_points = sort_index[:threshold]
        delete_input = origin_input.copy().reshape(-1)
        delete_input[delete_points] = deletion_baseline
        delete_input = delete_input.reshape(origin_input.shape)

        delete_batch.append(delete_input)
        deletion_percent_list.append(deletion_percent / percents)

        deletion_percent += deletion_step

    # 将delete_batch处理为torch形式，并进行预测
    delete_batch = torch.from_numpy(np.array(delete_batch)).to(device)
    delete_predictions1, delete_pred_labels1 = torch_predict(model1, delete_batch)
    delete_predictions2, delete_pred_labels2 = torch_predict(model2, delete_batch)
    delete_predictions = torch.abs(delete_predictions1 - delete_predictions2).sum(dim=-1)   # sum/mean
    delete_pred_labels = (delete_pred_labels1 != delete_pred_labels2)
    delete_predictions, delete_pred_labels = delete_predictions.cpu().numpy(), delete_pred_labels.cpu().numpy()

    auc = metrics.auc(deletion_percent_list, delete_predictions.squeeze())

    print('Compare_Deletion_Test:\tmodel1:{}\tmodel2:{}\tauc_compare:{}'.format(
        model1.__class__.__name__, model2.__class__.__name__, auc))

    return deletion_percent_list, delete_predictions, delete_pred_labels, auc


# 差异贡献插入测试。预期差异插入应该导致预测差异增大，共识插入应该导致预测差异？
def compare_insertion_test(model1: torch.nn, model2: torch.nn, origin_input: np.ndarray, attribution_maps: np.ndarray,
                          insertion_max: int = 30, insertion_step: int = 1, insertion_baseline: float or str = 0, consensus=True, percents=1000):
    device = next(model1.parameters()).device
    assert device == next(model2.parameters()).device

    attribution_map = np.abs(attribution_maps).sum(axis=-1)

    if insertion_baseline == 'mean':
        insertion_baseline = origin_input.mean()
    baseline_input = np.full_like(origin_input, insertion_baseline)

    sort_index = attribution_map.reshape(-1).argsort()  # 差异升序排序，即共识在前
    if not consensus:
        sort_index = sort_index[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)

    # 根据权重排序依次计算删除1%采样点后的预测置信度
    insertion_batch, insertion_percent_list = [], []
    insertion_percent = 0
    while insertion_percent <= insertion_max:
        threshold = int(index_num * insertion_percent / percents)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        insertion_points = sort_index[:threshold]
        insertion_input = baseline_input.copy().reshape(-1)
        insertion_input[insertion_points] = origin_input.reshape(-1)[insertion_points]
        insertion_input = insertion_input.reshape(origin_input.shape)

        insertion_batch.append(insertion_input)
        insertion_percent_list.append(insertion_percent / percents)

        insertion_percent += insertion_step

    # 将delete_batch处理为torch形式，并进行预测
    insertion_batch = torch.from_numpy(np.array(insertion_batch)).to(device)
    predictions1, pred_labels1 = torch_predict(model1, insertion_batch)
    predictions2, pred_labels2 = torch_predict(model2, insertion_batch)
    insertion_predictions = torch.abs(predictions1 - predictions2).sum(dim=-1)   # sum/mean
    insertion_pred_labels = (pred_labels1 != pred_labels2)
    insertion_predictions, insertion_pred_labels = insertion_predictions.cpu().numpy(), insertion_pred_labels.cpu().numpy()

    auc = metrics.auc(insertion_percent_list, insertion_predictions.squeeze())

    print('Compare_Insertion_Test:\tmodel1:{}\tmodel2:{}\tauc_compare:{}'.format(
        model1.__class__.__name__, model2.__class__.__name__, auc))

    return insertion_percent_list, insertion_predictions, insertion_pred_labels, auc


# 可视化归因结果
def generate_plot(sample_info: SampleInfo, attribution_maps: np.ndarray, channels_info: mne.Info,
                  top_channels: int=10,     # 在通道贡献地形图上，显示贡献最高的前top_channels个通道的名称
                  z_score:bool = True,     # 是否将贡献权重进行Z-Score标准化，更明显地突出贡献大小；如果进行了Z-Score标准化，则图片中的数值不代表准确的贡献值，而是贡献趋势
                  time_contribution:bool = True):   # 在时间贡献曲线上，默认（True）纵轴显示的是时间贡献数值；False时，纵轴显示的是原始时间曲线，即原始样本进行通道平均，此时曲线波动剧烈，可能不便于分析
    origin_input = sample_info.origin_input
    origin_time_input = origin_input.mean(axis=0)
    channels, points = origin_input.shape
    title = 'ID: {}   Label: {}'.format(sample_info.sample_id, sample_info.truth_label)

    heatmap = attribution_maps[:, :, sample_info.truth_label]
    heatmap_channel = heatmap.sum(axis=1)
    heatmap_time = heatmap.sum(axis=0)
    if z_score:
        heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap))
        heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))
        heatmap_time = (heatmap_time - np.mean(heatmap_time)) / (np.std(heatmap_time))

    # 计算地形图中需要突出显示的通道及名称，注意：由于在绘制地形图时两两合并为一个位置，需要保证TOP通道的名称一定显示，其余通道对显示第一个通道的名称
    mask_list = np.zeros(channels // 2, dtype=bool)  # 由于通道类型为Grad，在绘制地形图时两两合并为一个位置
    top_channel_index = np.argsort(-heatmap_channel)[:top_channels]
    names_list = []  # 两两合并后对应的通道名称
    for channel_index in range(channels // 2):
        if 2 * channel_index in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index] + '\n')  # 避免显示标记遮挡通道名称
            if 2 * channel_index + 1 in top_channel_index:
                names_list[channel_index] += channels_info.ch_names[2 * channel_index + 1] + '\n\n'
        elif 2 * channel_index + 1 in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index + 1] + '\n')
        else:
            names_list.append(channels_info.ch_names[2 * channel_index])

    # 打印TOP通道及其名称、贡献值
    print("index\tchannel name\tcontribution value")
    id = 0
    for index in top_channel_index:
        print(id, index, channels_info.ch_names[index], heatmap_channel[index])
        id += 1

    fig = plt.figure(figsize=(12, 12))
    gridlayout = gridspec.GridSpec(ncols=48, nrows=12, figure=fig, top=0.92, wspace=None, hspace=0.2)
    axs0 = fig.add_subplot(gridlayout[:, :20])
    axs1 = fig.add_subplot(gridlayout[:9, 20:47])
    axs1_colorbar = fig.add_subplot(gridlayout[2:8, 47])
    axs2 = fig.add_subplot(gridlayout[9:, 24:47])

    fontsize = 16
    linewidth = 2
    # 配色方案
    # 贡献由大到小颜色由深变浅：'plasma' 'viridis'
    # 有浅变深：'summer' 'YlGn' 'YlOrRd'
    # 'Oranges'
    cmap = 'Oranges'
    plt.rcParams['font.size'] = fontsize
    time_xticks = [0, 25, 50, 75, 100]
    time_xticklabels = ['-0.2', '0', '0.2', '0.4', '0.6(s)']

    fig.suptitle(title, y=0.99, fontsize=fontsize)

    # 绘制时间曲线图
    thespan = np.percentile(origin_input, 98)
    xx = np.arange(1, points + 1)

    for channel in range(channels):
        y = origin_input[channel, :] + thespan * (channels - 1 - channel)
        dydx = heatmap[channel, :]

        img_points = np.array([xx, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1), linewidths=(1,))
        lc.set_array(dydx)
        axs0.add_collection(lc)

    axs0.set_xlim([0, points + 1])
    axs0.set_xticks(time_xticks)
    axs0.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs0.set_xlabel('Time', fontsize=fontsize)
    axs0.set_title("(a)Contribution Map", fontsize=fontsize)

    inversechannelnames = []
    for channel in range(channels):
        inversechannelnames.append(channels_info.ch_names[channels - 1 - channel])

    yttics = np.zeros(channels)
    for gi in range(channels):
        yttics[gi] = gi * thespan

    axs0.set_ylim([-thespan, thespan * channels])
    plt.sca(axs0)
    plt.yticks(yttics, inversechannelnames, fontsize=fontsize // 3)

    # 绘制地形图
    # 地形图中TOP通道的显示参数
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=4)
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs1, outlines='head',
                         show=False, names=names_list, mask=mask_list, mask_params=mask_params)
    axs1.set_title("(b)Channel Contribution\n(Topomap)", y=0.9, fontsize=fontsize)
    # 设置颜色条带
    norm = colors.Normalize(vmin=heatmap_channel.min(), vmax=heatmap_channel.max())
    colorbar.ColorbarBase(axs1_colorbar, cmap=cmap, norm=norm)

    # 绘制时间贡献曲线
    xx = np.arange(1, points + 1)
    # 时间贡献曲线的纵轴显示原始信号的时间曲线还是时间贡献曲线（默认是时间贡献曲线，更直观）
    yy = heatmap_time if time_contribution else origin_time_input
    img_points = np.column_stack((xx.ravel(), yy.ravel())).reshape(-1, 1, 2)
    segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=(linewidth + 1,))
    lc.set_array(heatmap_time)
    axs2.patch.set_facecolor('lightgreen')
    axs2.set_title("(c)Time Contribution", fontsize=fontsize)
    axs2.add_collection(lc)
    # 设置x轴
    axs2.set_xticks(time_xticks)
    axs2.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs2.set_xlabel('Time', fontsize=fontsize)
    # 设置y轴
    axs2.set_ylim(yy.min(), yy.max())
    if time_contribution:
        if z_score:
            axs2.set_ylim(floor(yy.min()), ceil(yy.max()))
        axs2.set_ylabel('Contribution', fontsize=fontsize)
    else:
        axs2.set_ylabel('Gradient', fontsize=fontsize)  # 传感器应为梯度计，时间信号曲线的数值表示磁场梯度（即磁场随空间的变化率），单位通常为pT/m；由于原始样本信号可能进行了基线校准和标准化，暂时不显示单位

    plt.show()
    return fig, heatmap, heatmap_channel, heatmap_time


def similar_analysis(map1, map2):
    # map1 = map1 / (map1.sum() + 1e-8)
    # map2 = map2 / (map2.sum() + 1e-8)

    diff_map = map1 - map2

    # # 绝对差值
    # diff_map = np.abs(map1 - map2)
    #
    # # 平方差值
    # diff_map = (map1 - map2) ** 2
    # # 相对差值
    # epsilon = 1e-8
    # diff_map = np.abs(map1 - map2) / (np.abs(map1) + np.abs(map2) + epsilon)

    # import ot
    #
    # def wasserstein_block_diff(A, B, block_size=4):
    #     A = A[:, :, 0]
    #     B = B[:, :, 0]
    #     h, w = A.shape
    #     diff_map = np.zeros_like(A)
    #     for i in range(0, h, block_size):
    #         for j in range(0, w, block_size):
    #             # 提取块
    #             a = A[i:i + block_size, j:j + block_size].flatten()
    #             b = B[i:i + block_size, j:j + block_size].flatten()
    #             # 归一化为概率分布
    #             a = a / (a.sum() + 1e-8)
    #             b = b / (b.sum() + 1e-8)
    #             # 计算成本矩阵（假设位置为网格坐标）
    #             positions = np.array([[x, y] for x in range(block_size) for y in range(block_size)])
    #             M = ot.dist(positions, positions, metric='euclidean')
    #             # 计算Wasserstein距离
    #             W = ot.emd2(a, b, M)
    #             # 填充差异矩阵
    #             diff_map[i:i + block_size, j:j + block_size] = W
    #     return diff_map
    #
    # diff_map_ = wasserstein_block_diff(map1, map2)
    # diff_map = np.zeros_like(map1)
    # for i in range(map1.shape[2]):
    #     diff_map[:, :, i] = diff_map_

    return diff_map
