import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs

from differlib.engine.utils import log_msg, load_checkpoint, get_data_labels_from_dataset
from differlib.models import scikit_models, torch_models, model_dict


class MEGCounterfactualExplainer:
    def __init__(self, model, lambda_temp=0.1, lambda_spatial=0.05, lambda_frequency=0.05,
                 lambda_dist=0.01, learning_rate=0.001, max_iter=100, stop_target_prob=1.0,
                 connectivity_matrix=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        MEG反事实解释器 (PyTorch实现)

        参数:
        model: 预训练的PyTorch模型
        lambda_temp: 时间平滑约束强度
        lambda_spatial: 空间相关性约束强度
        lambda_dist: 修改距离约束强度
        learning_rate: 优化器学习率
        max_iter: 最大优化迭代次数
        connectivity_matrix: 通道间预期协方差矩阵
        device: 计算设备 (CPU/GPU)
        """
        self.model = model.to(device)
        self.model.eval()  # 设置为评估模式

        self.lambda_temp = lambda_temp
        self.lambda_spatial = lambda_spatial
        self.lambda_frequency = lambda_frequency
        self.lambda_dist = lambda_dist
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.stop_target_prob = stop_target_prob
        self.device = device

        # 默认使用单位矩阵作为连通性矩阵
        if connectivity_matrix is None:
            self.connectivity_matrix = torch.eye(204).to(device)
        else:
            self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32).to(device)

    def generate_counterfactual(self, X, target_class, verbose=True):
        """
        生成MEG反事实解释

        参数:
        X: 原始MEG样本 (204, 100)
        target_class: 目标类别
        verbose: 是否显示优化过程

        返回:
        X_cf: 反事实样本 (204, 100)
        """
        # 转换为PyTorch张量
        X_orig = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=False)

        # 初始化反事实为原始数据的克隆
        X_cf = X_orig.clone().detach().requires_grad_(True)

        # 选择优化器 (Adam通常效果较好)
        optimizer = optim.Adam([X_cf], lr=self.learning_rate)

        # 存储损失历史
        loss_history = []

        # 优化循环
        for i in range(self.max_iter):
            optimizer.zero_grad()

            # 计算损失
            total_loss, loss_components, target_prob = self._compute_loss(X_cf, X_orig, target_class)

            if target_prob > self.stop_target_prob:
                print(f"Iter {i}: Total Loss={total_loss.item():.4f} | "
                      f"Pred Loss={loss_components[0]:.4f} | "
                      f"Dist Loss={loss_components[1]:.4f} | "
                      f"Temp Loss={loss_components[2]:.4f} | "
                      f"Spatial Loss={loss_components[3]:.4f} | "
                      f"Frequency Loss={loss_components[4]:.4f} | "
                      f"Target Prob={target_prob:.4f}")
                break

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 应用值约束 (保持数据在合理范围内)
            with torch.no_grad():
                X_cf.data = torch.clamp(X_cf, X_orig.min(), X_orig.max())  # 假设数据标准化在[-3,3]范围

            # 记录损失
            loss_history.append(loss_components)

            # 打印进度
            if verbose and (i % 25 == 0 or i == self.max_iter - 1):
                print(f"Iter {i}: Total Loss={total_loss.item():.4f} | "
                      f"Pred Loss={loss_components[0]:.4f} | "
                      f"Dist Loss={loss_components[1]:.4f} | "
                      f"Temp Loss={loss_components[2]:.4f} | "
                      f"Spatial Loss={loss_components[3]:.4f} | "
                      f"Frequency Loss={loss_components[4]:.4f} | "
                      f"Target Prob={target_prob:.4f}")

        # 返回反事实数据
        return X_cf.detach().cpu().numpy(), loss_history

    def _compute_loss(self, X_cf, X_orig, target_class):
        """计算总损失函数"""
        # 预测损失 (鼓励模型预测目标类别)
        pred = self.model(X_cf.unsqueeze(0))
        target_prob = pred[0, target_class].item()
        pred_loss = 1 - pred[0, target_class]  # 最大化目标类概率

        # 距离损失 (最小化修改量)
        dist_loss = torch.mean((X_cf - X_orig) ** 2)

        # 时间平滑约束
        temp_penalty = self._temporal_smoothness_constraint(X_cf)

        # 空间相关性约束
        spatial_penalty = self._spatial_constraint(X_cf)

        # 频域约束
        frequency_penalty = self._frequency_domain_constraint(X_cf, X_orig)

        # 总损失
        total_loss = (pred_loss +
                      self.lambda_dist * dist_loss +
                      self.lambda_temp * temp_penalty +
                      self.lambda_spatial * spatial_penalty +
                      self.lambda_frequency * frequency_penalty)

        # 返回损失组件用于记录
        loss_components = (pred_loss.item(), dist_loss.item(),
                           temp_penalty.item(), spatial_penalty.item(), frequency_penalty.item())

        return total_loss, loss_components, target_prob

    def _attention_guided_loss(self, X_cf, X_orig, model):
        # 获取模型注意力图
        attention_map = self.get_model_attention(model, X_orig)

        # 在注意力高的区域允许更大修改
        weighted_diff = attention_map * (X_cf - X_orig)
        dist_loss = torch.mean(weighted_diff ** 2)
        return dist_loss

    def _temporal_smoothness_constraint(self, X):
        """时间平滑约束 (惩罚时间点间的不连续变化)"""
        # 计算时间差分 (沿时间维度)
        time_diffs = torch.diff(X, dim=1)

        # 惩罚大的时间变化
        penalty = torch.mean(time_diffs ** 2)

        # 添加对通道间时间模式一致性的惩罚
        channel_vars = torch.var(X, dim=0)  # 每个时间点上的通道方差
        penalty += 0.1 * torch.mean(channel_vars)  # 鼓励时间模式一致性

        return penalty

    def _spatial_constraint(self, X):
        """空间相关性约束 (保持通道间的空间相关性)"""
        # 计算当前样本的协方差矩阵
        X_centered = X - torch.mean(X, dim=1, keepdim=True)
        cov_matrix = torch.mm(X_centered, X_centered.t()) / (X.shape[1] - 1)

        # 计算与预期协方差矩阵的差异
        diff = cov_matrix - self.connectivity_matrix

        # 使用Frobenius范数作为惩罚
        penalty = torch.norm(diff, p='fro')

        return penalty

    def _frequency_domain_constraint(self, X_cf, X_orig):
        # 计算原始和反事实的频谱
        orig_spectrum = torch.fft.rfft(X_orig, dim=1).abs()
        cf_spectrum = torch.fft.rfft(X_cf, dim=1).abs()

        # 惩罚主要频带的变化
        alpha_band = slice(1, 45)  # Alpha频带
        penalty = torch.mean((cf_spectrum[:, alpha_band] - orig_spectrum[:, alpha_band]) ** 2)
        return penalty

    def visualize_differences(self, original, counterfactual, ch_names=None):
        """可视化原始数据与反事实的差异"""
        diff = counterfactual - original

        # 1. 总体差异热图
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(diff, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Difference')
        plt.title('Channel-Time Differences')
        plt.xlabel('Time Points')
        plt.ylabel('Channels')
        if ch_names:
            plt.yticks(range(len(ch_names)), ch_names, fontsize=8)

        # 2. 按通道的差异幅度
        plt.subplot(3, 1, 2)
        channel_diffs = np.mean(np.abs(diff), axis=1)
        plt.bar(range(len(channel_diffs)), channel_diffs)
        plt.title('Average Absolute Difference per Channel')
        plt.xlabel('Channel Index')
        plt.ylabel('Average Difference')
        if ch_names:
            plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=8)

        # 3. 按时间点的差异幅度
        plt.subplot(3, 1, 3)
        time_diffs = np.mean(np.abs(diff), axis=0)
        plt.plot(time_diffs)
        plt.title('Average Absolute Difference over Time')
        plt.xlabel('Time Points')
        plt.ylabel('Average Difference')

        plt.tight_layout()
        plt.show()

        # 4. 显示修改最大的通道
        max_diff_channel = np.argmax(channel_diffs)
        plt.figure(figsize=(12, 6))
        plt.plot(original[max_diff_channel], label='Original')
        plt.plot(counterfactual[max_diff_channel], label='Counterfactual')
        plt.title(f'Channel {max_diff_channel} Comparison ({ch_names[max_diff_channel] if ch_names else ""})')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

        return diff


def load_pretrained_model(model_type):
    print(log_msg("Loading model {}".format(model_type), "INFO"))
    model_class, model_pretrain_path = model_dict[dataset][model_type]
    assert (model_pretrain_path is not None), "no pretrain model {}".format(model_type)
    pretrained_model = None
    if model_type in scikit_models:
        pretrained_model = load_checkpoint(model_pretrain_path)
    elif model_type in torch_models:
        pretrained_model = model_class(channels=channels, points=points, num_classes=n_classes)
        pretrained_model.load_state_dict(load_checkpoint(model_pretrain_path))
        pretrained_model = pretrained_model.cuda()
    else:
        print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
    assert pretrained_model is not None
    return pretrained_model


# 示例使用
if __name__ == "__main__":
    # 1. 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # dataset = "CamCAN"
    # channels, points, n_classes = 204, 100, 2
    # sfreq, fmin, fmax = 125, 1, 45

    dataset = "DecMeg2014"
    channels, points, n_classes = 204, 250, 2
    sfreq, fmin, fmax = 250, 0.1, 20

    model = load_pretrained_model("atcnet")  # 实际使用时替换
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))

    meg_data = test_data

    connectivity_matrix_file = f"{dataset}_connectivity_matrix.npy"
    if os.path.exists(connectivity_matrix_file):
        connectivity_matrix = np.load(connectivity_matrix_file)
    else:
        # 计算连通性矩阵 (使用PLV方法)
        print("Calculating connectivity matrix...")
        con = spectral_connectivity_epochs(
            meg_data,
            sfreq=sfreq,
            method='plv',
            fmin=fmin,
            fmax=fmax,
            faverage=True
        )
        connectivity_matrix = con.get_data(output='dense')[:, :, 0]
        np.save(connectivity_matrix_file, connectivity_matrix)

    # 5. 创建解释器
    explainer = MEGCounterfactualExplainer(
        model,
        lambda_temp=0.05,
        lambda_spatial=0.05,
        lambda_frequency=0.05,
        lambda_dist=0.2,
        learning_rate=0.001,
        max_iter=300,
        stop_target_prob=1.0,
        connectivity_matrix=connectivity_matrix,
        device=device,
    )

    X_samples, cf_samples, diffs = np.zeros_like(meg_data), np.zeros_like(meg_data), np.zeros_like(meg_data)
    # 6. 选择一个样本进行解释
    for sample_idx, X_sample in enumerate(meg_data):
        # 获取原始预测
        with torch.no_grad():
            sample_tensor = torch.tensor(X_sample, dtype=torch.float32, device=device).unsqueeze(0)
            orig_pred = model(sample_tensor)
            orig_class = torch.argmax(orig_pred).item()
            target_class = 1 - orig_class  # 切换到相反的类别

        print(f"\nOriginal prediction: Class {orig_class} with prob {torch.softmax(orig_pred, dim=1)[0, orig_class].item():.4f}")
        print(f"Sample_idx: {sample_idx} Target class: {target_class}")

        # 7. 生成反事实
        cf_sample, loss_history = explainer.generate_counterfactual(X_sample, target_class)

        # 8. 验证反事实预测
        with torch.no_grad():
            cf_tensor = torch.tensor(cf_sample, dtype=torch.float32, device=device).unsqueeze(0)
            cf_pred = model(cf_tensor)
            cf_class = torch.argmax(cf_pred).item()
            cf_prob = torch.softmax(cf_pred, dim=1)[0, target_class].item()

        print(f"\nCounterfactual prediction: Class {cf_class} with target prob {cf_prob:.4f}")

        # # 9. 可视化结果
        # # 创建模拟通道名称
        # if sample_idx % 100 == 0:
        #     ch_names = [f"CH{i:03d}" for i in range(204)]
        #     diff = explainer.visualize_differences(X_sample, cf_sample, ch_names)

        # # 10. 绘制损失曲线
        # loss_history = np.array(loss_history)
        # plt.figure(figsize=(12, 6))
        # plt.plot(loss_history[:, 0], label='Prediction Loss')
        # plt.plot(loss_history[:, 1], label='Distance Loss')
        # plt.plot(loss_history[:, 2], label='Temporal Loss')
        # plt.plot(loss_history[:, 3], label='Spatial Loss')
        # plt.plot(np.sum(loss_history, axis=1), label='Total Loss', linewidth=2, linestyle='--')
        # plt.title('Loss Components During Optimization')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        X_samples[sample_idx] = X_sample
        cf_samples[sample_idx] = cf_sample
        # diffs[sample_idx] = diff

    # 11. 保存结果
    # np.save('original_sample.npy', X_samples)
    np.save(f'{dataset}_counterfactual_sample.npy', cf_samples)
    # np.save('difference.npy', diffs)
    print("Results saved to files.")