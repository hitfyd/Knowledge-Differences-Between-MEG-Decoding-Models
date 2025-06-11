import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs
from sklearn.metrics import pairwise_distances
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnx2torch import convert

from differlib.engine.utils import get_data_labels_from_dataset, log_msg, load_checkpoint
from differlib.models import scikit_models, torch_models, model_dict, CuMLWrapper, load_pretrained_model
from similarity.attribution.MEG_Shapley_Values import torch_predict


class DualMEGCounterfactualExplainer:
    def __init__(self, model1, model2, lambda_temp=0.1, lambda_spatial=0.05,
                 lambda_dist=0.01, lambda_frequency=0.5, learning_rate=0.1,
                 max_iter=500, connectivity_matrix=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        双模型MEG反事实解释器

        参数:
        model1, model2: 预训练的PyTorch模型
        lambda_temp: 时间平滑约束强度
        lambda_spatial: 空间相关性约束强度
        lambda_dist: 修改距离约束强度
        lambda_frequency: 频域相关性约束强度
        learning_rate: 优化器学习率
        max_iter: 最大优化迭代次数
        device: 计算设备 (CPU/GPU)
        """
        self.model1 = model1.to(device).eval()
        self.model2 = model2.to(device).eval()
        self.lambda_temp = lambda_temp
        self.lambda_spatial = lambda_spatial
        self.lambda_dist = lambda_dist
        self.lambda_frequency = lambda_frequency
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.device = device

        # 默认使用单位矩阵作为连通性矩阵
        if connectivity_matrix is None:
            self.connectivity_matrix = torch.eye(204).to(device)
        else:
            self.connectivity_matrix = torch.tensor(connectivity_matrix, dtype=torch.float32).to(device)

    def generate_counterfactual(self, X, mode='auto', target_model=None, verbose=True):
        """
        生成MEG反事实解释

        参数:
        X: 原始MEG样本 (204, 100)
        mode: 反事实模式 ('auto', 'different', 'same', 'flip_one')
        target_model: 当mode='flip_one'时指定要翻转的模型 (1或2)
        verbose: 是否显示优化过程

        返回:
        X_cf: 反事实样本 (204, 100)
        """
        # 转换为PyTorch张量
        X_orig = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=False)

        # 获取原始预测
        with torch.no_grad():
            orig_pred1 = self.model1(X_orig.unsqueeze(0))
            orig_pred2 = self.model2(X_orig.unsqueeze(0))
            orig_class1 = torch.argmax(orig_pred1, dim=1).item()
            orig_class2 = torch.argmax(orig_pred2, dim=1).item()

        # 确定反事实模式
        if mode == 'auto':
            if orig_class1 == orig_class2:
                mode = 'different'  # 原始一致，使其不一致
            else:
                mode = 'same'  # 原始不一致，使其一致

        # 初始化反事实
        X_cf = X_orig.clone().detach().requires_grad_(True).contiguous()

        # 选择优化器
        optimizer = optim.Adam([X_cf], lr=self.learning_rate)

        # 存储损失历史
        loss_history = []

        # 优化循环
        for i in range(self.max_iter):
            optimizer.zero_grad()

            # 计算损失
            total_loss, loss_components = self._compute_loss(
                X_cf, X_orig, orig_class1, orig_class2, mode, target_model
            )

            # 反向传播
            total_loss.backward()
            # # 应用特征权重调整梯度
            # with torch.no_grad():
            #     # 高权重特征应更难修改 - 减小梯度
            #     # 低权重特征应更容易修改 - 增大梯度
            #     gradient_adjustment = 1.0 / (1.0 + weights_tensor)
            #     X_cf.grad *= gradient_adjustment
            optimizer.step()

            # 应用值约束
            with torch.no_grad():
                X_cf.data = torch.clamp(X_cf, -3, 3)  # 假设数据标准化在[-3,3]范围

            # 记录损失
            loss_history.append(loss_components)

            # 打印进度
            if verbose and (i % 10 == 0 or i == self.max_iter - 1):
                with torch.no_grad():
                    pred1 = self.model1(X_cf.unsqueeze(0))
                    pred2 = self.model2(X_cf.unsqueeze(0))
                    class1 = torch.argmax(pred1, dim=1).item()
                    class2 = torch.argmax(pred2, dim=1).item()

                print(f"Iter {i}: Total Loss={total_loss.item():.4f} | "
                      f"Pred Loss={loss_components[0]:.4f} | "
                      f"Dist Loss={loss_components[1]:.4f} | "
                      f"Temp Loss={loss_components[2]:.4f} | "
                      f"Spatial Loss={loss_components[3]:.4f} | "
                      f"Classes: {class1} vs {class2}")

                if orig_class1 != class1:
                    break

        # 返回反事实数据和原始预测信息
        result = {
            'counterfactual': X_cf.detach().cpu().numpy(),
            'original_classes': (orig_class1, orig_class2),
            'counterfactual_classes': (class1, class2),
            'mode': mode,
            'loss_history': loss_history
        }

        return result

    def _compute_loss(self, X_cf, X_orig, orig_class1, orig_class2, mode, target_model=None):
        """计算总损失函数"""
        # 获取当前预测
        pred1 = self.model1(X_cf.unsqueeze(0))
        pred2 = self.model2(X_cf.unsqueeze(0))

        # 根据模式计算预测损失
        pred_loss = self._prediction_loss(
            pred1, pred2, orig_class1, orig_class2, mode, target_model
        )

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

        return total_loss, loss_components

    def _prediction_loss(self, pred1, pred2, orig_class1, orig_class2, mode, target_model=None):
        """根据选择的模式计算预测损失"""
        if mode == 'different':
            # 使两个模型的预测不同
            # 损失 = 鼓励两个模型预测相同类别的惩罚
            prob_same = torch.abs(pred1[:, orig_class1] - pred2[:, orig_class2])
            loss = prob_same.mean()

        elif mode == 'same':
            # 使两个模型的预测相同
            # 损失 = 1 - 两个模型预测相同类别的概率
            # 找到最可能达成一致的类别
            if orig_class1 == orig_class2:
                target_class = orig_class1
            else:
                # 选择原始概率更高的类别
                if pred1[0, orig_class1] > pred2[0, orig_class2]:
                    target_class = orig_class1
                else:
                    target_class = orig_class2

            # 鼓励两个模型都预测为该类别
            loss = (1 - pred1[0, target_class]) + (1 - pred2[0, target_class])

        elif mode == 'flip_one':
            # 翻转一个模型的预测，保持另一个不变
            if target_model == 1 or target_model is None:
                # 翻转模型1，保持模型2不变
                loss = (1 - pred1[0, 1 - orig_class1]) + torch.abs(pred2[0, orig_class2] - 0.9)
            else:
                # 翻转模型2，保持模型1不变
                loss = (1 - pred2[0, 1 - orig_class2]) + torch.abs(pred1[0, orig_class1] - 0.9)

        else:
            raise ValueError(f"未知模式: {mode}")

        return loss

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

    def compute_spatial_connectivity(self, X):
        """计算空间连通性矩阵"""
        # 转换为numpy以使用scikit-learn
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X

        # 计算通道间的相关性
        connectivity = 1 - pairwise_distances(X_np, metric='correlation')

        # 将对角线设为零
        np.fill_diagonal(connectivity, 0)

        return connectivity

    def _frequency_domain_constraint(self, X_cf, X_orig):
        # 计算原始和反事实的频谱
        orig_spectrum = torch.fft.rfft(X_orig, dim=1).abs()
        cf_spectrum = torch.fft.rfft(X_cf, dim=1).abs()

        # 惩罚主要频带的变化
        alpha_band = slice(1, 45)  # Alpha频带
        penalty = torch.mean((cf_spectrum[:, alpha_band] - orig_spectrum[:, alpha_band]) ** 2)
        return penalty

    def visualize_results(self, original, counterfactual, orig_classes, cf_classes, ch_names=None):
        """可视化原始数据与反事实的结果"""
        # 计算差异
        diff = counterfactual - original

        # 创建图表
        fig = plt.figure(figsize=(18, 12))

        # 1. 总体差异热图
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        im = ax1.imshow(diff, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax1, label='Difference')
        ax1.set_title('Channel-Time Differences')
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('Channels')
        if ch_names:
            ax1.set_yticks(range(len(ch_names)))
            ax1.set_yticklabels(ch_names, fontsize=6)

        # 2. 按通道的差异幅度
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        channel_diffs = np.mean(np.abs(diff), axis=1)
        top_channels = np.argsort(channel_diffs)[-10:]  # 显示差异最大的10个通道
        ax2.barh(range(10), channel_diffs[top_channels])
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([ch_names[i] for i in top_channels] if ch_names else top_channels)
        ax2.set_title('Top 10 Changed Channels')
        ax2.set_xlabel('Average Absolute Difference')

        # 3. 按时间点的差异幅度
        ax3 = plt.subplot2grid((3, 3), (1, 2))
        time_diffs = np.mean(np.abs(diff), axis=0)
        ax3.plot(time_diffs)
        ax3.set_title('Difference over Time')
        ax3.set_xlabel('Time Points')
        ax3.set_ylabel('Average Difference')

        # 4. 原始和反事实的连通性
        ax4 = plt.subplot2grid((3, 3), (2, 0))
        orig_conn = self.compute_spatial_connectivity(original)
        ax4.imshow(orig_conn, cmap='viridis')
        ax4.set_title(f'Original Connectivity (Classes: {orig_classes[0]}, {orig_classes[1]})')

        ax5 = plt.subplot2grid((3, 3), (2, 1))
        cf_conn = self.compute_spatial_connectivity(counterfactual)
        ax5.imshow(cf_conn, cmap='viridis')
        ax5.set_title(f'Counterfactual Connectivity (Classes: {cf_classes[0]}, {cf_classes[1]})')

        ax6 = plt.subplot2grid((3, 3), (2, 2))
        conn_diff = cf_conn - orig_conn
        ax6.imshow(conn_diff, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        ax6.set_title('Connectivity Difference')

        plt.tight_layout()
        plt.show()

        # 5. 显示修改最大的通道
        max_diff_channel = np.argmax(channel_diffs)
        plt.figure(figsize=(12, 6))
        plt.plot(original[max_diff_channel], label='Original')
        plt.plot(counterfactual[max_diff_channel], label='Counterfactual')
        plt.title(f'Channel {max_diff_channel} Comparison ({ch_names[max_diff_channel] if ch_names else ""})')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

        return fig


# 示例使用
if __name__ == "__main__":
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # dataset = "CamCAN"
    # channels, points, n_classes = 204, 100, 2
    # sfreq, fmin, fmax = 125, 1, 45

    dataset = "DecMeg2014"
    channels, points, n_classes = 204, 250, 2
    sfreq, fmin, fmax = 250, 0.1, 20

    model1 = load_pretrained_model("lr", dataset, channels, points, n_classes, device)  # 实际使用时替换，优先翻转PyTorch模型的预测结果
    model2 = load_pretrained_model("rf", dataset, channels, points, n_classes, device)
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
    explainer = DualMEGCounterfactualExplainer(
        model1,
        model2,
        lambda_temp=0.05,
        lambda_spatial=0.05,
        lambda_frequency=0.05,
        lambda_dist=0.2,
        learning_rate=0.003,
        max_iter=300,
        connectivity_matrix=connectivity_matrix,
        device=device
    )

    X_samples, cf_samples, diffs = np.zeros_like(meg_data), np.zeros_like(meg_data), np.zeros_like(meg_data)
    # 6. 选择一个样本进行解释
    for sample_idx, X_sample in enumerate(meg_data):
        # 获取原始预测
        with torch.no_grad():
            sample_tensor = torch.tensor(X_sample, dtype=torch.float32, device=device).unsqueeze(0)
            orig_pred1 = model1(sample_tensor)
            orig_pred2 = model2(sample_tensor)
            orig_class1 = torch.argmax(orig_pred1).item()
            orig_class2 = torch.argmax(orig_pred2).item()

        print(f"\n{sample_idx} 原始预测: 模型1={orig_class1}, 模型2={orig_class2}")

        # # 7. 根据原始预测选择模式
        # if orig_class1 == orig_class2:
        #     print("预测一致，生成不一致的反事实")
        #     mode = 'different'
        # else:
        #     print("预测不一致，生成一致的反事实")
        #     mode = 'same'

        # # 8. 生成反事实
        # result = explainer.generate_counterfactual(X_sample, mode=mode)
        # cf_sample = result['counterfactual']
        # cf_classes = result['counterfactual_classes']
        #
        # print(f"\n反事实预测: 模型1={cf_classes[0]}, 模型2={cf_classes[1]}")
        #
        # # 9. 可视化结果
        # # 创建模拟通道名称
        # ch_names = [f"CH{i:03d}" for i in range(204)]
        # explainer.visualize_results(
        #     X_sample, cf_sample,
        #     (orig_class1, orig_class2),
        #     cf_classes,
        #     ch_names
        # )

        # 10. 生成翻转一个模型的反事实
        print("\n生成只翻转模型1的反事实")
        result_flip = explainer.generate_counterfactual(
            X_sample, mode='flip_one', target_model=1
        )
        cf_flip = result_flip['counterfactual']
        cf_flip_classes = result_flip['counterfactual_classes']

        print(f"原始预测: 模型1={orig_class1}, 模型2={orig_class2}")
        print(f"反事实预测: 模型1={cf_flip_classes[0]}, 模型2={cf_flip_classes[1]}")

        # # 11. 可视化翻转结果
        # explainer.visualize_results(
        #     X_sample, cf_flip,
        #     (orig_class1, orig_class2),
        #     cf_flip_classes,
        #     ch_names
        # )

        X_samples[sample_idx] = X_sample
        cf_samples[sample_idx] = cf_flip
        # diffs[sample_idx] = diff

    # 11. 保存结果
    # np.save('original_sample.npy', X_samples)
    np.save(f'{dataset}_{model1.__class__.__name__}_{model2.__class__.__name__}_counterfactual_sample.npy', cf_samples)
    # np.save('difference.npy', diffs)
    print("Results saved to files.")