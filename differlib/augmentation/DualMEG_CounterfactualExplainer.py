import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mne_connectivity import spectral_connectivity_epochs
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


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
        # 如果模型是CuMLWrapper（RF），其无法计算梯度，反事实样本计算较为困难
        self.model1 = model1.to(device).eval()
        self.model2 = model2.to(device).eval()
        # 临时处理，HGRN包含RNN结果，必须在train模式下优化loss
        if self.model1.__class__.__name__ == 'HGRN':
            self.model1.train()
        if self.model2.__class__.__name__ == 'HGRN':
            self.model2.train()
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
        生成单个MEG反事实解释
        """
        # 添加批次维度并调用批量方法
        batch_result = self.generate_counterfactual_batch(
            X[np.newaxis, :, :],  # 添加批次维度
            [mode],  # 模式列表
            [target_model] if target_model is not None else [None],  # 目标模型列表
            verbose
        )

        # 提取并返回单个结果
        result = {
            'counterfactual': batch_result['counterfactuals'][0],
            'original_classes': batch_result['original_classes'][0],
            'counterfactual_classes': batch_result['cf_classes'][0],
            'mode': mode,
            'loss_history': batch_result['loss_histories'][0]
        }
        return result

    def generate_counterfactual_batch(self, X_batch, modes, target_models,
                                      n_cf_per_sample=3, diversity_strategy='noise_init', optimizer_type='adam',  # 新增参数：'adam' 或 'lbfgs'
                                      verbose=True):
        """
        批量生成多个不同的MEG反事实解释

        参数:
        X_batch: 原始MEG样本批次 (batch_size, 204, 100)
        modes: 反事实模式列表
        target_models: 目标模型列表
        n_cf_per_sample: 每个样本生成的反事实数量
        diversity_strategy: 多样性策略 ('noise_init', 'random_constraint', 'adversarial')
        verbose: 是否显示优化过程
        """
        # 1. 扩展批次以容纳多个反事实样本
        batch_size = X_batch.shape[0]
        X_orig = torch.tensor(X_batch, dtype=torch.float32, device=self.device, requires_grad=False)

        # 重复原始样本以生成多个反事实
        X_orig_expanded = X_orig.repeat_interleave(n_cf_per_sample, dim=0)

        # 2. 应用多样性策略
        if diversity_strategy == 'noise_init':
            # 策略1: 不同初始化噪声
            noise = (torch.randn_like(X_orig_expanded) * (X_orig_expanded.max() - X_orig_expanded.min()) + X_orig_expanded.min()) * 0.01
            X_cf = X_orig_expanded + noise
        else:
            # 默认策略: 无随机化
            X_cf = X_orig_expanded.clone()

        # 3. 扩展其他参数
        modes_expanded = []
        target_models_expanded = []
        orig_classes1_expanded = []
        orig_classes2_expanded = []

        with torch.no_grad():
            orig_preds1 = self.model1(X_orig)
            orig_preds2 = self.model2(X_orig)
            orig_classes1 = torch.argmax(orig_preds1, dim=1).cpu().numpy()
            orig_classes2 = torch.argmax(orig_preds2, dim=1).cpu().numpy()

        for i in range(batch_size):
            for j in range(n_cf_per_sample):
                modes_expanded.append(modes[i])
                target_models_expanded.append(target_models[i])
                orig_classes1_expanded.append(orig_classes1[i])
                orig_classes2_expanded.append(orig_classes2[i])

        # 转换为张量
        orig_classes1_expanded = torch.tensor(orig_classes1_expanded, device=self.device)
        orig_classes2_expanded = torch.tensor(orig_classes2_expanded, device=self.device)

        # 5. 优化循环
        expanded_batch_size = batch_size * n_cf_per_sample
        loss_histories = [[] for _ in range(expanded_batch_size)]

        X_cf = X_cf.detach().requires_grad_(True).contiguous()
        if optimizer_type.lower() == 'adam':
            # 4. 初始化优化器
            optimizer = optim.Adam([X_cf], lr=self.learning_rate)

            for iter_idx in range(self.max_iter):
                optimizer.zero_grad()

                # 计算总损失
                total_loss, loss_components = self._compute_loss_batch(
                    X_cf, X_orig_expanded,
                    orig_classes1_expanded, orig_classes2_expanded,
                    modes_expanded, target_models_expanded
                )

                # 反向传播
                total_loss.backward()

                # 应用多样性策略的梯度修改
                if diversity_strategy == 'random_constraint' and iter_idx % 10 == 0:
                    self._apply_random_constraints()

                optimizer.step()

                # 应用值约束
                with torch.no_grad():
                    X_cf.data = torch.clamp(X_cf, -3, 3)

                # 记录损失
                # for i in range(expanded_batch_size):
                #     loss_histories[i].append(loss_components[i])

                # 打印进度
                if verbose and (iter_idx % 10 == 0 or iter_idx == self.max_iter - 1):
                    with torch.no_grad():
                        preds1 = self.model1(X_cf)
                        preds2 = self.model2(X_cf)
                        classes1 = torch.argmax(preds1, dim=1).cpu().numpy()
                        classes2 = torch.argmax(preds2, dim=1).cpu().numpy()

                    # 计算成功率
                    success_count = 0
                    for i in range(expanded_batch_size):
                        orig_idx = i // n_cf_per_sample
                        if modes_expanded[i] == 'flip_one' and target_models_expanded[i] == 1:
                            if classes1[i] != orig_classes1[orig_idx]:
                                success_count += 1
                        elif modes_expanded[i] == 'flip_one' and target_models_expanded[i] == 2:
                            if classes2[i] != orig_classes2[orig_idx]:
                                success_count += 1
                        elif modes_expanded[i] == 'different':
                            if classes1[i] != classes2[i]:
                                success_count += 1
                        elif modes_expanded[i] == 'same':
                            if classes1[i] == classes2[i]:
                                success_count += 1

                    success_rate = success_count / expanded_batch_size
                    print(f"Iter {iter_idx}: Total Loss={total_loss.item():.4f} | "
                          f"Success Rate={success_rate:.2%}[{success_count}/{expanded_batch_size}]")

                    if success_count >= int(expanded_batch_size * 0.99):
                        if verbose:
                            print("满足停止条件，提前终止优化")
                        break
        elif optimizer_type.lower() == 'lbfgs':
            # """使用PyTorch的L-BFGS实现优化 (无边界约束)"""
            # X_cf_tensor = torch.tensor(X_cf, dtype=torch.float32, device=self.device, requires_grad=True)
            # 创建优化器
            optimizer = optim.LBFGS(
                [X_cf],
                lr=0.8,  # 较大的学习率
                max_iter=500,  # 每个样本的迭代次数
                history_size=10,
                line_search_fn='strong_wolfe'
            )

            def closure():
                optimizer.zero_grad()
                loss, _ = self._compute_loss_batch(
                    X_cf, X_orig_expanded,
                    orig_classes1_expanded, orig_classes2_expanded,
                    modes_expanded, target_models_expanded
                )
                loss.backward()
                return loss

            for iter_idx in range(5):
                loss = optimizer.step(closure)

                with torch.no_grad():
                    preds1 = self.model1(X_cf)
                    preds2 = self.model2(X_cf)
                    classes1 = torch.argmax(preds1, dim=1).cpu().numpy()
                    classes2 = torch.argmax(preds2, dim=1).cpu().numpy()
                # 计算成功率
                success_count = 0
                for i in range(expanded_batch_size):
                    orig_idx = i // n_cf_per_sample
                    if modes_expanded[i] == 'flip_one' and target_models_expanded[i] == 1:
                        if classes1[i] != orig_classes1[orig_idx]:
                            success_count += 1
                    elif modes_expanded[i] == 'flip_one' and target_models_expanded[i] == 2:
                        if classes2[i] != orig_classes2[orig_idx]:
                            success_count += 1
                    elif modes_expanded[i] == 'different':
                        if classes1[i] != classes2[i]:
                            success_count += 1
                    elif modes_expanded[i] == 'same':
                        if classes1[i] == classes2[i]:
                            success_count += 1

                success_rate = success_count / expanded_batch_size
                print(f"Iter {iter_idx}: Total Loss={loss.item():.4f} | "
                      f"Success Rate={success_rate:.2%}[{success_count}/{expanded_batch_size}]")

                # # 应用值约束 (近似边界约束)
                # with torch.no_grad():
                #     X_cf.data = torch.clamp(X_cf, -3, 3)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        # 6. 获取最终预测
        with torch.no_grad():
            preds1 = self.model1(X_cf)
            preds2 = self.model2(X_cf)
            cf_classes1 = torch.argmax(preds1, dim=1).cpu().numpy()
            cf_classes2 = torch.argmax(preds2, dim=1).cpu().numpy()

        # 7. 重组结果
        counterfactuals = X_cf.detach().cpu().numpy().reshape(batch_size, n_cf_per_sample, *X_batch.shape[1:])
        cf_classes = []
        loss_histories_reshaped = []

        for i in range(batch_size):
            sample_cf_classes = []
            sample_loss_hist = []

            for j in range(n_cf_per_sample):
                idx = i * n_cf_per_sample + j
                sample_cf_classes.append((cf_classes1[idx], cf_classes2[idx]))
                sample_loss_hist.append(loss_histories[idx])

            cf_classes.append(sample_cf_classes)
            loss_histories_reshaped.append(sample_loss_hist)

        # 返回结果
        return {
            'counterfactuals': counterfactuals,  # (batch_size, n_cf_per_sample, 204, 100)
            'original_classes': list(zip(orig_classes1, orig_classes2)),
            'cf_classes': cf_classes,  # 每个样本有n_cf_per_sample个结果
            'modes': modes,
            'loss_histories': loss_histories_reshaped
        }

    def _compute_loss_batch(self, X_cf, X_orig, orig_classes1, orig_classes2, modes, target_models):
        """计算批量总损失函数 - 向量化优化版"""
        batch_size = X_cf.shape[0]

        # 1. 获取当前预测
        preds1 = self.model1(X_cf)
        preds2 = self.model2(X_cf)

        # 2. 向量化计算距离损失
        dist_losses = torch.mean((X_cf - X_orig) ** 2, dim=(1, 2))

        # 3. 向量化计算时间平滑约束
        time_diffs = torch.diff(X_cf, dim=2)
        temp_penalties = torch.mean(time_diffs ** 2, dim=(1, 2))

        # 添加通道间时间模式一致性惩罚 (向量化)
        channel_vars = torch.var(X_cf, dim=1)
        temp_penalties += 0.1 * torch.mean(channel_vars, dim=1)

        # 4. 向量化计算空间相关性约束
        X_centered = X_cf - torch.mean(X_cf, dim=2, keepdim=True)
        cov_matrices = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (X_cf.shape[2] - 1)

        # 计算与预期协方差矩阵的差异 (广播机制)
        diff = cov_matrices - self.connectivity_matrix.unsqueeze(0)

        # 使用 Frobenius 范数作为惩罚 (向量化)
        spatial_penalties = torch.norm(diff, p='fro', dim=(1, 2))

        # 5. 向量化计算频域约束
        orig_spectrum = torch.fft.rfft(X_orig, dim=2).abs()
        cf_spectrum = torch.fft.rfft(X_cf, dim=2).abs()

        # Alpha 频带 (向量化选择)
        alpha_band = slice(1, 20)
        frequency_penalties = torch.mean(
            (cf_spectrum[:, :, alpha_band] - orig_spectrum[:, :, alpha_band]) ** 2,
            dim=(1, 2)
        )

        # 6. 向量化计算预测损失 - 这是最复杂的部分
        pred_losses = torch.zeros(batch_size, device=self.device)

        # 为不同模式创建掩码
        mask_different = torch.tensor([m == 'different' for m in modes], device=self.device)
        mask_same = torch.tensor([m == 'same' for m in modes], device=self.device)
        mask_flip1 = torch.tensor([m == 'flip_one' and tm == 1 for m, tm in zip(modes, target_models)],
                                  device=self.device)
        mask_flip2 = torch.tensor([m == 'flip_one' and tm == 2 for m, tm in zip(modes, target_models)],
                                  device=self.device)

        # 不同模式的处理
        if mask_different.any():
            # 计算相同类别的概率差异
            prob_same = torch.abs(
                preds1[torch.arange(batch_size), orig_classes1] -
                preds2[torch.arange(batch_size), orig_classes2]
            )
            pred_losses.masked_scatter_(mask_different, prob_same[mask_different])

        if mask_same.any():
            # 确定目标类别
            target_classes = torch.where(
                preds1[torch.arange(batch_size), orig_classes1] > preds2[torch.arange(batch_size), orig_classes2],
                torch.tensor(orig_classes1, device=self.device),
                torch.tensor(orig_classes2, device=self.device)
            )

            # 计算损失
            same_loss = (1 - preds1[torch.arange(batch_size), target_classes]) + \
                        (1 - preds2[torch.arange(batch_size), target_classes])
            pred_losses.masked_scatter_(mask_same, same_loss[mask_same])

        if mask_flip1.any():
            flip1_loss = (1 - preds1[torch.arange(batch_size), 1 - orig_classes1]) + \
                         torch.abs(preds2[torch.arange(batch_size), orig_classes2] - 0.9)
            pred_losses.masked_scatter_(mask_flip1, flip1_loss[mask_flip1])

        if mask_flip2.any():
            flip2_loss = (1 - preds2[torch.arange(batch_size), 1 - orig_classes2]) + \
                         torch.abs(preds1[torch.arange(batch_size), orig_classes1] - 0.9)
            pred_losses.masked_scatter_(mask_flip2, flip2_loss[mask_flip2])

        # 7. 组合所有损失分量
        total_per_sample = (
                pred_losses +
                self.lambda_dist * dist_losses +
                self.lambda_temp * temp_penalties +
                self.lambda_spatial * spatial_penalties +
                self.lambda_frequency * frequency_penalties
        )

        # 8. 计算批次的平均总损失
        total_loss = torch.mean(total_per_sample)

        # 9. 准备损失组件列表
        loss_components_list = []
        # for i in range(batch_size):
        #     loss_components = (
        #         pred_losses[i].item(),
        #         dist_losses[i].item(),
        #         temp_penalties[i].item(),
        #         spatial_penalties[i].item(),
        #         frequency_penalties[i].item()
        #     )
        #     loss_components_list.append(loss_components)

        return total_loss, loss_components_list

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


def counterfactual(model1, model2, dataset, meg_data, n_generate=5, batch_size=128, cover=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    save_path = "./counterfactuals/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, f'{dataset}_{model1.__class__.__name__}_{model2.__class__.__name__}_counterfactual_sample.npy')
    file_path_1 = os.path.join(save_path, f'{dataset}_{model2.__class__.__name__}_{model1.__class__.__name__}_counterfactual_sample.npy')
    if not cover and os.path.exists(file_path):
        cf_samples = np.load(file_path)
        print("counterfactual has been loaded")
    elif not cover and os.path.exists(file_path_1):
        cf_samples = np.load(file_path_1)
        print("counterfactual has been loaded")
    else:
        n_samples, channels, points = meg_data.shape
        cf_samples = np.zeros((n_samples, n_generate, channels, points), dtype=np.float32)

        connectivity_matrix_file = f"{dataset}_connectivity_matrix.npy"
        if os.path.exists(connectivity_matrix_file):
            connectivity_matrix = np.load(connectivity_matrix_file)
        else:
            if dataset == "CamCAN":
                sfreq, fmin, fmax = 125, 1, 45
            elif dataset == "DecMeg2014":
                sfreq, fmin, fmax = 250, 0.1, 20
            else:
                print("Not a valid dataset")
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

        explainer = DualMEGCounterfactualExplainer(
            model1,
            model2,
            lambda_dist=0.5,
            lambda_temp=0.5,
            lambda_spatial=0.01,
            lambda_frequency=0.5,
            learning_rate=0.03, # DecMeg2014 0.01   CamCAN 0.003
            max_iter=50,
            connectivity_matrix=connectivity_matrix,
            device=device
        )

        num_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            # 获取当前批次的样本
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            current_batch = meg_data[start_idx:end_idx]

            # 批量生成反事实样本
            batch_result = explainer.generate_counterfactual_batch(
                current_batch,
                modes=['flip_one'] * (end_idx - start_idx),  # 所有样本使用相同模式   flip_one    auto
                target_models=[1] * (end_idx - start_idx),  # 所有样本翻转模型1
                n_cf_per_sample=n_generate,
                diversity_strategy='noise_init',
                optimizer_type='adam',     # lbfgs adam
                verbose=True
            )
            cf_samples[start_idx:end_idx] = batch_result['counterfactuals']

        # 11. 保存结果
        np.save(file_path, cf_samples)
        print("Results saved to files.")

    return cf_samples