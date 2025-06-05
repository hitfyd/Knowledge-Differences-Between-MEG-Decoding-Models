import numpy as np
from scipy.optimize import minimize

from differlib.engine.utils import get_data_labels_from_dataset, log_msg, load_checkpoint
from differlib.models import model_dict, scikit_models, torch_models

# 输入样本: 204通道 x 100时间点
X = np.random.randn(204, 100)  # 示例数据


# 将每个通道分为10个特征组 (每组10个时间点)
def group_features(data):
    """将数据分为特征组"""
    n_channels, n_timepoints = data.shape
    n_groups = n_timepoints // 10
    groups = np.zeros((n_channels, n_groups, 10))

    for c in range(n_channels):
        for g in range(n_groups):
            start_idx = g * 10
            end_idx = (g + 1) * 10
            groups[c, g] = data[c, start_idx:end_idx]

    return groups


# 从分组重建完整数据
def reconstruct_data(groups):
    """从特征组重建完整时间序列"""
    n_channels, n_groups, _ = groups.shape
    reconstructed = np.zeros((n_channels, n_groups * 10))

    for c in range(n_channels):
        for g in range(n_groups):
            start_idx = g * 10
            end_idx = (g + 1) * 10
            reconstructed[c, start_idx:end_idx] = groups[c, g]

    return reconstructed


def temporal_smoothness_constraint(groups):
    """惩罚时间点间的不连续变化"""
    penalty = 0
    for c in range(groups.shape[0]):
        for g in range(groups.shape[1] - 1):
            # 检查组间过渡的平滑性
            last_point = groups[c, g, -1]
            next_point = groups[c, g + 1, 0]
            penalty += (last_point - next_point) ** 2

            # 组内平滑性
            diff = np.diff(groups[c, g])
            penalty += np.sum(diff ** 2) * 0.1
    return penalty


def spatial_constraint(groups, connectivity_matrix):
    """保持通道间的空间相关性"""
    penalty = 0
    n_channels, n_groups, _ = groups.shape

    for g in range(n_groups):
        group_data = groups[:, g, :]
        # 计算当前组通道间的协方差
        current_cov = np.cov(group_data)

        # 与预期协方差矩阵比较
        penalty += np.linalg.norm(current_cov - connectivity_matrix, 'fro')

    return penalty


import matplotlib.pyplot as plt
import seaborn as sns


def visualize_differences(original, counterfactual):
    """可视化原始数据与反事实的差异"""
    plt.figure(figsize=(15, 10))

    # 计算差异
    difference = counterfactual - original

    # 1. 总体差异热图
    plt.subplot(3, 1, 1)
    sns.heatmap(difference, cmap='coolwarm', center=0)
    plt.title('Channel-Time Differences')
    plt.xlabel('Time Points')
    plt.ylabel('Channels')

    # 2. 按通道的差异幅度
    plt.subplot(3, 1, 2)
    channel_diffs = np.mean(np.abs(difference), axis=1)
    plt.bar(range(len(channel_diffs)), channel_diffs)
    plt.title('Average Absolute Difference per Channel')
    plt.xlabel('Channel Index')
    plt.ylabel('Average Difference')

    # 3. 按时间点的差异幅度
    plt.subplot(3, 1, 3)
    time_diffs = np.mean(np.abs(difference), axis=0)
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
    plt.title(f'Channel {max_diff_channel} Comparison')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


# def attention_guided_modification(X_cf, X_orig, model, temperature=0.1):
#     """使用模型注意力引导修改"""
#     # 获取模型对原始样本的注意力图
#     attention_map = get_model_attention(model, X_orig)
#
#     # 在注意力高的区域允许更大修改
#     modification_capacity = 1 / (1 + np.exp(-attention_map / temperature))
#
#     return modification_capacity * (X_cf - X_orig) + X_orig


class MEGCounterfactualExplainer:
    def __init__(self, model, lambda_temp=0.1, lambda_spatial=0.05,
                 max_iter=100, connectivity_matrix=None):
        """
        MEG反事实解释器

        参数:
        model: 预训练的MEG分类模型
        lambda_temp: 时间平滑约束强度
        lambda_spatial: 空间相关性约束强度
        connectivity_matrix: 通道间预期协方差矩阵
        max_iter: 最大优化迭代次数
        """
        self.model = model
        self.lambda_temp = lambda_temp
        self.lambda_spatial = lambda_spatial
        self.max_iter = max_iter

        # 默认使用单位矩阵作为连通性矩阵
        if connectivity_matrix is None:
            self.connectivity_matrix = np.eye(204)
        else:
            self.connectivity_matrix = connectivity_matrix

    def generate_counterfactual(self, X, target_class):
        """
        生成MEG反事实解释

        参数:
        X: 原始MEG样本 (204, 100)
        target_class: 目标类别

        返回:
        X_cf: 反事实样本 (204, 100)
        """
        # 初始化为原始数据
        X_cf_flat = X.flatten()

        # 优化问题定义
        result = minimize(
            fun=self._loss_function,
            x0=X_cf_flat,
            args=(X, target_class),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'disp': True},
            callback=self._monitor_progress
        )

        # 重构反事实样本
        X_cf = result.x.reshape(204, 100)
        return X_cf

    def _loss_function(self, X_cf_flat, X_orig, target_class):
        """自定义损失函数"""
        # 重构为2D数组
        X_cf = X_cf_flat.reshape(204, 100)
        X_orig = X_orig.reshape(204, 100)

        # 分组特征
        groups_cf = group_features(X_cf)
        groups_orig = group_features(X_orig)

        # 预测损失 (鼓励模型预测目标类别)
        pred = self.model.predict_proba(X_cf.flatten()[np.newaxis, ...])[0]
        pred_loss = 1 - pred[target_class]

        # 距离损失 (最小化修改量)
        dist_loss = np.mean((X_cf - X_orig) ** 2)  # MSE

        # 时间平滑约束
        temp_penalty = temporal_smoothness_constraint(groups_cf)

        # 空间相关性约束
        spatial_penalty = 0 # spatial_constraint(groups_cf, self.connectivity_matrix)

        # 总损失
        total_loss = (pred_loss +
                      dist_loss +
                      self.lambda_temp * temp_penalty +
                      self.lambda_spatial * spatial_penalty)

        return total_loss

    def _monitor_progress(self, xk):
        """监控优化进度"""
        X_cf = xk.reshape(204, 100)
        current_pred = self.model.predict(X_cf.flatten()[np.newaxis, ...])[0]
        print(f"当前预测: {current_pred}", end='\r')
        return False


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


# def output_predict_targets(model_type, model, data: np.ndarray, num_classes=2, batch_size=512, softmax=True):
#     output, predict_targets = None, None
#     if model_type in scikit_models:
#         predict_targets = model.predict(data.reshape((len(data), -1)))
#         output = model.predict_proba(data.reshape((len(data), -1)))
#     elif model_type in torch_models:
#         output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
#         predict_targets = np.argmax(output, axis=1)
#     else:
#         print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
#     assert output is not None
#     assert predict_targets is not None
#     return output, predict_targets


# 示例使用
if __name__ == "__main__":
    # 1. 加载预训练模型和MEG数据
    dataset = "CamCAN"
    model_type = "rf"
    channels, points, n_classes = 204, 100, 2
    model = load_pretrained_model("rf")  # 实际使用时替换
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    X_sample = test_data[0]       # 实际使用时替换

    # 2. 创建解释器
    # 假设我们有通道连通性矩阵 (实际应用中应根据解剖知识构建)
    connectivity = None # np.load('meg_connectivity.npy')  # 204x204矩阵

    explainer = MEGCounterfactualExplainer(
        model,
        lambda_temp=0.2,
        lambda_spatial=0.1,
        connectivity_matrix=connectivity
    )

    # 3. 生成反事实
    # 假设原始预测为0，我们希望看到预测为1的反事实
    cf_sample = explainer.generate_counterfactual(X_sample, target_class=0)

    # 4. 结果分析
    print("\n原始预测:", model.predict(X_sample.flatten()[np.newaxis, ...])[0])
    print("反事实预测:", model.predict(cf_sample.flatten()[np.newaxis, ...])[0])

    # 5. 可视化修改 (重点显示修改最大的通道和时间段)
    visualize_differences(X_sample, cf_sample)