import numpy as np


def _calculate_variance(y):
    # 计算目标变量y的方差
    return np.var(y)


def _calculate_mse(y):
    # 计算目标变量y的均方误差（Mean Squared Error）
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)


def _split_dataset(X, y, feature_index, threshold):
    # 根据特征索引和阈值将数据集X和目标变量y划分为两个子集
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    return X_left, y_left, X_right, y_right


def _create_leaf_node(y):
    # 创建叶子节点
    leaf = {
        'is_leaf': True,
        'value': np.mean(y)
    }
    return leaf


def _create_internal_node(feature_index, threshold):
    # 创建内部节点
    node = {
        'is_leaf': False,
        'feature_index': feature_index,
        'threshold': threshold,
        'left': None,
        'right': None
    }
    return node


class RegressionDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth  # 最大深度限制
        self.min_samples_split = min_samples_split  # 分裂所需的最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶子节点所需的最小样本数
        self.tree = None  # 决策树的数据结构

    def _find_best_split(self, X, y):
        # 寻找最佳划分点
        best_feature_index = None
        best_threshold = None
        best_loss = float('inf')

        n_features = X.shape[1]  # 特征数量

        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])  # 当前特征的唯一值

            for threshold in unique_values:
                X_left, y_left, X_right, y_right = _split_dataset(X, y, feature_index, threshold)

                # 如果划分后的子集样本数小于阈值，则忽略该划分
                if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
                    continue

                loss_left = _calculate_variance(y_left)  # 左子集的方差
                loss_right = _calculate_variance(y_right)  # 右子集的方差
                weighted_loss = (len(X_left) * loss_left + len(X_right) * loss_right) / len(X)  # 加权平均方差

                # 如果加权平均方差小于当前最小损失，则更新最佳划分点
                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        # 递归构建决策树
        if depth == 0 or len(X) < self.min_samples_split:
            # 达到最大深度或样本数小于最小分裂样本数时，创建叶子节点
            return _create_leaf_node(y)

        best_feature_index, best_threshold = self._find_best_split(X, y)

        if best_feature_index is None or best_threshold is None:
            # 无法找到最佳划分点时，创建叶子节点
            return _create_leaf_node(y)

        # 根据最佳划分点划分子集
        X_left, y_left, X_right, y_right = _split_dataset(X, y, best_feature_index, best_threshold)

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            # 划分后的子集样本数小于叶子节点所需的最小样本数时，创建叶子节点
            return _create_leaf_node(y)

        # 创建内部节点
        node = _create_internal_node(best_feature_index, best_threshold)

        # 递归构建左子树和右子树
        node['left'] = self._build_tree(X_left, y_left, depth - 1)
        node['right'] = self._build_tree(X_right, y_right, depth - 1)

        return node

    def fit(self, X, y):
        # 构建决策树模型
        self.tree = self._build_tree(X, y, self.max_depth)

    def _predict_sample(self, sample, node):
        # 递归预测单个样本
        if node['is_leaf']:
            return node['value']

        feature_value = sample[node['feature_index']]
        if feature_value <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

    def predict(self, X):
        # 预测样本集
        predictions = []
        for sample in X:
            prediction = self._predict_sample(sample, self.tree)
            predictions.append(prediction)
        return np.array(predictions)


# 使用示例
from sklearn.datasets import make_regression
import numpy as np

# 生成模拟数据
X, y = make_regression(n_samples=100, n_features=2, noise=0.5)

# 构建决策树
root = RegressionDecisionTree(max_depth=5)
root.fit(X, y)

# 预测
print(root.predict([[0.1, 0.2]]))
