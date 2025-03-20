import numpy as np


def top_k_consensus(top_sort_1: np.ndarray, top_sort_2: np.ndarray, k: int) -> [np.ndarray, np.ndarray]:
    assert top_sort_1.shape == top_sort_2.shape
    assert k <= len(top_sort_1)
    consensus_list = []
    consensus_masks = np.zeros_like(top_sort_1, dtype=np.bool_)
    for i in range(k):
        if top_sort_1[i] in top_sort_2[:k]:
            consensus_list.append(top_sort_1[i])
            consensus_masks[top_sort_1[i]] = True
    return consensus_list, consensus_masks


def top_k_disagreement(top_sort_1: np.ndarray, top_sort_2: np.ndarray, k: int) -> [np.ndarray, np.ndarray]:
    assert top_sort_1.shape == top_sort_2.shape
    assert k <= len(top_sort_1)
    disagreement_list = []
    disagreement_masks = np.zeros_like(top_sort_1, dtype=np.bool_)
    for i in range(k):
        if top_sort_1[i] not in top_sort_2[:k]:
            disagreement_list.append(top_sort_1[i])
            disagreement_masks[top_sort_1[i]] = True
    return disagreement_list, disagreement_masks


# Rank Correlation、Top-k 特征一致性、符号排序一致性等

# 通道或时间平均贡献的rank correlation和成对排序一致性

# 使用RDMs评估两个模型的通道或时间特征间交互关系
