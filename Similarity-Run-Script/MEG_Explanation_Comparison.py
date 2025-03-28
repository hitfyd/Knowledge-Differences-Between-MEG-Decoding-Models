import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def top_k_consensus(top_sort_1: np.ndarray, top_sort_2: np.ndarray, k1: int, k2: int) -> [np.ndarray, np.ndarray]:
    assert top_sort_1.shape == top_sort_2.shape
    assert k1 <= len(top_sort_1)
    assert k2 <= len(top_sort_2)
    consensus_list = []
    consensus_masks = np.zeros_like(top_sort_1, dtype=np.bool_)
    for i in range(k1):
        if top_sort_1[i] in top_sort_2[:k2]:
            consensus_list.append(top_sort_1[i])
            consensus_masks[top_sort_1[i]] = True
    return consensus_list, consensus_masks


def top_k_disagreement(top_sort_1: np.ndarray, top_sort_2: np.ndarray, k1: int, k2: int) -> [np.ndarray, np.ndarray]:
    assert top_sort_1.shape == top_sort_2.shape
    assert k1 <= len(top_sort_1)
    assert k2 <= len(top_sort_2)
    disagreement_list = []
    disagreement_masks = np.zeros_like(top_sort_1, dtype=np.bool_)
    for i in range(k1):
        if top_sort_1[i] not in top_sort_2[:k2]:
            disagreement_list.append(top_sort_1[i])
            disagreement_masks[top_sort_1[i]] = True
    return disagreement_list, disagreement_masks


# Top-k 特征一致性
def feature_agreement(top_sort_1: np.ndarray, top_sort_2: np.ndarray, top_k: int) -> float:
    assert top_sort_1.shape == top_sort_2.shape
    assert top_k <= len(top_sort_1)
    consensus_list, _ = top_k_consensus(top_sort_1, top_sort_2, top_k, top_k)
    num_feature_agreements = len(consensus_list)
    similarity_score = num_feature_agreements / top_k
    print(f'Top-{top_k} Feature Agreement: {similarity_score}({num_feature_agreements}/{top_k})')
    return similarity_score


# 符号一致性，在特征一致性的基础上，符号也要一致
def sign_agreement(top_sort_1: np.ndarray, top_sort_2: np.ndarray, sign_sort_maps_1: np.ndarray, sign_sort_maps_2: np.ndarray, top_k: int) -> float:
    assert top_sort_1.shape == top_sort_2.shape
    assert top_k <= len(top_sort_1)
    _, n_classes = sign_sort_maps_1.shape
    consensus_list, _ = top_k_consensus(top_sort_1, top_sort_2, top_k, top_k)
    sign_1 = np.sign(sign_sort_maps_1[consensus_list])
    sign_2 = np.sign(sign_sort_maps_2[consensus_list])
    sign_consistent_mask = (sign_1 == sign_2)
    class_consensus_list = np.array(consensus_list).repeat(n_classes)   # 考虑所有类别下的符号一致性
    sign_consensus_list = class_consensus_list[np.where(sign_consistent_mask.reshape(-1))[0]]
    num_sign_agreements = len(sign_consensus_list)
    sign_similarity_score = num_sign_agreements / (top_k * n_classes)
    print(f'Top-{top_k} Sign Agreement: {sign_similarity_score}({num_sign_agreements}/{top_k})')
    return sign_similarity_score


# 通道或时间平均贡献的rank correlation和成对排序一致性

# 使用RDMs评估两个模型的通道或时间特征间交互关系


def plot_similarity_matrix(similarity_matrix, model_names, title=None, colorbar=True, vmin=0.0, vmax=1.0):
    assert similarity_matrix.shape == (len(model_names), len(model_names))
    disp = ConfusionMatrixDisplay(confusion_matrix=similarity_matrix, display_labels=model_names)
    disp.plot(cmap='Oranges', values_format='.3f', colorbar=colorbar, im_kw={'vmin': vmin, 'vmax': vmax})
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')
    disp.ax_.set_title(title)
    plt.show()
    return disp.ax_
