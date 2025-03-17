import argparse
import os
import shelve
import time

import numpy as np
import torch
from triton.language import dtype

from similarity.engine.cfg import CFG as cfg
from similarity.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, dataset_info_dict, predict)
from differlib.models import model_dict
from MEG_Shapley_Values import ShapleyValueExplainer, DatasetInfo, SampleInfo, deletion_test, \
    compare_deletion_test, similar_analysis, additive_efficient_normalization, compare_insertion_test, insertion_test, \
    IterationLogger, contribution_smooth, torch_individual_predict


def compute_sigma_based_on_std(a, b):
    data = np.vstack([a, b])  # 假设 a 和 b 是两个向量
    std = np.std(data)
    std = np.sqrt(std)
    return std if std != 0 else 1.0  # 避免除零错误


if __name__ == "__main__":
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS

    # 要分析的样本数量
    # datasets
    dataset = "DecMeg2014"  # "DecMeg2014", "CamCAN"
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    origin_data, labels = test_data, test_labels
    n_samples, channels, points = origin_data.shape
    n_classes = len(set(labels))
    assert channels == dataset_info_dict[dataset]["CHANNELS"]
    assert points == dataset_info_dict[dataset]["POINTS"]
    assert n_classes == dataset_info_dict[dataset]["NUM_CLASSES"]

    label_names = ['audio', 'visual']
    if dataset == 'DecMeg2014':
        label_names = ['Scramble', 'Face']
    dataset_info = DatasetInfo(dataset=dataset, label_names=label_names, channels=channels, points=points,
                               classes=n_classes)

    sample_num = 160
    model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]
    top_k = 0.25
    k = int(channels * points * top_k)

    # AttributionExplainer参数
    explainer = cfg.EXPLAINER.TYPE

    db_path = './output/Consensus/{}/{}_{}_attribution'.format(dataset, dataset, explainer)
    db = shelve.open(db_path)

    # 逐样本迭代
    # 计算所有的VARCNN归因结果，并计算最重要的top-k特征
    for model_name in model_names:
        all_maps = np.zeros([sample_num, channels, points, n_classes], dtype=np.float32)
        for sample_id in range(sample_num):
            attribution_id = f"{sample_id}_{model_name}"
            assert attribution_id in db
            maps = db[attribution_id]
            all_maps[sample_id] = maps
        mean_maps = np.abs(all_maps).mean(axis=0)
        feature_contribution = np.abs(mean_maps).sum(axis=-1).reshape(-1)
        top_sort = np.argsort(feature_contribution)[::-1]
        sort_contribution = feature_contribution[top_sort]

        file = '{}_{}_top_sort.npy'.format(dataset, model_name)
        np.save(file, top_sort)

    db.close()

    top_atcnet = np.load('{}_{}_top_sort.npy'.format(dataset, "MLP"))
    top_linear = np.load('{}_{}_top_sort.npy'.format(dataset, "Linear"))
    top_masks = np.zeros_like(feature_contribution, dtype=np.bool_)
    for id in range(k):
        if top_linear[id] in top_atcnet[:k]:
            top_masks[top_linear[id]] = True
    print("consensus_features:", top_masks.sum())

    top_masks = top_masks.reshape(channels, points)

    file = '{}_{}_top_k.npy'.format(dataset, "LFCNN")
    np.save(file, top_masks)
