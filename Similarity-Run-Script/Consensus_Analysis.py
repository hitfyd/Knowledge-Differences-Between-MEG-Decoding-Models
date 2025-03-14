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
    parser = argparse.ArgumentParser("analysis for attribution consensus.")
    parser.add_argument("--cfg", type=str, default="../configs/Consensus/DecMeg2014.yaml")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # init experiment
    project = cfg.EXPERIMENT.PROJECT
    experiment_name = cfg.EXPERIMENT.NAME
    tags = cfg.EXPERIMENT.TAG
    if experiment_name == "":
        experiment_name = tags
    tags = tags.split(',')
    if args.opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(args.opts[::2], args.opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(project, experiment_name)

    # init loggers
    log_prefix = cfg.LOG.PREFIX
    log_path = os.path.join(log_prefix, experiment_name)
    record_path = os.path.join(log_prefix, project)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    num_gpus = torch.cuda.device_count()

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    # train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    # train_loader = get_data_loader(train_data, train_labels)
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

    # 要分析的样本数量
    sample_num = cfg.NUM_SAMPLES

    model_types = cfg.MODELS
    num_models = len(model_types)
    model_list = torch.nn.ModuleList()
    for model_type in model_types:
        model_class, model_pretrain_path = model_dict[dataset][model_type]
        model_list.append(model_class(channels=channels, points=points, num_classes=n_classes))

    # AttributionExplainer参数
    explainer = cfg.EXPLAINER.TYPE

    db_path = log_path + '/{}_{}_attribution'.format(dataset, explainer)
    db = shelve.open(db_path)

    # 每个样本的特征归因共识
    all_consensus_maps = np.zeros([sample_num, channels, points, n_classes], dtype=np.float32)
    # 每个模型的特征归因与特征归因共识的相似度
    similarity_maps = np.zeros([sample_num, num_models], dtype=np.float32)

    # 逐样本迭代
    for sample_id in range(sample_num):
        origin_input, truth_label = origin_data[sample_id], labels[sample_id]
        sample_info = SampleInfo(sample_id=sample_id, origin_input=origin_input, truth_label=truth_label)
        print('sample_id:{}\ttruth_label:{}'.format(sample_id, truth_label))

        # 读取每个模型的特征归因结果
        all_model_maps = np.zeros([num_models, channels, points, n_classes], dtype=np.float32)
        for model_id in range(num_models):
            model_name = model_list[model_id].__class__.__name__
            attribution_id = f"{sample_id}_{model_name}"
            assert attribution_id in db
            maps = db[attribution_id]
            # maps = (maps - np.mean(maps)) / (np.std(maps))
            all_model_maps[model_id] = maps

        # # 计算当前样本上的模型共识
        # # 预先计算极值并保持维度以正确广播
        # min_val = all_model_maps.min(axis=0)
        # max_val = all_model_maps.max(axis=0)
        # # 计算数值安全的范围值（避免除零）
        # range_val = max_val - min_val
        # range_val[range_val == 0] = 1  # 将零范围位置设为1（即该位置所有模型值相同）
        # # 执行归一化并取平均
        # normalized_maps = (all_model_maps - min_val) / range_val
        # consensus_maps = normalized_maps.mean(axis=0)
        consensus_maps = all_model_maps.mean(axis=0)
        all_consensus_maps[sample_id] = consensus_maps

        # 计算每个模型与样本共识的相似性
        for model_id in range(num_models):
            a = all_model_maps[model_id].reshape(-1)
            b = consensus_maps.reshape(-1)
            sigma = a.max()     # compute_sigma_based_on_std(a, b)
            # 计算欧氏距离
            distance = np.linalg.norm(a - b)
            # 应用RBF公式进行标准化
            similarity_score = np.exp(- (distance / sigma) ** 2 / 2)
            similarity_maps[sample_id, model_id] = similarity_score
            print('model:{}\tsimilarity:{}'.format(model_list[model_id].__class__.__name__, similarity_score))

    # 计算每个模型的最终相似性
    model_similarities = similarity_maps.mean(axis=0)
    print('model_similarities:{}'.format(model_similarities))

    # 计算所有的VARCNN归因结果，并计算最重要的top-k特征
    k = 5100    # CamCAN 2040, DecMeg2014 5100
    all_varcnn_maps = np.zeros([sample_num, channels, points, n_classes], dtype=np.float32)
    for sample_id in range(sample_num):
        attribution_id = f"{sample_id}_VARCNN"
        assert attribution_id in db
        maps = db[attribution_id]
        all_varcnn_maps[sample_id] = maps
    mean_varcnn_maps = all_varcnn_maps.mean(axis=0)
    feature_contribution = np.abs(mean_varcnn_maps).sum(axis=-1).reshape(-1)
    top_sort = np.argsort(feature_contribution)[::-1]
    sort_contribution = feature_contribution[top_sort]
    top_k = top_sort[:k]
    top_masks = np.zeros_like(feature_contribution, dtype=np.bool_)
    top_masks[top_k] = 1
    top_masks = top_masks.reshape(channels, points)

    file = '{}_{}_top_k.npy'.format(dataset, "VARCNN")
    np.save(file, top_masks)
    top_masks = np.load(file)

    db.close()
