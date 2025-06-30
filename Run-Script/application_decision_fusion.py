import argparse
import os
import shelve
from datetime import datetime
from statistics import mean

import numpy as np
import pandas as pd
import sklearn
import torch
from mne.time_frequency import psd_array_multitaper
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from differlib.augmentation import am_dict
from differlib.augmentation.DualMEG_CounterfactualExplainer import counterfactual
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, dataset_info_dict, predict)
from differlib.explainer import explainer_dict
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps
from differlib.models import model_dict, scikit_models, torch_models, load_pretrained_model, output_predict_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="../configs/CamCAN/Logit.yaml")  # DecMeg2014    CamCAN      BCIIV2a
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

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    num_gpus = torch.cuda.device_count()
    num_cpus = cfg.EXPERIMENT.CPU_COUNT
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # 'cuda:1'
    print(f"Using device: {device}")

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    train_loader = get_data_loader(train_data, train_labels)
    data, labels = test_data, test_labels
    n_samples, channels, points = data.shape
    n_classes = len(set(labels))
    n_splits = cfg.NUM_SPLITS
    window_length = cfg.WINDOW_LENGTH
    feature_names = [f"C{c}T{t}" for c in range(channels) for t in range(points)]
    feature_names = np.array(feature_names)
    print(f"Dataset {dataset} {data.shape} has {n_classes} classes.")

    # init different models and load pre-trained checkpoints
    model_A_type = cfg.MODEL_A
    model_B_type = cfg.MODEL_B

    model_A = load_pretrained_model(model_A_type, dataset, channels, points, n_classes, device)
    model_B = load_pretrained_model(model_B_type, dataset, channels, points, n_classes, device)

    # # init feature selection
    # selection_type = cfg.SELECTION.TYPE
    # selection_method = fsm_dict[selection_type]()
    # selection_M = cfg.SELECTION.Diff.M
    # selection_threshold = cfg.SELECTION.Diff.THRESHOLD
    # # 预先计算所有样本的特征归因图，训练时只使用训练集样本的特征归因图
    # if selection_type in ["DiffShapley"]:
    #     all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M)

    # init explainer
    explainer_types = cfg.EXPLAINER.TYPE.split(";")
    # if isinstance(explainer_types, str):
    #     explainer_types = [explainer_types]
    max_depth = cfg.EXPLAINER.MAX_DEPTH
    min_samples_leaf = cfg.EXPLAINER.MIN_SAMPLES_LEAF
    # all initialization is ok

    # models predict differences
    output_A, pred_target_A = output_predict_targets(model_A_type, model_A, data, num_classes=n_classes, device=device)
    output_B, pred_target_B = output_predict_targets(model_B_type, model_B, data, num_classes=n_classes, device=device)
    delta_target = (pred_target_A != pred_target_B).astype(int)
    delta_weights = np.abs(output_A - output_B).mean(axis=1)


    def dynamic_fusion(data, model_A, model_B, explainer):
        data_ = data.reshape((len(data), -1))
        # For Feature Selection to Compute Feature Contributions
        # if selection_type in ["DiffShapley"]:
        #     # all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M)
        #     selection_method.fit(x_train, model_A, model_B, channels, points, n_classes,
        #                          window_length, selection_M, all_sample_feature_maps[train_index],
        #                          threshold=selection_threshold, num_gpus=num_gpus, num_cpus=num_cpus)
        #     # Execute Feature Selection
        #     data_, select_indices = selection_method.transform(data_)
        #     x_feature_names = feature_names[select_indices]
        select_indices = np.array([np.nonzero(feature_names == i)[0].item() for i in explainer.delta_tree.feature_names_in_])
        data_ = data_[:, select_indices]

        logit_delta_proxy = explainer.delta_tree.predict(data_)
        logit_delta = logit_delta_proxy[:, :n_classes]

        fusion_output, fusion_target = np.zeros_like(logit_delta), np.zeros_like(logit_delta[:, 0])
        for idx, x in enumerate(data):
            weight = np.abs(logit_delta[idx]).max()  # 取最大概率差作权重
            out_A, tag_A = output_predict_targets(model_A_type, model_A, x[np.newaxis, :], num_classes=n_classes, device=device)
            out_B, tag_B = output_predict_targets(model_B_type, model_B, x[np.newaxis, :], num_classes=n_classes, device=device)

            # fusion_output[idx] = (out_A + out_B) / 2

            weight = 0.5 + logit_delta[idx, 0] / 2
            # weight = 0.5 + logit_delta[idx, np.abs(logit_delta[idx]).argmax()] / 2
            fusion_output[idx] = weight * out_A + (1 - weight) * out_B
            # if weight > 0.05:
            #     if logit_delta[idx, 0] > 0:
            #         fusion_output[idx] = weight * out_A + (1 - weight) * out_B
            #     else:
            #         fusion_output[idx] = weight * out_B + (1 - weight) * out_A
            # else:
            #     fusion_output[idx] = (out_A + out_B) / 2

            # if logit_delta[idx, 0] > 0:
            #     fusion_output[idx] = weight * out_A + (1 - weight) * out_B
            # else:
            #     fusion_output[idx] = weight * out_B + (1 - weight) * out_A

            fusion_target = np.argmax(fusion_output, axis=1)

            # if logit_delta[idx, 0] > 0:
            #     fusion_output[idx], fusion_target[idx] = output_predict_targets(model_A_type, model_A,  x[np.newaxis, :], num_classes=n_classes, device=device)
            # else:
            #     fusion_output[idx], fusion_target[idx] = output_predict_targets(model_B_type, model_B,  x[np.newaxis, :], num_classes=n_classes, device=device)

        return fusion_output, fusion_target


    # K-Fold evaluation
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=cfg.TEST_SIZE, random_state=cfg.EXPERIMENT.SEED)   # 0.1   0.25
    # skf = StratifiedKFold(n_splits=n_splits)

    for explainer_type in explainer_types:
        explainer = explainer_dict[explainer_type]()

        log_path = os.path.join(log_prefix, f"{dataset}/{explainer_type}_train")
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)

        skf_id = 0
        # record metrics of i-th Fold
        acc_A_test_, acc_B_test_, fusion_acc_test_ = [], [], []
        # for train_index, test_index in skf.split(data, delta_target):
        x_train = train_data
        x_test, y_test = data, labels

        save_path = os.path.join(log_path, "{}".format(explainer_type))
        explainer = load_checkpoint(save_path)["explainer"]

        pred_target_A_test, pred_target_B_test = pred_target_A, pred_target_B
        acc_A_test = sklearn.metrics.accuracy_score(y_test, pred_target_A_test)
        acc_B_test = sklearn.metrics.accuracy_score(y_test, pred_target_B_test)

        fusion_output_test, fusion_target_test = dynamic_fusion(x_test, model_A, model_B, explainer)
        fusion_acc_test = sklearn.metrics.accuracy_score(y_test, fusion_target_test)

        print(f"skf_id", skf_id, "Explainer", explainer_type, acc_A_test, acc_B_test, fusion_acc_test)
        acc_A_test_.append(acc_A_test)
        acc_B_test_.append(acc_B_test)
        fusion_acc_test_.append(fusion_acc_test)

        skf_id += 1

        acc_A_test_, acc_B_test_, fusion_acc_test_ = np.array(acc_A_test_), np.array(acc_B_test_), np.array(fusion_acc_test_)
        print(f"acc_A_test: {acc_A_test_.mean()} {acc_A_test_.std()}")
        print(f"acc_B_test: {acc_B_test_.mean()} {acc_B_test_.std()}")
        print(f"fusion_acc_test: {fusion_acc_test_.mean()} {fusion_acc_test_.std()}")
        p_value = ttest_ind(acc_A_test_, fusion_acc_test_).pvalue
        print(f"p_value: {p_value}")
        p_value = ttest_ind(acc_B_test_, fusion_acc_test_).pvalue
        print(f"p_value: {p_value}")