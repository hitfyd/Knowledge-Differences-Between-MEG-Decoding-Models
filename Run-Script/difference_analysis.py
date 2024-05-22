import argparse
import os
import time
from datetime import datetime
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer, Binarizer

from differlib.augmentation import am_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, output_predict_targets, model_eval, sample_normalize,
                                    DatasetNormalization, dataset_info_dict)
from differlib.explainer import explainer_dict
from differlib.feature_extraction import feature_extraction
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps
from differlib.models import model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="")
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
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    num_gpus = torch.cuda.device_count()
    num_cpus = cfg.EXPERIMENT.CPU_COUNT

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    train_loader = get_data_loader(train_data, train_labels)
    test_loader = get_data_loader(test_data, test_labels)
    data, labels = test_data, test_labels
    n_samples, channels, points = data.shape
    n_classes = len(set(labels))
    assert channels == dataset_info_dict[dataset]["CHANNELS"]
    assert points == dataset_info_dict[dataset]["POINTS"]
    assert n_classes == dataset_info_dict[dataset]["NUM_CLASSES"]
    n_splits = cfg.NUM_SPLITS
    window_length = cfg.WINDOW_LENGTH
    feature_names = [f"C{c}T{t}" for c in range(channels) for t in range(points)]
    feature_names = np.array(feature_names)

    # init different models and load pre-trained checkpoints
    model_A_type = cfg.MODEL_A
    model_B_type = cfg.MODEL_B
    print(log_msg("Loading model A {}".format(model_A_type), "INFO"))
    model_A_class, model_A_pretrain_path = model_dict[dataset][model_A_type]
    assert (model_A_pretrain_path is not None), "no pretrain model A {}".format(model_A_type)
    model_A = model_A_class(channels=channels, points=points, num_classes=n_classes)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()
    # train_accuracy = model_eval(model_A, train_loader)
    # test_accuracy = model_eval(model_A, test_loader)
    # print(log_msg("Train Set: Accuracy {:.6f}".format(train_accuracy), "INFO"))
    # print(log_msg("Test Set: Accuracy {:.6f}".format(test_accuracy), "INFO"))

    print(log_msg("Loading model B {}".format(model_B_type), "INFO"))
    model_B_class, model_B_pretrain_path = model_dict[dataset][model_B_type]
    assert (model_B_pretrain_path is not None), "no pretrain model B {}".format(model_B_type)
    model_B = model_B_class(channels=channels, points=points, num_classes=n_classes)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()
    # train_accuracy = model_eval(model_B, train_loader)
    # test_accuracy = model_eval(model_B, test_loader)
    # print(log_msg("Train Set: Accuracy {:.6f}".format(train_accuracy), "INFO"))
    # print(log_msg("Test Set: Accuracy {:.6f}".format(test_accuracy), "INFO"))

    # init data augmentation
    augmentation_type = cfg.AUGMENTATION
    augmentation_method = am_dict[augmentation_type]()

    # Normalization
    normalize = cfg.NORMALIZATION

    # Extraction
    extract = cfg.EXTRACTION

    # init feature selection
    selection_type = cfg.SELECTION.TYPE
    selection_method = fsm_dict[selection_type]()
    selection_rate = cfg.SELECTION.RATE
    # 预先计算所有样本的特征归因图，训练时只使用训练集样本的特征归因图
    if selection_type in ["DiffShapley"]:
        all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B,
                                                                  n_classes, window_length, cfg.SELECTION.Diff.M,
                                                                  num_gpus=num_gpus, num_cpus=num_cpus)

    # init explainer
    explainer_type = cfg.EXPLAINER.TYPE
    explainer = explainer_dict[explainer_type]()
    max_depth = cfg.EXPLAINER.MAX_DEPTH
    min_samples_leaf = cfg.EXPLAINER.MIN_SAMPLES_LEAF
    # all initialization is ok

    # log config
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write("Run time: {}\n".format(datetime.now()))
        writer.write("CONFIG:\n{}".format(cfg.dump()))

    # models predict differences
    output_A, pred_target_A = output_predict_targets(model_A, data)
    output_B, pred_target_B = output_predict_targets(model_B, data)
    delta_target = (pred_target_A != pred_target_B).astype(int)

    # K-Fold evaluation
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4)
    skf_id = 0
    # record metrics of i-th Fold
    precision_l, recall_l, f1_l, num_rules_l, average_num_rule_preds_l, num_unique_preds_l = [], [], [], [], [], []
    for train_index, test_index in skf.split(data, delta_target):
        x_train = data[train_index]
        output_A_train = output_A[train_index]
        output_B_train = output_B[train_index]
        pred_target_A_train = pred_target_A[train_index]
        pred_target_B_train = pred_target_B[train_index]

        # selection_method.fit(x_train.reshape((-1, channels * points)), output_A[train_index], output_B[train_index])

        x_test = data[test_index]
        output_A_test = output_A[test_index]
        output_B_test = output_B[test_index]
        pred_target_A_test = pred_target_A[test_index]
        pred_target_B_test = pred_target_B[test_index]

        x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index])
        output_A_train, pred_target_A_train = output_predict_targets(model_A, x_train_aug)
        output_B_train, pred_target_B_train = output_predict_targets(model_B, x_train_aug)

        ydiff = (pred_target_A_train != pred_target_B_train).astype(int)
        print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")
        delta_diff = (ydiff != delta_target_aug).astype(int)
        print(
            f"delta_diffs in X_train = {delta_diff.sum()} / {len(delta_diff)} = {(delta_diff.sum() / len(delta_diff) * 100):.2f}%")

        x_train_aug = x_train_aug.reshape((len(x_train_aug), -1))
        x_test = x_test.reshape((len(test_index), -1))
        # 之后数据形状均为（n_samples, channels*points）

        # For Feature Selection to Compute Feature Contributions
        if selection_type in ["DiffShapley"]:
            selection_method.fit(x_train, model_A, model_B, channels, points, n_classes,
                                 window_length, cfg.SELECTION.Diff.M, all_sample_feature_maps[train_index],
                                 num_gpus=num_gpus, num_cpus=num_cpus)
        else:
            selection_method.fit(x_train_aug, output_A_train, output_B_train)

        # Normalization
        # if normalize:
        #     data_normalize = DatasetNormalization(x_train_aug)
        #     x_train_aug = data_normalize(x_train_aug)
        #     x_test = data_normalize(x_test)
        #     # x_train_aug = sample_normalize(x_train_aug)
        #     # x_test = sample_normalize(x_test)
        # from sklearn.preprocessing import Binarizer, KBinsDiscretizer
        #
        # transformer = Binarizer()
        # # transformer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=None)
        # transformer.fit(x_train_aug)
        # x_train_aug = transformer.transform(x_train_aug)
        # x_test = transformer.transform(x_test)

        # Execute Feature Selection
        x_train_aug, _ = selection_method.transform(x_train_aug, selection_rate)
        x_test, select_indices = selection_method.transform(x_test, selection_rate)
        x_feature_names = feature_names[select_indices]

        # Feature Extraction
        if extract:
            x_train_aug = feature_extraction(x_train_aug, window_length)
            x_test = feature_extraction(x_test, window_length)

        x_train = pd.DataFrame(x_train_aug, columns=x_feature_names)
        x_test = pd.DataFrame(x_test, columns=x_feature_names)
        print(x_train.shape, x_test.shape)

        if explainer_type in ["Logit"]:
            contributions = selection_method.computing_contribution()
            kth = int(len(contributions) * selection_rate)
            ind = np.argpartition(contributions, kth=-kth)[-kth:]
            explainer.fit(x_train, output_A_train, output_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf,
                          feature_weights=contributions[ind], feature_names=x_feature_names)
        else:
            explainer.fit(x_train, pred_target_A_train, pred_target_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf, feature_names=x_feature_names)

        diff_rules = explainer.explain()
        # print(diff_rules)

        # Computation of metrics on train and test set
        if explainer_type in ["Logit"]:
            train_metrics = explainer.metrics(x_train, output_A_train, output_B_train, name="train")
            test_metrics = explainer.metrics(x_test, output_A_test, output_B_test)
        else:
            train_metrics = explainer.metrics(x_train, pred_target_A_train, pred_target_B_train, name="train")
            test_metrics = explainer.metrics(x_test, pred_target_A_test, pred_target_B_test)

        print("skf_id", skf_id, "Explainer", explainer_type, "max_depth", max_depth, "min_samples_leaf",
              min_samples_leaf)
        print("Train set", train_metrics)
        print("Test set", test_metrics)
        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
            writer.write("skf_id {} Explainer {} max_depth {} min_samples_leaf {}\n".format(
                skf_id, explainer_type, max_depth, min_samples_leaf))
            writer.write("Train {}\n".format(train_metrics))
            writer.write("Test {}\n".format(test_metrics))
            # writer.write("train_index {}\n".format(train_index))
            writer.write("test_index {}\n".format(test_index))

        precision_l.append(test_metrics["test-precision"])
        recall_l.append(test_metrics["test-recall"])
        f1_l.append(test_metrics["test-f1"])
        num_rules_l.append(test_metrics["num-rules"])
        average_num_rule_preds_l.append(test_metrics["average-num-rule-preds"])
        num_unique_preds_l.append(test_metrics["num-unique-preds"])

        save_dict = {"explainer": explainer if explainer_type not in ["MERLIN"] else [],
                     "diff_rules": diff_rules,
                     "test_index": test_index,
                     "train_metrics": train_metrics,
                     "test_metrics": test_metrics,
                     }
        save_path = os.path.join(log_path, "{}_{}".format(explainer_type, skf_id))
        save_checkpoint(save_dict, save_path)

        skf_id += 1

    # print("test-precision(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(precision_l), pstdev(precision_l), precision_l))
    # print("test-recall(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(recall_l), pstdev(recall_l), recall_l))
    # print("test-f1(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(f1_l), pstdev(f1_l), f1_l))
    # print("num-rules(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(num_rules_l), pstdev(num_rules_l), num_rules_l))
    # print("average-num-rule-preds(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(average_num_rule_preds_l), pstdev(average_num_rule_preds_l), average_num_rule_preds_l))
    # print("num-unique-preds(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
    #     mean(num_unique_preds_l), pstdev(num_unique_preds_l), num_unique_preds_l))
    print("precision\trecall\tf1\tnum-rules\taverage-num-rule-preds\tnum-unique-preds")
    print("{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}"
          .format(mean(precision_l), pstdev(precision_l), mean(recall_l), pstdev(recall_l), mean(f1_l), pstdev(f1_l),
                  mean(num_rules_l), pstdev(num_rules_l), mean(average_num_rule_preds_l),
                  pstdev(average_num_rule_preds_l), mean(num_unique_preds_l), pstdev(num_unique_preds_l)))
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write(os.linesep + "-" * 25 + os.linesep)
        writer.write("test-precision(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(precision_l), pstdev(precision_l), precision_l))
        writer.write("test-recall(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(recall_l), pstdev(recall_l), recall_l))
        writer.write("test-f1(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(f1_l), pstdev(f1_l), f1_l))
        writer.write("num-rules(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(num_rules_l), pstdev(num_rules_l), num_rules_l))
        writer.write("average-num-rule-preds(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(average_num_rule_preds_l), pstdev(average_num_rule_preds_l), average_num_rule_preds_l))
        writer.write("num-unique-preds(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
            mean(num_unique_preds_l), pstdev(num_unique_preds_l), num_unique_preds_l))
        writer.write("precision\trecall\tf1\tnum-rules\taverage-num-rule-preds\tnum-unique-preds\n")
        writer.write("{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t"
                     "{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\n"
                     .format(mean(precision_l), pstdev(precision_l), mean(recall_l),
                             pstdev(recall_l), mean(f1_l), pstdev(f1_l),
                             mean(num_rules_l), pstdev(num_rules_l), mean(average_num_rule_preds_l),
                             pstdev(average_num_rule_preds_l), mean(num_unique_preds_l), pstdev(num_unique_preds_l)))
        writer.write("{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}\t"
                     "{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}\n"
                     .format(mean(precision_l), pstdev(precision_l), mean(recall_l),
                             pstdev(recall_l), mean(f1_l), pstdev(f1_l),
                             mean(num_rules_l), pstdev(num_rules_l), mean(average_num_rule_preds_l),
                             pstdev(average_num_rule_preds_l), mean(num_unique_preds_l), pstdev(num_unique_preds_l)))
        writer.write(os.linesep + "-" * 25 + os.linesep)
