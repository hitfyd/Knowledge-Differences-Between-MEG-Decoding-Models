import argparse
import os
import time
from datetime import datetime
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from differlib.augmentation import am_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, output_predict_targets, model_eval, sample_normalize)
from differlib.explainer import explainer_dict
from differlib.feature_extraction import feature_extraction
from differlib.feature_selection import fsm_dict
from differlib.models import model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # init loggers
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if args.opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(args.opts[::2], args.opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)
    n_repetitions = cfg.EXPERIMENT.NUM_REPETITIONS
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    num_gpus = torch.cuda.device_count()
    num_cpus = cfg.EXPERIMENT.CPU_COUNT

    # init dataset & models
    train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(cfg.DATASET.TYPE))
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE))
    train_loader = get_data_loader(train_data, train_labels)
    test_loader = get_data_loader(test_data, test_labels)

    dataset = cfg.DATASET.TYPE
    n_samples, channels, points = train_data.shape
    n_classes = len(set(train_labels))
    assert channels == cfg.DATASET.CHANNELS
    assert points == cfg.DATASET.POINTS
    assert n_classes == cfg.DATASET.NUM_CLASSES
    # n_splits = cfg.DATASET.NUM_SPLITS

    print(log_msg("Loading model A {}".format(cfg.MODELS.A), "INFO"))
    model_A_type, model_A_pretrain_path = model_dict[cfg.MODELS.A]
    assert (model_A_pretrain_path is not None), "no pretrain model A {}".format(cfg.MODELS.A)
    model_A = model_A_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()
    train_accuracy = model_eval(model_A, train_loader)
    test_accuracy = model_eval(model_A, test_loader)
    print(log_msg("Train Set: Accuracy {:.6f}".format(train_accuracy), "INFO"))
    print(log_msg("Test Set: Accuracy {:.6f}".format(test_accuracy), "INFO"))

    print(log_msg("Loading model B {}".format(cfg.MODELS.B), "INFO"))
    model_B_type, model_B_pretrain_path = model_dict[cfg.MODELS.B]
    assert (model_B_pretrain_path is not None), "no pretrain model B {}".format(cfg.MODELS.B)
    model_B = model_B_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()
    train_accuracy = model_eval(model_B, train_loader)
    test_accuracy = model_eval(model_B, test_loader)
    print(log_msg("Train Set: Accuracy {:.6f}".format(train_accuracy), "INFO"))
    print(log_msg("Test Set: Accuracy {:.6f}".format(test_accuracy), "INFO"))

    # init data augmentation
    augmentation_type = cfg.AUGMENTATION.TYPE
    augmentation_method = am_dict[augmentation_type]()

    # Normalization
    normalize = cfg.NORMALIZATION.FLAG

    # Extraction
    extract = cfg.EXTRACTION.FLAG

    # init feature selection
    selection_type = cfg.SELECTION.TYPE
    selection_method = fsm_dict[selection_type]()
    selection_rate = cfg.SELECTION.RATE

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

    # models predict differences for training
    train_output_A, train_predict_targets_A = output_predict_targets(model_A, train_data)
    train_output_B, train_predict_targets_B = output_predict_targets(model_B, train_data)
    train_delta_target = np.array(train_predict_targets_A != train_predict_targets_B).astype(int)

    # models predict differences for test
    test_output_A, test_predict_targets_A = output_predict_targets(model_A, test_data)
    test_output_B, test_predict_targets_B = output_predict_targets(model_B, test_data)
    test_delta_target = np.array(test_predict_targets_A != test_predict_targets_B).astype(int)

    # record metrics of i-th Fold
    precision_l, recall_l, f1_l, num_rules_l, average_num_rule_preds_l, num_unique_preds_l = [], [], [], [], [], []
    for repetition_id in range(n_repetitions):

        # TODO: augmentation_method.augment(train_data, model_A, model_B)
        train_data_aug, train_delta_target_aug = augmentation_method.augment(train_data, train_delta_target)
        train_output_A, train_predict_targets_A = output_predict_targets(model_A, train_data_aug)
        train_output_B, train_predict_targets_B = output_predict_targets(model_B, train_data_aug)

        ydiff = np.array(train_predict_targets_A != train_predict_targets_B).astype(int)
        print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")
        delta_diff = np.array(ydiff != train_delta_target_aug).astype(int)
        print(
            f"delta_diffs in X_train = {delta_diff.sum()} / {len(delta_diff)} = {(delta_diff.sum() / len(delta_diff) * 100):.2f}%")

        x_train_aug = train_data_aug.reshape((len(train_data_aug), -1))
        x_test = test_data.reshape((len(test_data), -1))
        # 之后数据形状均为（n_samples, channels*points）

        # For Feature Selection to Compute Feature Contributions
        time_start = time.perf_counter()
        if selection_type in ["DiffShapley"]:
            window_length, M = cfg.SELECTION.Diff.WINDOW_LENGTH, cfg.SELECTION.Diff.M
            selection_method.fit(x_train_aug, model_A, model_B, channels, points, n_classes, window_length, M,
                                 num_gpus=num_gpus, num_cpus=num_cpus)
        else:
            selection_method.fit(x_train_aug, train_output_A, train_output_B)
        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print("Feature Selection Run Time: {}".format(run_time))

        # Normalization
        if normalize:
            x_train_aug = sample_normalize(x_train_aug)
            x_test = sample_normalize(x_test)

        # Execute Feature Selection, 和Normalization二选一
        x_train_aug = selection_method.transform(x_train_aug, selection_rate)
        x_test = selection_method.transform(x_test, selection_rate)

        # Feature Extraction
        if extract:
            x_train_aug = feature_extraction(x_train_aug, cfg.EXTRACTION.WINDOW_LENGTH)
            x_test = feature_extraction(x_test, cfg.EXTRACTION.WINDOW_LENGTH)

        x_train = pd.DataFrame(x_train_aug)
        x_test = pd.DataFrame(x_test)
        print(x_train.shape, x_test.shape)

        for explainer_type in ["Logit", "Delta", "IMD", "SS"]:  # "Logit", "Delta", "IMD", "SS"
            explainer = explainer_dict[explainer_type]()
            if explainer_type in ["Logit"]:
                explainer.fit(x_train, train_output_A, train_output_B,
                              max_depth, min_samples_leaf=min_samples_leaf)
            else:
                explainer.fit(x_train, train_predict_targets_A, train_predict_targets_B,
                              max_depth, min_samples_leaf=min_samples_leaf)

            diffrules = explainer.explain()
            print(diffrules)

            # Computation of metrics
            if explainer_type in ["Logit"]:
                train_metrics = explainer.metrics(x_train, train_output_A, train_output_B, name="train")
                test_metrics = explainer.metrics(x_test, test_output_A, test_output_B)
            else:
                train_metrics = explainer.metrics(x_train, train_predict_targets_A, train_predict_targets_B, name="train")
                test_metrics = explainer.metrics(x_test, test_predict_targets_A, test_predict_targets_B)

            print("repetition_id", repetition_id, "Explainer", explainer_type,
                  "max_depth", max_depth, "min_samples_leaf", min_samples_leaf)
            print("Train set", train_metrics)
            print("Test set", test_metrics)
            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                writer.write("repetition_id {} Explainer {} max_depth {} min_samples_leaf {}\n".format(
                    repetition_id, explainer_type, max_depth, min_samples_leaf))
                writer.write("Train {}\n".format(train_metrics))
                writer.write("Test {}\n".format(test_metrics))

            precision_l.append(test_metrics["test-precision"])
            recall_l.append(test_metrics["test-recall"])
            f1_l.append(test_metrics["test-f1"])
            num_rules_l.append(test_metrics["num-rules"])
            average_num_rule_preds_l.append(test_metrics["average-num-rule-preds"])
            num_unique_preds_l.append(test_metrics["num-unique-preds"])

            save_checkpoint(explainer, os.path.join(log_path, f"{repetition_id}_{explainer_type}-{max_depth}"))
            save_checkpoint(diffrules, os.path.join(log_path, f"{repetition_id}_{explainer_type}_diffrules"))
            save_checkpoint(test_metrics, os.path.join(log_path, f"{repetition_id}_{explainer_type}_test_metrics"))

    print("test-precision(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(precision_l), pstdev(precision_l), precision_l))
    print("test-recall(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(recall_l), pstdev(recall_l), recall_l))
    print("test-f1(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(f1_l), pstdev(f1_l), f1_l))
    print("num-rules(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(num_rules_l), pstdev(num_rules_l), num_rules_l))
    print("average-num-rule-preds(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(average_num_rule_preds_l), pstdev(average_num_rule_preds_l), average_num_rule_preds_l))
    print("num-unique-preds(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
        mean(num_unique_preds_l), pstdev(num_unique_preds_l), num_unique_preds_l))
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
        writer.write(os.linesep + "-" * 25 + os.linesep)
