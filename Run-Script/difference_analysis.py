import argparse
import os
from datetime import datetime
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from differlib import explainer_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import log_msg, setup_seed, get_data_loader_from_dataset, load_checkpoint, \
    get_data_labels_from_dataset, save_checkpoint
from differlib.models import model_dict
from differlib.imd.imd import IMDExplainer, SeparateSurrogate
from differlib.imd.utils import visualize_jst
from differlib.DeltaXpainer import DeltaExplainer
from differlib.LogitDeltaRule import LogitDeltaRule

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

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # init dataset & models
    data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE))
    dataset = cfg.DATASET.TYPE
    n_samples, channels, points = data.shape
    n_classes = len(set(labels))
    assert channels == cfg.DATASET.CHANNELS
    assert points == cfg.DATASET.POINTS
    assert n_classes == cfg.DATASET.NUM_CLASSES
    n_splits = cfg.DATASET.NUM_SPLITS

    print(log_msg("Loading model A {}".format(cfg.MODELS.A), "INFO"))
    model_A_type, model_A_pretrain_path = model_dict[cfg.MODELS.A]
    assert (model_A_pretrain_path is not None), "no pretrain model A {}".format(cfg.MODELS.A)
    model_A = model_A_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()

    print(log_msg("Loading model B {}".format(cfg.MODELS.B), "INFO"))
    model_B_type, model_B_pretrain_path = model_dict[cfg.MODELS.B]
    assert (model_B_pretrain_path is not None), "no pretrain model B {}".format(cfg.MODELS.B)
    model_B = model_B_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()

    # init explainer
    explainer_name = cfg.EXPLAINER.TYPE
    explainer = explainer_dict[cfg.EXPLAINER.TYPE]
    max_depth = cfg.EXPLAINER.MAX_DEPTH
    min_samples_leaf = cfg.EXPLAINER.MIN_SAMPLES_LEAF
    # all initialization is ok

    # log config
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write("Run time: {}".format(datetime.now()))
        writer.write("CONFIG:\n{}".format(cfg.dump()))

    # models predict differences
    data_torch = torch.from_numpy(data).float().cuda()
    output_A = model_A(data_torch)
    _, pred_target_A = output_A.topk(1, 1, True, True)
    output_A = output_A.cpu().detach().numpy()
    pred_target_A = pred_target_A.squeeze().cpu().detach().numpy()
    if model_A.__class__.__name__ in ["LFCNN", "VARCNN", "HGRN"]:
        output_A = np.exp(output_A) / np.sum(np.exp(output_A), axis=-1, keepdims=True)

    output_B = model_B(data_torch)
    _, pred_target_B = output_B.topk(1, 1, True, True)
    output_B = output_B.cpu().detach().numpy()
    pred_target_B = pred_target_B.squeeze().cpu().detach().numpy()
    if model_B.__class__.__name__ in ["LFCNN", "VARCNN", "HGRN"]:
        output_B = np.exp(output_B) / np.sum(np.exp(output_B), axis=-1, keepdims=True)

    delta_target = pred_target_A ^ pred_target_B

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    from boruta import BorutaPy

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', perc=75, alpha=0.05, two_step=True, max_iter=100, verbose=2,
                             random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(data.reshape((-1, channels * points)), delta_target)

    # check selected features - first 5 features are selected
    print(feat_selector.support_)

    # check ranking of features
    print(feat_selector.ranking_)

    # call transform() on X to filter it down to selected features
    data_filtered = feat_selector.transform(data.reshape((-1, channels * points)))
    # data_filtered = data.reshape((n_samples, -1))
    # data_filtered = data[:, :, :].reshape((n_samples, -1))

    # K-Fold evaluation
    skf = StratifiedKFold(n_splits=n_splits)
    skf_id = 0
    # record metrics of i-th Fold
    precision_l, recall_l, f1_l, num_rules_l, average_num_rule_preds_l, num_unique_preds_l = [], [], [], [], [], []
    for train_index, test_index in skf.split(data_filtered, delta_target):
        x_train = pd.DataFrame(data_filtered[train_index])
        x_test = pd.DataFrame(data_filtered[test_index])

        if explainer_name == "LogitDeltaRule":
            explainer.fit(x_train, output_A[train_index], output_B[train_index], max_depth, min_samples_leaf=min_samples_leaf)
        else:
            explainer.fit(x_train, pred_target_A[train_index], pred_target_B[train_index], max_depth, min_samples_leaf=min_samples_leaf)

        diffrules = explainer.explain()
        print(diffrules)

        # Computation of metrics
        # on train set
        if explainer_name == "LogitDeltaRule":
            train_metrics = explainer.metrics(x_train, output_A[train_index], output_B[train_index], name="train")
        else:
            train_metrics = explainer.metrics(x_train, pred_target_A[train_index], pred_target_B[train_index], name="train")

        # on train set
        if explainer_name == "LogitDeltaRule":
            test_metrics = explainer.metrics(x_test, output_A[test_index], output_B[test_index])
        else:
            test_metrics = explainer.metrics(x_test, pred_target_A[test_index], pred_target_B[test_index])

        print("skf_id", skf_id, "Explainer", explainer_name, "max_depth", max_depth, "min_samples_leaf", min_samples_leaf)
        print("Train set", train_metrics)
        print("Test set", test_metrics)
        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
            writer.write("skf_id {} Explainer {} max_depth {} min_samples_leaf {}\n".format(
                skf_id, explainer_name, max_depth, min_samples_leaf))
            writer.write("Train {}\n".format(train_metrics))
            writer.write("Test {}\n".format(test_metrics))
            writer.write("train_index {}\n".format(train_index))
            writer.write("test_index {}\n".format(test_index))

        precision_l.append(test_metrics["test-precision"])
        recall_l.append(test_metrics["test-recall"])
        f1_l.append(test_metrics["test-f1"])
        num_rules_l.append(test_metrics["num-rules"])
        average_num_rule_preds_l.append(test_metrics["average-num-rule-preds"])
        num_unique_preds_l.append(test_metrics["num-unique-preds"])

        save_checkpoint(explainer, os.path.join(log_path, "{}_{}-{}".format(skf_id, explainer_name, max_depth)))
        save_checkpoint(diffrules, os.path.join(log_path, "{}_diffrules".format(skf_id)))
        save_checkpoint(test_index, os.path.join(log_path, "{}_test_index".format(skf_id)))
        save_checkpoint(test_metrics, os.path.join(log_path, "{}_test_metrics".format(skf_id)))

        skf_id += 1

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
