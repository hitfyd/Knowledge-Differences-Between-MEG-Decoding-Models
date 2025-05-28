import argparse
import os
import shelve
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
from numpy import argmax
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

from differlib.augmentation import am_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, dataset_info_dict, predict)
from differlib.explainer import explainer_dict
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps, diff_shapley
from differlib.models import model_dict, scikit_models, torch_models


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


def output_predict_targets(model_type, model, data: np.ndarray, num_classes=2, batch_size=512, softmax=True):
    output, predict_targets = None, None
    if model_type in scikit_models:
        predict_targets = model.predict(data.reshape((len(data), -1)))
        output = model.predict_proba(data.reshape((len(data), -1)))
    elif model_type in torch_models:
        output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
        predict_targets = np.argmax(output, axis=1)
    else:
        print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
    assert output is not None
    assert predict_targets is not None
    return output, predict_targets


class CustomBatchNorm(BaseEstimator, TransformerMixin):
    def __init__(self, gamma=1.0, beta=0.0):
        self.gamma = gamma
        self.beta = beta
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        X_normalized = (X - self.mean) / (self.std + 1e-5)
        return self.gamma * X_normalized + self.beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="../configs/CamCAN/Logit.yaml")
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
    # log_path = re.sub(r'[:]', '_', log_path)
    record_path = os.path.join(log_prefix, project)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS

    # init dataset & models
    dataset = cfg.DATASET
    train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    train_loader = get_data_loader(train_data, train_labels)
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    n_samples, channels, points = test_data.shape
    n_classes = len(set(test_labels))
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

    model_A = load_pretrained_model(model_A_type)
    model_B = load_pretrained_model(model_B_type)

    selection_type = cfg.SELECTION.TYPE
    selection_M = cfg.SELECTION.Diff.M
    selection_threshold = cfg.SELECTION.Diff.THRESHOLD

    # init explainer
    explainer_type = cfg.EXPLAINER.TYPE
    explainer = explainer_dict[explainer_type]()
    max_depth = cfg.EXPLAINER.MAX_DEPTH
    min_samples_leaf = cfg.EXPLAINER.MIN_SAMPLES_LEAF

    aug_data = np.load(f"/tmp/CourrgqpZb/OUTPUT/{dataset}/ddpm_fake_{dataset}.npy").astype(np.float32)
    # aug_data = aug_data.swapaxes(1, 2)
    aug_data = aug_data.reshape(-1, channels, points)
    train_data = np.concatenate((train_data, aug_data), axis=0)

    # models predict differences
    output_A, pred_target_A = output_predict_targets(model_A_type, model_A, train_data, num_classes=n_classes)
    output_B, pred_target_B = output_predict_targets(model_B_type, model_B, train_data, num_classes=n_classes)

    output_A_test, pred_target_A_test = output_predict_targets(model_A_type, model_A, test_data, num_classes=n_classes)
    output_B_test, pred_target_B_test = output_predict_targets(model_B_type, model_B, test_data, num_classes=n_classes)

    output_dalta = output_A - output_B  # output_A_test - output_B_test     output_A - output_B
    delta_output = np.abs(output_dalta).mean(axis=1)
    sort_index = np.argsort(delta_output)[::-1]

    # delta_threshold = 0.02
    # print((delta_output>delta_threshold).sum())
    # for idx in sort_index:
    #     print(idx, delta_output[idx])
    #     if delta_output[idx] < delta_threshold:
    #         break

    save_file = f"./feature_maps/{dataset}_{model_A.__class__.__name__}_{model_B.__class__.__name__}_{window_length}_{selection_M}"
    if os.path.exists(save_file):
        feature_maps = load_checkpoint(save_file)
        print("feature_maps has been loaded")
    else:
        feature_maps = diff_shapley(train_data[sort_index][:15000], model_A, model_B, window_length, selection_M, n_classes)
        if not isinstance(feature_maps, np.ndarray):
            feature_maps = feature_maps.detach().cpu().numpy()
        # np.save(save_file, feature_maps)

    n_samples, channels, points = train_data.shape
    label_logit_delta = abs(output_dalta).sum(axis=0)
    if output_dalta.shape[1] == 2:
        sample_weights = output_dalta[:, 0]
    else:
        sample_weights = output_dalta[:, argmax(label_logit_delta)]
    contributions = np.average(feature_maps, axis=0)#, weights=sample_weights[sort_index][:15000])
    if output_dalta.shape[1] == 2:
        contributions = contributions[:, 0]
    else:
        contributions = contributions[:, argmax(label_logit_delta)]
    contributions = np.repeat(contributions, window_length)
    mean = contributions.mean()
    std = contributions.std()
    print("mean", mean, "std", std)
    z_contributions = (contributions - mean) / std  # (self.contributions) / std
    abs_contributions = np.abs(z_contributions)
    indices = np.where(abs_contributions > selection_threshold)[0]

    # scaler = CustomBatchNorm()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)

    x_train_aug = train_data.reshape((len(train_data), -1))
    x_test = test_data.reshape((len(test_data), -1))

    x_train_aug = x_train_aug[:, indices]
    x_test = x_test[:, indices]
    feature_names = feature_names[indices]

    x_train = pd.DataFrame(x_train_aug, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)
    print(x_train.shape, x_test.shape)

    if explainer_type in ["Logit", "LogitRuleFit"]:
        explainer.fit(x_train, output_A, output_B, max_depth, min_samples_leaf=min_samples_leaf)
    elif explainer_type in ["SS", "IMD"]:
        jstobj, t1, t2 = explainer.fit_detail(x_train, pred_target_A, pred_target_B, max_depth, min_samples_leaf=min_samples_leaf)

        y_surrogate1 = jstobj.predict(test_data, t1)
        y_surrogate2 = jstobj.predict(test_data, t2)

        surrogate1_accuracy = sklearn.metrics.accuracy_score(test_labels, y_surrogate1) * 100
        surrogate2_accuracy = sklearn.metrics.accuracy_score(test_labels, y_surrogate2) * 100
        print('surrogate1_accuracy: {:.2f}\tsurrogate2_accuracy: {:.2f}'.format(surrogate1_accuracy, surrogate2_accuracy))

    else:
        explainer.fit(x_train, pred_target_A, pred_target_B, max_depth, min_samples_leaf=min_samples_leaf)

    # Computation of metrics on train and test set
    if explainer_type in ["Logit", "LogitRuleFit"]:
        train_metrics = explainer.metrics(x_train, output_A, output_B, name="train")
        test_metrics = explainer.metrics(x_test, output_A_test, output_B_test)
    else:
        train_metrics = explainer.metrics(x_train, pred_target_A, pred_target_A, name="train")
        test_metrics = explainer.metrics(x_test, pred_target_A_test, pred_target_B_test)

    diff_rules = explainer.explain()

    print(train_metrics)
    print(test_metrics)
