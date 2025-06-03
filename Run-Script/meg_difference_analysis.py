import argparse
import os
import shelve
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
from mne.time_frequency import psd_array_multitaper
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from differlib.augmentation import am_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, dataset_info_dict, predict)
from differlib.explainer import explainer_dict
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps
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
        self.mean = np.mean(X)
        self.std = np.std(X)
        return self

    def transform(self, X):
        X_normalized = (X - self.mean) / (self.std + 1e-5)
        return self.gamma * X_normalized + self.beta


# 方法 2: 获取所有 BatchNorm 层参数
def get_all_bn_params(model):
    """获取模型中所有 BatchNorm 层的参数"""
    bn_params = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            bn_params[name] = {
                'weight': module.weight.data.clone(),
                'bias': module.bias.data.clone(),
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'eps': module.eps,
                'momentum': module.momentum,
                'num_batches_tracked': module.num_batches_tracked
            }
    print(log_msg(bn_params, "INFO"))
    return bn_params


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
    num_gpus = torch.cuda.device_count()
    num_cpus = cfg.EXPERIMENT.CPU_COUNT

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    # train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    # train_loader = get_data_loader(train_data, train_labels)
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

    model_A = load_pretrained_model(model_A_type)
    model_B = load_pretrained_model(model_B_type)

    # init data augmentation
    augmentation_type = cfg.AUGMENTATION
    augment_factor = cfg.AUGMENT_FACTOR
    augmentation_method = am_dict[augmentation_type]()

    # init feature selection
    selection_type = cfg.SELECTION.TYPE
    selection_method = fsm_dict[selection_type]()
    selection_M = cfg.SELECTION.Diff.M
    selection_threshold = cfg.SELECTION.Diff.THRESHOLD
    # 预先计算所有样本的特征归因图，训练时只使用训练集样本的特征归因图
    if selection_type in ["DiffShapley"]:
        all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M)

        # db_path = f'./output/Consensus/{dataset}/{dataset}_ShapleyValueExplainer_attribution_testset'
        # db = shelve.open(db_path)
        # # 逐样本迭代
        # sample_num = len(data)
        # model1_name = model_A.__class__.__name__
        # model2_name = model_B.__class__.__name__
        # all_maps = np.zeros([sample_num, channels, points, n_classes], dtype=np.float32)
        # all_maps2 = np.zeros_like(all_maps)
        # for sample_id in range(sample_num):
        #     attribution_id = f"{sample_id}_{model1_name}"
        #     assert attribution_id in db
        #     all_maps[sample_id] = db[attribution_id]
        #
        #     all_maps2[sample_id] = db[f"{sample_id}_{model2_name}"]
        #
        # all_sample_feature_maps = all_maps - all_maps2
        # window_length = 1
        # all_sample_feature_maps = all_sample_feature_maps.reshape(sample_num, -1, n_classes)

        # abs_mean_maps = np.abs(all_maps).mean(axis=0)
        # abs_feature_contribution = abs_mean_maps.sum(axis=-1).reshape(-1)  # 合并一个特征对所有类别的绝对贡献
        # abs_top_sort = np.argsort(abs_feature_contribution)[::-1]
        # abs_mean_maps2 = np.abs(all_maps2).mean(axis=0)
        # abs_feature_contribution2 = abs_mean_maps2.sum(axis=-1).reshape(-1)  # 合并一个特征对所有类别的绝对贡献
        # abs_top_sort2 = np.argsort(abs_feature_contribution2)[::-1]
        # top_k = 1020
        # # consensus_list = abs_top_sort[:top_k]
        # consensus_list, consensus_masks = top_k_disagreement(abs_top_sort, abs_top_sort2, top_k, top_k)

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

    # scaler = CustomBatchNorm()
    # data = scaler.fit_transform(data)
    # test_mean, test_std = test_data.mean(), test_data.std()
    # train_mean, train_std = train_data.mean(), train_data.std()
    # data = (data - test_mean) / test_std * train_std + train_mean
    # new_data = np.zeros((len(data), channels, 36))
    # for idx, epoch in enumerate(data):
    #     psd, f = psd_array_multitaper(
    #         epoch,
    #         sfreq=125,
    #         fmin=1,
    #         fmax=45,
    #         bandwidth=3.0,  # 频带平滑
    #         adaptive=True,  # 自适应权重
    #         n_jobs=-1,
    #         normalization='length'  # 归一化
    #     )
    #     new_data[idx] = psd

    # models predict differences
    output_A, pred_target_A = output_predict_targets(model_A_type, model_A, data, num_classes=n_classes)
    output_B, pred_target_B = output_predict_targets(model_B_type, model_B, data, num_classes=n_classes)
    delta_target = (pred_target_A != pred_target_B).astype(int)
    delta_weights = np.abs(output_A - output_B).mean(axis=1)

    # aug = np.load(f"/home/fan/Diffusion-TS/OUTPUT/{dataset}/ddpm_fake_{dataset}.npy")
    # aug = aug.reshape(-1, channels, points)

    # K-Fold evaluation
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=cfg.EXPERIMENT.SEED)   # 0.1   0.25
    # skf = StratifiedKFold(n_splits=n_splits)
    skf_id = 0
    # record metrics of i-th Fold
    pd_test_metrics, pd_train_metrics = None, None

    # train_results = get_all_bn_params(model_A)
    # model_A = load_checkpoint("mlp.tmp")
    # test_results = get_all_bn_params(model_A)
    for train_index, test_index in skf.split(data, delta_target):
        x_train = data[train_index]
        x_test = data[test_index]

        # output_A_test, pred_target_A_test = output_predict_targets(model_A_type, model_A, x_test, num_classes=n_classes)
        # output_B_test, pred_target_B_test = output_predict_targets(model_B_type, model_B, x_test, num_classes=n_classes)

        x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index],
                                                                    augment_factor=augment_factor, )
        # if augmentation_type == "NONE":
        #     x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index], augment_factor=augment_factor,)
        # else:
        #     aug_save_path = f"./aug_data/{dataset}_{skf_id}_{augment_factor}"
        #     if os.path.exists(aug_save_path):
        #         x_train_aug = load_checkpoint(aug_save_path)
        #     else:
        #         x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index], augment_factor=augment_factor, )
        #         save_checkpoint(x_train_aug, aug_save_path)
        # x_train_aug = np.concatenate((x_train_aug, aug), axis=0)

        output_A_train, pred_target_A_train = output_predict_targets(model_A_type, model_A, x_train_aug, num_classes=n_classes)
        output_B_train, pred_target_B_train = output_predict_targets(model_B_type, model_B, x_train_aug, num_classes=n_classes)

        ydiff = (pred_target_A_train != pred_target_B_train).astype(int)
        print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.4f}%")

        x_train_aug = x_train_aug.reshape((len(x_train_aug), -1))
        x_test = x_test.reshape((len(x_test), -1))
        # x_train_aug = new_data[train_index].reshape((len(x_train_aug), -1))
        # x_test = new_data[test_index].reshape((len(x_test), -1))
        # 之后数据形状均为（n_samples, channels*points）

        # For Feature Selection to Compute Feature Contributions
        if selection_type in ["DiffShapley"]:
            # all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M)
            selection_method.fit(x_train, model_A, model_B, channels, points, n_classes,
                                 window_length, selection_M, all_sample_feature_maps[train_index],
                                 threshold=selection_threshold, num_gpus=num_gpus, num_cpus=num_cpus)
        else:
            selection_method.fit(x_train_aug, output_A_train, output_B_train)

        # Execute Feature Selection
        x_train_aug, _ = selection_method.transform(x_train_aug)
        x_test, select_indices = selection_method.transform(x_test)
        x_feature_names = feature_names[select_indices]

        x_train_aug = x_train_aug.reshape((len(x_train_aug), -1, window_length))
        x_test = x_test.reshape((len(x_test), -1, window_length))
        x_train_aug = x_train_aug.max(axis=-1)  # mean max
        x_test = x_test.max(axis=-1)
        x_feature_names = x_feature_names[::window_length]
        # x_train_aug = x_train_aug[:, consensus_list]
        # x_test = x_test[:, consensus_list]
        # x_feature_names = feature_names[consensus_list]

        x_train = pd.DataFrame(x_train_aug, columns=x_feature_names)
        x_test = pd.DataFrame(x_test, columns=x_feature_names)
        print(x_train.shape, x_test.shape)

        if explainer_type in ["Logit", "LogitRuleFit"]:
            contributions = selection_method.computing_contribution()
            # kth = int(len(contributions) * selection_rate)
            # ind = np.argpartition(contributions, kth=-kth)[-kth:]
            explainer.fit(x_train, output_A_train, output_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf,
                          # feature_weights=contributions[select_indices]
                          )
        elif explainer_type in ["SS", "IMD"]:
            jstobj, t1, t2 = explainer.fit_detail(x_train, pred_target_A_train, pred_target_B_train, max_depth, min_samples_leaf=min_samples_leaf)

            surrogate_test_data, _ = selection_method.transform(test_data.reshape((len(test_data), -1)))
            y_surrogate1 = jstobj.predict(surrogate_test_data, t1)
            y_surrogate2 = jstobj.predict(surrogate_test_data, t2)

            surrogate1_accuracy = sklearn.metrics.accuracy_score(test_labels, y_surrogate1) * 100
            surrogate2_accuracy = sklearn.metrics.accuracy_score(test_labels, y_surrogate2) * 100
            print('surrogate1_accuracy: {:.2f}\tsurrogate2_accuracy: {:.2f}'.format(surrogate1_accuracy,surrogate2_accuracy))

            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                writer.write('surrogate1_accuracy: {}\tsurrogate2_accuracy: {}'.format(surrogate1_accuracy,surrogate2_accuracy) + os.linesep)
        else:
            explainer.fit(x_train, pred_target_A_train, pred_target_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf)

        diff_rules = explainer.explain()
        # print(diff_rules)

        # Computation of metrics on train and test set
        if explainer_type in ["Logit", "LogitRuleFit"]:
            train_metrics = explainer.metrics(x_train, output_A_train, output_B_train, name="train")
            test_metrics = explainer.metrics(x_test, output_A[test_index], output_B[test_index])
        else:
            train_metrics = explainer.metrics(x_train, pred_target_A_train, pred_target_B_train, name="train")
            test_metrics = explainer.metrics(x_test, pred_target_A[test_index], pred_target_B[test_index])

        # 记录单次实验的训练和测试结果
        train_metrics['train-confusion_matrix'] = np.array2string(train_metrics['train-confusion_matrix'])
        if pd_train_metrics is None:
            pd_train_metrics = pd.DataFrame(columns=train_metrics.keys())
        pd_train_metrics.loc[len(pd_train_metrics)] = train_metrics.values()

        test_metrics['test-confusion_matrix'] = np.array2string(test_metrics['test-confusion_matrix'])
        if pd_test_metrics is None:
            pd_test_metrics = pd.DataFrame(columns=test_metrics.keys())
        pd_test_metrics.loc[len(pd_test_metrics)] = test_metrics.values()

        # 打印单次实验结果
        print("skf_id", skf_id, "Explainer", explainer_type)
        print(pd_train_metrics.to_string())
        print(pd_test_metrics.to_string())

        # 保存单次实验中的中间结果
        save_dict = {"explainer": explainer if explainer_type not in ["MERLIN"] else [],
                     "diff_rules": diff_rules,
                     "test_index": test_index,
                     "train_metrics": train_metrics,
                     "test_metrics": test_metrics,
                     }
        save_path = os.path.join(log_path, "{}_{}".format(explainer_type, skf_id))
        save_checkpoint(save_dict, save_path)

        skf_id += 1

    # 计算测试集上各个指标的均值和标准差
    assert len(pd_test_metrics.columns.tolist()) == 10
    partial_pd_metrics = pd_test_metrics.iloc[:, 3:]
    partial_pd_metrics_mean, partial_pd_metrics_std = partial_pd_metrics.mean(), partial_pd_metrics.std()
    record_mean_std = pd.Series(index=partial_pd_metrics_mean.index, dtype=str)
    for i in range(len(partial_pd_metrics_mean.values)):
        record_mean_std.iloc[i] = f"{partial_pd_metrics_mean.iloc[i]:.2f} ± {partial_pd_metrics_std.iloc[i]:.2f}"
    print(record_mean_std.to_string())
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write(os.linesep + "-" * 25 + os.linesep)
        writer.write(pd_train_metrics.to_string() + os.linesep)
        writer.write(pd_test_metrics.to_string() + os.linesep)
        writer.write(record_mean_std.to_string() + os.linesep)
        writer.write(partial_pd_metrics_mean.to_string() + os.linesep)
        writer.write(partial_pd_metrics_std.to_string() + os.linesep)
        for index in ["test-precision", "test-recall", "test-f1", "num-rules", "num-unique-preds"]:
            writer.write(f"{index}:\t{np.array2string(partial_pd_metrics[index].values, separator=', ')}" + os.linesep)
        writer.write(os.linesep + "-" * 25 + os.linesep)

    # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
    record_file = os.path.join(record_path, f"{model_A_type}_{model_B_type}_record.csv")
    record_mean_std['model_A'] = model_A_type
    record_mean_std['model_B'] = model_B_type
    record_mean_std['explainer'] = tags
    for index in ["test-precision", "test-recall", "test-f1", "num-rules", "num-unique-preds"]:
        record_mean_std[index] = partial_pd_metrics[index].values
    if os.path.exists(record_file):
        all_record_mean_std = pd.read_csv(record_file, encoding="utf_8_sig")
        assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
    else:
        all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
    all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
    all_record_mean_std.to_csv(record_file, index=False, encoding="utf_8_sig")
