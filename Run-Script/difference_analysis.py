import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from differlib.augmentation import am_dict
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (log_msg, setup_seed, load_checkpoint, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint, dataset_info_dict, predict)
from differlib.explainer import explainer_dict
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps
from differlib.models import model_dict, scikit_models, torch_models

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


    def load_pretrained_model(model_type):
        print(log_msg("Loading model {}".format(model_type), "INFO"))
        model_class, model_pretrain_path = model_dict[dataset][model_type]
        assert (model_pretrain_path is not None), "no pretrain model A {}".format(model_A_type)
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


    def output_predict_targets(model_type, model, data: np.ndarray, num_classes=2, batch_size=1024, softmax=True):
        output, predict_targets = None, None
        if model_type in scikit_models:
            predict_targets = model.predict(data.reshape((len(data), -1)))
            output = model.predict_proba(data.reshape((len(data), -1)))
        elif model_type in torch_models:
            output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
            predict_targets = np.argmax(output, axis=1)
            # _, predict_targets = output.topk(1, 1, True, True)
            # output = output.cpu().detach().numpy()
            # predict_targets = predict_targets.squeeze().cpu().detach().numpy()
        else:
            print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
        assert output is not None
        assert predict_targets is not None
        return output, predict_targets


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
        all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B,
                                                                  n_classes, window_length, selection_M,
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
    output_A, pred_target_A = output_predict_targets(model_A_type, model_A, data, num_classes=n_classes)
    output_B, pred_target_B = output_predict_targets(model_B_type, model_B, data, num_classes=n_classes)
    delta_target = (pred_target_A != pred_target_B).astype(int)

    # K-Fold evaluation
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25)
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.EXPERIMENT.SEED)
    skf_id = 0
    # record metrics of i-th Fold
    pd_test_metrics, pd_train_metrics = None, None
    for train_index, test_index in skf.split(data, delta_target):
        x_train = data[train_index]

        x_test = data[test_index]
        output_A_test = output_A[test_index]
        output_B_test = output_B[test_index]
        pred_target_A_test = pred_target_A[test_index]
        pred_target_B_test = pred_target_B[test_index]

        x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index], augment_factor=augment_factor,)

        output_A_train, pred_target_A_train = output_predict_targets(model_A_type, model_A, x_train_aug, num_classes=n_classes)
        output_B_train, pred_target_B_train = output_predict_targets(model_B_type, model_B, x_train_aug, num_classes=n_classes)

        ydiff = (pred_target_A_train != pred_target_B_train).astype(int)
        print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        x_train_aug = x_train_aug.reshape((len(x_train_aug), -1))
        x_test = x_test.reshape((len(test_index), -1))
        # 之后数据形状均为（n_samples, channels*points）

        # For Feature Selection to Compute Feature Contributions
        if selection_type in ["DiffShapley"]:
            selection_method.fit(x_train, model_A, model_B, channels, points, n_classes,
                                 window_length, selection_M, all_sample_feature_maps[train_index],
                                 threshold=selection_threshold, num_gpus=num_gpus, num_cpus=num_cpus)
        else:
            selection_method.fit(x_train_aug, output_A_train, output_B_train)

        # Execute Feature Selection
        x_train_aug, _ = selection_method.transform(x_train_aug)
        x_test, select_indices = selection_method.transform(x_test)
        x_feature_names = feature_names[select_indices]

        x_train = pd.DataFrame(x_train_aug, columns=x_feature_names)
        x_test = pd.DataFrame(x_test, columns=x_feature_names)
        print(x_train.shape, x_test.shape)

        if explainer_type in ["Logit", "LogitRuleFit"]:
            contributions = selection_method.computing_contribution()
            # kth = int(len(contributions) * selection_rate)
            # ind = np.argpartition(contributions, kth=-kth)[-kth:]
            explainer.fit(x_train, output_A_train, output_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf,
                          feature_weights=contributions[select_indices])
        else:
            explainer.fit(x_train, pred_target_A_train, pred_target_B_train,
                          max_depth, min_samples_leaf=min_samples_leaf)

        diff_rules = explainer.explain()
        # print(diff_rules)

        # Computation of metrics on train and test set
        if explainer_type in ["Logit", "LogitRuleFit"]:
            train_metrics = explainer.metrics(x_train, output_A_train, output_B_train, name="train")
            test_metrics = explainer.metrics(x_test, output_A_test, output_B_test)
        else:
            train_metrics = explainer.metrics(x_train, pred_target_A_train, pred_target_B_train, name="train")
            test_metrics = explainer.metrics(x_test, pred_target_A_test, pred_target_B_test)

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
        writer.write(os.linesep + "-" * 25 + os.linesep)

    # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
    record_file = os.path.join(record_path, f"{model_A_type}_{model_B_type}_record.csv")
    record_mean_std['model_A'] = model_A_type
    record_mean_std['model_B'] = model_B_type
    record_mean_std['explainer'] = tags
    if os.path.exists(record_file):
        all_record_mean_std = pd.read_csv(record_file)
        assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
    else:
        all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
    all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
    all_record_mean_std.to_csv(record_file, index=False)
