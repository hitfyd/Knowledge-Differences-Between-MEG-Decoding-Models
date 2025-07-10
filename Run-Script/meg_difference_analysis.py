import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
# from sklearnex import patch_sklearn
# patch_sklearn()
import sklearn
import torch
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit

from differlib.augmentation import am_dict
from differlib.augmentation.DualMEG_CounterfactualExplainer import counterfactual
from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import (setup_seed, get_data_labels_from_dataset, get_data_loader,
                                    save_checkpoint)
from differlib.explainer import explainer_dict
from differlib.feature_selection import fsm_dict
from differlib.feature_selection.DiffShapleyFS import compute_all_sample_feature_maps
from differlib.models import load_pretrained_model, output_predict_targets


def dynamic_fusion(data, model_A, model_B, explainer, device: torch.device = torch.device("cpu")):
    data_ = data.reshape((len(data), -1))
    select_indices = np.array([np.nonzero(feature_names == i)[0].item() for i in explainer.delta_tree.feature_names_in_])
    data_ = data_[:, select_indices]

    out_A, tag_A = output_predict_targets(model_A_type, model_A, data, num_classes=n_classes, device=device)
    out_B, tag_B = output_predict_targets(model_B_type, model_B, data, num_classes=n_classes, device=device)
    logit_delta_proxy = explainer.delta_tree.predict(data_)
    logit_delta = logit_delta_proxy[:, :n_classes]

    fusion_output, fusion_target = np.zeros_like(logit_delta), np.zeros_like(logit_delta[:, 0])
    for idx, x in enumerate(data):
        weight = np.abs(logit_delta[idx]).max()  # 取最大概率差作权重

        # weight = 0.5 + logit_delta[idx, 0] / 2
        weight = 0.5 + logit_delta[idx, np.abs(logit_delta[idx]).argmax()] / 2
        fusion_output[idx] = weight * out_A[idx] + (1 - weight) * out_B[idx]

        fusion_target = np.argmax(fusion_output, axis=1)

        # if logit_delta[idx, 0] > 0:
        #     fusion_output[idx], fusion_target[idx] = output_predict_targets(model_A_type, model_A,  x[np.newaxis, :], num_classes=n_classes, device=device)
        # else:
        #     fusion_output[idx], fusion_target[idx] = output_predict_targets(model_B_type, model_B,  x[np.newaxis, :], num_classes=n_classes, device=device)

    return fusion_output, fusion_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="../configs/DecMeg2014/Search_SS.yaml")  # DecMeg2014    CamCAN      BCIIV2a
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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # 'cuda:1'
    print(f"Using device: {device}")

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    # train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    # train_loader = get_data_loader(train_data, train_labels)
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

    # init data augmentation
    augmentation_type = cfg.AUGMENTATION
    augment_factor_list = cfg.AUGMENT_FACTOR
    augmentation_method = am_dict[augmentation_type]()

    # init feature selection
    selection_type = cfg.SELECTION.TYPE
    selection_method = fsm_dict[selection_type]()
    selection_M = cfg.SELECTION.Diff.M
    selection_threshold_list = cfg.SELECTION.Diff.THRESHOLD
    # 预先计算所有样本的特征归因图，训练时只使用训练集样本的特征归因图
    if selection_type in ["DiffShapley"]:
        all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M, device=device)

    if augmentation_type in ["Counterfactual"]:
        aug = counterfactual(model_A, model_B, dataset, data, cover=False, device=device, target_model=1)# if model_A_type != 'rf' else 2)

    # init explainer
    explainer_types = cfg.EXPLAINER.TYPE.split(";")
    # if isinstance(explainer_types, str):
    #     explainer_types = [explainer_types]
    max_depth_list = cfg.EXPLAINER.MAX_DEPTH
    min_samples_leaf = cfg.EXPLAINER.MIN_SAMPLES_LEAF
    # all initialization is ok

    # log config
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write("Run time: {}\n".format(datetime.now()))
        writer.write("CONFIG:\n{}".format(cfg.dump()))

    # models predict differences
    output_A, pred_target_A = output_predict_targets(model_A_type, model_A, data, num_classes=n_classes, device=device)
    output_B, pred_target_B = output_predict_targets(model_B_type, model_B, data, num_classes=n_classes, device=device)
    delta_target = (pred_target_A != pred_target_B).astype(int)
    delta_weights = np.abs(output_A - output_B).mean(axis=1)

    # K-Fold evaluation
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=cfg.TEST_SIZE, random_state=cfg.EXPERIMENT.SEED)   # 0.1   0.25
    # skf = StratifiedKFold(n_splits=n_splits)

    for explainer_type in explainer_types:
        explainer = explainer_dict[explainer_type]()

        for max_depth in max_depth_list:
            for augment_factor in augment_factor_list:
                for selection_threshold in selection_threshold_list:
                    setup_seed(cfg.EXPERIMENT.SEED) # 重新固定随机数
                    print(f"{explainer_type} {model_A_type} {model_B_type} max_depth:{max_depth} augment_factor:{augment_factor} selection_threshold:{selection_threshold}")
                    skf_id = 0
                    # record metrics of i-th Fold
                    pd_test_metrics, pd_train_metrics = None, None
                    acc_A_test_, acc_B_test_, fusion_acc_test_ = [], [], []
                    for train_index, test_index in skf.split(data, labels):#delta_target):
                        x_train = data[train_index]
                        x_test = data[test_index]

                        # output_A_test, pred_target_A_test = output_predict_targets(model_A_type, model_A, x_test, num_classes=n_classes, device=device)
                        # output_B_test, pred_target_B_test = output_predict_targets(model_B_type, model_B, x_test, num_classes=n_classes, device=device)

                        if augmentation_type == "Counterfactual":
                            if len(aug.shape) == 3:
                                x_train_aug = np.concatenate((x_train, aug[train_index]), axis=0)
                            elif len(aug.shape) == 4:
                                x_train_aug = np.concatenate((x_train, aug[train_index, :int(augment_factor)].reshape(-1, channels, points)), axis=0)
                                # x_train_aug = aug[train_index, :int(augment_factor)].reshape(-1, channels, points)
                            else:
                                print(aug.shape, "is not a valid augmentation type")
                        else:
                            x_train_aug, delta_target_aug = augmentation_method.augment(x_train, delta_target[train_index], augment_factor=augment_factor, )

                        output_A_train, pred_target_A_train = output_predict_targets(model_A_type, model_A, x_train_aug, num_classes=n_classes, device=device)
                        output_B_train, pred_target_B_train = output_predict_targets(model_B_type, model_B, x_train_aug, num_classes=n_classes, device=device)

                        ydiff = (pred_target_A_train != pred_target_B_train).astype(int)
                        print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.4f}%")

                        x_train_aug = x_train_aug.reshape((len(x_train_aug), -1))
                        x_test = x_test.reshape((len(x_test), -1))

                        # For Feature Selection to Compute Feature Contributions
                        if selection_type in ["DiffShapley"]:
                            # all_sample_feature_maps = compute_all_sample_feature_maps(dataset, data, model_A, model_B, n_classes, window_length, selection_M)
                            selection_method.fit(x_train, model_A, model_B, channels, points, n_classes,
                                                 window_length, selection_M, all_sample_feature_maps[train_index],
                                                 threshold=selection_threshold, device=device)
                        else:
                            selection_method.fit(x_train_aug, output_A_train, output_B_train)

                        # Execute Feature Selection
                        x_train_aug, _ = selection_method.transform(x_train_aug)
                        x_test, select_indices = selection_method.transform(x_test)
                        x_feature_names = feature_names[select_indices]

                        if cfg.Feature_SMOOTHING:
                            x_train_aug = x_train_aug.reshape((len(x_train_aug), -1, window_length))
                            x_test = x_test.reshape((len(x_test), -1, window_length))
                            x_train_aug = x_train_aug.max(axis=-1)  # mean max
                            x_test = x_test.max(axis=-1)
                            x_feature_names = x_feature_names[::window_length]

                        x_train = pd.DataFrame(x_train_aug, columns=x_feature_names)
                        x_test = pd.DataFrame(x_test, columns=x_feature_names)
                        print(x_train.shape, x_test.shape)

                        if explainer_type in ["Logit"]:
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
                        if explainer_type in ["Logit"]:
                            train_metrics = explainer.metrics(x_train, output_A_train, output_B_train, name="train")
                            test_metrics = explainer.metrics(x_test, output_A[test_index], output_B[test_index])
                        else:
                            train_metrics = explainer.metrics(x_train, pred_target_A_train, pred_target_B_train, name="train")
                            test_metrics = explainer.metrics(x_test, pred_target_A[test_index], pred_target_B[test_index])

                        # 记录单次实验的训练和测试结果
                        train_metrics['train_confusion_matrix'] = np.array2string(train_metrics['train_confusion_matrix'])
                        if pd_train_metrics is None:
                            pd_train_metrics = pd.DataFrame(columns=train_metrics.keys())
                        pd_train_metrics.loc[len(pd_train_metrics)] = train_metrics.values()

                        test_metrics['test_confusion_matrix'] = np.array2string(test_metrics['test_confusion_matrix'])
                        if pd_test_metrics is None:
                            pd_test_metrics = pd.DataFrame(columns=test_metrics.keys())
                        pd_test_metrics.loc[len(pd_test_metrics)] = test_metrics.values()

                        # 决策融合
                        if explainer_type in ["Logit"]:
                            x_test, y_test = data[test_index], labels[test_index]
                            pred_target_A_test, pred_target_B_test = pred_target_A[test_index], pred_target_B[test_index]
                            acc_A_test = sklearn.metrics.accuracy_score(y_test, pred_target_A_test)
                            acc_B_test = sklearn.metrics.accuracy_score(y_test, pred_target_B_test)

                            fusion_output_test, fusion_target_test = dynamic_fusion(x_test, model_A, model_B, explainer, device=device)
                            fusion_acc_test = sklearn.metrics.accuracy_score(y_test, fusion_target_test)

                            print(f"skf_id", skf_id, "Explainer", explainer_type, acc_A_test, acc_B_test, fusion_acc_test)
                            acc_A_test_.append(acc_A_test)
                            acc_B_test_.append(acc_B_test)
                            fusion_acc_test_.append(fusion_acc_test)

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
                        save_path = os.path.join(log_path, "{}_{}_{}_{}_{}_{}_{}".format(explainer_type, model_A_type, model_B_type, max_depth, augment_factor, selection_threshold, skf_id))
                        save_checkpoint(save_dict, save_path)

                        skf_id += 1

                    if explainer_type in ["Logit"]:
                        acc_A_test_, acc_B_test_, fusion_acc_test_ = np.array(acc_A_test_), np.array(acc_B_test_), np.array(fusion_acc_test_)
                        p_value_A = ttest_ind(acc_A_test_, fusion_acc_test_).pvalue
                        p_value_B = ttest_ind(acc_B_test_, fusion_acc_test_).pvalue
                        print(f"acc_A_test: {acc_A_test_.mean()} {acc_A_test_.std()} p_value: {p_value_A}")
                        print(f"acc_B_test: {acc_B_test_.mean()} {acc_B_test_.std()} p_value: {p_value_B}")
                        print(f"fusion_acc_test: {fusion_acc_test_.mean()} {fusion_acc_test_.std()}")
                        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                            writer.write(f"acc_A_test: {acc_A_test_.mean()} {acc_A_test_.std()} p_value: {p_value_A} {acc_A_test_}" + os.linesep)
                            writer.write(f"acc_B_test: {acc_B_test_.mean()} {acc_B_test_.std()} p_value: {p_value_B} {acc_B_test_}" + os.linesep)
                            writer.write(f"fusion_acc_test: {fusion_acc_test_.mean()} {fusion_acc_test_.std()} {fusion_acc_test_}" + os.linesep)

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
                        writer.write(f"{explainer_type} {model_A_type} {model_B_type} max_depth:{max_depth} augment_factor:{augment_factor} selection_threshold:{selection_threshold}" + os.linesep)
                        writer.write(pd_train_metrics.to_string() + os.linesep)
                        writer.write(pd_test_metrics.to_string() + os.linesep)
                        writer.write(record_mean_std.to_string() + os.linesep)
                        writer.write(partial_pd_metrics_mean.to_string() + os.linesep)
                        writer.write(partial_pd_metrics_std.to_string() + os.linesep)
                        for index in ["test_precision", "test_recall", "test_f1", "num_rules", "num_unique_preds"]:
                            writer.write(f"{index}:\t{np.array2string(partial_pd_metrics[index].values, separator=', ')}" + os.linesep)
                        writer.write(os.linesep + "-" * 25 + os.linesep)

                    # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
                    record_file = os.path.join(record_path, f"{cfg.EXPERIMENT.TAG}_{model_A_type}_{model_B_type}_record.csv")
                    record_mean_std['model_A'] = model_A_type
                    record_mean_std['model_B'] = model_B_type
                    record_mean_std['window_length'] = window_length
                    record_mean_std['explainer'] = explainer_type
                    record_mean_std['max_depth'] = max_depth
                    record_mean_std['augmentation_type'] = augmentation_type
                    record_mean_std['augment_factor'] = augment_factor
                    record_mean_std['selection_type'] = selection_type
                    record_mean_std['selection_M'] = selection_M
                    record_mean_std['selection_threshold'] = selection_threshold

                    for index in ["test_precision", "test_recall", "test_f1", "num_rules", "num_unique_preds"]:
                        record_mean_std[f"{index}_list"] = partial_pd_metrics[index].values
                    if os.path.exists(record_file):
                        all_record_mean_std = pd.read_csv(record_file, encoding="utf_8_sig")
                        assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
                    else:
                        all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
                    all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
                    all_record_mean_std.to_csv(record_file, index=False, encoding="utf_8_sig")
