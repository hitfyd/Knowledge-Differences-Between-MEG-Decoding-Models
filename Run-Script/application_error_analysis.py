import os

import numpy as np
import sklearn
import torch
from scipy.stats import ttest_ind

from differlib.engine.utils import load_checkpoint, dataset_info_dict, get_data_labels_from_dataset
from differlib.models import load_pretrained_model, output_predict_targets


def error_rules(rule, samples, sample_idxs):
    ok_sum = 0
    ok_sample_idxs = []
    for sample_idx, sample in zip(sample_idxs, samples):
        depth = len(rule.predicates)
        ok_depth = 0
        for cond in rule.predicates:
            feature = cond[0].strip()
            op = cond[1].strip()  # '<=' '>'
            value = cond[2]
            c, t = feature.split("T")
            c = int(c[1:])
            t = int(t)
            if op == '<=' and sample[c, t] <= value:
                ok_depth += 1
            elif op == '>' and sample[c, t] > value:
                ok_depth += 1
            else:
                continue
        if ok_depth == depth:
            ok_sum += 1
            ok_sample_idxs.append(sample_idx)
    return ok_sum, ok_sample_idxs


def stats_labes(labels, pred_targets_A, pred_targets_B, sample_idxs):
    all_yes_list, A_yes_list, B_yes_list, all_no_list = [], [], [], []
    for idx in range(len(labels)):
        if labels[idx] == pred_targets_A[idx] and labels[idx] == pred_targets_B[idx]:
            all_yes_list.append(sample_idxs[idx])
        elif labels[idx] == pred_targets_A[idx] and labels[idx] != pred_targets_B[idx]:
            A_yes_list.append(sample_idxs[idx])
        elif labels[idx] == pred_targets_B[idx] and labels[idx] != pred_targets_A[idx]:
            B_yes_list.append(sample_idxs[idx])
        else:
            all_no_list.append(sample_idxs[idx])
    return all_yes_list, A_yes_list, B_yes_list, all_no_list


dataset = "DecMeg2014"  # "DecMeg2014"  "CamCAN"
tags = "Search"
channels, points, num_classes = dataset_info_dict[dataset].values()
explainer_type = "Logit"
model_types = ["rf", "atcnet"]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # 'cuda:1'
print(f"Using device: {device}")
skf_ids = [0]
max_depth_list = [5]
augment_factor_list = [0]
selection_threshold_list = [3.0]   # [0, 1.0, 3.0, 5.0]
data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))

for model_A_type in model_types[:-1]:
    for model_B_type in model_types[model_types.index(model_A_type) + 1:]:
        log_path = os.path.join('./output/', f"{dataset}/{tags}MODEL_A:{model_A_type},MODEL_B:{model_B_type}")

        model_A = load_pretrained_model(model_A_type, dataset, channels, points, num_classes, device)
        model_B = load_pretrained_model(model_B_type, dataset, channels, points, num_classes, device)

        output_A, pred_target_A = output_predict_targets(model_A_type, model_A, data, num_classes=num_classes, device=device)
        output_B, pred_target_B = output_predict_targets(model_B_type, model_B, data, num_classes=num_classes, device=device)

        for max_depth in max_depth_list:
            for augment_factor in augment_factor_list:
                for selection_threshold in selection_threshold_list:
                    acc_A_test_, acc_B_test_, fusion_acc_test_ = [], [], []
                    for skf_id in skf_ids:
                        save_path = os.path.join(log_path, f"{explainer_type}_{model_A_type}_{model_B_type}_{max_depth}_{augment_factor}_{selection_threshold}_{skf_id}")
                        save_dict = load_checkpoint(save_path)
                        rules = save_dict["diff_rules"]
                        test_index = save_dict["test_index"]
                        train_index = [item for item in range(len(data)) if item not in test_index]
                        train_data, train_labels = data[train_index], labels[train_index]
                        test_data, test_labels = data[test_index], labels[test_index]

                        fusion_tagerts = np.zeros_like(labels)

                        for rule_idx, rule in enumerate(rules):
                            print("*"*25)
                            print(rule_idx, rule)
                            ok_sum, ok_sample_idxs = error_rules(rule, train_data, train_index)
                            # print("train", ok_sum, ok_sample_idxs)
                            all_yes_list, A_yes_list, B_yes_list, all_no_list = stats_labes(labels[ok_sample_idxs], pred_target_A[ok_sample_idxs], pred_target_B[ok_sample_idxs], ok_sample_idxs)
                            print("train", len(all_yes_list), len(A_yes_list), len(B_yes_list), len(all_no_list))
                            print("A_yes_list")
                            for idx in A_yes_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])
                            print("B_yes_list")
                            for idx in B_yes_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])
                            print("all_no_list")
                            for idx in all_no_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])

                            # 统计训练集中，符合某个规则下，哪个模型正确次数多，在测试集下符合该规则的样本即使用该模型的结果。
                            apply_A, apply_B = False, False
                            if len(A_yes_list) > len(B_yes_list):
                                print('apply_A')
                                apply_A = True
                                fusion_tagerts[ok_sample_idxs] = pred_target_A[ok_sample_idxs]
                            else:
                                apply_B = True
                                fusion_tagerts[ok_sample_idxs] = pred_target_B[ok_sample_idxs]

                            # 测试集分析
                            ok_sum, ok_sample_idxs = error_rules(rule, test_data, test_index)
                            # print('test', ok_sum, ok_sample_idxs)
                            all_yes_list, A_yes_list, B_yes_list, all_no_list = stats_labes(labels[ok_sample_idxs], pred_target_A[ok_sample_idxs], pred_target_B[ok_sample_idxs], ok_sample_idxs)
                            print('test', len(all_yes_list), len(A_yes_list), len(B_yes_list), len(all_no_list))
                            print("A_yes_list")
                            for idx in A_yes_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])
                            print("B_yes_list")
                            for idx in B_yes_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])
                            print("all_no_list")
                            for idx in all_no_list:
                                print(idx, labels[idx], pred_target_A[idx], output_A[idx], pred_target_B[idx], output_B[idx])

                            # 选择模型决策融合后的结果
                            if apply_A:
                                fusion_tagerts[ok_sample_idxs] = pred_target_A[ok_sample_idxs]
                            else:
                                fusion_tagerts[ok_sample_idxs] = pred_target_B[ok_sample_idxs]

                        # 统计模型A，模型B，决策融合后的精度
                        acc_A_test = sklearn.metrics.accuracy_score(test_labels, pred_target_A[test_index])
                        acc_B_test = sklearn.metrics.accuracy_score(test_labels, pred_target_B[test_index])
                        acc_fusion = sklearn.metrics.accuracy_score(test_labels, fusion_tagerts[test_index])
                        print("skf_id", skf_id, "acc_A_test", acc_A_test, "acc_B_test", acc_B_test, "acc_fusion", acc_fusion)
                        acc_A_test_.append(acc_A_test)
                        acc_B_test_.append(acc_B_test)
                        fusion_acc_test_.append(acc_fusion)

                    print(max_depth, augment_factor, selection_threshold)
                    acc_A_test_, acc_B_test_, fusion_acc_test_ = np.array(acc_A_test_), np.array(acc_B_test_), np.array(fusion_acc_test_)
                    print(f"acc_A_test: {acc_A_test_.mean()} {acc_A_test_.std()} {acc_A_test_}")
                    print(f"acc_B_test: {acc_B_test_.mean()} {acc_B_test_.std()} {acc_B_test_}")
                    print(f"fusion_acc_test: {fusion_acc_test_.mean()} {fusion_acc_test_.std()} {fusion_acc_test_}")
                    p_value = ttest_ind(acc_A_test_, fusion_acc_test_).pvalue
                    print(f"p_value: {p_value}")
                    p_value = ttest_ind(acc_B_test_, fusion_acc_test_).pvalue
                    print(f"p_value: {p_value}")
