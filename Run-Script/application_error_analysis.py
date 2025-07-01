import os

import torch

from differlib.engine.utils import load_checkpoint, dataset_info_dict, get_data_labels_from_dataset
from differlib.models import load_pretrained_model, output_predict_targets

dataset = "DecMeg2014"  # "DecMeg2014"  "CamCAN"
channels, points, num_classes = dataset_info_dict[dataset].values()
explainer_type = "Logit"
model_types = ["rf", "mlp", "varcnn", "hgrn", "atcnet"]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # 'cuda:1'
print(f"Using device: {device}")
skf_ids = 1

data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))

for model_A_type in model_types[:-1]:
    for model_B_type in model_types[model_types.index(model_A_type) + 1:]:
        log_path = os.path.join('./output/', f"{dataset}/{explainer_type}MODEL_A:{model_A_type},MODEL_B:{model_B_type}")

        model_A = load_pretrained_model(model_A_type, dataset, channels, points, num_classes, device)
        model_B = load_pretrained_model(model_B_type, dataset, channels, points, num_classes, device)

        feature_importance = {}
        for skf_id in range(skf_ids):
            save_path = os.path.join(log_path, "{}_{}".format(explainer_type, skf_id))
            save_dict = load_checkpoint(save_path)
            rules = save_dict["diff_rules"]
            test_index = save_dict["test_index"]

            test_data, test_labels = data[test_index], labels[test_index]

            output_A, pred_target_A = output_predict_targets(model_A_type, model_A, test_data, num_classes=num_classes, device=device)
            output_B, pred_target_B = output_predict_targets(model_B_type, model_B, test_data, num_classes=num_classes, device=device)

            for sample_idx, sample in enumerate(test_data):
                print(test_index[sample_idx], test_labels[sample_idx], pred_target_A[sample_idx], pred_target_B[sample_idx])

                if pred_target_A[sample_idx] != test_labels[sample_idx] and  pred_target_B[sample_idx] != test_labels[sample_idx]:
                    for rule in rules:
                        depth = len(rule.predicates)
                        ok_depth = 0
                        for cond in rule.predicates:
                            feature = cond[0].strip()
                            op = cond[1].strip()    # '<=' '>'
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
                            print(rule)
                            continue

            # # 计算规则相似度矩阵
            # sim_matrix = np.zeros((len(rules), len(rules)))
            # for i, r1 in enumerate(rules):
            #     for j, r2 in enumerate(rules):
            #         shared_conds = len(set(r1.predicates) & set(r2.predicates))
            #         sim_matrix[i,j] = shared_conds / max(len(r1.predicates), len(r2.predicates))
            #
            # # 聚类可视化
            # plt.figure(figsize=(15,8))
            # dn = hierarchy.dendrogram(hierarchy.linkage(sim_matrix, method='ward'),
            #                           labels=[f"Rule{i}" for i in range(len(rules))],
            #                           leaf_rotation=90)
            # plt.title("Rule Clustering by Condition Similarity")
            # plt.show()
            #
            # # 提取规则核心特征
            # rule_vectors = []
            # for rule in rules:
            #     vec = [rule.class_label[0], rule.class_label[1], len(rule.predicates), max(abs(rule.class_label[:num_classes]))]
            #     rule_vectors.append(vec)
            #
            # # t-SNE降维
            # tsne = TSNE(n_components=2, perplexity=5)
            # embedding = tsne.fit_transform(np.array(rule_vectors))
            #
            # # 交互式3D散点图
            # fig = px.scatter_3d(x=embedding[:,0], y=embedding[:,1], z=embedding[:,2],
            #                     color=[f"Cluster{dn['leaves_color_list'][i]}" for i in range(len(rules))],
            #                     hover_name=[f"Rule{i}" for i in range(len(rules))],
            #                     size=[10*max(abs(r.class_label[:num_classes])) for r in rules])
            # fig.show()
