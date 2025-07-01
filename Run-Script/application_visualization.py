import os
import shelve

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE

from differlib.engine.utils import load_checkpoint, dataset_info_dict, save_figure
from similarity.attribution.MEG_Shapley_Values import topomap_plot, time_curve_plot

dataset = "CamCAN"  # "DecMeg2014"  "CamCAN"
channels, points, num_classes = dataset_info_dict[dataset].values()
explainer_type = "Logit"
model_types = ["rf", "mlp", "varcnn", "hgrn", "atcnet"]
skf_ids = 5

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

for model_A_type in model_types[:-1]:
    for model_B_type in model_types[model_types.index(model_A_type) + 1:]:
        log_path = os.path.join('./output/', f"{dataset}/{explainer_type}MODEL_A:{model_A_type},MODEL_B:{model_B_type}")
        feature_importance = {}
        for skf_id in range(skf_ids):
            save_path = os.path.join(log_path, "{}_{}".format(explainer_type, skf_id))
            rules = load_checkpoint(save_path)["diff_rules"]

            # 统计特征在规则中的出现频率和平均深度
            for rule in rules:
                depth = len(rule.predicates)
                for cond in rule.predicates:
                    feature = cond[0].strip()
                    feature_importance[feature] = feature_importance.get(feature, 0) + 1/depth

        attribution_maps = np.zeros((channels, points))
        for feature, importance in feature_importance.items():
            c, t = feature.split("T")
            c = int(c[1:])
            t = int(t)
            attribution_maps[c, t] = importance
        # 规则重要性热力图
        title = f"{model_A_type} vs. {model_B_type}"
        fig, heatmap_channel, top_channels = topomap_plot(title, attribution_maps, channels_info, channels=channels, top_channel_num=0)
        save_figure(fig, './images/', f"{dataset}_{model_A_type}_{model_B_type}_topomap")

        # fig, heatmap_time =  time_curve_plot(title, attribution_maps, points=points)
        # save_figure(fig, './images/', f"{dataset}_{model_A_type}_{model_B_type}_time_curve")

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
