import os
import shelve

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE

from differlib.engine.utils import load_checkpoint, dataset_info_dict, save_figure
from similarity.attribution.MEG_Shapley_Values import topomap_plot, time_curve_plot

dataset = "CamCAN"  # "DecMeg2014"  "CamCAN"
label_names = ['Audio', 'Visual']
if dataset == 'DecMeg2014':
    label_names = ['Scramble', 'Face']
channels, points, num_classes = dataset_info_dict[dataset].values()
models = ['rf', 'varcnn', 'hgrn', 'atcnet'] # 'rf', 'mlp', 'lfcnn', 'varcnn', 'hgrn', 'atcnet', 'ctnet'
model_names= {
    'rf': 'Random Forest',
    'varcnn': 'VARCNN',
    'hgrn': 'HGRN',
    'atcnet': 'ATCNet',
}
skf_ids = 5

max_depth = 6
augment_factor_list = [3.0]
selection_threshold_list = [3.0]
tags = "Logit"
explainer_dict = {'Logit': 'BO-RPPD',
                  # 'Delta': 'DeltaXplainer',
                  # 'SS': 'Separate Surrogates',
                  # 'IMD': 'IMD',
                  # 'MERLIN': 'MERLIN',
                  }

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

for model_A_type in models[:-1]:
    for model_B_type in models[models.index(model_A_type) + 1:]:
        log_path = os.path.join('./output/', f"{dataset}_benchmark/{tags}MODEL_A:{model_A_type},MODEL_B:{model_B_type}")
        for explainer, explainer_name in explainer_dict.items():
            for augment_factor in augment_factor_list:
                for selection_threshold in selection_threshold_list:
                    feature_importance = {}
                    for skf_id in range(skf_ids):
                        saved_diff = "{}_{}".format(explainer, skf_id)
                        save_path = os.path.join(log_path, saved_diff)
                        rules = load_checkpoint(save_path)["diff_rules"]

                        # 统计特征在规则中的出现频率和平均深度
                        for rule in rules:
                            depth = len(rule.predicates)
                            for cond in rule.predicates:
                                feature = cond[0].strip()
                                operator = cond[1]  # 提取运算符 > 或 <
                                value = cond[2]  # 提取阈值

                                # 更新重要性（按深度加权）
                                weight = 1 / depth
                                # 初始化特征记录
                                if feature not in feature_importance:
                                    feature_importance[feature] = {
                                        'importance': 0,
                                        'thresholds': [],
                                        'directions': []  # 记录方向（大于/小于）
                                    }
                                feature_importance[feature]['importance'] += weight
                                feature_importance[feature]['thresholds'].append(value)
                                feature_importance[feature]['directions'].append(1 if operator == '>' else -1)


                    attribution_maps = np.zeros((channels, points))
                    for feature, values in feature_importance.items():
                        c, t = feature.split("T")
                        c = int(c[1:])
                        t = int(t)
                        attribution_maps[c, t] = values['importance']
                    # 规则重要性热力图
                    title = f'{model_names[model_A_type]} vs {model_names[model_B_type]}'
                    figure_name = f"{dataset}_{explainer}_{model_A_type}_{model_B_type}_{max_depth}_{augment_factor}_{selection_threshold}"
                    fig, heatmap_channel, top_channels = topomap_plot(title, attribution_maps, channels_info, channels=channels, top_channel_num=0)
                    save_figure(fig, './images/', f"{figure_name}_topomap")

                    # fig, heatmap_time =  time_curve_plot(title, attribution_maps, points=points)
                    # save_figure(fig, './images/', f"{figure_name}_time_curve")
