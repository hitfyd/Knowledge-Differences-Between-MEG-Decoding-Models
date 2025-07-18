import os
import shelve

import numpy as np
from matplotlib import pyplot as plt, gridspec

from differlib.engine.utils import load_checkpoint, dataset_info_dict, save_figure
from similarity.attribution.MEG_Shapley_Values import topomap_plot_axis

dataset = "DecMeg2014"  # "DecMeg2014"  "CamCAN"
label_names = ['Audio', 'Visual']
window_length = 10
M=2
if dataset == 'DecMeg2014':
    label_names = ['Scramble', 'Face']
    window_length = 25
channels, points, num_classes = dataset_info_dict[dataset].values()
models = ['rf', 'varcnn', 'hgrn', 'atcnet'] # 'rf', 'mlp', 'lfcnn', 'varcnn', 'hgrn', 'atcnet', 'ctnet'
model_names= {
    'rf': 'RandomForestClassifier',
    'varcnn': 'VARCNN',
    'hgrn': 'HGRN',
    'atcnet': 'ATCNet',
}
skf_ids = 5

max_depth = 6
augment_factor_list = [3.0]
selection_threshold_list = [3.0]
tags = "Logit"
explainer_dict = {
    'Delta': 'DeltaXplainer',
    'SS': 'Separate Surrogates',
    'IMD': 'IMD',
    'MERLIN': 'MERLIN',
    'Logit': 'BO-RPPD',
    'attribution': 'Feature Attribution',
}

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

fig = plt.figure(figsize=(6*6, 6*len(explainer_dict)))
fig.suptitle(f'Topographic maps of model differencing on {dataset}', fontsize=24, fontweight='bold', y=0.98)
fig.subplots_adjust(top=0.98, bottom=0.05)
gridlayout = gridspec.GridSpec(ncols=25*6, nrows=6*len(explainer_dict), figure=fig, top=None, bottom=None, wspace=None, hspace=0)

explainer_idx = 0
for explainer, explainer_name in explainer_dict.items():
    column_idx = 0
    for model_A_type in models[:-1]:
        for model_B_type in models[models.index(model_A_type) + 1:]:
            log_path = os.path.join('./output/', f"{dataset}_benchmark/{tags}MODEL_A:{model_A_type},MODEL_B:{model_B_type}")
            for augment_factor in augment_factor_list:
                for selection_threshold in selection_threshold_list:
                    if explainer == "attribution":
                        save_file = os.path.join("./feature_maps/", f"{dataset}_{model_names[model_A_type]}_{model_names[model_B_type]}_{window_length}_{M}")
                        attribution_maps = load_checkpoint(save_file)
                        attribution_maps = attribution_maps.mean(axis=0)
                        attribution_maps = np.abs(attribution_maps).sum(axis=1)
                        attribution_maps = np.repeat(attribution_maps, window_length)
                        attribution_maps = attribution_maps.reshape(channels, points)
                    else:
                        feature_importance = {}
                        for skf_id in range(skf_ids):
                            saved_diff = f"{explainer}_{skf_id}"
                            save_path = os.path.join(log_path, saved_diff)
                            rules = load_checkpoint(save_path)["diff_rules"]

                            # 统计特征在规则中的出现频率和平均深度
                            for rule in rules:
                                depth = len(rule.predicates)
                                for cond in rule.predicates:
                                    if explainer in ['MERLIN']:
                                        _feature, value = cond.split("LEQ")
                                        _feature = _feature.strip()
                                        if _feature[0] == '~':
                                            feature = _feature[1:]
                                            operator = '>'
                                        else:
                                            feature = _feature
                                            operator = '<='
                                    else:
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
                    figure_name = f"{dataset}_{model_A_type}_{model_B_type}_{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"
                    axis = fig.add_subplot(gridlayout[6*explainer_idx:6*(explainer_idx+1), 25*column_idx+1:25*column_idx + 24])
                    axis_colorbar = fig.add_subplot(gridlayout[6*explainer_idx+1:6*(explainer_idx+1), 25*column_idx + 24])
                    topomap_plot_axis(axis, axis_colorbar, attribution_maps, channels_info, channels=channels, top_channel_num=0)
                    # fig, heatmap_channel, top_channels = topomap_plot(title, attribution_maps, channels_info, channels=channels, top_channel_num=0)
                    # save_figure(fig, f'./images/{tags}/', f"{figure_name}_topomap")

                    # fig, heatmap_time =  time_curve_plot(title, attribution_maps, points=points)
                    # save_figure(fig, './images/', f"{figure_name}_time_curve")

                    if explainer_idx == 0:
                        title = f'{model_names[model_A_type]} vs {model_names[model_B_type]}'
                        axis.set_title(title, y=0.98, fontsize=20, fontweight='bold')

                    if column_idx == 0:
                        axis.set_ylabel(explainer_name, fontsize=20, fontweight='bold', labelpad=14, x=0.05)

            column_idx += 1
    explainer_idx += 1
# plt.show()
save_figure(fig, f'./images/{tags}/', f"{dataset}_topomap")
