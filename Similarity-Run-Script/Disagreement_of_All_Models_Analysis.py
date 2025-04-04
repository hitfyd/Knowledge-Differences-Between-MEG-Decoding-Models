import os
import shelve

import numpy as np
import mne
from sklearn.metrics.pairwise import cosine_similarity

from similarity.analyzer.MEG_Explanation_Comparison import top_k_consensus, top_k_disagreement
from similarity.attribution.MEG_Shapley_Values import topomap_plot
from differlib.engine.utils import dataset_info_dict, save_figure

# datasets
datasets = ["CamCAN"]     # "DecMeg2014", "CamCAN"
# top-k
top_k_list = [0.1]    # 0.05, 0.1, 0.2
compared_model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]    # "Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"
num_models = len(compared_model_names)
assert num_models >= 2

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

# init dataset & models
for dataset in datasets:
    # save config
    save_path = f"./output/Consensus_and_Disagreement_{dataset}/"

    dataset_info = dataset_info_dict[dataset]
    channels, points, num_classes = dataset_info["CHANNELS"], dataset_info["POINTS"], dataset_info["NUM_CLASSES"]

    for i in range(num_models):
        model_i = compared_model_names[i]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        abs_top_sort_i, abs_sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], \
            npz_i['sign_sort_maps']
        # 关键步骤：创建逆向索引
        inverse_indices_i = np.argsort(abs_top_sort_i)
        # 恢复原始数组
        abs_mean_contribution_i = abs_sort_contribution_i[inverse_indices_i].reshape(channels, points)
        sign_mean_maps_i = sign_sort_maps_i[inverse_indices_i].reshape(channels, points, num_classes)

        for top_k in top_k_list:
            consensus_file = '{}/{}_top_{}_union_consensus.npz'.format(save_path, dataset, top_k)
            consensus = np.load(consensus_file)
            union_consensus, union_consensus_masks = consensus['union_consensus'], consensus['union_consensus_masks']

            top_k_percent = int(top_k*100)
            k = int(channels * points * top_k)
            abs_top_sort_k = abs_top_sort_i[:k]

            # 计算与全模型共识的分歧
            disagreement_top_sort_k = []
            for c in abs_top_sort_k:
                if c not in union_consensus:
                    disagreement_top_sort_k.append(c)

            disagreement_mask = np.zeros_like(abs_top_sort_i, dtype=np.bool_)
            disagreement_mask[disagreement_top_sort_k] = True
            disagreement_mask = disagreement_mask.reshape(channels, points)
            for label in range(1, num_classes):
                disagreement_contribution = sign_mean_maps_i[:, :, label] * disagreement_mask

                consensus_title = f'Top-{top_k_percent}% Disagreement of {model_i} (Label {label})'
                fig, heatmap_channel, _ = topomap_plot(consensus_title, disagreement_contribution, channels_info, channels=channels, top_channel_num=5)
                save_figure(fig, save_path, '{}_top_{}_{}_{}_all_disagreement'.format(dataset, top_k, model_i, label))

                # 读取先验
                evoked_feature_db = shelve.open('../dataset/{}_evoked_feature'.format(dataset))
                if dataset == "CamCAN":
                    if label == 0:
                        evoked_feature = evoked_feature_db["aud"]
                    else:
                        evoked_feature = evoked_feature_db["vis"]
                else:
                    if label == 0:
                        evoked_feature = evoked_feature_db["scramble"]
                    else:
                        evoked_feature = evoked_feature_db["face"]
                evoked_feature = (evoked_feature - np.mean(evoked_feature)) / (np.std(evoked_feature))

                cos_sim = cosine_similarity(evoked_feature.reshape(1, -1), heatmap_channel.reshape(1, -1))
                print('Dataset', dataset, 'Model', model_i, "label", label, 'cos_sim', cos_sim)

                # 保存
                file = '{}/{}_top_{}_{}_{}_all_disagreement.npz'.format(save_path, dataset, top_k, model_i, label)
                np.savez(file, label=label, cos_sim=cos_sim, disagreement_contribution=disagreement_contribution, disagreement_top_sort_k=disagreement_top_sort_k)
