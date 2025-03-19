import os
import shelve

import numpy as np
import mne

from MEG_Explanation_Comparison import top_k_consensus, top_k_disagreement
from MEG_Shapley_Values import topomap_plot
from differlib.engine.utils import dataset_info_dict, save_figure

# datasets
datasets = ["CamCAN"]     # "DecMeg2014", "CamCAN"
# top-k
top_k_list = [0.1]    # 0.05, 0.1, 0.2
compared_model_names = ["Linear", "LFCNN", "ATCNet"]    # "Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"
num_models = len(compared_model_names)
assert num_models >= 2

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

# init dataset & models
for dataset in datasets:
    # save config
    save_path = f"./output/Consensus_and_Disagreement_{datasets}/"

    dataset_info = dataset_info_dict[dataset]
    channels, points, num_classes = dataset_info["CHANNELS"], dataset_info["POINTS"], dataset_info["NUM_CLASSES"]

    for i in range(num_models-1):
        model_i = compared_model_names[i]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        abs_top_sort_i, abs_sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], \
            npz_i['sign_sort_maps']
        # 关键步骤：创建逆向索引
        inverse_indices_i = np.argsort(abs_top_sort_i)
        # 恢复原始数组
        abs_mean_contribution_i = abs_sort_contribution_i[inverse_indices_i].reshape(channels, points)
        sign_mean_maps_i = sign_sort_maps_i[inverse_indices_i].reshape(channels, points, num_classes)   # 后续可用于分类别分析

        for j in range(i+1, num_models):
            model_j = compared_model_names[j]
            npz_j = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_j))
            abs_top_sort_j, abs_sort_contribution_j, sign_sort_maps_j = npz_j['abs_top_sort'], npz_j['abs_sort_contribution'], \
                npz_j['sign_sort_maps']
            # 关键步骤：创建逆向索引
            inverse_indices_j = np.argsort(abs_top_sort_j)
            # 恢复原始数组
            abs_mean_contribution_j = abs_sort_contribution_j[inverse_indices_j].reshape(channels, points)
            sign_mean_maps_j = sign_sort_maps_j[inverse_indices_j].reshape(channels, points, num_classes)


            for top_k in top_k_list:
                k = int(channels * points * top_k)

                # 计算共识
                consensus_list, consensus_masks = top_k_consensus(abs_top_sort_i, abs_top_sort_j, k)
                consensus_masks = consensus_masks.reshape(channels, points)
                print("consensus_features:", len(consensus_list))
                consensus_contribution = abs_mean_contribution_i * consensus_masks
                consensus_title = 'Consensus of {} and {}'.format(model_i, model_j)
                fig, _, _ = topomap_plot(consensus_title, consensus_contribution, channels_info, channels=channels, top_channel_num=5)
                save_figure(fig, save_path, '{}_{}_{}_{}_consensus'.format(dataset, model_i, model_j, top_k))

                # 计算差异
                disagreement_list, disagreement_masks = top_k_disagreement(abs_top_sort_i, abs_top_sort_j, k)
                disagreement_masks = disagreement_masks.reshape(channels, points)
                print("disagreement_features:", len(disagreement_list))
                disagreement_contribution_i = abs_mean_contribution_i * disagreement_masks
                disagreement_title = '{} Disagreement with {}'.format(model_i, model_j)
                fig, _, _ = topomap_plot(disagreement_title, disagreement_contribution_i, channels_info, channels=channels, top_channel_num=5)
                save_figure(fig, save_path, '{}_{}_{}_{}_disagreement'.format(dataset, model_i, model_j, top_k))

                disagreement_contribution_j = abs_mean_contribution_j * disagreement_masks
                disagreement_title = '{} Disagreement with {}'.format(model_j, model_i)
                fig, _, _ = topomap_plot(disagreement_title, disagreement_contribution_j, channels_info, channels=channels, top_channel_num=5)
                save_figure(fig, save_path, '{}_{}_{}_{}_disagreement'.format(dataset, model_j, model_i, top_k))

