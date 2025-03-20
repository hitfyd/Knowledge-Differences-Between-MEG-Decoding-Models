import os
import shelve

import numpy as np
import mne

from MEG_Explanation_Comparison import top_k_consensus, top_k_disagreement
from MEG_Shapley_Values import topomap_plot
from differlib.engine.utils import dataset_info_dict, save_figure

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
# top-k
top_k_list = [0.05, 0.1, 0.2]    # 0.05, 0.1, 0.2
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

    abs_top_sort = np.zeros((num_models, channels*points), dtype=np.int32)
    abs_mean_contribution = np.zeros((num_models, channels, points), dtype=np.float32)
    for i in range(num_models):
        model_i = compared_model_names[i]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        abs_top_sort_i, abs_sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], \
            npz_i['sign_sort_maps']
        abs_top_sort[i] = abs_top_sort_i
        # 关键步骤：创建逆向索引
        inverse_indices_i = np.argsort(abs_top_sort_i)
        # 恢复原始数组
        abs_mean_contribution_i = abs_sort_contribution_i[inverse_indices_i].reshape(channels, points)
        sign_mean_maps_i = sign_sort_maps_i[inverse_indices_i].reshape(channels, points, num_classes)
        abs_mean_contribution[i] = abs_mean_contribution_i



    for top_k in top_k_list:
        k = int(channels * points * top_k)
        abs_top_sort_k = abs_top_sort[:, :k]

        # 计算共识
        union_consensus = abs_top_sort_k[0]
        for arr in abs_top_sort_k[1:]:
            union_consensus = np.intersect1d(union_consensus, arr, assume_unique=False)
            if union_consensus.size == 0:  # 提前终止优化
                break

        union_consensus_masks = np.zeros_like(abs_top_sort[0], dtype=np.bool_)
        union_consensus_masks[union_consensus] = True
        union_consensus_masks = union_consensus_masks.reshape(channels, points)
        print("union_consensus:", len(union_consensus))
        model_mean_contribution = abs_mean_contribution.mean(axis=0)
        union_consensus_contribution = model_mean_contribution * union_consensus_masks
        consensus_title = 'Consensus of All Models'
        fig, _, _ = topomap_plot(consensus_title, union_consensus_contribution, channels_info, channels=channels, top_channel_num=5)
        save_figure(fig, save_path, '{}_{}_all_models_consensus'.format(dataset, top_k))

        # 保存union_consensus
        file = '{}/{}_top_{}_union_consensus.npz'.format(save_path, dataset, top_k)
        np.savez(file, union_consensus=union_consensus, union_consensus_masks=union_consensus_masks, union_consensus_contribution=union_consensus_contribution)

