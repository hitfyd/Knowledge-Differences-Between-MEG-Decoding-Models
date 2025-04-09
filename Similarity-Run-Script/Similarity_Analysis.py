import shelve

import numpy as np
from matplotlib import pyplot as plt, gridspec, colors, colorbar
from scipy.spatial.distance import euclidean, correlation
from sklearn.metrics import pairwise_distances

from similarity.analyzer.MEG_Explanation_Comparison import feature_agreement, plot_similarity_matrix, sign_agreement, rank_correlation, \
    pairwise_rank_agreement, feature_contribution_correlation
from differlib.engine.utils import dataset_info_dict, save_figure
from similarity.engine.cfg import CFG as cfg

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]    # "MEEGNet",
n_models = len(model_names)
top_k = 0.15
top_k_percent = int(top_k * 100)
explainer = cfg.EXPLAINER.TYPE

for dataset in datasets:
    db_path = './output/Consensus/{}/{}_{}_attribution'.format(dataset, dataset, explainer)
    db = shelve.open(db_path)
    save_path = f"./output/Attribution_Similarity_{dataset}/"

    sample_num = 2000
    if dataset == 'DecMeg2014':
        sample_num = 300

    channels, points, num_classes = dataset_info_dict[dataset]['CHANNELS'], dataset_info_dict[dataset]['POINTS'], dataset_info_dict[dataset]['NUM_CLASSES']
    k = int(channels * points * top_k)
    # 计算Feature Agreement
    feature_agreement_matrix = np.ones((n_models, n_models), dtype=float)
    sign_agreement_matrix = np.ones((n_models, n_models), dtype=float)
    channel_rank_correlation_matrix = np.ones((n_models, n_models), dtype=float)
    time_rank_correlation_matrix = np.ones((n_models, n_models), dtype=float)
    channel_pairwise_rank_agreement_matrix = np.ones((n_models, n_models), dtype=float)
    time_pairwise_rank_agreement_matrix = np.ones((n_models, n_models), dtype=float)

    # 汇总
    fig = plt.figure(figsize=(18, 12))
    gridlayout = gridspec.GridSpec(nrows=2, ncols=35, figure=fig,  wspace=None, hspace=0.2)
    axs00 = fig.add_subplot(gridlayout[:1, :10])
    axs01 = fig.add_subplot(gridlayout[:1, 12:22])
    axs02 = fig.add_subplot(gridlayout[:1, 24:34])
    axs10 = fig.add_subplot(gridlayout[1:, :10])
    axs11 = fig.add_subplot(gridlayout[1:, 12:22])
    axs12 = fig.add_subplot(gridlayout[1:, 24:34])
    axs_colorbar = fig.add_subplot(gridlayout[1:, 34:])
    # 设置颜色条带
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    colorbar.ColorbarBase(axs_colorbar, cmap='Oranges', norm=norm)

    for i_th in range(n_models-1):
        model_i = model_names[i_th]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        top_sort_i, sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], npz_i['sign_sort_maps']
        # 关键步骤：创建逆向索引
        inverse_indices_i = np.argsort(top_sort_i)
        # 恢复原始数组
        abs_mean_contribution_i = sort_contribution_i[inverse_indices_i].reshape(channels, points)
        sign_mean_maps_i = sign_sort_maps_i[inverse_indices_i].reshape(channels, points, num_classes)  # 后续可用于分类别分析
        heatmap_channel_i, heatmap_time_i = abs_mean_contribution_i.sum(axis=1), abs_mean_contribution_i.sum(axis=0)
        channel_sort_i, time_sort_i = np.argsort(heatmap_channel_i)[::-1], np.argsort(heatmap_time_i)[::-1]

        for j_th in range(i_th+1, n_models):
            model_j = model_names[j_th]
            npz_j = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_j))
            top_sort_j, sort_contribution_j, sign_sort_maps_j = npz_j['abs_top_sort'], npz_j['abs_sort_contribution'], npz_j['sign_sort_maps']
            # 关键步骤：创建逆向索引
            inverse_indices_j = np.argsort(top_sort_j)
            # 恢复原始数组
            abs_mean_contribution_j = sort_contribution_j[inverse_indices_j].reshape(channels, points)
            sign_mean_maps_j = sign_sort_maps_j[inverse_indices_j].reshape(channels, points, num_classes)
            heatmap_channel_j, heatmap_time_j = abs_mean_contribution_j.sum(axis=1), abs_mean_contribution_j.sum(axis=0)
            channel_sort_j, time_sort_j = np.argsort(heatmap_channel_j)[::-1], np.argsort(heatmap_time_j)[::-1]

            # feature_agreement_scores = []
            # sign_agreement_scores = []
            # for sample_id in range(sample_num):
            #     attribution_id = f"{sample_id}_{model_i}"
            #     assert attribution_id in db
            #     maps_i = db[attribution_id]
            #     attribution_id = f"{sample_id}_{model_j}"
            #     assert attribution_id in db
            #     maps_j = db[attribution_id]
            #
            #     abs_maps_i = np.abs(maps_i).sum(axis=-1).reshape(-1)
            #     abs_top_sort_i = np.argsort(abs_maps_i)[::-1]
            #     abs_sort_contribution_i = abs_maps_i[abs_top_sort_i]
            #     sign_sort_maps_i = maps_i.reshape(-1, n_classes)[abs_top_sort_i]
            #
            #     abs_maps_j = np.abs(maps_j).sum(axis=-1).reshape(-1)
            #     abs_top_sort_j = np.argsort(abs_maps_j)[::-1]
            #     abs_sort_contribution_j = abs_maps_j[abs_top_sort_j]
            #     sign_sort_maps_j = maps_j.reshape(-1, n_classes)[abs_top_sort_j]
            #
            #     feature_agreement_scores.append(feature_agreement(abs_top_sort_i, abs_top_sort_j, k))
            #     sign_agreement_scores.append(sign_agreement(abs_top_sort_i, abs_top_sort_j, sign_sort_maps_i, sign_sort_maps_j, k))

            # feature_agreement_matrix[i_th, j_th] = feature_agreement_matrix[j_th, i_th] = np.mean(feature_agreement_scores)
            # sign_agreement_matrix[i_th, j_th] = sign_agreement_matrix[j_th, i_th] = np.mean(sign_agreement_scores)

            # 计算Feature Agreement
            feature_agreement_matrix[i_th, j_th] = feature_agreement_matrix[j_th, i_th] = feature_agreement(time_sort_i, time_sort_j, 30)
            # feature_agreement_matrix[i_th, j_th] = feature_agreement_matrix[j_th, i_th] = feature_agreement(top_sort_i, top_sort_j, k)

            # 计算Sign Agreement
            sign_agreement_matrix[i_th, j_th] = sign_agreement_matrix[j_th, i_th] = sign_agreement(top_sort_i, top_sort_j, sign_sort_maps_i, sign_sort_maps_j, k)

            # 计算Rank Correlation
            # channel_rank_correlation_matrix[i_th, j_th] = channel_rank_correlation_matrix[j_th, i_th] = feature_contribution_correlation(abs_mean_contribution_i.flatten(), abs_mean_contribution_j.flatten())[2]
            # time_rank_correlation_matrix[i_th, j_th] = time_rank_correlation_matrix[j_th, i_th] = feature_contribution_correlation(sign_mean_maps_i.flatten(), sign_mean_maps_j.flatten())[2]
            channel_rank_correlation_matrix[i_th, j_th] = channel_rank_correlation_matrix[j_th, i_th] = feature_contribution_correlation(heatmap_channel_i, heatmap_channel_j)[2]
            time_rank_correlation_matrix[i_th, j_th] = time_rank_correlation_matrix[j_th, i_th] = feature_contribution_correlation(heatmap_time_i, heatmap_time_j)[2]
            # channel_rank_correlation_matrix[i_th, j_th] = channel_rank_correlation_matrix[j_th, i_th] = rank_correlation(channel_sort_i, channel_sort_j)
            # time_rank_correlation_matrix[i_th, j_th] = time_rank_correlation_matrix[j_th, i_th] = rank_correlation(time_sort_i, time_sort_j)

            # 计算Pairwise Rank Agreement
            channel_pairwise_rank_agreement_matrix[i_th, j_th] = channel_pairwise_rank_agreement_matrix[j_th, i_th] = pairwise_rank_agreement(heatmap_channel_i, heatmap_channel_j)
            time_pairwise_rank_agreement_matrix[i_th, j_th] = time_pairwise_rank_agreement_matrix[j_th, i_th] = pairwise_rank_agreement(heatmap_time_i, heatmap_time_j)

    feature_agreement_axis = plot_similarity_matrix(feature_agreement_matrix, model_names, title=f"Feature Agreement (Top-k={top_k_percent}%)", ax=axs00, colorbar=False)
    sign_agreement_axis = plot_similarity_matrix(sign_agreement_matrix, model_names, title=f"Sign Agreement (Top-k={top_k_percent}%)", ax=axs10, colorbar=False)
    channel_rank_correlation_axis = plot_similarity_matrix(channel_rank_correlation_matrix, model_names, title=f"Channel Rank Correlation", ax=axs01, colorbar=False)
    time_rank_correlation_axis = plot_similarity_matrix(time_rank_correlation_matrix, model_names, title=f"Time Rank Correlation", ax=axs11, colorbar=False)
    # channel_pairwise_rank_agreement_axis = plot_similarity_matrix(channel_pairwise_rank_agreement_matrix, model_names, title=f"Channel Pairwise Rank Agreement", ax=axs01, colorbar=False)
    # time_pairwise_rank_agreement_axis = plot_similarity_matrix(time_pairwise_rank_agreement_matrix, model_names, title=f"Time Pairwise Rank Agreement", ax=axs11, colorbar=False)


    related_matrix = np.ones((n_models, n_models), dtype=float)
    related_matrix_1 = np.ones((n_models, n_models), dtype=float)
    rdms = []
    sample_num = 100
    for i_th in range(n_models):
        model_i = model_names[i_th]
        # 计算样本间关系
        map_1_list = []
        for sample_id in range(sample_num):
            attribution_id = f"{sample_id}_{model_i}"
            assert attribution_id in db
            maps_i = db[attribution_id]

            abs_maps_i = np.abs(maps_i).sum(axis=-1).reshape(-1)
            abs_top_sort_i = np.argsort(abs_maps_i)[::-1]
            abs_sort_contribution_i = abs_maps_i[abs_top_sort_i]
            sign_sort_maps_i = maps_i.reshape(-1, num_classes)[abs_top_sort_i]

            map_1_list.append(abs_sort_contribution_i)

        rdm = pairwise_distances(map_1_list, metric='correlation')  # 'euclidean', 'correlation'
        rdm_minmax_scaler = (rdm - np.min(rdm)) / (np.max(rdm) - np.min(rdm))
        # plot_similarity_matrix(rdm_minmax_scaler, [str("") for _id in range(sample_num)], title=f"RDM_{model_i}", include_values=False)
        rdms.append(rdm)

    for i_th in range(n_models-1):
        for j_th in range(i_th+1, n_models):
            cos_sim, corr, tau = feature_contribution_correlation(rdms[i_th].flatten(), rdms[j_th].flatten())
            related_matrix[i_th, j_th] = related_matrix[j_th, i_th] = corr
            related_matrix_1[i_th, j_th] = related_matrix_1[j_th, i_th] = tau
    related_axis = plot_similarity_matrix(related_matrix, model_names, title=f"Relation-Based Similarity Analysis (Pearson)", ax=axs02, colorbar=False)
    related_1_axis = plot_similarity_matrix(related_matrix_1, model_names, title=f"Relation-Based Similarity Analysis (Kendall)",
                                          ax=axs12, colorbar=False)

    save_figure(fig, save_path, '{}_similarity'.format(dataset))
    plt.show()
