import shelve

import numpy as np

from MEG_Explanation_Comparison import feature_agreement, plot_similarity_matrix, sign_agreement, rank_correlation, \
    pair_rank_agreement
from differlib.engine.utils import dataset_info_dict
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

    channels, points, n_classes = dataset_info_dict[dataset]['CHANNELS'], dataset_info_dict[dataset]['POINTS'], dataset_info_dict[dataset]['NUM_CLASSES']
    k = int(channels * points * top_k)
    # 计算Feature Agreement
    feature_agreement_matrix = np.ones((n_models, n_models), dtype=float)
    rank_correlation_matrix = np.ones((n_models, n_models), dtype=float)
    sign_agreement_matrix = np.ones((n_models, n_models), dtype=float)

    for i_th in range(n_models-1):
        model_i = model_names[i_th]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        top_sort_i, sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], npz_i['sign_sort_maps']

        for j_th in range(i_th+1, n_models):
            model_j = model_names[j_th]
            npz_j = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_j))
            top_sort_j, sort_contribution_j, sign_sort_maps_j = npz_j['abs_top_sort'], npz_j['abs_sort_contribution'], npz_j['sign_sort_maps']

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
            feature_agreement_matrix[i_th, j_th] = feature_agreement_matrix[j_th, i_th] = feature_agreement(top_sort_i, top_sort_j, k)

            # 计算Sign Agreement
            sign_agreement_matrix[i_th, j_th] = sign_agreement_matrix[j_th, i_th] = sign_agreement(top_sort_i, top_sort_j, sign_sort_maps_i, sign_sort_maps_j, k)

            # 计算Rank Correlation
            rank_correlation_matrix[i_th, j_th] = rank_correlation_matrix[j_th, i_th] = pair_rank_agreement(top_sort_i, top_sort_j, k)

            # 计算Pairwise Rank Agreement

            # 计算

            # 计算

    feature_agreement_axis = plot_similarity_matrix(feature_agreement_matrix, model_names, title=f"Feature Agreement (Top-k={top_k_percent}%)")
    sign_agreement_axis = plot_similarity_matrix(sign_agreement_matrix, model_names, title=f"Sign Agreement (Top-k={top_k_percent}%)")
    rank_correlation_axis = plot_similarity_matrix(rank_correlation_matrix, model_names, title=f"Rank Correlation (Top-k={top_k_percent}%)")
