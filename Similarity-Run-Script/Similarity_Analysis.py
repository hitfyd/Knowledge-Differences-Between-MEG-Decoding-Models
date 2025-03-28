import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt

from differlib.engine.utils import dataset_info_dict
from MEG_Explanation_Comparison import feature_agreement, plot_similarity_matrix, sign_agreement

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]    # "MEEGNet",
n_models = len(model_names)
top_k = 0.5
top_k_percent = int(top_k * 100)

for dataset in datasets:
    channels, points = dataset_info_dict[dataset]['CHANNELS'], dataset_info_dict[dataset]['POINTS']
    k = int(channels * points * top_k)
    # 计算Feature Agreement
    feature_agreement_matrix = np.ones((n_models, n_models), dtype=float)
    sign_agreement_matrix = np.ones((n_models, n_models), dtype=float)

    for i_th in range(n_models-1):
        model_i = model_names[i_th]
        npz_i = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_i))
        top_sort_i, sort_contribution_i, sign_sort_maps_i = npz_i['abs_top_sort'], npz_i['abs_sort_contribution'], npz_i['sign_sort_maps']

        for j_th in range(i_th+1, n_models):
            model_j = model_names[j_th]
            npz_j = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_j))
            top_sort_j, sort_contribution_j, sign_sort_maps_j = npz_j['abs_top_sort'], npz_j['abs_sort_contribution'], npz_j['sign_sort_maps']

            # 计算Feature Agreement
            feature_agreement_matrix[i_th, j_th] = feature_agreement_matrix[j_th, i_th] = feature_agreement(top_sort_i, top_sort_j, k)

            # 计算Rank Agreement

            # 计算Sign Agreement
            sign_agreement_matrix[i_th, j_th] = sign_agreement_matrix[j_th, i_th] = sign_agreement(top_sort_i, top_sort_j, sign_sort_maps_i, sign_sort_maps_j, k)

            # 计算Signed Rank Agreement

            # 计算Rank Correlation

            # 计算Pairwise Rank Agreement

            # 计算

            # 计算

    feature_agreement_axis = plot_similarity_matrix(feature_agreement_matrix, model_names, title=f"Feature Agreement (Top-k={top_k_percent}%)")
    sign_agreement_axis = plot_similarity_matrix(sign_agreement_matrix, model_names, title=f"Sign Agreement (Top-k={top_k_percent}%)")
