import os
import shelve

import numpy as np
from kneed import KneeLocator

from differlib.engine.utils import setup_seed, get_data_labels_from_dataset, get_data_loader, dataset_info_dict, \
    save_figure, model_eval, load_checkpoint, accuracy
from differlib.models import model_dict
from similarity.engine.cfg import CFG as cfg
from similarity.attribution.MEG_Shapley_Values import topomap_plot, time_curve_plot

if __name__ == "__main__":
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS

    # 要分析的样本数量
    # datasets
    datasets = ["DecMeg2014", "CamCAN"]
    for dataset in datasets:
        channels = dataset_info_dict[dataset]["CHANNELS"]
        points = dataset_info_dict[dataset]["POINTS"]
        n_classes = dataset_info_dict[dataset]["NUM_CLASSES"]

        # 读取通道可视化信息
        channel_db = shelve.open('../dataset/grad_info')
        channels_info = channel_db['info']
        channel_db.close()


        model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]
        accuracies = [94.0549, 94.3598, 95.1728, 95.6131, 95.6640, 96.1551]
        sample_num = 160
        if dataset == 'DecMeg2014':
            sample_num = 100
            accuracies = [74.4108, 75.7576, 80.4714, 81.6498, 79.2929, 82.9966]
        n_x = 100   # 百分比划分特征

        # AttributionExplainer参数
        explainer = cfg.EXPLAINER.TYPE

        db_path = './output/Consensus/{}/{}_{}_attribution_train'.format(dataset, dataset, explainer)
        db = shelve.open(db_path)
        save_path = f"./output/Attribution_Plot_{dataset}/"

        # 逐样本迭代
        for model_name, acc in zip(model_names, accuracies):
            all_maps = np.zeros([sample_num, channels, points, n_classes], dtype=np.float32)
            for sample_id in range(sample_num):
                attribution_id = f"{sample_id}_{model_name}"
                assert attribution_id in db
                maps = db[attribution_id]
                all_maps[sample_id] = maps
            sign_mean_maps = all_maps.mean(axis=0)
            abs_mean_maps = np.abs(all_maps).mean(axis=0)
            abs_feature_contribution = abs_mean_maps.sum(axis=-1).reshape(-1)   # 合并一个特征对所有类别的绝对贡献
            abs_top_sort = np.argsort(abs_feature_contribution)[::-1]
            abs_sort_contribution = abs_feature_contribution[abs_top_sort]
            sign_sort_maps = sign_mean_maps.reshape(-1, n_classes)[abs_top_sort]

            y = abs_sort_contribution.reshape(-1, len(abs_sort_contribution) // n_x).sum(axis=-1)
            x = range(1, n_x+1)
            kl = KneeLocator(x, y, curve="convex", direction="decreasing")
            top_k = kl.knee / n_x

            file = './output/Consensus/{}/{}_{}_top_sort_train.npz'.format(dataset, dataset, model_name)
            np.savez(file, abs_top_sort=abs_top_sort, abs_sort_contribution=abs_sort_contribution, sign_sort_maps=sign_sort_maps, top_k=top_k)

            # 绘制特征绝对贡献地形图和时间曲线
            attribution_maps = abs_mean_maps.mean(axis=-1)
            title = '{} (Acc: {:.4f}%)'.format(model_name, acc)
            fig, _, _ = topomap_plot(title, attribution_maps, channels_info, channels=channels, top_channel_num=5, z_score=False, minmax_scaler=True)
            # save_figure(fig, save_path, '{}_{}_attribution_topomap'.format(dataset, model_name))

            # fig, _ = time_curve_plot(title, attribution_maps, points=points, z_score=False, minmax_scaler=True)
            # save_figure(fig, save_path, '{}_{}_attribution_time_curve'.format(dataset, model_name))

        db.close()
