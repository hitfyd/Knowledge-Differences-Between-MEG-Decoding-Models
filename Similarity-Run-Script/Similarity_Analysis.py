import os
import shelve

import numpy as np

from differlib.engine.utils import setup_seed, get_data_labels_from_dataset, get_data_loader, dataset_info_dict
from similarity.engine.cfg import CFG as cfg

if __name__ == "__main__":
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS

    # 要分析的样本数量
    # datasets
    dataset = "CamCAN"  # "DecMeg2014", "CamCAN"
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)
    origin_data, labels = test_data, test_labels
    n_samples, channels, points = origin_data.shape
    n_classes = len(set(labels))
    assert channels == dataset_info_dict[dataset]["CHANNELS"]
    assert points == dataset_info_dict[dataset]["POINTS"]
    assert n_classes == dataset_info_dict[dataset]["NUM_CLASSES"]

    sample_num = 3000
    if dataset == 'DecMeg2014':
        sample_num = 300

    model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]

    # AttributionExplainer参数
    explainer = cfg.EXPLAINER.TYPE

    db_path = './output/Consensus/{}/{}_{}_attribution'.format(dataset, dataset, explainer)
    db = shelve.open(db_path)

    # 逐样本迭代
    # 计算所有的VARCNN归因结果，并计算最重要的top-k特征
    for model_name in model_names:
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

        file = './output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_name)
        np.savez(file, abs_top_sort=abs_top_sort, abs_sort_contribution=abs_sort_contribution, sign_sort_maps=sign_sort_maps)

    db.close()
