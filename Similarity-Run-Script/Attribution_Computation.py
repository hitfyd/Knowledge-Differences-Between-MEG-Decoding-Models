import argparse
import os
import shelve
import time

import numpy as np
import torch

from MEG_Shapley_Values import ShapleyValueExplainer, DatasetInfo, SampleInfo, deletion_test, \
    additive_efficient_normalization, IterationLogger, torch_individual_predict
from differlib.engine.utils import (setup_seed, load_checkpoint, get_data_labels_from_dataset,
                                    get_data_loader,
                                    dataset_info_dict, model_eval)
from differlib.models import model_dict
from similarity.engine.cfg import CFG as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for attribution consensus.")
    parser.add_argument("--cfg", type=str, default="../configs/Consensus/DecMeg2014_atcnet.yaml")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # init experiment
    project = cfg.EXPERIMENT.PROJECT
    experiment_name = cfg.EXPERIMENT.NAME
    tags = cfg.EXPERIMENT.TAG
    if experiment_name == "":
        experiment_name = tags
    tags = tags.split(',')
    if args.opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(args.opts[::2], args.opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(project, experiment_name)

    # init loggers
    log_prefix = cfg.LOG.PREFIX
    log_path = os.path.join(log_prefix, experiment_name)
    record_path = os.path.join(log_prefix, project)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # set GPUs, CPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    num_gpus = torch.cuda.device_count()

    # init dataset & models
    dataset = cfg.DATASET
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    train_data, train_labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    train_loader = get_data_loader(train_data, train_labels)
    test_loader = get_data_loader(test_data, test_labels)
    origin_data, labels = train_data, train_labels
    n_samples, channels, points = origin_data.shape
    n_classes = len(set(labels))
    assert channels == dataset_info_dict[dataset]["CHANNELS"]
    assert points == dataset_info_dict[dataset]["POINTS"]
    assert n_classes == dataset_info_dict[dataset]["NUM_CLASSES"]

    label_names = ['audio', 'visual']
    if dataset == 'DecMeg2014':
        label_names = ['Scramble', 'Face']
    dataset_info = DatasetInfo(dataset=dataset, label_names=label_names, channels=channels, points=points,
                               classes=n_classes)

    model_types = cfg.MODELS
    model_list = torch.nn.ModuleList()
    for model_type in model_types:
        model_class, model_pretrain_path = model_dict[dataset][model_type]
        assert (model_pretrain_path is not None), "no pretrain model A {}".format(model_class)
        trained_model = model_class(channels=channels, points=points, num_classes=n_classes)
        trained_model.load_state_dict(load_checkpoint(model_pretrain_path))
        model_list.append(trained_model)
        # 评估模型精度
        acc = model_eval(trained_model, test_loader)
        print("{}: {:.4f}".format(model_type, acc))

    # 建立零基线
    zero_baseline_input = np.zeros([channels, points])

    # # 读取通道可视化信息
    # channel_db = shelve.open(get_project_path() + '/dataset/grad_info')
    # channels_info = channel_db['info']
    # channel_db.close()

    # 要分析的样本数量
    sample_num = cfg.NUM_SAMPLES

    # AttributionExplainer参数
    explainer = cfg.EXPLAINER.TYPE
    window_length = cfg.EXPLAINER.W
    M = cfg.EXPLAINER.M
    reference_num = cfg.EXPLAINER.NUM_REFERENCES
    reference_dataset = origin_data[cfg.EXPLAINER.RANGE_REFERENCES:]     # 原始的参考数据集
    reference_filter = False    # 对于模型比较来说，不应该启用；对于单一模型的特征归因有效果
    antithetic_variables = False

    db_path = log_path + '/{}_{}_attribution_train'.format(dataset, explainer)
    db = shelve.open(db_path)
    joint_explainer = ShapleyValueExplainer(dataset_info, model_list, reference_dataset,
                                            reference_num, window_length, M, reference_filter, antithetic_variables)

    logger = IterationLogger()
    for sample_id in range(sample_num):
        origin_input, truth_label = origin_data[sample_id], labels[sample_id]
        sample_info = SampleInfo(sample_id=sample_id, origin_input=origin_input, truth_label=truth_label)
        print('sample_id:{}\ttruth_label:{}'.format(sample_id, truth_label))

        time_start = time.perf_counter()
        joint_maps, joint_maps_all = joint_explainer(origin_input)
        time_end = time.perf_counter()  # 记录结束时间
        joint_run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        # 区分模型1和模型2的归因图，并计算对比归因图
        for model_id in range(len(model_list)):
            model = model_list[model_id]
            # 计算样本的期望
            prediction, pred_label = torch_individual_predict(model, torch.from_numpy(origin_input))
            # 计算基线样本的预测，作为基线期望值
            baseline, _ = torch_individual_predict(model, torch.from_numpy(zero_baseline_input))
            prediction, pred_label, baseline = prediction.cpu().numpy(), pred_label.cpu().numpy(), baseline.cpu().numpy()
            print('model:{}\tpredicted_label:{}\tprediction:{}\tbaseline_predicted_label:{}'.format(
                model.__class__.__name__, pred_label, prediction, baseline))
            attribution_maps = joint_maps[model_id]
            attribution_maps = additive_efficient_normalization(prediction, baseline, attribution_maps)
            # attribution_maps = contribution_smooth(attribution_maps)

            _, _, _, del_auc = deletion_test(model, origin_input, attribution_maps)

            attribution_id = f"{sample_id}_{model.__class__.__name__}"
            db[attribution_id] = attribution_maps

    db.close()
