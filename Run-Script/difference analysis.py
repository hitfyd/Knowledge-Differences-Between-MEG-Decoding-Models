import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import log_msg, setup_seed, get_data_loader_from_dataset, load_checkpoint, \
    get_data_labels_from_dataset
from differlib.models import model_dict
from differlib.imd.imd import IMDExplainer
from differlib.imd.utils import visualize_jst
from differlib.DeltaXpainer import DeltaExplainer
from differlib.LogitDeltaRule import LogitDeltaRule

if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis for knowledge differences.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # init dataset & models
    data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE))
    dataset = cfg.DATASET.TYPE
    channels = cfg.DATASET.CHANNELS
    points = cfg.DATASET.POINTS
    n_classes = cfg.DATASET.NUM_CLASSES

    print(log_msg("Loading model A", "INFO"))
    model_A_type, model_A_pretrain_path = model_dict[cfg.MODELS.A]
    assert (model_A_pretrain_path is not None), "no pretrain model A {}".format(cfg.MODELS.A)
    model_A = model_A_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()

    print(log_msg("Loading model B", "INFO"))
    model_B_type, model_B_pretrain_path = model_dict[cfg.MODELS.B]
    assert (model_B_pretrain_path is not None), "no pretrain model B {}".format(cfg.MODELS.B)
    model_B = model_B_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()

    # models predict differences
    data_torch = torch.from_numpy(data).float().cuda()
    output_A = model_A(data_torch)
    _, pred_target_A = output_A.topk(1, 1, True, True)
    output_A = output_A.cpu().detach().numpy()
    pred_target_A = pred_target_A.squeeze().cpu().detach().numpy()

    output_B = model_B(data_torch)
    _, pred_target_B = output_B.topk(1, 1, True, True)
    output_B = output_B.cpu().detach().numpy()
    pred_target_B = pred_target_B.squeeze().cpu().detach().numpy()

    max_depth = 7

    x = pd.DataFrame(data.reshape((-1, channels*points)))
    imd = IMDExplainer()
    # imd = DeltaExplainer()
    imd.fit(x, pred_target_A, pred_target_B, max_depth=max_depth)

    # imd = LogitDeltaRule()
    # output_A = np.exp(output_A)/np.sum(np.exp(output_A), axis=-1, keepdims=True)
    # imd.fit(x, output_A, output_B, max_depth=max_depth)

    diffrules = imd.explain()
    print(diffrules)

    # rule_idx = -1
    #
    # rule = diffrules[rule_idx]
    # filtered_data = x[rule.apply(x)]
    # print(filtered_data)

    # Computation of metrics
    # on train set
    metrics = imd.metrics(x, pred_target_A, pred_target_B, name="train")
    # metrics = imd.metrics(x, output_A, output_B, name="train")
    print(metrics)