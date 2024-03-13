import os
import argparse

import numpy as np
import pandas as pd
# import graphviz
# import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
# from graphviz import Digraph
from matplotlib.collections import LineCollection
from sklearn.metrics import accuracy_score, confusion_matrix

# import torchsummary
# from torchviz import make_dot

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import log_msg, setup_seed, get_data_loader_from_dataset, load_checkpoint, accuracy, \
    get_data_labels_from_dataset
from differlib.models import model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # init dataloader & models
    # train_loader = get_data_loader_from_dataset('../dataset/{}_train.npz'.format(cfg.DATASET.TYPE),
    #                                             cfg.SOLVER.BATCH_SIZE)
    val_loader = get_data_loader_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE),
                                              cfg.DATASET.TEST.BATCH_SIZE)
    val_data, val_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE))

    print(log_msg("Loading teacher model", "INFO"))
    model_A_type, model_A_pretrain_path = model_dict[cfg.MODELS.A]
    assert (model_A_pretrain_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.A)
    model_A = model_A_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()

    model_B_type, model_B_pretrain_path = model_dict[cfg.MODELS.B]
    assert (model_B_pretrain_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.B)
    model_B = model_B_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()

    # validate
    data_, targets, pred_A, pred_B = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(val_loader):
        data_.extend(data.numpy())
        targets.extend(target.numpy())
        data = data.float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output_A = model_A(data)
        _, pred_target_A = output_A.topk(1, 1, True, True)
        pred_target_A = pred_target_A.squeeze().cpu().detach().numpy()
        pred_A.extend(pred_target_A)
        acc_A = accuracy_score(y_true=target.cpu().detach().numpy(), y_pred=pred_target_A)
        print(f"{idx}: model_A test accuracy: {(acc_A * 100):.2f}%")
        # loss_A = criterion(output_A, target)
        # acc_A, _ = accuracy(output_A, target, topk=(1, 2))
        # print(acc_A, loss_A)

        output_B = model_B(data)
        _, pred_target_B = output_B.topk(1, 1, True, True)
        pred_target_B = pred_target_B.squeeze().cpu().detach().numpy()
        pred_B.extend(pred_target_B)
        acc_B = accuracy_score(y_true=target.cpu().detach().numpy(), y_pred=pred_target_B)
        print(f"{idx}: model_B test accuracy: {(acc_B * 100):.2f}%")
        # loss_B = criterion(output_B, target)
        # acc_B, _ = accuracy(output_B, target, topk=(1, 2))
        # print(acc_B, loss_B)

    data_, targets, pred_A, pred_B = np.array(data_), np.array(targets), np.array(pred_A), np.array(pred_B)

    acc_A = accuracy_score(y_true=targets, y_pred=pred_A)
    print(f"sum_model_A test accuracy: {(acc_A * 100):.2f}%")
    cm_A = confusion_matrix(y_true=targets, y_pred=pred_A)
    print('cm_A is:\n', cm_A)

    acc_B = accuracy_score(y_true=targets, y_pred=pred_B)
    print(f"sum_model_B test accuracy: {(acc_B * 100):.2f}%")
    cm_B = confusion_matrix(y_true=targets, y_pred=pred_B)
    print('cm_B is:\n', cm_B)

    ydiff = (pred_A != pred_B).astype(int)
    print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff)):.2f}")

    from differlib.imd.imd import IMDExplainer

    max_depth = 6

    imd = IMDExplainer()
    imd.fit(pd.DataFrame(data.cpu().detach().numpy().reshape((-1, 204*100))), pred_target_A, pred_target_B, max_depth=max_depth)
    # imd.fit(pd.DataFrame(data.cpu().detach().numpy()[:, :, 0]), pred_target_A, pred_target_B,
    #         max_depth=max_depth)

    diffrules = imd.explain()
    print(diffrules)

    rule_idx = -1

    rule = diffrules[rule_idx]
    filtered_data = data_[rule.apply(data_)]
    print(filtered_data)

    # Computation of metrics
    # on train set
    metrics = imd.metrics(x_train, y1, y2, name="train")
    print(metrics)

    # on test set
    metrics = imd.metrics(x_test, model1.predict(x_test), model2.predict(x_test), name="test")
    print(metrics)

    from differlib.imd.utils import visualize_jst

    visualize_jst(imd.jst, path="joint.jpg")
