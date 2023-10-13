import os
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics, tree, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.parallel import Parallel, delayed

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import get_data_labels_from_dataset, log_msg, load_checkpoint, setup_seed
from differlib.models import model_dict


def predict(model, data, batch_size=1024):
    model.cuda()
    model.eval()
    data = torch.from_numpy(data)
    data_split = torch.split(data, batch_size, dim=0)

    pred_target, output = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in data_split:
            batch_data = batch_data.float()
            batch_data = batch_data.cuda(non_blocking=True)

            output_i = model(batch_data)
            _, pred_target_i = output_i.topk(1, 1, True, True)
            output_i = output_i.cpu().detach().numpy()
            pred_target_i = pred_target_i.squeeze().cpu().detach().numpy()
            pred_target.extend(pred_target_i)
            output.extend(output_i)

    pred_target, output = np.array(pred_target), np.array(output)
    return pred_target, output


# def predict(model, val_loader):
#     model.cuda()
#     model.eval()
#     pred_target, output = [], []
#     with torch.no_grad():
#         for idx, (data, target) in enumerate(val_loader):
#             data = data.float()
#             data = data.cuda(non_blocking=True)
#
#             output_i = model(data)
#             _, pred_target_i = output_i.topk(1, 1, True, True)
#             output_i = output_i.cpu().detach().numpy()
#             pred_target_i = pred_target_i.squeeze().cpu().detach().numpy()
#             pred_target.extend(pred_target_i)
#             output.extend(output_i)
#
#     pred_target, output = np.array(pred_target), np.array(output)
#     return pred_target, output


def evaluate(target, pred_target):
    confusion_matrix = metrics.confusion_matrix(target, pred_target)
    accuracy = metrics.accuracy_score(target, pred_target)
    precision = metrics.precision_score(target, pred_target, average=None)
    recall = metrics.recall_score(target, pred_target, average=None)
    f1 = metrics.f1_score(target, pred_target, average=None)
    return confusion_matrix, accuracy, precision, recall, f1


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
    # val_loader = get_data_loader_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE),
    #                                           cfg.DATASET.TEST.BATCH_SIZE)
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

    pred_target_A, _ = predict(model_A, val_data)
    pred_target_B, _ = predict(model_B, val_data)

    data_len = len(val_data)
    val_data_clf = val_data.reshape(data_len, -1)
    delta_target = pred_target_A ^ pred_target_B
    for i in range(data_len):
        if pred_target_A[i] == 0:
            if pred_target_B[i] == 0:
                delta_target[i] = 0
            else:
                delta_target[i] = 1
        else:
            if pred_target_B[i] == 0:
                delta_target[i] = 2
            else:
                delta_target[i] = 3
    print("0: {}\t 1: {}".format(data_len - delta_target.sum(), delta_target.sum()))


    def clf2parallel(clf, X, y, train_index, test_index, save_path=None):
        clf_clone = clone(clf)
        clf_clone = clf_clone.fit(X[train_index], y[train_index])

        # 保存到当前工作目录中的文件
        if save_path is not None:
            joblib.dump(clf_clone, save_path)

        pred_y = clf_clone.predict(X[test_index])
        scores = evaluate(y[test_index], pred_y)
        return scores


    skf = StratifiedKFold(n_splits=5)
    min_samples_leaf = 1
    clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    parallel = Parallel(n_jobs=-1)
    all_results = parallel(delayed(clf2parallel)
                           (clf, val_data_clf, delta_target, train_index, test_index,
                            "{}_DT_{}_{}.sav".format(cfg.DATASET.TYPE, min_samples_leaf, test_index[0]))
                           for train_index, test_index in skf.split(val_data_clf, delta_target))
    for result in all_results:
        print(result)


    # # 交叉验证
    # skf = StratifiedKFold(n_splits=5)
    # k_id = 0
    # min_samples_leaf = 5
    # for train, test in skf.split(val_data_clf, delta_target):
    #     joblib_file = "{}_DT_{}_{}.sav".format(cfg.DATASET.TYPE, min_samples_leaf, k_id)
    #     k_id += 1
    #
    #     clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    #     clf = clf.fit(val_data_clf[train], delta_target[train])
    #
    #     # 保存到当前工作目录中的文件
    #     joblib.dump(clf, joblib_file)
    #
    #     pred = clf.predict(val_data_clf[test])
    #     print(evaluate(delta_target[test], pred))
    #     tree.plot_tree(clf)
    #
    #     # # 从文件中加载
    #     # joblib_model = joblib.load(joblib_file)
    #     #
    #     # pred = clf.predict(val_data_clf[test])
    #     # print(evaluate(delta_target[test], pred))
    #     # tree.plot_tree(joblib_model)
    #
    #     plt.show()
