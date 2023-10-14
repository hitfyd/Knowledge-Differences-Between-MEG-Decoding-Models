import os
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import StratifiedKFold
from torch import optim

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import get_data_labels_from_dataset, log_msg, load_checkpoint, setup_seed, get_data_loader
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


criterion = nn.MSELoss()

def train(model, train_loader, epoch, lr=3e-4):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, penalty = model(data, is_training_data=True)
        loss = criterion(output, target) + penalty
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        # correct += pred.eq(target.view_as(pred)).sum().item()
        correct += pred.eq(target.max(1, keepdim=True)[1].view_as(pred)).sum().item()

    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f}%)]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), train_accuracy, train_loss))
    return train_accuracy, train_loss


def evaluate(target, output):
    output_A = output[:, :2]
    output_B = np.copy(output_A)
    output_B[:, 0] = output_A[:, 0] - output[:, 2]
    output_B[:, 1] = output_A[:, 1] + output[:, 2]
    pred_target_A = output_A.argmax(axis=1)
    pred_target_B = output_B.argmax(axis=1)
    pred_target = pred_target_A ^ pred_target_B
    confusion_matrix = metrics.confusion_matrix(target, pred_target)
    accuracy = metrics.accuracy_score(target, pred_target)
    precision = metrics.precision_score(target, pred_target)
    recall = metrics.recall_score(target, pred_target)
    f1 = metrics.f1_score(target, pred_target)
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

    pred_target_A, output_A = predict(model_A, val_data)
    pred_target_B, output_B = predict(model_B, val_data)

    data_len = len(val_data)
    delta_target = pred_target_A ^ pred_target_B
    delta_output = output_A - output_B
    delta_output = np.append(output_A, delta_output, axis=1)[:, :3]
    print("0: {}\t 1: {}".format(data_len - delta_target.sum(), delta_target.sum()))

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(val_data, delta_target):
        delta_model = model_A_type(
            channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES+1)
        delta_model = delta_model.cuda()

        # train_loader = get_data_loader(val_data[train_index], delta_target[train_index], cfg.SOLVER.BATCH_SIZE)
        train_loader = get_data_loader(val_data, delta_output, cfg.SOLVER.BATCH_SIZE)

        for epoch in range(100):
            train_accuracy, train_loss = train(delta_model, train_loader, epoch)
            pred, output = predict(delta_model, val_data[test_index])
            print(evaluate(delta_target[test_index], output))
        print("Train:")
        pred, output = predict(delta_model, val_data[train_index])
        print(evaluate(delta_target[train_index], output))
        print("Test:")
        pred, output = predict(delta_model, val_data[test_index])
        print(evaluate(delta_target[test_index], output))

