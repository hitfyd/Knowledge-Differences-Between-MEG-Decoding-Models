import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics


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


def predict(model, val_loader):
    model.cuda()
    model.eval()
    pred_target, output = [], []
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)

            output_i = model(data)
            _, pred_target_i = output_i.topk(1, 1, True, True)
            output_i = output_i.cpu().detach().numpy()
            pred_target_i = pred_target_i.squeeze().cpu().detach().numpy()
            pred_target.extend(pred_target_i)
            output.extend(output_i)

    pred_target, output = np.array(pred_target), np.array(output)
    return pred_target, output


def evaluate(target, pred_target):
    confusion_matrix = metrics.confusion_matrix(target, pred_target)
    accuracy = metrics.accuracy_score()
    precision = metrics.precision_score()
    recall = metrics.recall_score()
    f1 = metrics.f1_score()
    return confusion_matrix, accuracy, precision, recall, f1
