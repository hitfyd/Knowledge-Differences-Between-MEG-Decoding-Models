import os
import random

import torch
import torch.nn as nn
import numpy as np
import sys
import time

from tqdm import tqdm


dataset_info_dict = {
    "CamCAN": {"CHANNELS": 204, "POINTS": 100, "NUM_CLASSES": 2},
    "DecMeg2014": {"CHANNELS": 204, "POINTS": 250, "NUM_CLASSES": 2},
    "BCIIV2a": {"CHANNELS": 22, "POINTS": 1125, "NUM_CLASSES": 4},
}


# 设置全局随机数种子，同时用于记录实验数据
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 尽可能提高确定性
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


# 配置Pytorch批处理数据集
def get_data_loader(data, label, batch_size=256, shuffle=True):
    assert isinstance(data, np.ndarray) and isinstance(label, np.ndarray) and len(data) == len(label)
    assert data.dtype == np.float32 #and label.dtype == np.longlong
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# 从数据集文件读取样本和标签
def get_data_labels_from_dataset(dataset_path):
    dataset = np.load(dataset_path)
    data = dataset['data']
    labels = dataset['labels']
    return data, labels


# 从数据集文件读取，生成数据集
def get_data_loader_from_dataset(dataset_path, batch_size=256, shuffle=True):
    dataset = np.load(dataset_path)
    data = dataset['data']
    labels = dataset['labels']
    return get_data_loader(data, labels, batch_size, shuffle)


def sample_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu)/std


class DatasetNormalization(object):

    def __init__(self, data):
        self.mu = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def __call__(self, data):
        return (data - self.mu) / self.std

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1 = [AverageMeter() for _ in range(3)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(data=data)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 2))
            batch_size = data.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Loss:{loss:.4f}| Top-1:{top1:.4f}".format(
                loss=losses.avg,
                top1=top1.avg
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, losses.avg


def predict(model, data, num_classes=2, batch_size=512, eval=False, softmax=True, device: torch.device = torch.device("cpu")):
    # if model.__class__.__name__ in ["GaussianNB", "RandomForestClassifier", "LogisticRegression"]:
    #     output = model.predict_proba(data.reshape((len(data), -1)))
    # else:
    model.to(device)
    model.eval()
    data = torch.from_numpy(data)
    data_split = torch.split(data, batch_size, dim=0)
    output = torch.zeros(len(data), num_classes).to(device)  # 预测的置信度和置信度最大的标签编号
    start = 0
    if eval:
        with torch.no_grad():
            for batch_data in data_split:
                batch_data = batch_data.to(device)
                batch_data = batch_data.float()
                output[start:start+len(batch_data)] = model(batch_data)
                start += len(batch_data)
    else:
        for batch_data in data_split:
            batch_data = batch_data.to(device)
            batch_data = batch_data.float()
            output[start:start + len(batch_data)] = model(batch_data)
            start += len(batch_data)
            del batch_data
    if softmax:
        if model.__class__.__name__ in ["LFCNN", "VARCNN", "CTNet"]:
            output = torch.exp(output) / torch.sum(torch.exp(output), dim=-1, keepdim=True)
        if model.__class__.__name__ in ["HGRN", "ATCNet"]:  #, "EEGNetv4", "NewEEGNetv1"
            output = torch.exp(output)
    output = output.cpu().detach().numpy()
    return output


def model_eval(model, data_loader):
    model.cuda()
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            data = data.float()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy


def output_predict_targets(model: torch.nn, data: np.ndarray, num_classes=2, batch_size=1024, softmax=True):
    # data_torch = torch.from_numpy(data).float().cuda()
    # model.eval()
    # with torch.no_grad():
    #     output = model(data_torch)
    output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
    _, predict_targets = output.topk(1, 1, True, True)
    output = output.cpu().detach().numpy()
    predict_targets = predict_targets.squeeze().cpu().detach().numpy()
    # if softmax:
    #     if model.__class__.__name__ in ["LFCNN", "VARCNN"]:
    #         output = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
    #     if model.__class__.__name__ in ["HGRN", "ATCNet"]:
    #         output = np.exp(output)
    return output, predict_targets


# def individual_predict(model, individual_data, eval=True):
#     pred = predict(model, np.expand_dims(individual_data, 0), eval=eval)
#     return pred[0]


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path, device: torch.device = torch.device("cpu")):
    with open(path, "rb") as f:
        return torch.load(f, map_location=device, weights_only=False)


def save_figure(fig, save_dir, figure_name, save_dpi=400, format_list=None):
    from matplotlib import pyplot as plt
    # EPS format for LaTeX
    # PDF format for LaTeX/Display
    # SVG format for Web
    # JPG format for display
    if format_list is None:
        format_list = ["pdf", "svg"]     # "eps", "pdf", "svg"
    plt.rcParams['savefig.dpi'] = save_dpi  # 图片保存像素
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 确保路径存在
    for save_format in format_list:
        fig.savefig('{}{}.{}'.format(save_dir, figure_name, save_format), format=save_format,
                    bbox_inches='tight', transparent=False)
