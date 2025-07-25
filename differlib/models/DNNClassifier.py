from collections import OrderedDict

import torch
import torch.nn as nn
from braindecode.models import EEGNetv4, EEGNetv1

# global network parameters
# CamCAN data parameters
global_channels = 204
global_points = 100
global_classes = 2
# DecMeg2014 data parameters
# global_channels = 204
# global_points = 250
# global_classes = 2

global_spatial_sources = 32
global_conv_pool = 2
global_conv_dropout = 0.5
global_active_func = nn.ReLU()

global_gru1_hidden = 100
global_gru2_hidden = 10
global_gru_pool = 5
global_gru_dropout = 0.1

global_mlp_hidden_features = 500
global_mlp_dropout = 0.2


# init/reset global network parameters
def init_global_network_parameters(channels=204, points=100, num_classes=2,
                                   spatial_sources=32, conv_pool=2, conv_dropout=0.5, active_func=nn.ReLU(),
                                   gru1_hidden=100, gru2_hidden=10, gru_pool=5, gru_dropout=0.1,
                                   mlp_hidden_features=500, mlp_dropout=0.2):
    global global_channels, global_points, global_classes, \
        global_spatial_sources, global_conv_pool, global_conv_dropout, global_active_func, \
        global_gru1_hidden, global_gru2_hidden, global_gru_pool, global_gru_dropout, \
        global_mlp_hidden_features, global_mlp_dropout
    global_channels = channels
    global_points = points
    global_classes = num_classes
    global_spatial_sources = spatial_sources
    global_conv_pool = conv_pool
    global_conv_dropout = conv_dropout
    global_active_func = active_func
    global_gru1_hidden = gru1_hidden
    global_gru2_hidden = gru2_hidden
    global_gru_pool = gru_pool
    global_gru_dropout = gru_dropout
    global_mlp_hidden_features = mlp_hidden_features
    global_mlp_dropout = mlp_dropout


# 初始化基准模型
def init_models():
    lfcnn, varcnn, hgrn = LFCNN(), VARCNN(), HGRN()
    return [lfcnn, varcnn, hgrn]


def lfcnn(channels=204, points=100, num_classes=2, **kwargs):
    init_global_network_parameters(channels=channels, points=points, num_classes=num_classes)
    return LFCNN()


def varcnn(channels=204, points=100, num_classes=2, **kwargs):
    init_global_network_parameters(channels=channels, points=points, num_classes=num_classes)
    return VARCNN()


def hgrn(channels=204, points=100, num_classes=2, **kwargs):
    init_global_network_parameters(channels=channels, points=points, num_classes=num_classes)
    return HGRN()


def mlp(channels=204, points=100, num_classes=2, **kwargs):
    init_global_network_parameters(channels=channels, points=points, num_classes=num_classes)
    return MLP()


def linear(channels=204, points=100, num_classes=2, **kwargs):
    init_global_network_parameters(channels=channels, points=points, num_classes=num_classes)
    return Linear()


def eegnetv4(channels=204, points=100, num_classes=2, **kwargs):
    return EEGNetv4(channels, num_classes, points)


def eegnetv1(channels=204, points=100, num_classes=2, **kwargs):
    return NewEEGNetv1(channels, num_classes, points)


class NewEEGNetv1(nn.Module):
    def __init__(self, channels, num_classes, points):
        super().__init__()
        self.output = EEGNetv1(channels, num_classes, points)

    # input data shape：(batch * channels * points)
    def forward(self, x, is_training_data=False):
        out = self.output(x)
        if is_training_data:
            return out, 0.0
        return out


# 转换非torch.nn类型操作，以适应Sequential
# define torch.transpose in torch.nn
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


# define torch.unsqueeze in torch.nn
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


# define torch.Tensor.view in torch.nn
class TensorView(nn.Module):
    def __init__(self):
        super(TensorView, self).__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


# GRU操作返回一个二维元组，提取Sequential只需要的第一个数据
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, x):
        return x[self.item_index]


class LFCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('transpose0', Transpose(1, 2)),
            ('Spatial', nn.Linear(global_channels, global_spatial_sources)),
            ('transpose1', Transpose(1, 2)),
            ('unsqueeze', Unsqueeze(-2)),
            ('Temporal_LF', nn.Conv2d(global_spatial_sources, global_spatial_sources, (1, 7), padding=(0, 3),
                                      groups=global_spatial_sources)),
            ('active', global_active_func),
            ('transpose2', Transpose(1, 2)),
            ('pool', nn.MaxPool2d((1, global_conv_pool), (1, global_conv_pool))),
            ('view', TensorView()),
            ('dropout', nn.Dropout(p=global_conv_dropout))
        ]))
        # 输出层
        self.output = nn.Linear(global_spatial_sources * int(global_points / 2), global_classes)

    # input data shape：(batch * channels * points)
    def forward(self, x, is_training_data=False):
        x = self.features(x)
        out = self.output(x)
        if is_training_data:
            return out, 0.0
        return out


class VARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('transpose0', Transpose(1, 2)),
            ('Spatial', nn.Linear(global_channels, global_spatial_sources)),
            ('transpose1', Transpose(1, 2)),
            ('Temporal_VAR', nn.Conv1d(global_spatial_sources, global_spatial_sources, (7,), padding=(3,))),
            ('unsqueeze', Unsqueeze(-3)),
            ('active', global_active_func),
            ('pool', nn.MaxPool2d((1, global_conv_pool), (1, global_conv_pool))),
            ('view', TensorView()),
            ('dropout', nn.Dropout(p=global_conv_dropout))
        ]))
        # 输出层
        self.output = nn.Linear(global_spatial_sources * int(global_points / 2), global_classes)

    # input data shape：(batch * channels * points)
    def forward(self, x, is_training_data=False):
        x = self.features(x)
        out = self.output(x)
        if is_training_data:
            return out, 0.0
        return out


class HGRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('transpose', Transpose(1, 2)),
            ('GRU_1', nn.GRU(input_size=global_channels, hidden_size=global_gru1_hidden, batch_first=True)),
            ('selectItem1', SelectItem(0)),
            ('dropout1', nn.Dropout(p=global_gru_dropout)),
            ('pool1', nn.MaxPool2d((global_gru_pool, 1), (global_gru_pool, 1))),
            ('GRU_2', nn.GRU(input_size=global_gru1_hidden, hidden_size=global_gru2_hidden, batch_first=True)),
            ('selectItem2', SelectItem(0)),
            ('dropout2', nn.Dropout(p=global_gru_dropout)),
            ('pool2', nn.MaxPool2d((global_gru_pool, 1), (global_gru_pool, 1))),
            ('view', TensorView())
        ]))
        # 输出层
        self.output = nn.Linear(global_gru2_hidden * int(global_points / global_gru_pool / global_gru_pool),
                                global_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    # input data shape：(batch * channels * points)
    def forward(self, x, is_training_data=False):
        x = self.features(x)
        out = self.output(x)
        out = self.softmax(out)
        if is_training_data:
            return out, 0.0
        return out


class MLP(nn.Sequential):
    def __init__(self):
        super().__init__(
            TensorView(),
            nn.Linear(global_channels * global_points, global_mlp_hidden_features),
            nn.BatchNorm1d(global_mlp_hidden_features),     # 非eval模式下，会累计更新，影响后续预测结果
            nn.Dropout(global_mlp_dropout),
            nn.ReLU(),
            nn.Linear(global_mlp_hidden_features, global_mlp_hidden_features),
            nn.BatchNorm1d(global_mlp_hidden_features),
            nn.Dropout(global_mlp_dropout),
            nn.ReLU(),
            nn.Linear(global_mlp_hidden_features, global_classes),
            nn.BatchNorm1d(global_classes),
            nn.Dropout(global_mlp_dropout),
            nn.Softmax(dim=1)
        )


class Linear(nn.Sequential):
    def __init__(self):
        super().__init__(
            TensorView(),
            nn.Linear(global_channels * global_points, global_classes),
            nn.Softmax(dim=1)
        )
