import torch
import torch.nn as nn
from torch.nn import functional as F


def megnet(channels=204, points=100, num_classes=2, **kwargs):
    return MEGNet(nb_classes=num_classes, Chans=channels, Samples=points)


class MEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(MEGNet, self).__init__()

        # 输入形状处理 (PyTorch使用NCHW格式)
        self.input_shape = (1, Chans, Samples)  # [channels, height, width]

        # Block 1 实现
        self.block1 = nn.Sequential(
            # Conv2D(F1, (1, kernLength))
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),

            # DepthwiseConv2D实现 (分组卷积)
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropoutRate) if dropoutType == 'SpatialDropout2D' else nn.Dropout(dropoutRate)
        )

        # Block 2 实现
        self.block2 = nn.Sequential(
            # SeparableConv2D实现 (深度可分离卷积)
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding='same', bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(dropoutRate) if dropoutType == 'SpatialDropout2D' else nn.Dropout(dropoutRate)
        )

        # 全连接层
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self._get_flatten_size(), nb_classes)

        # 权重约束 (在forward中实现)
        self.norm_rate = norm_rate

    def _get_flatten_size(self):
        # 自动计算flatten后的维度
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.block1(x)
            x = self.block2(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        # 输入形状转换 [N, C, H, W]
        x = x.permute(0, 3, 1, 2)  # 从NHWC转为NCHW

        # Block 1
        x = self.block1(x)

        # Block 2
        x = self.block2(x)

        # 全连接层
        x = self.flatten(x)
        x = self.dense(x)

        # 权重约束 (max norm)
        with torch.no_grad():
            norm = self.dense.weight.norm(2, dim=1, keepdim=True).clamp(min=self.norm_rate)
            self.dense.weight.data.div_(norm)

        return F.softmax(x, dim=1)
