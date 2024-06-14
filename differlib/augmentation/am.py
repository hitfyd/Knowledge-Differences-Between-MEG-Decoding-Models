# Augmentation Method

import abc
import sys

from differlib.engine.utils import setup_seed

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class AMethod(ABC):
    """
    AMethod is the base class for Augmentation Method (AM).
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a AMethod object.
        """

    @abc.abstractmethod
    def augment(self, *argv, **kwargs):
        """
        Augment the input data.
        """
        raise NotImplementedError


class NoneAM(AMethod):
    def augment(self, data, labels, *argv, **kwargs):
        return data, labels


import random
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import signal


@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    awgn_time: MinMax = field(default_factory=lambda: MinMax(.0, 0.01))  # SNR = 1 / level
    awgn_frequency: MinMax = field(default_factory=lambda: MinMax(.0, 0.01))  # SNR = 1 / level
    reduce_waveform: MinMax = field(default_factory=lambda: MinMax(.0, .1))
    expand_waveform: MinMax = field(default_factory=lambda: MinMax(.0, .1))
    translation_left: MinMax = field(default_factory=lambda: MinMax(.0, .1))
    translation_right: MinMax = field(default_factory=lambda: MinMax(.0, .1))
    mask: MinMax = field(default_factory=lambda: MinMax(.0, .1))


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return '<' + self.name + '>'

    def __name__(self):
        return self.name

    def meg_transformer(self, probability, level):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level)
            if len(im.shape) < 2:
                print(2)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


def _compute_shape_size(array):
    size = 1
    for i in array.shape:
        size *= i
    return size


# 单图像输入
# 时域噪声添加
def _awgn_time(epoch_data, level):
    """
    加性高斯白噪声 Additive White Gaussian Noise
    level即为SNR
    """
    if level == 0:
        return epoch_data
    x = epoch_data
    snr = 1 / float_parameter(level, min_max_vals.awgn_time.max)
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / _compute_shape_size(x)
    npower = xpower / snr
    noise = np.random.randn(*x.shape) * np.sqrt(npower)
    aug_data = x + noise
    return aug_data


# 频域噪声添加
def _awgn_frequency(epoch_data, level):
    sfreq = 125
    if level == 0:
        return epoch_data
    f, t, Zxx = signal.stft(epoch_data, fs=sfreq, nperseg=epoch_data.shape[-1])
    _, origin = signal.istft(Zxx)
    x = Zxx
    snr = 1 / float_parameter(level, min_max_vals.awgn_frequency.max)
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / _compute_shape_size(x)
    npower = xpower / snr
    noise = np.random.randn(*x.shape) * np.sqrt(npower)
    aug_data = x + noise
    _, aug_data_time = signal.istft(aug_data, fs=sfreq)
    return aug_data_time


PARAMETER_MAX = 10
min_max_vals = MinMaxVals()
awgn_time = TransformT('awgn_time', _awgn_time)
awgn_frequency = TransformT('awgn_frequency', _awgn_frequency)
# 波形缩放
reduce_waveform = TransformT('reduce_waveform', lambda epoch_data, level: epoch_data * (
        1 - float_parameter(level, min_max_vals.reduce_waveform.max)))
expand_waveform = TransformT('expand_waveform', lambda epoch_data, level: epoch_data * (
        1 + float_parameter(level, min_max_vals.expand_waveform.max)))


# 时域左右平移，空缺位置为常数或者随机数填充
def _translation_left(epoch_data, level):
    if level == 0:
        return epoch_data
    level = float_parameter(level, min_max_vals.translation_left.max)
    translation_length = int(level * epoch_data.shape[1])
    fill_matrix = np.zeros((epoch_data.shape[0], translation_length), dtype=float)
    aug_data = np.concatenate((epoch_data[:, translation_length:], fill_matrix), axis=1)
    return aug_data


translation_left = TransformT('translation_left', _translation_left)


def _translation_right(epoch_data, level):
    if level == 0:
        return epoch_data
    level = float_parameter(level, min_max_vals.translation_right.max)
    translation_length = int(level * epoch_data.shape[1])
    fill_matrix = np.zeros((epoch_data.shape[0], translation_length), dtype=float)
    aug_data = np.concatenate((fill_matrix, epoch_data[:, :-translation_length]), axis=1)
    return aug_data


translation_right = TransformT('translation_right', _translation_right)


# 随机缺失，缺失位置为常数或者随机数填充
def _mask(epoch_data, level):
    channels, points = epoch_data.shape[0], epoch_data.shape[1]
    level = float_parameter(level, min_max_vals.mask.max)
    mask_length = int(level * points)
    mask_start_index = random.randint(0, points - mask_length)
    for i in range(channels):
        for j in range(points):
            if mask_start_index <= j < mask_start_index + mask_length:
                epoch_data[i][j] = 0
    return epoch_data


mask = TransformT('mask', _mask)


# 多图像输入
def divide_by_labels(data, labels):
    labels_set = set(labels)
    for label in labels_set:
        exec("data_%s=[]" % label)
    for index in range(len(labels)):
        exec("data_%s.append(data[index])" % labels[index])
    data_dict = {}
    for label in labels_set:
        exec("data_dict[%d] = data_%s" % (label, label))
    return data_dict


# 时域分割重组
def segment_recombine_time(train_data, ag_number, segment=4):
    """
    根据原始数据集，使用时域分割重组生成数据增强数据集
    :param train_data: 原始数据集
    :param ag_number: 每一个类的生成的个数
    :param segment：分割数量
    """
    # 生成重组随机矩阵
    train_data = np.array(train_data)
    assert len(train_data.shape) == 3
    random_matrix = np.random.randint(0, train_data.shape[0], (ag_number, segment))

    # 生成重组epoch
    ag_data = np.zeros((ag_number, train_data.shape[1], train_data.shape[2]), dtype=np.float32)
    segment_length = int(train_data.shape[-1] / segment)
    for i in range(ag_number):
        for j in range(segment):
            if j < segment - 1:
                ag_data[i, :, j * segment_length:(j + 1) * segment_length] = \
                    train_data[random_matrix[i][j], :, j * segment_length:(j + 1) * segment_length]
            else:
                ag_data[i, :, j * segment_length:] = train_data[random_matrix[i][j], :, j * segment_length:]

    return ag_data


# 信道分割重组
def segment_recombine_channel(train_data, ag_number, segment=4):
    """
    根据原始数据集，使用信道分割重组生成数据增强数据集
    :param train_data: 原始数据集
    :param ag_number: 每一个类的生成的个数
    :param segment：分割数量
    """
    # 生成重组随机矩阵
    train_data = np.array(train_data)
    assert len(train_data.shape) == 3
    random_matrix = np.random.randint(0, train_data.shape[0], (ag_number, segment))

    # 生成重组epoch
    ag_data = np.zeros((ag_number, train_data.shape[1], train_data.shape[2]), dtype=np.float32)
    segment_length = int(train_data.shape[-1] / segment)
    for i in range(ag_number):
        for j in range(segment):
            if j < segment - 1:
                ag_data[i, j * segment_length:(j + 1) * segment_length, :] = \
                    train_data[random_matrix[i][j], j * segment_length:(j + 1) * segment_length, :]
            else:
                ag_data[i, :, j * segment_length:] = train_data[random_matrix[i][j], :, j * segment_length:]

    return ag_data


# 默认的分割频带及其范围
bandpass_frequency = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 14},
    {'name': 'Beta', 'fmin': 14, 'fmax': 30},
    {'name': 'Gamma', 'fmin': 30, 'fmax': 90}
]


# 频域分割重组
def segment_recombine_frequency(train_data, ag_number, sfreq=125):
    """
    根据原始数据集，使用频域分割重组生成数据增强数据集
    :param train_data: 原始数据集
    :param ag_number: 每一个类的生成的个数
    :param sfreq：采样频率
    """
    # 区分标签并将单个epoch分割
    train_data_f, train_data_Zxx = [], []
    for i in range(len(train_data)):
        f, t, Zxx = signal.stft(train_data[i], fs=sfreq, nperseg=train_data[i].shape[-1])
        train_data_f.append(f)
        train_data_Zxx.append(Zxx)

    # 生成重组随机矩阵
    random_matrix = np.random.randint(0, len(train_data_f), (ag_number, len(bandpass_frequency)))

    # 生成重组epoch
    gen_data = []
    for i in range(ag_number):
        recombine0 = np.zeros(train_data_Zxx[0].shape, dtype=np.complex_)
        for j in range(len(bandpass_frequency)):
            iter_subject = random_matrix[i][j]
            iter_freq = bandpass_frequency[j]
            # 定位有效频率的索引
            index = np.where(
                (iter_freq['fmin'] <= train_data_f[iter_subject]) & (train_data_f[iter_subject] <= iter_freq['fmax']))
            recombine0[:, index, :] = train_data_Zxx[iter_subject][:, index, :]
        _, recombine0_data = signal.istft(recombine0, fs=sfreq)
        gen_data.append(recombine0_data)

    gen_data = np.array(gen_data)
    # print(gen_data.shape)

    return gen_data


# 随机同标签数据取平均
def average(epoch_1, epoch_2, level=0):
    return (epoch_1 + epoch_2) / 2


def epochs_average(train_data, ag_number, average_number=2):
    # 生成重组随机矩阵
    train_data = np.array(train_data)
    assert len(train_data.shape) == 3
    random_matrix = np.random.randint(0, train_data.shape[0], (ag_number, average_number))

    ag_data = np.zeros((ag_number, train_data.shape[1], train_data.shape[2]), dtype=np.float32)
    for i in range(ag_number):
        for j in range(average_number):
            ag_data[i] += train_data[random_matrix[i][j]]
        ag_data[i] /= average_number

    return ag_data


ALL_TRANSFORMS = [
    awgn_time,
    awgn_frequency,
    reduce_waveform,
    expand_waveform,
    # translation_left,
    # translation_right,
    # mask,
    # segment_recombine_time,
    # segment_recombine_frequency,
]

multi_input_algorithm = [
    segment_recombine_time,
    # segment_recombine_channel,
    segment_recombine_frequency,
    # epochs_average,
]


class BaseAM(AMethod):

    def augment(self, origin_data, delta_labels, *argv, augment_factor=0.5, label_ratio="1", **kwargs):
        # label_ratio："balance"表示数据增广后，预测标签一致和不一致的样本比例调整为[0.5:0.5]，否则为原始标签比例
        n_labels = len(delta_labels)
        n_true_labels = delta_labels.sum()
        n_false_labels = n_labels - n_true_labels
        if label_ratio == "balance":
            n_augmented = int(n_labels * augment_factor)
            n_aug_true = (n_augmented + n_labels) // 2 - n_true_labels
            n_aug_false = n_augmented - n_aug_true
        else:
            n_aug_true = int(n_true_labels * augment_factor)
            n_aug_false = int(n_false_labels * augment_factor)

        true_data = origin_data[np.where(delta_labels == 1)]
        false_data = origin_data[np.where(delta_labels == 0)]

        def _aug_label_data(data, number):
            ag_data, ag_label = [], []
            for i in range(number):
                op = ALL_TRANSFORMS[random.randint(0, len(ALL_TRANSFORMS)-1)]
                if op in multi_input_algorithm:
                    ag_data.extend(op(data, 1))
                    ag_label.append(1)
                else:
                    data_index = random.randint(0, len(data) - 1)
                    ag_data.append(op.meg_transformer(1., PARAMETER_MAX - 1)(data[data_index]))
                    ag_label.append(delta_labels[data_index])

                # augment_func = multi_input_algorithm[random.randint(0, len(multi_input_algorithm)-1)]
                # ag_data.extend(augment_func(data, 1))
                # ag_label.append(1)

            # n = number // len(multi_input_algorithm)
            # for augment_func in multi_input_algorithm:
            #     ag_data.extend(augment_func(data, n))
            #     ag_label.extend(np.full(n, 1))
            return np.array(ag_data), np.array(ag_label)

        ag_data, ag_label = _aug_label_data(true_data, n_aug_true)
        all_data = np.concatenate((origin_data, ag_data), axis=0)
        all_label = np.concatenate((delta_labels, ag_label), axis=0)
        ag_data, ag_label = _aug_label_data(false_data, n_aug_false)
        all_data = np.concatenate((all_data, ag_data), axis=0)
        all_label = np.concatenate((all_label, ag_label), axis=0)
        return all_data, all_label


        # ag_data, ag_label = [], []
        # data_dict = divide_by_labels(origin_data, delta_labels)
        # for label in data_dict.keys():
        #     label_data = data_dict[label]
        #
        #     for augment_func in multi_input_algorithm:
        #         ag_data.extend(augment_func(label_data, len(label_data)))
        #         ag_label.extend(np.full(len(label_data), label))
        #
        #
        # ag_data, ag_label = [], []
        # for i in range(len(delta_labels)):
        #     # if delta_labels[i] == 0:
        #     #     continue
        #     for op in ALL_TRANSFORMS:
        #         ag_data.append(op.meg_transformer(1., PARAMETER_MAX - 1)(origin_data[i]))
        #         ag_label.append(delta_labels[i])
        #         # ag_data.append(op.meg_transformer(1., PARAMETER_MAX - 1)(origin_data[i]))
        #         # ag_label.append(delta_labels[i])
        #
        # data_dict = divide_by_labels(origin_data, delta_labels)
        # for label in data_dict.keys():
        #     label_data = data_dict[label]
        #     for augment_func in multi_input_algorithm:
        #         ag_data.extend(augment_func(label_data, len(label_data)))
        #         ag_label.extend(np.full(len(label_data), label))
        #
        # ag_data, ag_label = np.array(ag_data), np.array(ag_label)
        # all_data = np.concatenate((origin_data, ag_data), axis=0)
        # all_label = np.concatenate((delta_labels, ag_label), axis=0)
        # return all_data, all_label
