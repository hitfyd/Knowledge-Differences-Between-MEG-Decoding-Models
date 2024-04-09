# Augmentation Method

import abc
import sys

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
ALL_TRANSFORMS = [
    awgn_time,
    awgn_frequency,
    reduce_waveform,
    expand_waveform,
]


class BaseAM(AMethod):

    def augment(self, origin_data, delta_labels, *argv, **kwargs):
        ag_data, ag_label = [], [], [], []
        for i in range(len(delta_labels)):
            if delta_labels[i] == 0:
                continue
            for op in ALL_TRANSFORMS:
                ag_data.append(op.meg_transformer(1., PARAMETER_MAX-1)(origin_data[i]))
                ag_label.append(delta_labels[i])
        ag_data, ag_label = np.array(ag_data), np.array(ag_label)
        all_data = np.concatenate((origin_data, ag_data), axis=0)
        all_label = np.concatenate((delta_labels, ag_label), axis=0)
        return all_data, all_label
