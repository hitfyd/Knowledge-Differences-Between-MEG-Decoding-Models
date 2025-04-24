# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import register_parametrization


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class CausalConv1d(nn.Conv1d):
    """Causal 1-dimensional convolution

    Code modified from [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels (number of filters).
    kernel_size : int
        Kernel size.
    dilation : int, optional
        Dilation (number of elements to skip within kernel multiplication).
        Default to 1.
    **kwargs :
        Other keyword arguments to pass to torch.nn.Conv1d, except for
        `padding`!!

    References
    ----------
    .. [1] https://discuss.pytorch.org/t/causal-convolution/3456/4
    .. [2] https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        assert "padding" not in kwargs, (
            "The padding parameter is controlled internally by "
            f"{type(self).__name__} class. You should not try to override this"
            " parameter."
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, X):
        out = F.conv1d(
            X,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out[..., : -self.padding[0]]


class MaxNorm(nn.Module):
    def __init__(self, max_norm_val=2.0, eps=1e-5):
        super().__init__()
        self.max_norm_val = max_norm_val
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        return X * (number / (denom + self.eps))

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        # Assuming the forward scales X by a factor s,
        # the right inverse would scale it back by 1/s.
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        scale = number / (denom + self.eps)
        return X / scale


class MaxNormLinear(nn.Linear):
    """Linear layer with MaxNorm constraining on weights.

    Equivalent of Keras tf.keras.Dense(..., kernel_constraint=max_norm())
    [1]_ and [2]_. Implemented as advised in [3]_.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    .. [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    .. [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/
           MaxNorm
    .. [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-
           max-norm-constraint/96769
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps
        register_parametrization(self, "weight", MaxNorm(self._max_norm_val, self._eps))
