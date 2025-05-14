# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import register_parametrization
from torch.fft import fftfreq


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


class GeneralizedGaussianFilter(nn.Module):
    """Generalized Gaussian Filter from Ludwig et al (2024) [eegminer]_.

    Implements trainable temporal filters based on generalized Gaussian functions
    in the frequency domain.

    This module creates filters in the frequency domain using the generalized
    Gaussian function, allowing for trainable center frequency (`f_mean`),
    bandwidth (`bandwidth`), and shape (`shape`) parameters.

    The filters are applied to the input signal in the frequency domain, and can
    be optionally transformed back to the time domain using the inverse
    Fourier transform.

    The generalized Gaussian function in the frequency domain is defined as:

    .. math::

        F(x) = \\exp\\left( - \\left( \\frac{abs(x - \\mu)}{\\alpha} \\right)^{\\beta} \\right)

    where:
      - μ (mu) is the center frequency (`f_mean`).

      - α (alpha) is the scale parameter, reparameterized in terms of the full width at half maximum (FWHM) `h` as:

      .. math::

          \\alpha = \\frac{h}{2 \\left( \\ln(2) \\right)^{1/\\beta}}

      - β (beta) is the shape parameter (`shape`), controlling the shape of the filter.

    The filters are constructed in the frequency domain to allow full control
    over the magnitude and phase responses.

    A linear phase response is used, with an optional trainable group delay (`group_delay`).

      - Copyright (C) Cogitat, Ltd.
      - Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
      - Patent GB2609265 - Learnable filters for eeg classification
      - https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. Must be a multiple of `in_channels`.
    sequence_length : int
        Length of the input sequences (time steps).
    sample_rate : float
        Sampling rate of the input signals in Hz.
    inverse_fourier : bool, optional
        If True, applies the inverse Fourier transform to return to the time domain after filtering.
        Default is True.
    affine_group_delay : bool, optional
        If True, makes the group delay parameter trainable. Default is False.
    group_delay : tuple of float, optional
        Initial group delay(s) in milliseconds for the filters. Default is (20.0,).
    f_mean : tuple of float, optional
        Initial center frequency (frequencies) of the filters in Hz. Default is (23.0,).
    bandwidth : tuple of float, optional
        Initial bandwidth(s) (full width at half maximum) of the filters in Hz. Default is (44.0,).
    shape : tuple of float, optional
        Initial shape parameter(s) of the generalized Gaussian filters. Must be >= 2.0. Default is (2.0,).
    clamp_f_mean : tuple of float, optional
        Minimum and maximum allowable values for the center frequency `f_mean` in Hz.
        Specified as (min_f_mean, max_f_mean). Default is (1.0, 45.0).


    Notes
    -----
    The model and the module **have a patent** [eegminercode]_, and the **code is CC BY-NC 4.0**.

    .. versionadded:: 0.9

    References
    ----------
    .. [eegminer] Ludwig, S., Bakas, S., Adamos, D. A., Laskaris, N., Panagakis,
       Y., & Zafeiriou, S. (2024). EEGMiner: discovering interpretable features
       of brain activity with learnable filters. Journal of Neural Engineering,
       21(3), 036010.
    .. [eegminercode] Ludwig, S., Bakas, S., Adamos, D. A., Laskaris, N., Panagakis,
       Y., & Zafeiriou, S. (2024). EEGMiner: discovering interpretable features
       of brain activity with learnable filters.
       https://github.com/SMLudwig/EEGminer/.
       Cogitat, Ltd. "Learnable filters for EEG classification."
       Patent GB2609265.
       https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        sequence_length,
        sample_rate,
        inverse_fourier=True,
        affine_group_delay=False,
        group_delay=(20.0,),
        f_mean=(23.0,),
        bandwidth=(44.0,),
        shape=(2.0,),
        clamp_f_mean=(1.0, 45.0),
    ):
        super(GeneralizedGaussianFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.inverse_fourier = inverse_fourier
        self.affine_group_delay = affine_group_delay
        self.clamp_f_mean = clamp_f_mean
        assert out_channels % in_channels == 0, (
            "out_channels has to be multiple of in_channels"
        )
        assert len(f_mean) * in_channels == out_channels
        assert len(bandwidth) * in_channels == out_channels
        assert len(shape) * in_channels == out_channels

        # Range from 0 to half sample rate, normalized
        self.n_range = nn.Parameter(
            torch.tensor(
                list(
                    fftfreq(n=sequence_length, d=1 / sample_rate)[
                        : sequence_length // 2
                    ]
                )
                + [sample_rate / 2]
            )
            / (sample_rate / 2),
            requires_grad=False,
        )

        # Trainable filter parameters
        self.f_mean = nn.Parameter(
            torch.tensor(f_mean * in_channels) / (sample_rate / 2), requires_grad=True
        )
        self.bandwidth = nn.Parameter(
            torch.tensor(bandwidth * in_channels) / (sample_rate / 2),
            requires_grad=True,
        )  # full width half maximum
        self.shape = nn.Parameter(torch.tensor(shape * in_channels), requires_grad=True)

        # Normalize group delay so that group_delay=1 corresponds to 1000ms
        self.group_delay = nn.Parameter(
            torch.tensor(group_delay * in_channels) / 1000,
            requires_grad=affine_group_delay,
        )

        # Construct filters from parameters
        self.filters = self.construct_filters()

    @staticmethod
    def exponential_power(x, mean, fwhm, shape):
        """
        Computes the generalized Gaussian function:

        .. math::

             F(x) = \\exp\\left( - \\left( \\frac{|x - \\mu|}{\\alpha} \\right)^{\\beta} \\right)

        where:

          - :math:`\\mu` is the mean (`mean`).

          - :math:`\\alpha` is the scale parameter, reparameterized using the FWHM :math:`h` as:

            .. math::

                \\alpha = \\frac{h}{2 \\left( \\ln(2) \\right)^{1/\\beta}}

          - :math:`\\beta` is the shape parameter (`shape`).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing frequencies, normalized between 0 and 1.
        mean : torch.Tensor
            The center frequency (`f_mean`), normalized between 0 and 1.
        fwhm : torch.Tensor
            The full width at half maximum (`bandwidth`), normalized between 0 and 1.
        shape : torch.Tensor
            The shape parameter (`shape`) of the generalized Gaussian.

        Returns
        -------
        torch.Tensor
            The computed generalized Gaussian function values at frequencies `x`.

        """
        mean = mean.unsqueeze(1)
        fwhm = fwhm.unsqueeze(1)
        shape = shape.unsqueeze(1)
        log2 = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        scale = fwhm / (2 * log2 ** (1 / shape))
        # Add small constant to difference between x and mean since grad of 0 ** shape is nan
        return torch.exp(-((((x - mean).abs() + 1e-8) / scale) ** shape))

    def construct_filters(self):
        """
        Constructs the filters in the frequency domain based on current parameters.

        Returns
        -------
        torch.Tensor
            The constructed filters with shape `(out_channels, freq_bins, 2)`.

        """
        # Clamp parameters
        self.f_mean.data = torch.clamp(
            self.f_mean.data,
            min=self.clamp_f_mean[0] / (self.sample_rate / 2),
            max=self.clamp_f_mean[1] / (self.sample_rate / 2),
        )
        self.bandwidth.data = torch.clamp(
            self.bandwidth.data, min=1.0 / (self.sample_rate / 2), max=1.0
        )
        self.shape.data = torch.clamp(self.shape.data, min=2.0, max=3.0)

        # Create magnitude response with gain=1 -> (channels, freqs)
        mag_response = self.exponential_power(
            self.n_range, self.f_mean, self.bandwidth, self.shape * 8 - 14
        )
        mag_response = mag_response / mag_response.max(dim=-1, keepdim=True)[0]

        # Create phase response, scaled so that normalized group_delay=1
        # corresponds to group delay of 1000ms.
        phase = torch.linspace(
            0,
            self.sample_rate,
            self.sequence_length // 2 + 1,
            device=mag_response.device,
            dtype=mag_response.dtype,
        )
        phase = phase.expand(mag_response.shape[0], -1)  # repeat for filter channels
        pha_response = -self.group_delay.unsqueeze(-1) * phase * torch.pi

        # Create real and imaginary parts of the filters
        real = mag_response * torch.cos(pha_response)
        imag = mag_response * torch.sin(pha_response)

        # Stack real and imaginary parts to create filters
        # -> (channels, freqs, 2)
        filters = torch.stack((real, imag), dim=-1)

        return filters

    def forward(self, x):
        """
        Applies the generalized Gaussian filters to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., in_channels, sequence_length)`.

        Returns
        -------
        torch.Tensor
            The filtered signal. If `inverse_fourier` is True, returns the signal in the time domain
            with shape `(..., out_channels, sequence_length)`. Otherwise, returns the signal in the
            frequency domain with shape `(..., out_channels, freq_bins, 2)`.

        """
        # Construct filters from parameters
        self.filters = self.construct_filters()
        # Preserving the original dtype.
        dtype = x.dtype
        # Apply FFT -> (..., channels, freqs, 2)
        x = torch.fft.rfft(x, dim=-1)
        x = torch.view_as_real(x)  # separate real and imag

        # Repeat channels in case of multiple filters per channel
        x = torch.repeat_interleave(x, self.out_channels // self.in_channels, dim=-3)

        # Apply filters in the frequency domain
        x = x * self.filters

        # Apply inverse FFT if requested
        if self.inverse_fourier:
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, n=self.sequence_length, dim=-1)

        x = x.to(dtype)

        return x
