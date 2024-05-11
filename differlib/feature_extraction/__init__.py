import numpy as np
from tqdm import tqdm


def feature_extraction(x: np.ndarray, window_length=10):
    assert x.ndim == 2
    num_samples, num_times = x.shape
    num_windows = num_times // window_length
    points = 250
    channels = num_times // points
    x_features = []
    for i in tqdm(range(num_samples)):
        sample_features = []
        for window in np.array_split(x[i], num_windows):
            # Time Domain Features
            # maximum, minimum, mean, standard deviation
            maximum = np.max(window)
            minimum = np.min(window)
            mean = np.mean(window)
            standard_deviation = np.std(window)
            # diff = np.diff(window)
            # first_difference = np.mean(diff)
            # norm_first_difference = first_difference / standard_deviation
            # second_diff = window[2:] - window[:-2]
            # second_difference = np.mean(second_diff)
            # norm_second_difference = second_difference / standard_deviation

            time_features = [maximum, minimum, mean, standard_deviation,]
                             # first_difference, norm_first_difference, second_difference, norm_second_difference]
            sample_features.extend(time_features)
            # sample_features.extend(window)

            # Frequency Domain Features：针对平稳信号（静息态）
            pass

        # for c in range(channels):
        #     # Time-Frequency Domain：针对非平稳信号
        #     channel_data = x[i, c*points:(c+1)*points]
        #     window_type = 'hann'  # 窗口类型
        #     overlap = 0  # 重叠比例
        #     from scipy.signal import stft
        #     frequencies, time_points, magnitude = stft(channel_data, fs=250, window=window_type, nperseg=window_length,
        #                                                noverlap=overlap)
        #     power = np.abs(magnitude) ** 2
        #
        #     # 按频带范围求平均能量
        #     # delta_power = np.mean(power[np.where((frequencies >= 0.5) & (frequencies < 4))[0], :], axis=0)
        #     theta_power = np.mean(power[np.where((frequencies >= 4) & (frequencies < 8))[0], :], axis=0)
        #     alpha_power = np.mean(power[np.where((frequencies >= 8) & (frequencies < 12))[0], :], axis=0)
        #     beta_power = np.mean(power[np.where((frequencies >= 12) & (frequencies < 30))[0], :], axis=0)
        #     gamma_power = np.mean(power[np.where((frequencies >= 30) & (frequencies <= 50))[0], :], axis=0)
        #     # sample_features.extend(delta_power)   # 容易为nan
        #     sample_features.extend(theta_power)
        #     sample_features.extend(alpha_power)
        #     sample_features.extend(beta_power)
        #     sample_features.extend(alpha_power / beta_power)
        #     sample_features.extend(gamma_power)
        #
        #     # sample_features.extend(power.reshape(-1))
        x_features.append(sample_features)
    return np.array(x_features)


def bandpower(data, sf, band, window_sec=None, relative=False):
    window_length = 50  # 窗口长度
    window_type = 'hann'  # 窗口类型
    overlap = 0  # 重叠比例
    from scipy.signal import stft
    frequencies, time_points, magnitude = stft(data, fs=250, window=window_type, nperseg=window_length, noverlap=overlap)
    power = np.abs(magnitude) ** 2
    start_freq = 8  # 起始频率
    end_freq = 12  # 结束频率

    freq_indices = np.where((frequencies >= start_freq) & (frequencies <= end_freq))[0]
    band_power = np.mean(power[freq_indices, :], axis=0)  # 按频带范围求平均能量
    return band_power
