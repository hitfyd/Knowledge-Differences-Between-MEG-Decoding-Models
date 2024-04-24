import numpy as np


def feature_extraction(x: np.ndarray, window_length=25):
    assert x.ndim == 2
    num_samples, num_times = x.shape
    num_windows = int(num_times / window_length)
    x_features = []
    for i in range(num_samples):
        sample_features = []
        for j in range(num_windows):
            window = x[i, j * window_length:(j + 1) * window_length]
            # Time Domain Features
            # maximum, minimum, mean, standard deviation
            maximum = np.max(window)
            minimum = np.min(window)
            mean = np.mean(window)
            standard_deviation = np.std(window)
            diff = np.diff(window)
            first_difference = np.mean(diff)
            norm_first_difference = first_difference / standard_deviation
            second_diff = window[2:] - window[:-2]
            second_difference = np.mean(second_diff)
            norm_second_difference = second_difference / standard_deviation

            # Frequency Domain Features

            # Time-Frequency Domain

            window_features = [maximum, minimum, mean, standard_deviation,
                               first_difference, norm_first_difference, second_difference, norm_second_difference]
            sample_features.extend(window_features)
        x_features.append(sample_features)
    return np.array(x_features)
