import numpy as np


def feature_extraction(x: np.ndarray, window_length=25):
    assert x.ndim == 2
    num_samples, num_times = x.shape
    num_windows = int(num_times / window_length)
    x_features = []
    for i in range(num_samples):
        x_i = x[i]
        x_i_features = []
        for j in range(num_windows):
            window = x_i[j * window_length:(j + 1) * window_length]
            # Time Domain Features
            # maximum, minimum, mean, standard deviation
            maximum = np.max(window)
            minimum = np.min(window)
            mean = np.mean(window)
            standard_deviation = np.std(window)

            # Frequency Domain Features

            # Time-Frequency Domain

            window_features = [maximum, minimum, mean, standard_deviation]
            x_i_features.extend(window_features)
        x_features.append(x_i_features)
    return np.array(x_features)
