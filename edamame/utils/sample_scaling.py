import numpy as np

def z_norm(data: np.ndarray) -> np.ndarray:
    # shape of data is (N, T, C) or (N, T)
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True) + 1e-8  # avoid division by zero
    return (data - means) / stds

def mean_scaling(data: np.ndarray) -> np.ndarray:
    s = np.mean(abs(data), axis=1, keepdims=True) + 1e-8  # avoid division by zero
    return data / s

def min_max_scaling(data: np.ndarray) -> np.ndarray:
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    ranges = max_vals - min_vals + 1e-8  # avoid division by zero
    return (data - min_vals) / ranges