import numpy as np

def z_norm(data: np.ndarray, *args, **kwargs) -> np.ndarray:
    # shape of data is (N, T, C) or (N, T)
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True) + 1e-8  # avoid division by zero
    return (data - means) / stds

def mean_scaling(data: np.ndarray, *args, **kwargs) -> np.ndarray:
    s = np.mean(abs(data), axis=1, keepdims=True) + 1e-8  # avoid division by zero
    return data / s

def min_max_scaling(data: np.ndarray, *args, **kwargs) -> np.ndarray:
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    ranges = max_vals - min_vals + 1e-8  # avoid division by zero
    return (data - min_vals) / ranges

def min_max_scaling_per_user(data: np.ndarray, user_ids: np.ndarray, *args, **kwargs) -> np.ndarray:
    scaled_data = np.zeros_like(data)
    unique_users = np.unique(user_ids)
    for user in unique_users:
        user_mask = (user_ids == user)
        user_data = data[user_mask]
        min_vals = np.min(user_data, axis=(0,1), keepdims=True)
        max_vals = np.max(user_data, axis=(0,1), keepdims=True)
        ranges = max_vals - min_vals + 1e-8  # avoid division by zero
        scaled_data[user_mask] = (user_data - min_vals) / ranges
    return scaled_data