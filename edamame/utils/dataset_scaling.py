import torch
import numpy as np
from torch.utils.data import Subset, Dataset


def get_empty_scaling_params(dataset: Dataset) -> None:
    """
    Return default scaling parameters (no scaling).

    Args:
        dataset: PyTorch Dataset or Subset (not used here)

    Returns:
        Dictionary with keys 'mean' and 'std' set to 0.0 and 1.0 respectively
    """
    return None


def get_standard_scaling_params(dataset: Dataset) -> dict[str, float]:
    """
    Compute mean and standard deviation from a dataset.

    Args:
        dataset: PyTorch Dataset or Subset where each item is a dict with 'x' key
                 and x has shape (T,) for single channel or (T, C) for multiple channels

    Returns:
        Dictionary with keys 'mean' and 'std'

    Example:
        >>> params = get_scaling_params(train_set)
        >>> print(params)
        {'mean': 0.5, 'std': 1.2}
    """
    all_data = []

    # Collect all samples
    for i in range(len(dataset)):
        sample = dataset[i]
        x = sample["x"]

        # Convert to numpy if tensor
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        all_data.append(x.flatten())  # Flatten to 1D

    # Concatenate all data
    all_data = np.concatenate(all_data)

    # Compute statistics
    mean = float(np.mean(all_data))
    std = float(np.std(all_data))

    return {"mean": mean, "std": std}


def get_mean_scaling_params(dataset: Dataset) -> dict[str, float]:
    """
    Compute mean from a dataset for mean-centering (without scaling).

    Args:
        dataset: PyTorch Dataset or Subset where each item is a dict with 'x' key
                 and x has shape (T,) for single channel or (T, C) for multiple channels

    Returns:
        Dictionary with keys 'mean' (computed from data) and 'std' (fixed at 1.0)
        This allows mean-centering without standard deviation scaling.

    Example:
        >>> params = get_mean_scaling_params(train_set)
        >>> print(params)
        {'mean': 0.5, 'std': 1.0}
    """
    res = get_standard_scaling_params(dataset)
    res["std"] = 1.0  # Override std to 1.0

    return res
