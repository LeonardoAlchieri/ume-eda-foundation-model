"""
Generic dataset class for loading multiple .npy files.
"""

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
from typing import List, Optional, Union, Callable
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class EDAtaNoLabel(Dataset):
    """
    Generic dataset for loading and combining multiple .npy files.

    This dataset is agnostic to the model/engine and simply loads data from
    multiple .npy files, concatenates them, and optionally shuffles.

    Args:
        base_path: Base directory path containing the dataset folders
        dataset_names: List of dataset folder names to load
        shuffle: Whether to shuffle the data after loading
        seed: Random seed for shuffling (for reproducibility)
        channels: Channel indices to select. Can be:
            - None: Use all channels (default)
            - int: Select single channel, output shape will be (T,)
            - List[int]: Select multiple channels, output shape will be (T, C)

    Example:
        >>> dataset = EDAtaNoLabel(
        ...     base_path='/path/to/data',
        ...     dataset_names=['USILaughs', 'BiHeartS', 'APSYNC'],
        ...     shuffle=True,
        ...     channels=0  # Select only first channel
        ... )
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        dataset_names: List[str],
        shuffle: bool = True,
        train_val_test_size: Optional[tuple] = None,
        seed: Optional[int] = None,
        channels: Optional[Union[int, List[int]]] = None,
        sample_scaling_method: Optional[Callable] = None,
    ):
        """Initialize the dataset."""
        self.base_path = Path(base_path)
        self.dataset_names = self._check_dataset_names(dataset_names)
        self.shuffle = shuffle
        self.seed = seed
        self.channels = channels
        self.sample_scaling_method = sample_scaling_method

        # Load all data
        self.data, self.user_ids = self._load_data()

        self.train_val_test_size = train_val_test_size

        # Shuffle if requested
        if self.shuffle:
            self._shuffle_data()

        logger.info(
            f"Loaded dataset with {len(self.data)} samples from {len(self.dataset_names)} dataset(s)"
        )

    def _check_dataset_names(self, dataset_names: List[str]):
        """Check if the provided dataset names exist in the base path."""
        if dataset_names == "all":
            dataset_names = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        for name in dataset_names:
            dataset_path = self.base_path / name
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise ValueError(
                    f"Dataset path does not exist or is not a directory: {dataset_path}"
                )
        return dataset_names

    def get_splits(self):
        """
        Split the dataset into train, validation, and test sets.

        Returns:
            train_set, val_set, test_set: Subsets of the dataset
        """
        if self.train_val_test_size is None:
            raise ValueError(
                "train_val_test_size must be provided for splitting the dataset."
            )

        train_size, val_size, test_size = self.train_val_test_size
        if train_size + val_size + test_size != 1:
            raise ValueError("train_val_test_size must sum to 1.")

        train_set, val_set, test_set = random_split(
            self,
            [train_size, val_size, test_size],
            generator=(
                torch.Generator().manual_seed(self.seed)
                if self.seed is not None
                else None
            ),
        )

        return train_set, val_set, test_set

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from all specified datasets.

        Returns:
            Concatenated numpy array of all loaded data
        """
        all_data = []
        all_user_ids = []

        for dataset_name in self.dataset_names:
            dataset_path = self.base_path / dataset_name

            if not dataset_path.exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue

            # Check if file_pattern is a specific file or a glob pattern
            file_paths = dataset_path.glob("*.npz")

            # Load each file
            for file_path in file_paths:
                if not file_path.exists():
                    logger.warning(f"File does not exist: {file_path}")
                    continue
                try:
                    # Load .npz file
                    npz_data = np.load(file_path)
                    # .npz files are dictionary-like, get the first array
                    # Usually saved with np.savez(filename, array) which stores as 'arr_0'
                    if len(npz_data.files) > 0:
                        data: np.ndarray = npz_data["values"]
                        user_id: np.ndarray = npz_data["groups"]
                        # NOTE: adding the dataset name, since some datasets have overlapping user IDs
                        user_id = np.asarray([f"{dataset_name}_{val}"for val in user_id])
                        all_data.append(data)
                        all_user_ids.append(user_id)
                        logger.info(f"Loaded {data.shape} from {file_path}")
                        logger.info(f"Loaded {user_id.shape} user IDs from {file_path}")
                    else:
                        logger.warning(f"No arrays found in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    raise e

        if len(all_data) == 0:
            raise ValueError(
                f"No data loaded from {self.base_path} with datasets {self.dataset_names}"
            )

        # Concatenate all data along first axis
        concatenated_data = np.concatenate(all_data, axis=0)
        user_ids_data = np.concatenate(all_user_ids, axis=0)
        
        # NOTE: we need to convert the user ids to integers. We use sklearn
        user_ids_data = LabelEncoder().fit_transform(user_ids_data)

        # Apply channel selection if specified
        if self.channels is not None:
            if isinstance(self.channels, int) or isinstance(self.channels, Iterable):
                # Single channel: output shape (N, T)
                concatenated_data = concatenated_data[..., self.channels]
                logger.info(
                    f"Selected channel {self.channels}, new shape: {concatenated_data.shape}"
                )
            else:
                raise ValueError(
                    f"channels must be None, int, or Iterable[int], got {type(self.channels)}"
                )
        else:
            # Use all channels, squeeze if only one channel
            if concatenated_data.shape[-1] == 1:
                concatenated_data = concatenated_data[..., 0]
                logger.info(
                    f"Squeezed single channel, new shape: {concatenated_data.shape}"
                )

        if self.sample_scaling_method is not None:
            concatenated_data = self.sample_scaling_method(concatenated_data)

        logger.info(f"Concatenated data shape: {concatenated_data.shape}")
        
        if concatenated_data.shape[0] != user_ids_data.shape[0]:
            raise ValueError(
                f"Data samples ({concatenated_data.shape[0]}) and user IDs ({user_ids_data.shape[0]}) have different lengths."
            )

        return concatenated_data, user_ids_data
    
    

    def _shuffle_data(self):
        """Shuffle the data in-place."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate random permutation
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.user_ids = self.user_ids[indices]
        logger.info(f"Shuffled data with seed={self.seed}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with 'x' key containing the data sample as a torch tensor
            Shape: (T,) if single channel, (T, C) if multiple channels
        """
        # Load sample and convert to tensor
        sample = self.data[idx]
        sample_user_id = self.user_ids[idx]

        # Convert to torch tensor
        x = torch.from_numpy(sample).float()

        # Return as dictionary (generic format)
        return {"x": x, "user": sample_user_id}

    def reshuffle(self):
        """Reshuffle the dataset (useful between epochs)."""
        self._shuffle_data()

    def get_data_shape(self) -> tuple:
        """Return the shape of a single data sample."""
        return self.data[0].shape

    def get_dataset_info(self) -> dict:
        """Return information about the dataset."""
        return {
            "num_samples": len(self.data),
            "data_shape": self.data.shape,
            "sample_shape": self.get_data_shape(),
            "dataset_names": self.dataset_names,
            "base_path": str(self.base_path),
            "shuffled": self.shuffle,
            "channels": self.channels,
            "unique_user_ids": np.unique(self.user_ids).tolist(),
        }
