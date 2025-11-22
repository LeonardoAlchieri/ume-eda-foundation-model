"""
Generic dataset class for loading multiple .npy files.
"""

from collections.abc import Iterable
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
from typing import List, Optional, Union, Callable
import logging
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EDAtaNoLabelOnlineSegmentation(Dataset):
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
        segment_length: int,
        shift: Optional[int] = 1,
        shuffle: bool = True,
        train_val_test_size: Optional[tuple] = None,
        seed: Optional[int] = None,
        channels: Optional[Union[int, List[int]]] = None,
        sample_scaling_method: Optional[Callable] = None,
        splitting_method: Optional[Callable] = random_split,
    ):
        """Initialize the dataset."""
        self.base_path = Path(base_path)
        self.dataset_names = self._check_dataset_names(dataset_names)
        self.segment_length = segment_length
        self.shift = shift

        self.shuffle = shuffle
        self.seed = seed
        self.channels = channels
        self.sample_scaling_method = sample_scaling_method
        self.splitting_method = splitting_method

        # Load all data
        self.data = self._load_data()
        self._set_index_mapping()

        self.train_val_test_size = train_val_test_size

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

        train_set, val_set, test_set = self.splitting_method(
            dataset=self,
            lengths=[train_size, val_size, test_size],
            generator=(
                torch.Generator().manual_seed(self.seed)
                if self.seed is not None
                else None
            ),
        )

        return train_set, val_set, test_set

    def _set_index_mapping(self) -> None:
        # TODO: create a test to double check that this is working as intended
        # 3. with random split, get everything, reorder and check that everything is working as indented.

        sample_counts = np.array(
            [
                (data["len_data"] - self.segment_length) // self.shift + 1
                for data in self.data
            ]
        )

        # Create arrays for O(1) lookup
        idx_to_series = np.repeat(np.arange(len(self.data)), sample_counts)
        idx_to_position = np.concatenate(
            [np.arange(0, count * self.shift, self.shift) for count in sample_counts]
        )

        if len(idx_to_series) != len(idx_to_position):
            raise ValueError("Index mapping arrays have mismatched lengths.")

        self.idx_to_series = torch.from_numpy(idx_to_series)
        self.idx_to_position = torch.from_numpy(idx_to_position)

        self.series_to_idx = [
            torch.where(self.idx_to_series == i)[0] for i in range(len(self.data))
        ]

    def _load_data(self) -> list:
        """
        Load data from all specified datasets.

        Returns:
            List of data dictionaries and index mapping
        """
        all_data: list = []

        for dataset_name in self.dataset_names:
            dataset_path = self.base_path / dataset_name

            if not dataset_path.exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue

            # Check if file_pattern is a specific file or a glob pattern
            file_paths = dataset_path.glob("*.h5")

            # Load each file
            for file_path in file_paths:
                logger.info(f"Loading data from {file_path}")

                # Read HDF5 file
                with pd.HDFStore(file_path, mode="r") as store:
                    # Get all keys in the store
                    keys = store.keys()

                    for key in tqdm(keys, desc=f"Loading {file_path.name}"):
                        # Load the data
                        data_df = store.get(key)

                        data_df = data_df.drop(columns=["Timestamp"])

                        # Load metadata from attributes
                        metadata = store.get_storer(key).attrs.metadata

                        # Append to all_data
                        all_data.append(
                            {
                                "user": metadata["user"],
                                "session": metadata["session"],
                                "dataset_name": metadata["dataset_name"],
                                "len_data": metadata["len_data"],
                                "data": data_df.values,
                            }
                        )

        user_ids_data = [data["user"] for data in all_data]

        # NOTE: we need to convert the user ids to integers. We use sklearn
        user_ids_data = LabelEncoder().fit_transform(user_ids_data)

        all_data = [
            {
                "user": user_ids_data[i],
                "data": data["data"],
                "len_data": data["len_data"],
            }
            for i, data in enumerate(all_data)
        ]

        if self.sample_scaling_method is not None:
            raise NotImplementedError(
                "Sample scaling method not implemented yet for the new oneline dataset format"
            )
            # concatenated_data = self.sample_scaling_method(concatenated_data)

        return all_data

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.idx_to_series)

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
        # series_idx = self.idx_mapping[idx]['series']
        # idx_in_series = self.idx_mapping[idx]['idx_in_series']

        series_idx = self.idx_to_series[idx]
        idx_in_series = self.idx_to_position[idx]

        series = self.data[series_idx]["data"]

        # NOTE: the shift is already considered in the idx_mapping creation
        segment = series[
            idx_in_series : idx_in_series + self.segment_length, self.channels
        ]
        if len(segment) < self.segment_length:
            raise IndexError(
                f"Index {idx} out of range for segment length {self.segment_length}"
            )
        sample_user_id = torch.tensor(self.data[series_idx]["user"], dtype=torch.long)

        x = torch.from_numpy(segment).float()

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
