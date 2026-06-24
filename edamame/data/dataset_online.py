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
        subsample: Optional fraction of available samples to keep (0 < subsample <= 1)
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
        subsample: Optional[float] = None,
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
        self.subsample = subsample
        if self.subsample is not None and not (0 < self.subsample <= 1):
            raise ValueError("subsample must be in the range (0, 1].")

        self.shuffle = shuffle
        self.seed = seed
        self.channels = channels
        self.sample_scaling_method = sample_scaling_method
        self.splitting_method = splitting_method

        # Load all data
        self.data = self._load_data()
        if len(self.data) == 0:
            raise ValueError("No data loaded. Please check the dataset paths.")
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

        if self.subsample is not None:
            num_total = len(idx_to_series)
            num_keep = max(1, int(num_total * self.subsample))
            if num_keep < num_total:
                if self.shuffle:
                    rng = np.random.default_rng(self.seed)
                    keep_idx = rng.choice(num_total, size=num_keep, replace=False)
                else:
                    keep_idx = np.arange(num_keep)
                idx_to_series = idx_to_series[keep_idx]
                idx_to_position = idx_to_position[keep_idx]
                logger.info(
                    "Subsample active: keeping %d/%d samples (%.2f%%), shuffle=%s, seed=%s",
                    num_keep,
                    num_total,
                    100 * num_keep / num_total,
                    self.shuffle,
                    self.seed,
                )
            else:
                logger.info(
                    "Subsample active: keeping all %d samples (subsample=%s).",
                    num_total,
                    self.subsample,
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

        for dataset_name in tqdm(self.dataset_names, desc="Loading datasets", position=0):
            dataset_path = self.base_path / dataset_name

            if not dataset_path.exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue

            # Check if file_pattern is a specific file or a glob pattern
            file_paths = list(dataset_path.glob("*.h5"))
            if len(file_paths) == 0:
                raise ValueError(f"No .h5 files found for dataset {dataset_name}")

            # Load each file
            for file_path in file_paths:
                logger.info(f"Loading data from {file_path}")

                # Read HDF5 file
                with pd.HDFStore(file_path, mode="r") as store:
                    # Get all keys in the store
                    keys = store.keys()

                    for key in tqdm(keys, desc=f"Loading {file_path.name}", position=1):
                        # Load the data
                        data_df = store.get(key)

                        data_df = data_df.drop(columns=["Timestamp"])

                        # Load metadata from attributes
                        metadata = store.get_storer(key).attrs.metadata
                        if metadata["len_data"] < self.segment_length:
                            logger.warning(
                                f"Data length {metadata['len_data']} is less than segment length {self.segment_length}. Skipping."
                            )
                            continue
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
        self.unique_user_ids = np.unique(user_ids_data)

        all_data = [
            {
                "user": user_ids_data[i],
                "data": data["data"],
                "len_data": data["len_data"],
            }
            for i, data in enumerate(all_data)
        ]

        if self.sample_scaling_method is not None:
            if self.sample_scaling_method.func.__name__ == "min_max_scaling_per_user":
                maxs: dict[int, np.ndarray] = {}
                mins: dict[int, np.ndarray] = {}
                for data in all_data:
                    user_id = data["user"]
                    
                    values = data["data"][..., self.channels]
                    
                    # Ensure values is 2D (time, channels) for consistent handling
                    if values.ndim == 1:
                        values = values[:, np.newaxis]
                    
                    # Compute min/max per channel (axis=0 reduces over time)
                    min_val = np.min(values, axis=0)  # shape: (num_channels,)
                    max_val = np.max(values, axis=0)  # shape: (num_channels,)
                    
                    if user_id not in maxs:
                        maxs[user_id] = max_val.copy()
                        mins[user_id] = min_val.copy()
                    else:
                        # Element-wise min/max across channels
                        mins[user_id] = np.minimum(mins[user_id], min_val)
                        maxs[user_id] = np.maximum(maxs[user_id], max_val)
                        
                self.user_maxs = maxs
                self.user_mins = mins
                    
            else:
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
        sample_user_id = self.data[series_idx]["user"]
        
        if self.sample_scaling_method is not None:
            if self.sample_scaling_method.func.__name__ == "min_max_scaling_per_user":
                segment = (segment - self.user_mins[sample_user_id]) / (self.user_maxs[sample_user_id] - self.user_mins[sample_user_id] + 1e-8)
            else:
                raise NotImplementedError(
                    "Sample scaling method not implemented yet for the new oneline dataset format"
                )
        
        sample_user_id = torch.tensor(sample_user_id, dtype=torch.long)

        x = torch.from_numpy(segment).float()

        return {"x": x, "user": sample_user_id}

    # def reshuffle(self):
    #     """Reshuffle the dataset (useful between epochs)."""
    #     self._shuffle_data()

    def get_data_shape(self) -> tuple:
        """Return the shape of a single data sample."""
        return self.__getitem__(0)["x"].shape

    def get_dataset_info(self) -> dict:
        """Return information about the dataset."""
        return {
            "num_samples": len(self.data),
            "data_shape": self.get_data_shape(),
            "dataset_names": self.dataset_names,
            "base_path": str(self.base_path),
            "shuffled": self.shuffle,
            "channels": self.channels,
            "unique_user_ids": self.unique_user_ids,
        }


class EDAtaTNCOnlineSegmentation(EDAtaNoLabelOnlineSegmentation):
    """Online-segmentation dataset for Temporal Neighborhood Coding (TNC).

    TNC (Tonekaboni et al., 2021, https://arxiv.org/abs/2106.00750) is a
    *non-augmentation* contrastive objective: instead of building two augmented
    views of a window, it contrasts windows by their **temporal proximity inside
    the same recording**. For a reference window starting at offset ``t`` we draw

    - **neighbors** (positives): windows whose start offset lies within
      ``±neighbor_range`` samples of ``t`` (a fixed-distance neighborhood);
    - **non-neighbors** (negatives): windows from the *same* recording whose
      start offset is farther than ``non_neighbor_gap`` samples from ``t``.

    Cross-recording (easy) negatives are *not* produced here — they are formed
    on the fly from other samples in the batch by the TNC loss. This class only
    supplies the same-recording structure that batching cannot guarantee.

    ``__getitem__`` returns, in addition to the reference window ``x`` and
    ``user`` of :class:`EDAtaNoLabelOnlineSegmentation`::

        {
            "x":                FloatTensor (T, C)        # reference window
            "x_neighbors":      FloatTensor (M, T, C)     # M positives
            "x_non_neighbors":  FloatTensor (M, T, C)     # M same-recording negatives
            "non_neighbor_mask":FloatTensor (M,)          # 1 where the negative is valid
            "user":             LongTensor  ()
        }

    Recordings too short to supply same-recording non-neighbors get an all-zero
    ``non_neighbor_mask`` (the placeholder windows are ignored by the loss, which
    falls back to the cross-recording negatives for those samples).

    Args:
        mc_sample_size: Number ``M`` of neighbor / non-neighbor windows sampled
            per reference (Monte-Carlo sample size, as in the original TNC).
        neighbor_range: Half-width ``η`` (in samples) of the temporal
            neighborhood; neighbors are sampled from ``[t-η, t+η]``.
        non_neighbor_gap: Minimum offset distance ``δ`` (in samples) for a window
            to count as a non-neighbor. Should be ``>= neighbor_range``; set it to
            a multiple of ``segment_length`` to make negatives non-overlapping with
            the reference.
    """

    def __init__(
        self,
        *args,
        mc_sample_size: int = 5,
        neighbor_range: int = 240,
        non_neighbor_gap: int = 480,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mc_sample_size < 1:
            raise ValueError("mc_sample_size must be >= 1.")
        if neighbor_range < 0 or non_neighbor_gap < 0:
            raise ValueError("neighbor_range and non_neighbor_gap must be >= 0.")
        if non_neighbor_gap < neighbor_range:
            logger.warning(
                "non_neighbor_gap (%d) < neighbor_range (%d): neighbor and "
                "non-neighbor regions overlap.",
                non_neighbor_gap,
                neighbor_range,
            )
        self.mc_sample_size = mc_sample_size
        self.neighbor_range = neighbor_range
        self.non_neighbor_gap = non_neighbor_gap

    def _window(self, series: np.ndarray, user_id: int, offset: int) -> np.ndarray:
        """Extract and (optionally) scale a single window at ``offset``."""
        segment = series[offset : offset + self.segment_length, self.channels]
        if self.sample_scaling_method is not None:
            if self.sample_scaling_method.func.__name__ == "min_max_scaling_per_user":
                segment = (segment - self.user_mins[user_id]) / (
                    self.user_maxs[user_id] - self.user_mins[user_id] + 1e-8
                )
            else:
                raise NotImplementedError(
                    "Sample scaling method not implemented yet for the new online dataset format"
                )
        return segment

    @staticmethod
    def _sample_within(center: int, radius: int, max_start: int, n: int) -> torch.Tensor:
        """Sample ``n`` start offsets in ``[center-radius, center+radius]`` (clamped)."""
        low = max(0, center - radius)
        high = min(max_start, center + radius)  # inclusive
        return torch.randint(low, high + 1, (n,))

    @staticmethod
    def _sample_outside(
        center: int, gap: int, max_start: int, n: int
    ) -> tuple[torch.Tensor, bool]:
        """Sample ``n`` start offsets at distance > ``gap`` from ``center``.

        Returns the offsets and a validity flag. When the recording cannot supply
        any far-enough offset the flag is ``False`` and the offsets are placeholder
        zeros (to be masked out by the loss).
        """
        left_high = center - gap  # valid left offsets: [0, left_high]
        right_low = center + gap  # valid right offsets: [right_low, max_start]
        left_size = left_high + 1 if left_high >= 0 else 0
        right_size = (max_start - right_low + 1) if right_low <= max_start else 0
        total = left_size + right_size
        if total <= 0:
            return torch.zeros(n, dtype=torch.long), False
        r = torch.randint(0, total, (n,))
        offsets = torch.where(r < left_size, r, right_low + (r - left_size))
        return offsets, True

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)

        series_idx = int(self.idx_to_series[idx])
        ref_offset = int(self.idx_to_position[idx])
        series = self.data[series_idx]["data"]
        user_id = self.data[series_idx]["user"]
        max_start = series.shape[0] - self.segment_length  # inclusive max offset
        m = self.mc_sample_size

        neighbor_offsets = self._sample_within(
            ref_offset, self.neighbor_range, max_start, m
        )
        non_neighbor_offsets, valid = self._sample_outside(
            ref_offset, self.non_neighbor_gap, max_start, m
        )

        neighbors = np.stack(
            [self._window(series, user_id, int(o)) for o in neighbor_offsets]
        )
        non_neighbors = np.stack(
            [self._window(series, user_id, int(o)) for o in non_neighbor_offsets]
        )

        sample["x_neighbors"] = torch.from_numpy(neighbors).float()
        sample["x_non_neighbors"] = torch.from_numpy(non_neighbors).float()
        sample["non_neighbor_mask"] = torch.full(
            (m,), float(valid), dtype=torch.float
        )
        return sample
