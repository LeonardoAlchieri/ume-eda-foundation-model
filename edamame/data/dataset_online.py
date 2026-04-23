"""Multimodal on-the-fly segmentation dataset for H5 stores.

The dataset expects H5 keys with modality suffixes, for example:
``/APSYNC/S015/2017-10-05 08:59:41+02:00/ACC``.

Data are grouped by ``(dataset, user, session)`` and stored as one entry where
``data`` is a modality metadata dictionary, e.g.
``{"ACC": {"file_path": "...", "key": "/.../ACC"}}``.
Segmentation is done on-the-fly by index mapping, avoiding materializing all
windows in memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Callable, Optional, Union
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

_PerModalityInt = Union[int, Mapping[str, int]]
_ChannelSelector = Optional[Union[int, Iterable[int]]]
_PerModalityChannels = Union[_ChannelSelector, Mapping[str, _ChannelSelector]]
_Seconds = Union[int, float]


class DataOnline(Dataset):
    """Multimodal H5 dataset with on-the-fly segmentation.

    Args:
        base_path: Base path containing one folder per dataset.
        dataset_names: Dataset folder names or "all".
        segment_length: Segment length in seconds.
        sampling_rates_hz: Mapping ``modality -> sampling rate (Hz)`` used to
            convert ``segment_length`` from seconds to samples per modality.
        shift: Sliding window step in samples. Can be an int or
            ``modality -> int`` mapping.
        shuffle: Reserved for API compatibility. Index mapping is deterministic.
        train_val_test_size: Split fractions used by ``get_splits``.
        seed: Random seed for split generator.
        channels: Optional channel selector. Can be:
            - None: keep all channels for every modality.
            - int / iterable of int: apply same selector to every modality.
            - mapping ``modality -> selector`` for modality-specific selection.
        sample_scaling_method: Optional scaling method. Currently supports only
            ``min_max_scaling_per_user``.
        splitting_method: Split function compatible with dataset and lengths.
        modalities: Optional ordered list of required modalities. Sessions that
            do not contain all required modalities are discarded.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        dataset_names: list[str],
        segment_length: _Seconds,
        sampling_rates_hz: Mapping[str, _Seconds],
        shift: _PerModalityInt = 1,
        shuffle: bool = True,
        train_val_test_size: Optional[tuple] = None,
        seed: Optional[int] = None,
        channels: _PerModalityChannels = None,
        sample_scaling_method: Optional[Callable] = None,
        splitting_method: Optional[Callable] = random_split,
        modalities: Optional[list[str]] = None,
    ):
        self.base_path = Path(base_path)
        self.dataset_names = self._check_dataset_names(dataset_names)
        self.shuffle = shuffle
        self.seed = seed
        self.sample_scaling_method = sample_scaling_method
        self.splitting_method = splitting_method
        self.train_val_test_size = train_val_test_size
        self._store_cache: dict[str, pd.HDFStore] = {}

        all_data = self._load_data()
        if len(all_data) == 0:
            raise ValueError("No data loaded. Please check the dataset paths and H5 keys.")

        self.data, self.modalities = self._standardize_modalities(all_data, modalities)
        self.segment_length_seconds = self._validate_segment_length_seconds(segment_length)
        self.sampling_rates_hz = self._resolve_sampling_rates_hz(sampling_rates_hz)
        self.segment_length_by_modality = self._resolve_segment_length_samples()
        self.shift_by_modality = self._resolve_per_modality_int_param(
            name="shift",
            value=shift,
        )
        self.channels_by_modality = self._resolve_channels_per_modality(channels)

        self._validate_segmentation_feasibility()

        if self.sample_scaling_method is not None:
            self._init_sample_scaling()

        self._set_index_mapping()
        self.unique_user_ids = np.unique([entry["user"] for entry in self.data])

        logger.info(
            "Loaded multimodal dataset with %d session-series, %d modalities, %d windows",
            len(self.data),
            len(self.modalities),
            len(self.idx_to_series),
        )

    def _check_dataset_names(self, dataset_names: list[str]) -> list[str]:
        if dataset_names == "all":
            dataset_names = [d.name for d in self.base_path.iterdir() if d.is_dir()]

        for name in dataset_names:
            dataset_path = self.base_path / name
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise ValueError(
                    f"Dataset path does not exist or is not a directory: {dataset_path}"
                )

        return dataset_names

    @staticmethod
    def _parse_h5_key(key: str) -> Optional[tuple[str, str, str, str]]:
        parts = key.strip("/").split("/")
        if len(parts) < 4:
            return None

        dataset_name = parts[0]
        user = parts[1]
        modality = parts[-1]
        session = "/".join(parts[2:-1])

        if session == "" or modality == "":
            return None

        return dataset_name, user, session, modality

    def _load_data(self) -> list[dict]:
        grouped: dict[str, dict] = {}

        for dataset_name in tqdm(self.dataset_names, desc="Loading datasets", position=0):
            dataset_path = self.base_path / dataset_name
            if not dataset_path.exists():
                logger.warning("Dataset path does not exist: %s", dataset_path)
                continue

            file_paths = list(dataset_path.glob("*.h5"))
            if len(file_paths) == 0:
                raise ValueError(f"No .h5 files found for dataset {dataset_name}")

            for file_path in file_paths:
                logger.info("Loading data from %s", file_path)
                with pd.HDFStore(file_path, mode="r") as store:
                    keys = store.keys()
                    for key in tqdm(keys, desc=f"Loading {file_path.name}", position=1):
                        parsed = self._parse_h5_key(key)
                        if parsed is None:
                            logger.warning("Skipping malformed H5 key: %s", key)
                            continue

                        dataset_from_key, user_from_key, session_from_key, modality = parsed
                        group_id = f"{dataset_from_key}/{user_from_key}/{session_from_key}"

                        storer = store.get_storer(key)
                        metadata = getattr(storer.attrs, "metadata", {}) or {}
                        length = getattr(storer, "nrows", None)
                        if length is None:
                            shape = getattr(storer, "shape", None)
                            if isinstance(shape, tuple) and len(shape) > 0:
                                length = shape[0]
                        if length is None:
                            length = int(store.get(key).shape[0])

                        record = grouped.setdefault(
                            group_id,
                            {
                                "user": metadata.get("user", user_from_key),
                                "session": metadata.get("session", session_from_key),
                                "dataset_name": metadata.get("dataset_name", dataset_from_key),
                                "data": {},
                                "len_data_per_modality": {},
                            },
                        )

                        record["data"][modality] = {
                            "file_path": str(file_path),
                            "key": key,
                        }
                        record["len_data_per_modality"][modality] = int(length)

        all_data = list(grouped.values())
        if len(all_data) == 0:
            return []

        encoded_users = LabelEncoder().fit_transform([entry["user"] for entry in all_data])
        for idx, entry in enumerate(all_data):
            entry["user"] = int(encoded_users[idx])
            entry["len_data"] = int(min(entry["len_data_per_modality"].values()))

        return all_data

    @staticmethod
    def _validate_requested_modalities(modalities: Optional[list[str]]) -> Optional[list[str]]:
        if modalities is None:
            return None

        if len(modalities) == 0:
            raise ValueError("'modalities' cannot be an empty list.")

        if any(not isinstance(modality, str) or modality.strip() == "" for modality in modalities):
            raise ValueError("All modality names must be non-empty strings.")

        if len(set(modalities)) != len(modalities):
            raise ValueError(f"Duplicate modality names in 'modalities': {modalities}")

        return modalities

    def _standardize_modalities(
        self,
        all_data: list[dict],
        modalities: Optional[list[str]],
    ) -> tuple[list[dict], list[str]]:
        requested_modalities = self._validate_requested_modalities(modalities)

        if requested_modalities is None:
            requested_modalities = list(all_data[0]["data"].keys())

        standardized: list[dict] = []
        dropped = 0
        for entry in all_data:
            missing = [m for m in requested_modalities if m not in entry["data"]]
            if missing:
                dropped += 1
                continue

            entry["data"] = {m: entry["data"][m] for m in requested_modalities}
            entry["len_data_per_modality"] = {
                m: int(entry["len_data_per_modality"][m]) for m in requested_modalities
            }
            entry["len_data"] = int(min(entry["len_data_per_modality"].values()))
            standardized.append(entry)

        if dropped > 0:
            logger.warning(
                "Dropped %d session-series missing required modalities %s.",
                dropped,
                requested_modalities,
            )

        if len(standardized) == 0:
            raise ValueError(
                "No multimodal session-series left after modality filtering. "
                f"Requested modalities: {requested_modalities}."
            )

        return standardized, requested_modalities

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")

    @staticmethod
    def _validate_segment_length_seconds(segment_length: _Seconds) -> float:
        if not isinstance(segment_length, (int, float)) or float(segment_length) <= 0:
            raise ValueError(
                "segment_length must be a positive number of seconds, "
                f"got {segment_length!r}."
            )
        return float(segment_length)

    def _resolve_sampling_rates_hz(
        self,
        sampling_rates_hz: Mapping[str, _Seconds],
    ) -> dict[str, float]:
        if not isinstance(sampling_rates_hz, Mapping):
            raise TypeError(
                "sampling_rates_hz must be a mapping of modality name to Hz value."
            )

        rate_map = dict(sampling_rates_hz)
        unexpected = [m for m in rate_map if m not in self.modalities]
        if unexpected:
            raise ValueError(
                f"Unexpected modalities in 'sampling_rates_hz': {unexpected}. "
                f"Expected only {self.modalities}."
            )

        missing = [m for m in self.modalities if m not in rate_map]
        if missing:
            raise ValueError(f"Missing modalities in 'sampling_rates_hz': {missing}.")

        resolved = {}
        for modality in self.modalities:
            rate = rate_map[modality]
            if not isinstance(rate, (int, float)) or float(rate) <= 0:
                raise ValueError(
                    "Each sampling rate must be a positive number. "
                    f"Invalid value for modality '{modality}': {rate!r}."
                )
            resolved[modality] = float(rate)
        return resolved

    def _resolve_segment_length_samples(self) -> dict[str, int]:
        resolved = {}
        for modality in self.modalities:
            raw_samples = self.segment_length_seconds * self.sampling_rates_hz[modality]
            samples = int(round(raw_samples))
            if samples < 1:
                raise ValueError(
                    "Computed segment length in samples must be >= 1 for every modality. "
                    f"Modality '{modality}' produced {samples} samples from "
                    f"segment_length={self.segment_length_seconds}s and "
                    f"sampling_rate={self.sampling_rates_hz[modality]}Hz."
                )
            resolved[modality] = samples
        return resolved

    def _resolve_per_modality_int_param(self, name: str, value: _PerModalityInt) -> dict[str, int]:
        if isinstance(value, Mapping):
            param_map = dict(value)
            unexpected = [m for m in param_map if m not in self.modalities]
            if unexpected:
                raise ValueError(
                    f"Unexpected modalities in '{name}': {unexpected}. "
                    f"Expected only {self.modalities}."
                )
            missing = [m for m in self.modalities if m not in param_map]
            if missing:
                raise ValueError(f"Missing modalities in '{name}': {missing}.")

            resolved = {}
            for modality in self.modalities:
                self._require_positive_int(f"{name}[{modality}]", param_map[modality])
                resolved[modality] = int(param_map[modality])
            return resolved

        self._require_positive_int(name, value)
        return {modality: int(value) for modality in self.modalities}

    @staticmethod
    def _normalize_channel_selector(selector: _ChannelSelector, modality: str):
        if selector is None:
            return None

        if isinstance(selector, int):
            return int(selector)

        if isinstance(selector, Iterable) and not isinstance(selector, (str, bytes, dict)):
            normalized = [int(ch) for ch in selector]
            if any(not isinstance(ch, int) for ch in normalized):
                raise ValueError(
                    f"Channel selector for modality '{modality}' must contain integers."
                )
            return normalized

        raise ValueError(
            f"Invalid channel selector for modality '{modality}': {selector!r}."
        )

    def _resolve_channels_per_modality(self, channels: _PerModalityChannels) -> dict[str, _ChannelSelector]:
        if isinstance(channels, Mapping):
            channel_map = dict(channels)
            unexpected = [m for m in channel_map if m not in self.modalities]
            if unexpected:
                raise ValueError(
                    f"Unexpected modalities in 'channels': {unexpected}. "
                    f"Expected only {self.modalities}."
                )
            return {
                modality: self._normalize_channel_selector(channel_map.get(modality), modality)
                for modality in self.modalities
            }

        normalized_selector = self._normalize_channel_selector(channels, "all")
        return {modality: normalized_selector for modality in self.modalities}

    @staticmethod
    def _num_windows(length: int, segment_length: int, shift: int) -> int:
        if length < segment_length:
            return 0
        return (length - segment_length) // shift + 1

    def _validate_segmentation_feasibility(self) -> None:
        feasible = False
        for entry in self.data:
            windows = [
                self._num_windows(
                    entry["len_data_per_modality"][modality],
                    self.segment_length_by_modality[modality],
                    self.shift_by_modality[modality],
                )
                for modality in self.modalities
            ]
            if min(windows) > 0:
                feasible = True
                break

        if not feasible:
            raise ValueError(
                "No valid windows found. Segment/shift settings are incompatible with "
                "available modality lengths."
            )

    def _set_index_mapping(self) -> None:
        sample_counts = np.array(
            [
                min(
                    self._num_windows(
                        entry["len_data_per_modality"][modality],
                        self.segment_length_by_modality[modality],
                        self.shift_by_modality[modality],
                    )
                    for modality in self.modalities
                )
                for entry in self.data
            ],
            dtype=np.int64,
        )

        sample_counts = np.maximum(sample_counts, 0)
        idx_to_series = np.repeat(np.arange(len(self.data), dtype=np.int64), sample_counts)

        if len(idx_to_series) == 0:
            raise ValueError("No segment windows were created from the loaded data.")

        idx_to_position = np.concatenate(
            [np.arange(0, count, dtype=np.int64) for count in sample_counts if count > 0]
        )

        if len(idx_to_series) != len(idx_to_position):
            raise ValueError("Index mapping arrays have mismatched lengths.")

        self.idx_to_series = torch.from_numpy(idx_to_series)
        self.idx_to_position = torch.from_numpy(idx_to_position)
        self.series_to_idx = [
            torch.where(self.idx_to_series == series_idx)[0]
            for series_idx in range(len(self.data))
        ]

    @staticmethod
    def _coerce_hdf_values(data_obj: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        data_df = data_obj.to_frame() if isinstance(data_obj, pd.Series) else data_obj
        if "Timestamp" in data_df.columns:
            data_df = data_df.drop(columns=["Timestamp"])

        values = np.asarray(data_df.values)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        return values

    def _get_store(self, file_path: str) -> pd.HDFStore:
        store = self._store_cache.get(file_path)
        if store is None or not store.is_open:
            store = pd.HDFStore(file_path, mode="r")
            self._store_cache[file_path] = store
        return store

    def _close_store_cache(self) -> None:
        store_cache = getattr(self, "_store_cache", None)
        if not isinstance(store_cache, dict):
            return

        for _, store in store_cache.items():
            try:
                store.close()
            except Exception:
                logger.debug("Failed to close HDFStore handle cleanly.", exc_info=True)
        self._store_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store_cache"] = {}
        return state

    def __del__(self):
        try:
            self._close_store_cache()
        except Exception:
            # Destructors should never raise during interpreter shutdown or
            # partial object initialization.
            logger.debug("Failed to close DataOnline resources in __del__.", exc_info=True)

    def _read_modality_slice(
        self,
        entry: dict,
        modality: str,
        start: int,
        stop: int,
    ) -> np.ndarray:
        ref = entry["data"][modality]
        file_path = ref["file_path"]
        key = ref["key"]
        store = self._get_store(file_path)

        try:
            data_obj = store.select(key, start=start, stop=stop)
            return self._coerce_hdf_values(data_obj)
        except (TypeError, ValueError, NotImplementedError):
            values = self._coerce_hdf_values(store.get(key))
            return values[start:stop]

    def _read_full_modality(self, entry: dict, modality: str) -> np.ndarray:
        length = int(entry["len_data_per_modality"][modality])
        return self._read_modality_slice(entry=entry, modality=modality, start=0, stop=length)

    def _select_channels(self, values: np.ndarray, modality: str) -> np.ndarray:
        selector = self.channels_by_modality[modality]
        if selector is None:
            return values

        try:
            selected = values[..., selector]
        except Exception as exc:
            raise ValueError(
                f"Channel selection failed for modality '{modality}' with selector "
                f"{selector!r} and shape {tuple(values.shape)}."
            ) from exc

        return np.asarray(selected)

    @staticmethod
    def _ensure_2d(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return values[:, np.newaxis]
        if values.ndim == 2:
            return values
        raise ValueError(f"Expected 1D/2D array after channel selection, got {values.shape}.")

    def _init_sample_scaling(self) -> None:
        scaling_name = getattr(getattr(self.sample_scaling_method, "func", None), "__name__", "")
        if scaling_name != "min_max_scaling_per_user":
            raise NotImplementedError(
                "Sample scaling method not implemented for multimodal online dataset. "
                f"Received: {scaling_name}."
            )

        user_maxs: dict[int, dict[str, np.ndarray]] = {}
        user_mins: dict[int, dict[str, np.ndarray]] = {}

        for entry in self.data:
            user_id = int(entry["user"])
            if user_id not in user_maxs:
                user_maxs[user_id] = {}
                user_mins[user_id] = {}

            for modality in self.modalities:
                values = self._read_full_modality(entry, modality)
                values = self._select_channels(values, modality)
                values = self._ensure_2d(values)

                min_val = np.min(values, axis=0)
                max_val = np.max(values, axis=0)

                if modality not in user_maxs[user_id]:
                    user_maxs[user_id][modality] = max_val.copy()
                    user_mins[user_id][modality] = min_val.copy()
                else:
                    user_mins[user_id][modality] = np.minimum(
                        user_mins[user_id][modality], min_val
                    )
                    user_maxs[user_id][modality] = np.maximum(
                        user_maxs[user_id][modality], max_val
                    )

        self.user_maxs = user_maxs
        self.user_mins = user_mins

    def get_splits(self):
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

    def __len__(self) -> int:
        return len(self.idx_to_series)

    def __getitem__(self, idx: int) -> dict:
        series_idx = int(self.idx_to_series[idx].item())
        window_idx = int(self.idx_to_position[idx].item())
        entry = self.data[series_idx]

        sample_user_id = int(entry["user"])
        sample = {}
        for modality in self.modalities:
            start = window_idx * self.shift_by_modality[modality]
            stop = start + self.segment_length_by_modality[modality]
            segment = self._read_modality_slice(
                entry=entry,
                modality=modality,
                start=start,
                stop=stop,
            )

            if segment.shape[0] < self.segment_length_by_modality[modality]:
                raise IndexError(
                    f"Index {idx} out of range for modality '{modality}' with "
                    f"segment_length={self.segment_length_by_modality[modality]}."
                )

            segment = self._select_channels(segment, modality)

            if self.sample_scaling_method is not None:
                denom = self.user_maxs[sample_user_id][modality] - self.user_mins[sample_user_id][modality] + 1e-8
                segment = (segment - self.user_mins[sample_user_id][modality]) / denom

            sample[modality] = torch.from_numpy(np.asarray(segment)).float()

        return {"x": sample, "user": torch.tensor(sample_user_id, dtype=torch.long)}

    def get_data_shape(self) -> dict[str, tuple]:
        sample = self.__getitem__(0)["x"]
        return {modality: tuple(value.shape) for modality, value in sample.items()}

    def get_dataset_info(self) -> dict:
        return {
            "num_session_series": len(self.data),
            "num_samples": len(self),
            "sample_shape": self.get_data_shape(),
            "dataset_names": self.dataset_names,
            "base_path": str(self.base_path),
            "shuffled": self.shuffle,
            "modalities": self.modalities,
            "channels": self.channels_by_modality,
            "segment_length_seconds": self.segment_length_seconds,
            "segment_length": self.segment_length_by_modality,
            "sampling_rates_hz": self.sampling_rates_hz,
            "shift": self.shift_by_modality,
            "unique_user_ids": self.unique_user_ids,
        }
