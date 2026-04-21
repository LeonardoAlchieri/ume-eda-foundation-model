"""Index mapping tests for multimodal online H5 segmentation dataset."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from edamame.data.dataset_online import DataOnline


def _make_sensor_df(length: int, n_channels: int, offset: float) -> pd.DataFrame:
    data = {
        "Timestamp": pd.date_range("2021-01-01", periods=length, freq="1s"),
    }
    base = np.arange(length, dtype=np.float32) + np.float32(offset)
    for channel_idx in range(n_channels):
        data[f"c{channel_idx}"] = base + np.float32(channel_idx * 0.01)
    return pd.DataFrame(data)


@pytest.fixture
def mapping_h5_base_path():
    temp_dir = Path(tempfile.mkdtemp())
    dataset_dir = temp_dir / "APSYNC"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_path = dataset_dir / "data_unknown.h5"
    sessions = {
        "session_1": {
            "ACC": (25, 3, 0.0),
            "PPG": (60, 4, 100.0),
            "EEG": (90, 8, 200.0),
        },
        "session_2": {
            "ACC": (30, 3, 300.0),
            "PPG": (72, 4, 400.0),
            "EEG": (108, 8, 500.0),
        },
    }

    with pd.HDFStore(file_path, mode="w") as store:
        for session, modality_specs in sessions.items():
            for modality, (length, n_channels, offset) in modality_specs.items():
                key = f"/APSYNC/S001/{session}/{modality}"
                df = _make_sensor_df(length=length, n_channels=n_channels, offset=offset)
                store.put(key, df, format="table")
                store.get_storer(key).attrs.metadata = {
                    "user": "S001",
                    "session": session,
                    "dataset_name": "APSYNC",
                    "len_data": length,
                }

    yield temp_dir
    shutil.rmtree(temp_dir)


def _build_dataset(base_path: Path, cache_max_ram_mb: float = 512.0) -> DataOnline:
    return DataOnline(
        base_path=base_path,
        dataset_names=["APSYNC"],
        modalities=["ACC", "PPG", "EEG"],
        segment_length=5,
        sampling_rates_hz={"ACC": 2.0, "PPG": 4.0, "EEG": 6.0},
        shift={"ACC": 5, "PPG": 10, "EEG": 15},
        channels={"ACC": [0, 1], "PPG": 1, "EEG": [0, 2, 4, 6]},
        shuffle=False,
        seed=42,
        train_val_test_size=(0.5, 0.25, 0.25),
        cache_max_ram_mb=cache_max_ram_mb,
    )


def _num_windows(length: int, segment_length: int, shift: int) -> int:
    if length < segment_length:
        return 0
    return (length - segment_length) // shift + 1


def _expected_windows_for_series(dataset: DataOnline, series_idx: int) -> int:
    entry = dataset.data[series_idx]
    return min(
        _num_windows(
            entry["len_data_per_modality"][modality],
            dataset.segment_length_by_modality[modality],
            dataset.shift_by_modality[modality],
        )
        for modality in dataset.modalities
    )


def test_index_mapping_length_matches_manual_count(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    expected_total = sum(
        _expected_windows_for_series(dataset, series_idx)
        for series_idx in range(len(dataset.data))
    )

    assert expected_total == 9
    assert len(dataset) == expected_total
    assert len(dataset.idx_to_series) == expected_total
    assert len(dataset.idx_to_position) == expected_total


def test_series_to_idx_is_consistent_with_idx_to_series(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    collected_indices: list[int] = []
    for series_idx in range(len(dataset.data)):
        series_indices = dataset.series_to_idx[series_idx]
        expected_windows = _expected_windows_for_series(dataset, series_idx)

        assert len(series_indices) == expected_windows
        if expected_windows > 0:
            assert torch.all(dataset.idx_to_series[series_indices] == series_idx)
        collected_indices.extend(series_indices.tolist())

    assert sorted(collected_indices) == list(range(len(dataset)))


def test_window_positions_are_sequential_per_series(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    for series_idx in range(len(dataset.data)):
        series_indices = dataset.series_to_idx[series_idx]
        if len(series_indices) == 0:
            continue

        positions = dataset.idx_to_position[series_indices].numpy()
        expected = np.arange(len(series_indices), dtype=np.int64)
        np.testing.assert_array_equal(positions, expected)


def test_getitem_extracts_expected_segments(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    series_indices = dataset.series_to_idx[0]
    assert len(series_indices) >= 2

    idx_0 = int(series_indices[0].item())
    idx_1 = int(series_indices[1].item())

    sample_0 = dataset[idx_0]["x"]
    sample_1 = dataset[idx_1]["x"]
    entry = dataset.data[0]

    for modality in dataset.modalities:
        shift = dataset.shift_by_modality[modality]
        segment_length = dataset.segment_length_by_modality[modality]
        selector = dataset.channels_by_modality[modality]

        expected_0 = dataset._read_modality_slice(
            entry=entry,
            modality=modality,
            start=0,
            stop=segment_length,
        )
        expected_1 = dataset._read_modality_slice(
            entry=entry,
            modality=modality,
            start=shift,
            stop=shift + segment_length,
        )

        if selector is not None:
            expected_0 = expected_0[..., selector]
            expected_1 = expected_1[..., selector]

        np.testing.assert_allclose(
            sample_0[modality].numpy(),
            np.asarray(expected_0),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sample_1[modality].numpy(),
            np.asarray(expected_1),
            rtol=1e-6,
            atol=1e-6,
        )


def test_single_channel_selector_returns_1d_segment(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    sample = dataset[0]["x"]
    assert sample["PPG"].shape == (20,)


def test_get_splits_cover_all_indices_without_overlap(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path)

    train_set, val_set, test_set = dataset.get_splits()
    split_indices = list(train_set.indices) + list(val_set.indices) + list(test_set.indices)

    assert len(split_indices) == len(dataset)
    assert len(set(split_indices)) == len(dataset)
    assert set(split_indices) == set(range(len(dataset)))


def test_chunk_cache_respects_ram_budget(mapping_h5_base_path):
    dataset = _build_dataset(mapping_h5_base_path, cache_max_ram_mb=0.01)

    _ = dataset[0]
    _ = dataset[1]
    _ = dataset[2]

    assert dataset._chunk_cache_nbytes <= dataset.cache_max_ram_bytes
