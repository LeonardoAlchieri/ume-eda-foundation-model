"""
Tests for EDAtaNoLabelOnlineSegmentation index mapping and splitting.

Uses small synthetic HDF5 stores written to a tmp dir in the same format the
loader reads: one ``.h5`` file per dataset folder, one store key per
recording, each a DataFrame with a ``Timestamp`` column (dropped on load)
plus signal channels, and a ``metadata`` storer attribute with
``user``/``session``/``dataset_name``/``len_data``.

Verifies:
1. The number of windows matches ``(len - segment_length) // shift + 1`` per recording
2. The index mapping maps global indices to the right (recording, offset)
3. Segments are extracted correctly, with the right shift between neighbours
4. ``user_session_splitting`` keeps each recording in exactly one split
5. ``subsample`` keeps the expected fraction of windows
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from edamame.data.dataset_online import EDAtaNoLabelOnlineSegmentation
from edamame.utils.splitting import user_session_splitting

N_CHANNELS = 3

# (dataset_name, user, session, length)
RECORDINGS = [
    ("DatasetA", "u1", "s1", 1000),
    ("DatasetA", "u1", "s2", 730),
    ("DatasetA", "u2", "s1", 600),
    ("DatasetA", "u3", "s1", 100),  # shorter than every tested segment_length -> skipped
    ("DatasetB", "u4", "s1", 900),
    ("DatasetB", "u5", "s1", 481),
    ("DatasetB", "u5", "s2", 1200),
    ("DatasetB", "u6", "s1", 777),
    ("DatasetB", "u7", "s1", 640),
    ("DatasetB", "u8", "s1", 850),
    ("DatasetB", "u9", "s1", 520),
]
# 10 usable recordings -> an 0.8/0.1/0.1 recording-level split is 8/1/1

CONFIGS = [
    {"segment_length": 120, "shift": 60},
    {"segment_length": 240, "shift": 120},
    {"segment_length": 240, "shift": 240},  # no overlap
    {"segment_length": 480, "shift": 120},
]


def _write_h5_datasets(base_path):
    rng = np.random.default_rng(0)
    by_dataset = {}
    for rec in RECORDINGS:
        by_dataset.setdefault(rec[0], []).append(rec)

    for dataset_name, recs in by_dataset.items():
        folder = base_path / dataset_name
        folder.mkdir(parents=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pytables NaturalNameWarning etc.
            with pd.HDFStore(folder / "data.h5", mode="w") as store:
                for i, (_, user, session, length) in enumerate(recs):
                    values = rng.standard_normal((length, N_CHANNELS)).astype(np.float32)
                    df = pd.DataFrame(values, columns=["mixed", "phasic", "tonic"])
                    df.insert(0, "Timestamp", np.arange(length, dtype=np.int64))
                    key = f"rec_{i}"
                    store.put(key, df)
                    store.get_storer(key).attrs.metadata = {
                        "user": user,
                        "session": session,
                        "dataset_name": dataset_name,
                        "len_data": length,
                    }


@pytest.fixture(scope="module")
def base_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("online_h5")
    _write_h5_datasets(path)
    return path


def _make_dataset(base_path, segment_length=240, shift=120, **kwargs):
    kwargs.setdefault("shuffle", False)
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("channels", 0)
    kwargs.setdefault("train_val_test_size", [0.8, 0.1, 0.1])
    kwargs.setdefault("splitting_method", user_session_splitting)
    return EDAtaNoLabelOnlineSegmentation(
        base_path=base_path,
        dataset_names=["DatasetA", "DatasetB"],
        segment_length=segment_length,
        shift=shift,
        **kwargs,
    )


def _expected_windows(dataset):
    return [
        (len(item["data"]) - dataset.segment_length) // dataset.shift + 1
        for item in dataset.data
    ]


@pytest.fixture(params=CONFIGS, ids=lambda c: f"L{c['segment_length']}_S{c['shift']}")
def dataset(request, base_path):
    return _make_dataset(base_path, **request.param)


class TestLoading:
    def test_dataset_loads(self, dataset):
        assert len(dataset.data) > 0

    def test_data_structure(self, dataset):
        for i, item in enumerate(dataset.data):
            assert {"user", "data", "len_data"} <= set(item), f"item {i}"
            assert isinstance(item["data"], np.ndarray)
            # Timestamp column is dropped
            assert item["data"].shape[1] == N_CHANNELS

    def test_short_recordings_skipped(self, dataset):
        for item in dataset.data:
            assert item["len_data"] >= dataset.segment_length
        n_long_enough = sum(r[3] >= dataset.segment_length for r in RECORDINGS)
        assert len(dataset.data) == n_long_enough

    def test_users_label_encoded(self, dataset):
        users = sorted({int(item["user"]) for item in dataset.data})
        assert users == list(range(len(users)))
        assert np.array_equal(dataset.unique_user_ids, np.array(users))


class TestIndexMapping:
    def test_index_mapping_length(self, dataset):
        expected = sum(_expected_windows(dataset))
        assert len(dataset.idx_to_series) == expected
        assert len(dataset) == expected

    def test_index_mapping_consistency(self, dataset):
        assert len(dataset.idx_to_series) == len(dataset.idx_to_position)

    def test_series_to_idx(self, dataset):
        for series_idx, n_windows in enumerate(_expected_windows(dataset)):
            idxs = dataset.series_to_idx[series_idx]
            assert len(idxs) == n_windows
            assert torch.all(dataset.idx_to_series[idxs] == series_idx)

    def test_all_series_covered(self, dataset):
        unique_series = torch.unique(dataset.idx_to_series).numpy()
        assert np.array_equal(unique_series, np.arange(len(dataset.data)))

    def test_sequential_segments_have_shift(self, dataset):
        for series_idx in range(len(dataset.data)):
            positions = dataset.idx_to_position[dataset.series_to_idx[series_idx]]
            assert positions[0].item() == 0
            diffs = torch.diff(positions)
            assert torch.all(diffs == dataset.shift)
            # last window fits inside the recording
            last_end = positions[-1].item() + dataset.segment_length
            assert last_end <= len(dataset.data[series_idx]["data"])

    def test_segment_extraction(self, dataset):
        for idx in list(range(min(5, len(dataset)))) + [len(dataset) - 1]:
            sample = dataset[idx]
            series_idx = dataset.idx_to_series[idx].item()
            start = dataset.idx_to_position[idx].item()
            original = torch.from_numpy(dataset.data[series_idx]["data"]).float()
            expected = original[start : start + dataset.segment_length, dataset.channels]
            assert sample["x"].shape == (dataset.segment_length,)
            assert torch.allclose(sample["x"], expected)
            assert sample["user"].dtype == torch.long
            assert sample["user"].item() == int(dataset.data[series_idx]["user"])

    def test_multichannel_shape(self, base_path):
        ds = _make_dataset(base_path, channels=[0, 1, 2])
        assert ds[0]["x"].shape == (240, 3)
        assert ds.get_data_shape() == (240, 3)


class TestSplitting:
    def test_user_session_split_partitions_indices(self, dataset):
        train_set, val_set, test_set = dataset.get_splits()
        all_idx = torch.cat(
            [torch.as_tensor(s.indices) for s in (train_set, val_set, test_set)]
        ).tolist()
        assert len(all_idx) == len(dataset)
        assert set(all_idx) == set(range(len(dataset)))

    def test_recordings_do_not_leak_across_splits(self, dataset):
        splits = dataset.get_splits()
        series_per_split = [
            set(dataset.idx_to_series[torch.as_tensor(s.indices)].tolist()) for s in splits
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                assert series_per_split[i].isdisjoint(series_per_split[j])

    def test_split_samples_match_dataset(self, dataset):
        for subset in dataset.get_splits():
            for pos in range(min(3, len(subset))):
                orig_idx = int(subset.indices[pos])
                assert torch.equal(subset[pos]["x"], dataset[orig_idx]["x"])

    def test_splits_reproducible_with_seed(self, base_path):
        a = _make_dataset(base_path).get_splits()
        b = _make_dataset(base_path).get_splits()
        for sa, sb in zip(a, b):
            assert torch.equal(torch.as_tensor(sa.indices), torch.as_tensor(sb.indices))

    def test_split_with_zero_recordings_in_a_split(self, base_path):
        ds = _make_dataset(base_path, train_val_test_size=[0.9, 0.05, 0.05])
        train_set, val_set, test_set = ds.get_splits()
        assert len(train_set) + len(val_set) + len(test_set) == len(ds)

    def test_get_splits_requires_sizes(self, base_path):
        ds = _make_dataset(base_path, train_val_test_size=None)
        with pytest.raises(ValueError):
            ds.get_splits()

    def test_get_splits_sizes_must_sum_to_one(self, base_path):
        ds = _make_dataset(base_path, train_val_test_size=[0.5, 0.2, 0.1])
        with pytest.raises(ValueError):
            ds.get_splits()


class TestSubsample:
    def test_subsample_reduces_windows(self, base_path):
        full = _make_dataset(base_path)
        half = _make_dataset(base_path, subsample=0.5, shuffle=True)
        assert len(half) == max(1, int(len(full) * 0.5))
        # every kept (series, position) pair exists in the full mapping
        full_pairs = set(zip(full.idx_to_series.tolist(), full.idx_to_position.tolist()))
        half_pairs = set(zip(half.idx_to_series.tolist(), half.idx_to_position.tolist()))
        assert half_pairs <= full_pairs

    def test_subsample_one_keeps_all(self, base_path):
        assert len(_make_dataset(base_path, subsample=1.0)) == len(_make_dataset(base_path))

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_subsample_raises(self, base_path, bad):
        with pytest.raises(ValueError):
            _make_dataset(base_path, subsample=bad)


def test_missing_dataset_folder_raises(base_path):
    with pytest.raises(ValueError):
        EDAtaNoLabelOnlineSegmentation(
            base_path=base_path, dataset_names=["DoesNotExist"], segment_length=240
        )
