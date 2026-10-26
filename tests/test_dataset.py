"""
Test suite for EDAtaNoLabel dataset class.

This module contains comprehensive tests for the EDAtaNoLabel dataset,
including data loading, concatenation, shuffling, and error handling.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from ume.data.dataset import EDAtaNoLabel


def _groups(n, n_users=5):
    """Per-segment user IDs, cycling through ``n_users`` users."""
    return np.array([f"user{i % n_users}" for i in range(n)])


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock data for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create dataset directories
    dataset1_dir = Path(temp_dir) / "dataset1"
    dataset2_dir = Path(temp_dir) / "dataset2"
    dataset3_dir = Path(temp_dir) / "dataset3"
    
    dataset1_dir.mkdir()
    dataset2_dir.mkdir()
    dataset3_dir.mkdir()
    
    # Create mock .npz files in the format EDAtaNoLabel reads:
    #   'values': (N, T, C) float array of segments
    #   'groups': (N,) per-segment user IDs (prefixed with the dataset name on load)
    # Dataset 1: 100 samples, 240 time points, 1 channel
    data1 = np.random.randn(100, 240, 1).astype(np.float32)
    np.savez(dataset1_dir / "data.npz", values=data1, groups=_groups(100))
    
    # Dataset 2: 150 samples, 240 time points, 1 channel
    data2 = np.random.randn(150, 240, 1).astype(np.float32)
    np.savez(dataset2_dir / "data.npz", values=data2, groups=_groups(150))
    
    # Dataset 3: 80 samples, 240 time points, 1 channel
    data3 = np.random.randn(80, 240, 1).astype(np.float32)
    np.savez(dataset3_dir / "data.npz", values=data3, groups=_groups(80))
    
    # Additional files for testing multiple files per dataset
    np.savez(
        dataset1_dir / "extra.npz",
        values=np.random.randn(50, 240, 1).astype(np.float32),
        groups=_groups(50),
    )
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def empty_data_dir():
    """Create a temporary directory with no data files."""
    temp_dir = tempfile.mkdtemp()
    dataset_dir = Path(temp_dir) / "empty_dataset"
    dataset_dir.mkdir()
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


class TestEDAtaNoLabelInitialization:
    """Test dataset initialization."""
    
    def test_basic_initialization(self, temp_data_dir):
        """Test basic dataset initialization with single dataset."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset1"],
            shuffle=False,
        )
        
        # Should load data.npz and extra.npz = 100 + 50 = 150 samples
        assert len(dataset) == 150
        # With channels=None a trailing singleton channel is squeezed: (N, T)
        assert dataset.data.shape == (150, 240)
    
    def test_multiple_datasets(self, temp_data_dir):
        """Test loading and concatenating multiple datasets."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset1", "dataset2", "dataset3"],
            shuffle=False,
        )
        
        # Total: dataset1 (100+50) + dataset2 (150) + dataset3 (80) = 380
        assert len(dataset) == 380
        assert dataset.data.shape == (380, 240)
    
    def test_initialization_with_shuffle(self, temp_data_dir):
        """Test dataset initialization with shuffling enabled."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],  # Use dataset2 (no extra files)
            shuffle=True,
            seed=42,
        )
        
        assert len(dataset) == 150
        assert dataset.data.shape == (150, 240)
    
    def test_initialization_with_seed(self, temp_data_dir):
        """Test that same seed produces same shuffle order."""
        dataset1 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=True,
            seed=42,
        )
        
        dataset2 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=True,
            seed=42,
        )
        
        # Same seed should produce same order
        np.testing.assert_array_equal(dataset1.data, dataset2.data)
    
    def test_different_seeds_produce_different_shuffles(self, temp_data_dir):
        """Test that different seeds produce different shuffle orders."""
        dataset1 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=True,
            seed=42,
        )
        
        dataset2 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=True,
            seed=123,
        )
        
        # Different seeds should produce different orders
        assert not np.array_equal(dataset1.data, dataset2.data)


class TestEDAtaNoLabelDataAccess:
    """Test data access methods."""
    
    def test_getitem_returns_dict(self, temp_data_dir):
        """Test that __getitem__ returns dictionary with 'x' key."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        sample = dataset[0]
        
        assert isinstance(sample, dict)
        assert 'x' in sample
        assert isinstance(sample['x'], torch.Tensor)
    
    def test_getitem_returns_float_tensor(self, temp_data_dir):
        """Test that returned tensor is float type."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        sample = dataset[0]
        
        assert sample['x'].dtype == torch.float32
    
    def test_getitem_shape(self, temp_data_dir):
        """Test that sample has correct shape."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        sample = dataset[0]
        
        # Single-channel data is squeezed: shape (T,)
        assert sample['x'].shape == (240,)
    
    def test_getitem_multiple_indices(self, temp_data_dir):
        """Test accessing multiple different indices."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        sample1 = dataset[0]
        sample2 = dataset[50]
        sample3 = dataset[99]
        
        assert all(isinstance(s['x'], torch.Tensor) for s in [sample1, sample2, sample3])
        # Samples should be different
        assert not torch.allclose(sample1['x'], sample2['x'])
    
    def test_len(self, temp_data_dir):
        """Test __len__ method."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2", "dataset3"],
            shuffle=False,
        )
        
        assert len(dataset) == 230  # 150 + 80


class TestEDAtaNoLabelChannelsAndUsers:
    """Test channel selection and user-ID handling."""

    @pytest.fixture
    def multichannel_dir(self, tmp_path):
        folder = tmp_path / "mc"
        folder.mkdir()
        values = np.random.randn(20, 240, 3).astype(np.float32)
        np.savez(folder / "data.npz", values=values, groups=_groups(20, n_users=4))
        return tmp_path, values

    def test_all_channels_kept(self, multichannel_dir):
        base, values = multichannel_dir
        ds = EDAtaNoLabel(base_path=base, dataset_names=["mc"], shuffle=False)
        assert ds[0]['x'].shape == (240, 3)
        np.testing.assert_allclose(ds[0]['x'].numpy(), values[0])

    def test_single_channel_selection(self, multichannel_dir):
        base, values = multichannel_dir
        ds = EDAtaNoLabel(base_path=base, dataset_names=["mc"], shuffle=False, channels=1)
        assert ds[0]['x'].shape == (240,)
        np.testing.assert_allclose(ds[0]['x'].numpy(), values[0, :, 1])

    def test_multi_channel_selection(self, multichannel_dir):
        base, values = multichannel_dir
        ds = EDAtaNoLabel(
            base_path=base, dataset_names=["mc"], shuffle=False, channels=[0, 2]
        )
        assert ds[0]['x'].shape == (240, 2)
        np.testing.assert_allclose(ds[0]['x'].numpy(), values[0][:, [0, 2]])

    def test_users_encoded_per_dataset(self, temp_data_dir):
        """Same raw group IDs in different datasets map to different users."""
        ds = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2", "dataset3"],
            shuffle=False,
        )
        users2 = set(ds.user_ids[:150].tolist())
        users3 = set(ds.user_ids[150:].tolist())
        assert len(users2) == 5 and len(users3) == 5
        assert users2.isdisjoint(users3)
        assert 'user' in ds[0]

    def test_shuffle_keeps_users_aligned(self, multichannel_dir):
        base, _ = multichannel_dir
        plain = EDAtaNoLabel(base_path=base, dataset_names=["mc"], shuffle=False)
        shuffled = EDAtaNoLabel(base_path=base, dataset_names=["mc"], shuffle=True, seed=1)
        lookup = {plain.data[i].tobytes(): plain.user_ids[i] for i in range(len(plain))}
        for i in range(len(shuffled)):
            assert lookup[shuffled.data[i].tobytes()] == shuffled.user_ids[i]


class TestEDAtaNoLabelUtilityMethods:
    """Test utility methods."""
    
    def test_get_data_shape(self, temp_data_dir):
        """Test get_data_shape method."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        shape = dataset.get_data_shape()
        
        # Single-channel data is squeezed: shape (T,)
        assert shape == (240,)
    
    def test_get_dataset_info(self, temp_data_dir):
        """Test get_dataset_info method."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2", "dataset3"],
            shuffle=True,
            seed=42,
        )
        
        info = dataset.get_dataset_info()
        
        assert info['num_samples'] == 230
        assert info['data_shape'] == (230, 240)  # (N, T), channel squeezed
        assert info['sample_shape'] == (240,)     # (T,)
        # 5 users per dataset, prefixed with the dataset name -> 10 unique
        assert info['unique_user_ids'] == list(range(10))
        assert info['dataset_names'] == ["dataset2", "dataset3"]
        assert info['shuffled'] == True
        assert 'base_path' in info
    
    def test_reshuffle(self, temp_data_dir):
        """Test reshuffle method."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
            seed=42,
        )
        
        # Get initial data
        data_before = dataset.data.copy()
        
        # Reshuffle
        dataset.reshuffle()
        
        # Data should be different after reshuffling
        assert not np.array_equal(data_before, dataset.data)
        # But should have same shape
        assert data_before.shape == dataset.data.shape


class TestEDAtaNoLabelErrorHandling:
    """Test error handling and edge cases."""
    
    def test_nonexistent_base_path(self):
        """Test error when base path doesn't exist."""
        with pytest.raises(ValueError, match="Dataset path does not exist or is not a directory"):
            EDAtaNoLabel(
                base_path="/nonexistent/path",
                dataset_names=["dataset1"],
                shuffle=False,
            )
    
    def test_nonexistent_dataset(self, temp_data_dir):
        """Test error when dataset folder doesn't exist."""
        # With the current implementation, this raises an error immediately
        with pytest.raises(ValueError, match="Dataset path does not exist or is not a directory"):
            EDAtaNoLabel(
                base_path=temp_data_dir,
                dataset_names=["dataset2", "nonexistent_dataset"],
                shuffle=False,
            )
    
    def test_empty_dataset_list(self, empty_data_dir):
        """Test error when no datasets have data."""
        with pytest.raises(ValueError, match="No data loaded"):
            EDAtaNoLabel(
                base_path=empty_data_dir,
                dataset_names=["empty_dataset"],
                shuffle=False,
            )
    
    def test_out_of_bounds_index(self, temp_data_dir):
        """Test error when accessing out of bounds index."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        with pytest.raises(IndexError):
            _ = dataset[1000]  # Only 150 samples


class TestEDAtaNoLabelConcatenation:
    """Test data concatenation from multiple sources."""
    
    def test_concatenation_order(self, temp_data_dir):
        """Test that datasets are concatenated in order."""
        # Load each dataset individually
        dataset2 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        dataset3 = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset3"],
            shuffle=False,
        )
        
        # Load combined dataset
        combined = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2", "dataset3"],
            shuffle=False,
        )
        
        # First 150 samples should match dataset2
        np.testing.assert_array_equal(
            combined.data[:150],
            dataset2.data
        )
        
        # Next 80 samples should match dataset3
        np.testing.assert_array_equal(
            combined.data[150:230],
            dataset3.data
        )


class TestEDAtaNoLabelWithDataLoader:
    """Test dataset integration with PyTorch DataLoader."""
    
    def test_dataloader_integration(self, temp_data_dir):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Get first batch
        batch = next(iter(loader))
        
        assert 'x' in batch
        # Batch shape should be (batch_size, T) = (16, 240)
        assert batch['x'].shape == (16, 240)
        assert batch['user'].shape == (16,)
        assert batch['x'].dtype == torch.float32
    
    def test_dataloader_full_iteration(self, temp_data_dir):
        """Test iterating through full dataloader."""
        from torch.utils.data import DataLoader
        
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        total_samples = 0
        for batch in loader:
            assert 'x' in batch
            assert batch['x'].dim() == 2  # (batch_size, T)
            assert batch['x'].shape[1] == 240  # T dimension
            total_samples += batch['x'].shape[0]
        
        assert total_samples == 150


class TestEDAtaNoLabelSplits:
    """Test train/val/test splitting functionality."""
    
    def test_get_splits_basic(self, temp_data_dir):
        """Test basic train/val/test splitting."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
            train_val_test_size=(0.6, 0.2, 0.2),
            seed=42,
        )
        
        train, val, test = dataset.get_splits()
        
        # Fractions of 150 samples
        assert len(train) == 90
        assert len(val) == 30
        assert len(test) == 30
    
    def test_get_splits_without_size_raises_error(self, temp_data_dir):
        """Test that get_splits raises error when train_val_test_size not provided."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
        )
        
        with pytest.raises(ValueError, match="train_val_test_size must be provided"):
            dataset.get_splits()
    
    def test_get_splits_wrong_size_raises_error(self, temp_data_dir):
        """Test that get_splits raises error when sizes don't match total."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
            train_val_test_size=(0.6, 0.2, 0.3),  # Fractions sum to 1.1
            seed=42,
        )
        
        with pytest.raises(ValueError, match="train_val_test_size must sum to 1"):
            dataset.get_splits()

    def test_get_splits_float_rounding(self, temp_data_dir):
        """Fractions that sum to 1 up to float rounding should be accepted."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],
            shuffle=False,
            train_val_test_size=(0.7, 0.2, 0.1),
            seed=42,
        )
        train, val, test = dataset.get_splits()
        assert len(train) + len(val) + len(test) == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
