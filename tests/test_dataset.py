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
from src.data.dataset import EDAtaNoLabel


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
    
    # Create mock .npz files with 'values' key containing (N, T, C) arrays
    # Dataset 1: 100 samples, 240 time points, 1 channel
    data1 = np.random.randn(100, 240, 1).astype(np.float32)
    np.savez(dataset1_dir / "data.npz", values=data1)
    
    # Dataset 2: 150 samples, 240 time points, 1 channel
    data2 = np.random.randn(150, 240, 1).astype(np.float32)
    np.savez(dataset2_dir / "data.npz", values=data2)
    
    # Dataset 3: 80 samples, 240 time points, 1 channel
    data3 = np.random.randn(80, 240, 1).astype(np.float32)
    np.savez(dataset3_dir / "data.npz", values=data3)
    
    # Additional files for testing multiple files per dataset
    np.savez(dataset1_dir / "extra.npz", values=np.random.randn(50, 240, 1).astype(np.float32))
    
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
        assert dataset.data.shape[1] == 240  # T dimension
        assert dataset.data.shape[2] == 1    # C dimension
    
    def test_multiple_datasets(self, temp_data_dir):
        """Test loading and concatenating multiple datasets."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset1", "dataset2", "dataset3"],
            shuffle=False,
        )
        
        # Total: dataset1 (100+50) + dataset2 (150) + dataset3 (80) = 380
        assert len(dataset) == 380
        assert dataset.data.shape[1] == 240  # T dimension
        assert dataset.data.shape[2] == 1    # C dimension
    
    def test_initialization_with_shuffle(self, temp_data_dir):
        """Test dataset initialization with shuffling enabled."""
        dataset = EDAtaNoLabel(
            base_path=temp_data_dir,
            dataset_names=["dataset2"],  # Use dataset2 (no extra files)
            shuffle=True,
            seed=42,
        )
        
        assert len(dataset) == 150
        assert dataset.data.shape[1] == 240  # T dimension
        assert dataset.data.shape[2] == 1    # C dimension
    
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
        
        # Shape should be (T, C) = (240, 1)
        assert sample['x'].shape == (240, 1)
    
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
        
        # Shape should be (T, C) = (240, 1)
        assert shape == (240, 1)
    
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
        assert info['data_shape'] == (230, 240, 1)  # (N, T, C)
        assert info['sample_shape'] == (240, 1)     # (T, C)
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
        # Batch shape should be (batch_size, T, C) = (16, 240, 1)
        assert batch['x'].shape == (16, 240, 1)
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
            assert batch['x'].dim() == 3  # (batch_size, T, C)
            assert batch['x'].shape[1] == 240  # T dimension
            assert batch['x'].shape[2] == 1    # C dimension
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
            train_val_test_size=(100, 30, 20),
            seed=42,
        )
        
        train, val, test = dataset.get_splits()
        
        assert len(train) == 100
        assert len(val) == 30
        assert len(test) == 20
    
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
            train_val_test_size=(100, 30, 30),  # Total = 160, but dataset has 150
            seed=42,
        )
        
        with pytest.raises(ValueError, match="Sum of train, val, and test sizes"):
            dataset.get_splits()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
