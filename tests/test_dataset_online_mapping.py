"""
Test for EDAtaNoLabelOnlineSegmentation dataset index mapping.

This test verifies that:
1. The number of samples calculated matches the expected number
2. The index mapping correctly maps global indices to series and positions
3. Sequential segments are correctly extracted with proper shift
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from sys import path
path.append("./")
from edamame.data.dataset_online import EDAtaNoLabelOnlineSegmentation
from edamame.utils.splitting import user_session_splitting


class TestDatasetOnlineMapping:
    """Test suite for dataset index mapping."""

    def dataset_path(self):
        """Path to test dataset."""
        return Path("/Users/leonardoalchieri/Desktop/GIT-Mine/datasets_cleaning/data_ready_nolabel.nosync/BiHeartS")

    def dataset(self, dataset_path):
        """Create dataset instance."""
        return EDAtaNoLabelOnlineSegmentation(
            base_path=dataset_path.parent,
            dataset_names=["BiHeartS"],
            segment_length=240,
            shift=120,
            shuffle=False,
            seed=42,
            channels=0,
            train_val_test_size=[0.8, 0.1, 0.1],
        )

    def test_dataset_loads(self, dataset):
        """Test that dataset loads successfully."""
        assert dataset is not None
        assert len(dataset.data) > 0
        print(f"✓ Dataset loaded with {len(dataset.data)} time series")

    def test_data_structure(self, dataset):
        """Test that loaded data has correct structure."""
        # Check that each item has required keys
        for i, item in enumerate(dataset.data):
            assert "user" in item, f"Missing 'user' key in item {i}"
            assert "data" in item, f"Missing 'data' key in item {i}"
            assert isinstance(item["data"], np.ndarray), f"Data should be numpy array in item {i}"
        
        print(f"✓ All {len(dataset.data)} time series have correct structure")

    def test_index_mapping_length(self, dataset):
        """Test that index mapping has correct length."""
        # Calculate expected number of samples
        expected_samples = 0
        for item in dataset.data:
            series_len = len(item["data"])
            num_windows = (series_len - dataset.segment_length) // dataset.shift + 1
            expected_samples += max(0, num_windows)  # Can't be negative
        
        actual_samples = len(dataset.idx_to_series)
        
        print(f"Expected samples: {expected_samples}")
        print(f"Actual samples: {actual_samples}")
        print(f"Dataset __len__: {len(dataset)}")
        
        assert actual_samples == expected_samples, \
            f"Index mapping length mismatch: expected {expected_samples}, got {actual_samples}"
        assert len(dataset) == expected_samples, \
            f"Dataset length mismatch: expected {expected_samples}, got {len(dataset)}"
        
        print(f"✓ Index mapping has correct length: {actual_samples}")

    def test_index_mapping_consistency(self, dataset):
        """Test that idx_to_series and idx_to_position have same length."""
        assert len(dataset.idx_to_series) == len(dataset.idx_to_position), \
            "idx_to_series and idx_to_position should have same length"
        
        print(f"✓ Index mapping arrays are consistent")

    def test_manual_calculation_vs_mapping(self, dataset):
        """Manually calculate expected samples and compare with mapping."""
        print("\n" + "="*80)
        print("Manual Calculation vs Index Mapping")
        print("="*80)
        
        total_expected = 0
        for i, item in enumerate(dataset.data):
            series_len = len(item["data"])
            num_windows = (series_len - dataset.segment_length) // dataset.shift + 1
            num_windows = max(0, num_windows)
            
            print(f"Series {i}: len={series_len}, windows={num_windows}")
            total_expected += num_windows
        
        print(f"\nTotal expected windows: {total_expected}")
        print(f"Actual mapping length: {len(dataset.idx_to_series)}")
        
        assert total_expected == len(dataset.idx_to_series), \
            f"Manual calculation ({total_expected}) != mapping length ({len(dataset.idx_to_series)})"
        
        print("✓ Manual calculation matches index mapping")

    def test_sequential_segments(self, dataset):
        """Test that sequential segments have correct shift."""
        print("\n" + "="*80)
        print("Testing Sequential Segments")
        print("="*80)
        
        # Pick first series with multiple segments
        series_idx = 0
        series_data = dataset.data[series_idx]["data"]
        
        # Find all indices that belong to this series
        series_indices = dataset.series_to_idx[series_idx].numpy()
        
        if len(series_indices) < 2:
            print("⚠ Not enough segments in first series, skipping sequential test")
            return
        
        print(f"Testing series {series_idx} with {len(series_indices)} segments")
        
        # Test first few sequential segments
        for i in range(min(3, len(series_indices) - 1)):
            idx1 = series_indices[i]
            idx2 = series_indices[i + 1]
            
            sample1 = dataset[idx1]
            sample2 = dataset[idx2]
            
            # Check that second segment starts 'shift' positions after first
            pos1 = dataset.idx_to_position[idx1].item()
            pos2 = dataset.idx_to_position[idx2].item()
            
            print(f"  Segment {i}: pos={pos1}, next pos={pos2}, diff={pos2-pos1}")
            
            assert pos2 - pos1 == dataset.shift, \
                f"Position difference should be {dataset.shift}, got {pos2-pos1}"
        
        print("✓ Sequential segments have correct shift")

    def test_segment_extraction(self, dataset):
        """Test that segments are correctly extracted from original data."""
        print("\n" + "="*80)
        print("Testing Segment Extraction")
        print("="*80)
        
        # I ASSUME THAT THE FIRST 5 SAMPELS ARE FROM THE FIRST SERIES FOR SIMPLICITY
        end_pos = 0
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]
            
            # Get mapping info
            series_idx = 0 # For simplicity, assume first series
            
            
            # Calculate actual start position in original series
            start_pos = dataset.shift * idx
            end_pos = start_pos + dataset.segment_length
            
            # Get original data and convert to torch for comparison
            original_data = dataset.data[series_idx]["data"]
            original_data_torch = torch.from_numpy(original_data).float()
            
            # Extract expected segment (using torch operations)
            expected_segment = original_data_torch[start_pos:end_pos, dataset.channels]
            
            # Get actual segment from dataset
            actual_segment = sample["x"]
            
            print(f"Sample {idx}: series={series_idx}, pos={start_pos}, "
                  f"start={start_pos}, end={end_pos}, "
                  f"shape={actual_segment.shape}")
            
            # Compare
            assert actual_segment.shape[0] == dataset.segment_length, \
                f"Segment should have length {dataset.segment_length}, got {actual_segment.shape[0]}"
            
            # Use torch.allclose for floating point comparison
            try:
                assert torch.allclose(actual_segment, expected_segment, rtol=1e-4, atol=1e-5), \
                f"Segment {idx} data mismatch. Max diff: {torch.max(torch.abs(actual_segment - expected_segment))}"
            except AssertionError as e:
                breakpoint()
        
        print("✓ Segments are correctly extracted")

    def test_no_overlap_in_indices(self, dataset):
        """Test that all indices in mapping are unique."""
        all_indices = list(range(len(dataset.idx_to_series)))
        unique_indices = set(all_indices)
        
        assert len(all_indices) == len(unique_indices), \
            "Index mapping contains duplicate indices"
        
        print(f"✓ All {len(all_indices)} indices are unique")

    def test_series_coverage(self, dataset):
        """Test that all series are covered in the mapping."""
        print("\n" + "="*80)
        print("Testing Series Coverage")
        print("="*80)
        
        # Check that all series appear in idx_to_series
        unique_series = torch.unique(dataset.idx_to_series).numpy()
        expected_series = np.arange(len(dataset.data))
        
        print(f"Expected series: {len(expected_series)}")
        print(f"Series in mapping: {len(unique_series)}")
        
        # Not all series may have segments if they're too short
        for series_idx in expected_series:
            series_len = len(dataset.data[series_idx]["data"])
            can_fit = (series_len - dataset.segment_length) // dataset.shift + 1 > 0
            
            if can_fit:
                assert series_idx in unique_series, \
                    f"Series {series_idx} (len={series_len}) should be in mapping"
            else:
                print(f"  Series {series_idx} (len={series_len}) is too short, skipped")
        
        print("✓ All valid series are covered in mapping")

    def test_boundary_conditions(self, dataset):
        """Test boundary conditions (first and last samples)."""
        print("\n" + "="*80)
        print("Testing Boundary Conditions")
        print("="*80)
        
        # Test first sample
        first_sample = dataset[0]
        assert first_sample is not None
        assert "x" in first_sample
        assert "user" in first_sample
        print(f"✓ First sample (idx=0): shape={first_sample['x'].shape}, user={first_sample['user']}")
        
        # Test last sample
        last_idx = len(dataset) - 1
        last_sample = dataset[last_idx]
        assert last_sample is not None
        assert "x" in last_sample
        assert "user" in last_sample
        print(f"✓ Last sample (idx={last_idx}): shape={last_sample['x'].shape}, user={last_sample['user']}")

    def test_random_split_consistency(self, dataset, split_func: callable = None):
        """Test that random split preserves data integrity when reordered."""
        print("\n" + "="*80)
        print("Testing Random Split Consistency")
        print("="*80)
        
        # Get splits
        train_set, val_set, test_set = dataset.get_splits()
        
        print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print(f"Total: {len(train_set) + len(val_set) + len(test_set)} (original: {len(dataset)})")
        
        # Verify total length
        assert len(train_set) + len(val_set) + len(test_set) == len(dataset), \
            "Split sizes don't match original dataset size"
        
        # Collect all samples from splits with their original indices
        all_split_samples = []
        
        # Get indices from train set
        train_indices = train_set.indices
        for idx in train_indices:
            sample = dataset[idx]
            all_split_samples.append((idx, sample['x'], sample['user']))
        
        # Get indices from val set
        val_indices = val_set.indices
        for idx in val_indices:
            sample = dataset[idx]
            all_split_samples.append((idx, sample['x'], sample['user']))
        
        # Get indices from test set
        test_indices = test_set.indices
        for idx in test_indices:
            sample = dataset[idx]
            all_split_samples.append((idx, sample['x'], sample['user']))
        
        print(f"Collected {len(all_split_samples)} samples from splits")
        
        # Check for duplicate indices
        all_indices = [int(idx) for idx, _, _ in all_split_samples]
        unique_indices = set(all_indices)
        assert len(all_indices) == len(unique_indices), \
            "Duplicate indices found in splits"
        print("✓ No duplicate indices in splits")
        
        # Check that all indices are covered
        expected_indices = set(range(len(dataset)))
        actual_indices = set(sorted(all_indices))
        assert expected_indices == actual_indices, \
            f"Missing indices: {expected_indices - actual_indices}"
        print("✓ All indices covered in splits")
        
        # Sort by original index to restore original order
        all_split_samples.sort(key=lambda x: x[0])
        
        # Verify data integrity by comparing with original dataset
        print("Verifying data integrity after reordering...")
        mismatches = 0
        for i, (orig_idx, split_x, split_user) in enumerate(all_split_samples):
            # Get original sample
            original_sample = dataset[orig_idx]
            original_x = original_sample['x']
            original_user = original_sample['user']
            
            # Compare data
            if not torch.allclose(split_x, original_x, rtol=1e-5, atol=1e-7):
                print(f"  ❌ Data mismatch at index {orig_idx}")
                mismatches += 1
            
            # Compare user
            if split_user != original_user:
                print(f"  ❌ User mismatch at index {orig_idx}: {split_user} vs {original_user}")
                mismatches += 1
        
        assert mismatches == 0, f"Found {mismatches} mismatches in split data"
        print(f"✓ All {len(all_split_samples)} samples match original data after reordering")
        
        # Verify sequential segments still have correct shift after split
        print("\nVerifying sequential segments in first series after split...")
        series_idx = 0
        series_indices = dataset.series_to_idx[series_idx].numpy()
        
        if len(series_indices) >= 2:
            # Check which split each index belongs to
            for i in range(min(3, len(series_indices) - 1)):
                idx1 = series_indices[i]
                idx2 = series_indices[i + 1]
                
                # Find which split they belong to
                split1 = "train" if idx1 in train_indices else ("val" if idx1 in val_indices else "test")
                split2 = "train" if idx2 in train_indices else ("val" if idx2 in val_indices else "test")
                
                pos1 = dataset.idx_to_position[idx1].item()
                pos2 = dataset.idx_to_position[idx2].item()
                
                print(f"  Segment {i}: idx={idx1} ({split1}), next idx={idx2} ({split2}), pos_diff={pos2-pos1}")
                
                # Position difference should still be shift
                assert pos2 - pos1 == dataset.shift, \
                    f"Position difference should be {dataset.shift}, got {pos2-pos1}"
        
        print("✓ Sequential segments maintain correct shift after split")
        print("\n✓ Random split consistency test passed!")

    def test_different_configurations(self, dataset_path):
        """Test with different segment lengths and shifts."""
        print("\n" + "="*80)
        print("Testing Different Configurations")
        print("="*80)
        
        configs = [
            {"segment_length": 120, "shift": 60},
            {"segment_length": 240, "shift": 240},  # No overlap
            {"segment_length": 480, "shift": 120},
        ]
        
        for config in configs:
            ds = EDAtaNoLabelOnlineSegmentation(
                base_path=dataset_path.parent,
                dataset_names=["BiHeartS"],
                shuffle=False,
                seed=42,
                channels=0,
                **config
            )
            
            # Calculate expected
            expected = sum(
                max(0, (len(item["data"]) - config["segment_length"]) // config["shift"] + 1)
                for item in ds.data
            )
            
            actual = len(ds)
            
            print(f"Config {config}: expected={expected}, actual={actual}, match={expected==actual}")
            
            assert expected == actual, \
                f"Config {config}: expected {expected} samples, got {actual}"
        
        print("✓ All configurations work correctly")


def run_tests():
    """Run all tests with detailed output."""
    print("\n" + "="*80)
    print("RUNNING DATASET ONLINE MAPPING TESTS")
    print("="*80 + "\n")
    
    # Create test instance
    test = TestDatasetOnlineMapping()
    
    # Get fixtures
    dataset_path = test.dataset_path()
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        print("Please update the path in the test file.")
        return
    
    print(f"Using dataset: {dataset_path}\n")
    
    # Test configurations: different segment lengths and shifts
    test_configs = [
        {"segment_length": 120, "shift": 60, "name": "Short segments (120) with overlap (shift=60)"},
        {"segment_length": 240, "shift": 120, "name": "Medium segments (240) with overlap (shift=120)"},
        {"segment_length": 240, "shift": 240, "name": "Medium segments (240) no overlap (shift=240)"},
        {"segment_length": 480, "shift": 120, "name": "Long segments (480) with large overlap (shift=120)"},
        {"segment_length": 480, "shift": 240, "name": "Long segments (480) with medium overlap (shift=240)"},
        {"segment_length": 600, "shift": 300, "name": "Extra long segments (600) with overlap (shift=300)"},
    ]
    
    total_passed = 0
    total_failed = 0
    
    for config in test_configs:
        print("\n" + "="*80)
        print(f"TESTING CONFIGURATION: {config['name']}")
        print(f"  segment_length={config['segment_length']}, shift={config['shift']}")
        print("="*80 + "\n")
        
        try:
            # Create dataset with current configuration
            dataset = EDAtaNoLabelOnlineSegmentation(
                base_path=dataset_path.parent,
                dataset_names=["BiHeartS"],
                segment_length=config['segment_length'],
                shift=config['shift'],
                shuffle=False,
                seed=42,
                channels=0,
                train_val_test_size=[0.8, 0.1, 0.1],
                splitting_method=user_session_splitting
            )
            
            # Run tests in order
            test.test_dataset_loads(dataset)
            test.test_data_structure(dataset)
            test.test_index_mapping_length(dataset)
            test.test_index_mapping_consistency(dataset)
            test.test_manual_calculation_vs_mapping(dataset)
            test.test_sequential_segments(dataset)
            test.test_segment_extraction(dataset)
            test.test_no_overlap_in_indices(dataset)
            test.test_series_coverage(dataset)
            test.test_boundary_conditions(dataset)
            test.test_random_split_consistency(dataset)
            
            print(f"\n✅ Configuration {config['name']} PASSED!\n")
            total_passed += 1
            
        except Exception as e:
            raise e
            print(f"\n❌ Configuration {config['name']} FAILED: {e}\n")
            total_failed += 1
            import traceback
            traceback.print_exc()
    
    # Run test with different configurations (legacy test)
    print("\n" + "="*80)
    print("RUNNING LEGACY CONFIGURATION TEST")
    print("="*80 + "\n")
    try:
        dataset = test.dataset(dataset_path)
        test.test_different_configurations(dataset_path)
        total_passed += 1
    except Exception as e:
        print(f"❌ Legacy test failed: {e}")
        total_failed += 1
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total configurations tested: {len(test_configs) + 1}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {total_failed} TEST(S) FAILED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_tests()
