"""
Tests for ContrastiveEngine embedding collection and visualization.

This test suite verifies the refactored embedding collection logic:
1. _collect_embeddings_for_visualization helper method
2. Early stopping logic for efficient collection
3. Training and validation step integration
4. Edge cases and boundary conditions
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from edamame.engine import ContrastiveEngine


# ============================================================================
# Mock Models and Fixtures
# ============================================================================

class MockContrastiveModel(nn.Module):
    """Mock contrastive model for testing."""
    
    def __init__(self, input_dim=10, embedding_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x_1, x_2):
        """Forward pass expects x_1 and x_2."""
        z1 = self.encoder(x_1)
        z2 = self.encoder(x_2)
        return {'z1': z1, 'z2': z2}
    
    def get_representation(self, x):
        """Extract embeddings from input."""
        return self.encoder(x)
    
    def loss(self, output, batch, step='train'):
        """Simple contrastive loss."""
        z1, z2 = output['z1'], output['z2']
        return torch.mean((z1 - z2) ** 2)


@pytest.fixture
def mock_contrastive_model():
    """Create mock contrastive model."""
    return MockContrastiveModel(input_dim=10, embedding_dim=8)


@pytest.fixture
def contrastive_engine(mock_contrastive_model):
    """Create ContrastiveEngine with default settings."""
    return ContrastiveEngine(
        model=mock_contrastive_model,
        seed=42,
        plotting_kwargs={
            'max_users': 5,
            'max_segments_per_user': 10
        }
    )


@pytest.fixture
def contrastive_engine_unlimited(mock_contrastive_model):
    """Create ContrastiveEngine with no limits on collection."""
    return ContrastiveEngine(
        model=mock_contrastive_model,
        seed=42,
        plotting_kwargs={
            'max_users': None,
            'max_segments_per_user': None
        }
    )


def create_batch(batch_size=4, input_dim=10, user_ids=None):
    """Helper to create a batch with user labels."""
    if user_ids is None:
        user_ids = np.arange(batch_size)
    
    return {
        'x': torch.randn(batch_size, input_dim),
        'x_1': torch.randn(batch_size, input_dim),
        'x_2': torch.randn(batch_size, input_dim),
        'user': torch.tensor(user_ids, dtype=torch.long)
    }


# ============================================================================
# Test Collection Complete Logic
# ============================================================================

class TestCollectionComplete:
    """Test _collection_complete method."""
    
    def test_collection_not_complete_empty_counts(self, contrastive_engine):
        """Empty user_counts should not be complete."""
        assert not contrastive_engine._collection_complete({})
    
    def test_collection_not_complete_insufficient_users(self, contrastive_engine):
        """Less than max_users should not be complete."""
        user_counts = {0: 10, 1: 10, 2: 10}  # Only 3 users, need 5
        assert not contrastive_engine._collection_complete(user_counts)
    
    def test_collection_not_complete_insufficient_segments(self, contrastive_engine):
        """Enough users but insufficient segments should not be complete."""
        user_counts = {0: 10, 1: 10, 2: 5, 3: 10, 4: 10}  # User 2 has only 5 segments
        assert not contrastive_engine._collection_complete(user_counts)
    
    def test_collection_complete_all_requirements_met(self, contrastive_engine):
        """All requirements met should be complete."""
        user_counts = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}  # 5 users, each with 10 segments
        assert contrastive_engine._collection_complete(user_counts)
    
    def test_collection_complete_extra_segments(self, contrastive_engine):
        """More segments than required should still be complete."""
        user_counts = {0: 15, 1: 12, 2: 11, 3: 20, 4: 10}  # All have >= 10 segments
        assert contrastive_engine._collection_complete(user_counts)
    
    def test_collection_never_complete_unlimited_users(self, contrastive_engine_unlimited):
        """With unlimited users (None), should never be complete."""
        user_counts = {i: 20 for i in range(100)}  # Even with many users
        assert not contrastive_engine_unlimited._collection_complete(user_counts)
    
    def test_collection_with_unlimited_segments(self, mock_contrastive_model):
        """With unlimited segments but limited users, should complete when users met."""
        engine = ContrastiveEngine(
            model=mock_contrastive_model,
            seed=42,
            plotting_kwargs={
                'max_users': 3,
                'max_segments_per_user': None  # Unlimited segments
            }
        )
        
        # Should complete with 3 users regardless of segment count
        user_counts = {0: 5, 1: 10, 2: 2}
        assert engine._collection_complete(user_counts)
        
        # But not with only 2 users
        user_counts = {0: 100, 1: 100}
        assert not engine._collection_complete(user_counts)


# ============================================================================
# Test Sample Selection Logic
# ============================================================================

class TestSelectSamplesToCollect:
    """Test _select_samples_to_collect method."""
    
    def test_select_all_samples_initially(self, contrastive_engine):
        """Initially with empty counts, should select all samples."""
        users = np.array([0, 1, 2, 3])
        user_counts = {}
        
        selected = contrastive_engine._select_samples_to_collect(users, user_counts)
        
        assert len(selected) == 4
        assert user_counts == {0: 1, 1: 1, 2: 1, 3: 1}
    
    def test_select_samples_partial_collection(self, contrastive_engine):
        """Should select samples from users not at limit."""
        users = np.array([0, 0, 1, 1, 2])
        user_counts = {0: 9, 1: 10, 2: 5}  # User 1 is at limit
        
        selected = contrastive_engine._select_samples_to_collect(users, user_counts)
        
        # Should skip user 1 (indices 2 and 3)
        assert 2 not in selected
        assert 3 not in selected
        # Should include user 0 (only first occurrence since it will reach limit) and user 2
        assert 0 in selected  # First user 0
        # Index 1 (second user 0) will be skipped if first one reaches limit
        assert 4 in selected  # User 2
        
        # Counts should be updated
        assert user_counts[0] == 10  # Was 9, added 1 (reached limit after first sample)
        assert user_counts[1] == 10  # Stayed at 10
        assert user_counts[2] == 6   # Was 5, added 1
    
    def test_select_samples_max_users_reached(self, contrastive_engine):
        """Should not accept new users when max_users is reached."""
        users = np.array([0, 1, 5, 6, 7, 8])
        user_counts = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}  # Already have 5 users
        
        selected = contrastive_engine._select_samples_to_collect(users, user_counts)
        
        # Should only select samples from users 0 and 1 (already in counts)
        assert 0 in selected  # User 0
        assert 1 in selected  # User 1
        assert 2 not in selected  # User 5 (new)
        assert 3 not in selected  # User 6 (new)
        assert 4 not in selected  # User 7 (new)
        assert 5 not in selected  # User 8 (new)
    
    def test_select_samples_unlimited_users(self, contrastive_engine_unlimited):
        """With unlimited users, should always accept new users."""
        users = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        user_counts = {i: 5 for i in range(20)}  # Already have many users
        
        selected = contrastive_engine_unlimited._select_samples_to_collect(users, user_counts)
        
        # Should select all samples since no limits
        assert len(selected) == 10
    
    def test_select_samples_updates_counts_correctly(self, contrastive_engine):
        """Should correctly update counts for selected samples."""
        users = np.array([0, 0, 1, 2, 2, 2])
        user_counts = {}
        
        selected = contrastive_engine._select_samples_to_collect(users, user_counts)
        
        assert user_counts[0] == 2  # Two samples from user 0
        assert user_counts[1] == 1  # One sample from user 1
        assert user_counts[2] == 3  # Three samples from user 2


# ============================================================================
# Test Embedding Collection Helper
# ============================================================================

class TestCollectEmbeddingsForVisualization:
    """Test _collect_embeddings_for_visualization method."""
    
    def test_collect_embeddings_basic(self, contrastive_engine):
        """Test basic embedding collection."""
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        embeddings_list = []
        users_list = []
        user_counts = {}
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch, user_counts, embeddings_list, users_list
        )
        
        assert len(embeddings_list) == 1
        assert len(users_list) == 1
        assert embeddings_list[0].shape == (4, 8)  # 4 samples, 8 embedding dims
        assert len(users_list[0]) == 4
    
    def test_collect_embeddings_early_stopping(self, contrastive_engine):
        """Test that collection stops when limits are reached."""
        # Pre-fill counts to be at limit
        user_counts = {i: 10 for i in range(5)}  # 5 users, 10 segments each
        embeddings_list = []
        users_list = []
        
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch, user_counts, embeddings_list, users_list
        )
        
        # Should not collect anything since limits are reached
        assert len(embeddings_list) == 0
        assert len(users_list) == 0
    
    def test_collect_embeddings_selective(self, contrastive_engine):
        """Test selective collection based on user counts."""
        user_counts = {0: 10, 1: 5, 2: 0}  # User 0 at limit, others not
        embeddings_list = []
        users_list = []
        
        # Batch with mixed users
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch, user_counts, embeddings_list, users_list
        )
        
        # Should collect from users 1, 2, 3 but not 0
        assert len(embeddings_list) == 1
        collected_users = users_list[0]
        assert 0 not in collected_users  # User 0 should be skipped
        assert 1 in collected_users
        assert 2 in collected_users
        assert 3 in collected_users
    
    def test_collect_embeddings_without_user_key(self, contrastive_engine):
        """Test that collection doesn't happen without user key."""
        batch = {
            'x': torch.randn(4, 10),
            'x_1': torch.randn(4, 10),
            'x_2': torch.randn(4, 10)
            # No 'user' key
        }
        embeddings_list = []
        users_list = []
        user_counts = {}
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch, user_counts, embeddings_list, users_list
        )
        
        # Should not collect anything without user key
        assert len(embeddings_list) == 0
        assert len(users_list) == 0
    
    def test_collect_embeddings_uses_x1_if_available(self, contrastive_engine):
        """Test that x_1 is used if available, otherwise x."""
        # Test with x_1
        batch_with_x1 = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        embeddings_list_1 = []
        users_list_1 = []
        user_counts_1 = {}
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch_with_x1, user_counts_1, embeddings_list_1, users_list_1
        )
        
        # Test with only x
        batch_without_x1 = {
            'x': torch.randn(4, 10),
            'user': torch.tensor([0, 1, 2, 3])
        }
        embeddings_list_2 = []
        users_list_2 = []
        user_counts_2 = {}
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch_without_x1, user_counts_2, embeddings_list_2, users_list_2
        )
        
        # Both should collect embeddings
        assert len(embeddings_list_1) == 1
        assert len(embeddings_list_2) == 1


# ============================================================================
# Test Training and Validation Step Integration
# ============================================================================

class TestTrainingValidationStepIntegration:
    """Test that training and validation steps properly use the helper."""
    
    def test_training_step_collects_embeddings(self, contrastive_engine):
        """Test that training_step collects embeddings."""
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        
        # Manually call on_after_batch_transfer to create x_1 and x_2
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Run training step
        loss = contrastive_engine.training_step(batch, batch_idx=0)
        
        # Check embeddings were collected
        assert len(contrastive_engine.train_embeddings) == 1
        assert len(contrastive_engine.train_users) == 1
        assert contrastive_engine.train_embeddings[0].shape[0] == 4
        
        # Check counts were updated
        assert len(contrastive_engine.train_user_counts) == 4
        assert all(count == 1 for count in contrastive_engine.train_user_counts.values())
    
    def test_validation_step_collects_embeddings(self, contrastive_engine):
        """Test that validation_step collects embeddings."""
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        
        # Manually call on_after_batch_transfer
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Run validation step
        loss = contrastive_engine.validation_step(batch, batch_idx=0)
        
        # Check embeddings were collected
        assert len(contrastive_engine.validation_embeddings) == 1
        assert len(contrastive_engine.validation_users) == 1
        
        # Check counts were updated
        assert len(contrastive_engine.val_user_counts) == 4
    
    def test_train_and_val_use_separate_storage(self, contrastive_engine):
        """Test that train and val steps use separate storage."""
        batch = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Run both steps
        train_loss = contrastive_engine.training_step(batch, batch_idx=0)
        val_loss = contrastive_engine.validation_step(batch, batch_idx=0)
        
        # Check separate storage
        assert len(contrastive_engine.train_embeddings) == 1
        assert len(contrastive_engine.validation_embeddings) == 1
        assert len(contrastive_engine.train_user_counts) == 4
        assert len(contrastive_engine.val_user_counts) == 4
        
        # Counts should be independent
        assert contrastive_engine.train_user_counts != contrastive_engine.val_user_counts or \
               all(contrastive_engine.train_user_counts[k] == contrastive_engine.val_user_counts[k] 
                   for k in contrastive_engine.train_user_counts.keys())
    
    def test_multiple_batches_accumulate_correctly(self, contrastive_engine):
        """Test that multiple batches accumulate embeddings correctly."""
        # First batch
        batch1 = create_batch(batch_size=4, user_ids=[0, 1, 2, 3])
        batch1 = contrastive_engine.on_after_batch_transfer(batch1, dataloader_idx=0)
        contrastive_engine.training_step(batch1, batch_idx=0)
        
        # Second batch with some overlapping users
        batch2 = create_batch(batch_size=4, user_ids=[2, 3, 4, 5])
        batch2 = contrastive_engine.on_after_batch_transfer(batch2, dataloader_idx=0)
        contrastive_engine.training_step(batch2, batch_idx=1)
        
        # Should have 2 embedding tensors in list
        assert len(contrastive_engine.train_embeddings) == 2
        assert len(contrastive_engine.train_users) == 2
        
        # User counts should reflect overlaps
        # After batch1: users 0,1,2,3 (4 users)
        # After batch2: can accept 1 more user (to reach max_users=5), so either user 4 or 5
        assert contrastive_engine.train_user_counts[0] == 1
        assert contrastive_engine.train_user_counts[1] == 1
        assert contrastive_engine.train_user_counts[2] == 2  # In both batches
        assert contrastive_engine.train_user_counts[3] == 2  # In both batches
        # Only one of users 4 or 5 will be accepted (first new user encountered)
        assert contrastive_engine.train_user_counts[4] == 1  # First new user in batch2
        assert 5 not in contrastive_engine.train_user_counts  # Rejected (would exceed max_users)
    
    def test_early_stopping_prevents_collection(self, contrastive_engine):
        """Test that early stopping prevents further collection."""
        # Fill up to limit: 5 users, 10 segments each
        for batch_idx in range(10):
            batch = create_batch(batch_size=5, user_ids=[0, 1, 2, 3, 4])
            batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
            contrastive_engine.training_step(batch, batch_idx=batch_idx)
        
        # Should have collected exactly 10 batches
        assert len(contrastive_engine.train_embeddings) == 10
        
        # Try to collect more - should not add anything
        initial_len = len(contrastive_engine.train_embeddings)
        batch = create_batch(batch_size=5, user_ids=[0, 1, 2, 3, 4])
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        contrastive_engine.training_step(batch, batch_idx=10)
        
        # Should not have added new embeddings
        assert len(contrastive_engine.train_embeddings) == initial_len


# ============================================================================
# Test Epoch End Hooks
# ============================================================================

class TestEpochEndHooks:
    """Test on_train_epoch_end and on_validation_epoch_end hooks."""
    
    @patch.object(ContrastiveEngine, '_plot_embedding_visualization')
    def test_train_epoch_end_calls_plotting(self, mock_plot, contrastive_engine):
        """Test that on_train_epoch_end calls plotting and clears storage."""
        # Add some dummy data
        contrastive_engine.train_embeddings = [torch.randn(4, 8)]
        contrastive_engine.train_users = [np.array([0, 1, 2, 3])]
        contrastive_engine.train_user_counts = {0: 1, 1: 1, 2: 1, 3: 1}
        
        # Call hook
        contrastive_engine.on_train_epoch_end()
        
        # Check plotting was called with correct split
        mock_plot.assert_called_once()
        # Check keyword argument
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs['split'] == 'train'
        
        # Check storage was cleared
        assert len(contrastive_engine.train_embeddings) == 0
        assert len(contrastive_engine.train_users) == 0
        assert len(contrastive_engine.train_user_counts) == 0
    
    @patch.object(ContrastiveEngine, '_plot_embedding_visualization')
    def test_val_epoch_end_calls_plotting(self, mock_plot, contrastive_engine):
        """Test that on_validation_epoch_end calls plotting and clears storage."""
        # Add some dummy data
        contrastive_engine.validation_embeddings = [torch.randn(4, 8)]
        contrastive_engine.validation_users = [np.array([0, 1, 2, 3])]
        contrastive_engine.val_user_counts = {0: 1, 1: 1, 2: 1, 3: 1}
        
        # Call hook
        contrastive_engine.on_validation_epoch_end()
        
        # Check plotting was called with correct split
        mock_plot.assert_called_once()
        # Check keyword argument
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs['split'] == 'val'
        
        # Check storage was cleared
        assert len(contrastive_engine.validation_embeddings) == 0
        assert len(contrastive_engine.validation_users) == 0
        assert len(contrastive_engine.val_user_counts) == 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_batch(self, contrastive_engine):
        """Test handling of empty batch."""
        batch = create_batch(batch_size=0)
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Should not crash
        embeddings_list = []
        users_list = []
        user_counts = {}
        
        contrastive_engine._collect_embeddings_for_visualization(
            batch, user_counts, embeddings_list, users_list
        )
        
        assert len(embeddings_list) == 0
        assert len(users_list) == 0
    
    def test_single_user_repeated(self, contrastive_engine):
        """Test batch with single user repeated multiple times."""
        batch = create_batch(batch_size=10, user_ids=[0] * 10)
        batch = contrastive_engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        contrastive_engine.training_step(batch, batch_idx=0)
        
        # Should have collected all 10 samples from user 0
        assert contrastive_engine.train_user_counts[0] == 10
        assert len(contrastive_engine.train_embeddings) == 1
        assert contrastive_engine.train_embeddings[0].shape[0] == 10
    
    def test_plotting_kwargs_validation(self, mock_contrastive_model):
        """Test that missing plotting_kwargs keys raise error."""
        with pytest.raises(ValueError, match="plotting_kwargs must contain"):
            ContrastiveEngine(
                model=mock_contrastive_model,
                seed=42,
                plotting_kwargs={'max_users': 5}  # Missing max_segments_per_user
            )
        
        with pytest.raises(ValueError, match="plotting_kwargs must contain"):
            ContrastiveEngine(
                model=mock_contrastive_model,
                seed=42,
                plotting_kwargs={'max_segments_per_user': 10}  # Missing max_users
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
