"""
Tests for data augmentation in ContrastiveEngine and ReconstructionEngine.
"""

import pytest
import numpy as np
import torch
from torch import nn

from src.engine import ContrastiveEngine, ReconstructionEngine
from src.model.matton_constrastive import ContrastiveModel


class DummyAugmentation:
    """Dummy augmentation that adds a constant value to the input."""
    
    def __init__(self, offset=1.0):
        self.offset = offset
    
    def __call__(self, x):
        """Add offset to input array."""
        return x + self.offset


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, input_dim=100):
        super().__init__()
        self.linear = nn.Linear(input_dim, 64)
    
    def forward(self, x_1, x_2):
        """Forward pass."""
        # Simple forward for testing
        return {'z_1': self.linear(x_1.mean(dim=1)), 'z_2': self.linear(x_2.mean(dim=1))}
    
    def loss(self, output, batch, step='train'):
        """Dummy loss."""
        return torch.tensor(0.5)


class TestContrastiveEngineAugmentation:
    """Test data augmentation in ContrastiveEngine."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model."""
        return DummyModel(input_dim=100)
    
    @pytest.fixture
    def augmentation(self):
        """Create a dummy augmentation."""
        return DummyAugmentation(offset=2.0)
    
    def test_augmentation_2d_training(self, dummy_model, augmentation):
        """Test augmentation with 2D input (N, T) during training."""
        # Create engine
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=augmentation
        )
        engine.train()  # Set to training mode
        
        # Create batch with 2D data (N=4, T=100)
        batch_size = 4
        time_steps = 100
        x = torch.randn(batch_size, time_steps)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are created
        assert 'x_1' in augmented_batch
        assert 'x_2' in augmented_batch
        assert 'batch_size' in augmented_batch
        
        # Check shapes
        assert augmented_batch['x_1'].shape == (batch_size, time_steps)
        assert augmented_batch['x_2'].shape == (batch_size, time_steps)
        
        # Check that augmentation was applied (offset of 2.0)
        # Since augmentation adds 2.0, we can verify this
        x_1_np = augmented_batch['x_1'].cpu().numpy()
        x_2_np = augmented_batch['x_2'].cpu().numpy()
        x_np = x.cpu().numpy()
        
        # Check that x_1 and x_2 are different from x (augmented)
        np.testing.assert_allclose(x_1_np, x_np + 2.0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(x_2_np, x_np + 2.0, rtol=1e-5, atol=1e-5)
        
        # Check batch_size
        assert augmented_batch['batch_size'] == batch_size
    
    def test_augmentation_3d_training(self, dummy_model, augmentation):
        """Test augmentation with 3D input (N, T, C) during training."""
        # Create engine
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=augmentation
        )
        engine.train()  # Set to training mode
        
        # Create batch with 3D data (N=4, T=100, C=3)
        batch_size = 4
        time_steps = 100
        channels = 3
        x = torch.randn(batch_size, time_steps, channels)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are created
        assert 'x_1' in augmented_batch
        assert 'x_2' in augmented_batch
        assert 'batch_size' in augmented_batch
        
        # Check shapes
        assert augmented_batch['x_1'].shape == (batch_size, time_steps, channels)
        assert augmented_batch['x_2'].shape == (batch_size, time_steps, channels)
        
        # Check that augmentation was applied (offset of 2.0)
        x_1_np = augmented_batch['x_1'].cpu().numpy()
        x_2_np = augmented_batch['x_2'].cpu().numpy()
        x_np = x.cpu().numpy()
        
        # Check that x_1 and x_2 are different from x (augmented)
        np.testing.assert_allclose(x_1_np, x_np + 2.0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(x_2_np, x_np + 2.0, rtol=1e-5, atol=1e-5)
        
        # Check batch_size
        assert augmented_batch['batch_size'] == batch_size
    
    def test_no_augmentation_validation(self, dummy_model, augmentation):
        """Test that augmentation is NOT applied during validation."""
        # Create engine
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=augmentation
        )
        engine.eval()  # Set to evaluation mode (validation/test)
        
        # Create batch with 2D data
        batch_size = 4
        time_steps = 100
        x = torch.randn(batch_size, time_steps)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are created but NOT augmented
        assert 'x_1' in augmented_batch
        assert 'x_2' in augmented_batch
        
        # Check that x_1 and x_2 are identical to x (no augmentation)
        torch.testing.assert_close(augmented_batch['x_1'], x)
        torch.testing.assert_close(augmented_batch['x_2'], x)
    
    def test_no_augmentation_none(self, dummy_model):
        """Test when data_augmentation is None."""
        # Create engine without augmentation
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=None
        )
        engine.train()
        
        # Create batch
        batch_size = 4
        time_steps = 100
        x = torch.randn(batch_size, time_steps)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are identical to x (no augmentation)
        torch.testing.assert_close(augmented_batch['x_1'], x)
        torch.testing.assert_close(augmented_batch['x_2'], x)
    
    def test_augmentation_preserves_device(self, dummy_model, augmentation):
        """Test that augmentation preserves device (CPU/GPU)."""
        # Create engine
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=augmentation
        )
        engine.train()
        
        # Create batch on CPU
        x = torch.randn(4, 100)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check device
        assert augmented_batch['x_1'].device == x.device
        assert augmented_batch['x_2'].device == x.device
        
        # If CUDA is available, test with GPU
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            batch_cuda = {'x': x_cuda}
            
            augmented_batch_cuda = engine.on_after_batch_transfer(batch_cuda, dataloader_idx=0)
            
            assert augmented_batch_cuda['x_1'].device == x_cuda.device
            assert augmented_batch_cuda['x_2'].device == x_cuda.device
    
    def test_augmentation_different_views(self, dummy_model):
        """Test that x_1 and x_2 are different when using stochastic augmentation."""
        
        class StochasticAugmentation:
            """Augmentation that adds random noise."""
            def __call__(self, x):
                return x + np.random.randn(*x.shape) * 0.1
        
        # Create engine with stochastic augmentation
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=StochasticAugmentation()
        )
        engine.train()
        
        # Create batch
        x = torch.randn(4, 100)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # x_1 and x_2 should be different (stochastic augmentation)
        # We can't use exact comparison, but they should not be identical
        assert not torch.allclose(augmented_batch['x_1'], augmented_batch['x_2'])
    
    def test_augmentation_batch_independence(self, dummy_model, augmentation):
        """Test that augmentation is applied independently to each sample in batch."""
        # Create engine
        engine = ContrastiveEngine(
            model=dummy_model,
            data_augmentation=augmentation
        )
        engine.train()
        
        # Create batch where each sample has different values
        x = torch.arange(12).reshape(3, 4).float()  # (N=3, T=4)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that augmentation was applied to each sample
        expected = x + 2.0
        torch.testing.assert_close(augmented_batch['x_1'], expected)
        torch.testing.assert_close(augmented_batch['x_2'], expected)


class TestReconstructionEngineAugmentation:
    """Test data augmentation in ReconstructionEngine."""
    
    @pytest.fixture
    def dummy_reconstruction_model(self):
        """Create a dummy reconstruction model."""
        class DummyReconstructionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 100)
            
            def forward(self, x_aug):
                return {'reconstructed': x_aug}
            
            def loss(self, output, batch, step='train'):
                return torch.tensor(0.5)
        
        return DummyReconstructionModel()
    
    @pytest.fixture
    def augmentation(self):
        """Create a dummy augmentation that returns a dict."""
        class DictAugmentation:
            def __call__(self, batch, prep_keys=None):
                # Augmentation that adds 3.0 to the tensor
                x = batch['x']
                return {'x': x + 3.0}
        
        return DictAugmentation()
    
    def test_reconstruction_augmentation_training(self, dummy_reconstruction_model, augmentation):
        """Test augmentation in ReconstructionEngine during training."""
        # Create engine
        engine = ReconstructionEngine(
            model=dummy_reconstruction_model,
            data_augmentation=augmentation
        )
        engine.train()
        
        # Create batch
        x = torch.randn(4, 100)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_aug is created
        assert 'x_aug' in augmented_batch
        
        # Check that augmentation was applied
        expected = x + 3.0
        torch.testing.assert_close(augmented_batch['x_aug'], expected)
    
    def test_reconstruction_no_augmentation_validation(self, dummy_reconstruction_model, augmentation):
        """Test that augmentation is NOT applied during validation in ReconstructionEngine."""
        # Create engine
        engine = ReconstructionEngine(
            model=dummy_reconstruction_model,
            data_augmentation=augmentation
        )
        engine.eval()
        
        # Create batch
        x = torch.randn(4, 100)
        batch = {'x': x}
        
        # Apply augmentation
        augmented_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_aug equals x (no augmentation in eval mode)
        torch.testing.assert_close(augmented_batch['x_aug'], x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
