"""
Tests for the SelfSupervisedEngine class.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import Accuracy, MeanSquaredError
import pytorch_lightning as pl

from src.engine import (
    BaseSelfSupervisedEngine,
    ContrastiveEngine,
    ReconstructionEngine,
    SelfSupervisedEngine,  # Backward compatibility alias
)


# ============================================================================
# Mock Models for Testing
# ============================================================================

class MockDataAugmentation(nn.Module):
    """Mock data augmentation pipeline."""
    
    def __init__(self, augmentation_value=1.0):
        super().__init__()
        self.augmentation_value = augmentation_value
        
    def forward(self, batch, prep_keys=None):
        """Apply augmentation by adding a constant to specified keys."""
        if prep_keys is None:
            return batch
        
        augmented_batch = batch.copy()
        for key in prep_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                # Add noise so that multiple calls produce different results
                noise = torch.randn_like(batch[key]) * 0.01
                augmented_batch[key] = batch[key] + self.augmentation_value + noise
        
        return augmented_batch


class MockModel(nn.Module):
    """Mock model with loss method."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, batch):
        x = batch['x']
        h = self.encoder(x)
        out = self.decoder(h)
        return {'output': out, 'latent': h}
    
    def loss(self, output, batch, step='train'):
        # Simple MSE loss for testing
        target = batch.get('target', batch['x'][:, :self.output_dim])
        loss = nn.functional.mse_loss(output['output'], target)
        return loss


class MockClassificationModel(nn.Module):
    """Mock classification model with custom metric extraction."""
    
    def __init__(self, input_dim=10, num_classes=3):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)
        self.output_dim = num_classes
        
    def forward(self, batch):
        x = batch['x']
        logits = self.net(x)
        return {'logits': logits}
    
    def loss(self, output, batch, step='train'):
        target = batch['y']
        loss = nn.functional.cross_entropy(output['logits'], target)
        return loss
    
    def get_predictions_for_metrics(self, output, batch):
        """Extract predictions and targets for metrics."""
        if output is None:
            return None, None
        predictions = output['logits'].argmax(dim=1)
        targets = batch['y']
        return predictions, targets


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_batch():
    """Create a simple batch for testing."""
    return {
        'x': torch.randn(4, 10),
        'target': torch.randn(4, 5)
    }


@pytest.fixture
def classification_batch():
    """Create a classification batch for testing."""
    return {
        'x': torch.randn(4, 10),
        'y': torch.randint(0, 3, (4,))
    }


@pytest.fixture
def mock_model():
    """Create a basic mock model."""
    return MockModel()


@pytest.fixture
def mock_classification_model():
    """Create a mock classification model with metrics support."""
    return MockClassificationModel()


@pytest.fixture
def mock_data_augmentation():
    """Create a mock data augmentation module."""
    return MockDataAugmentation(augmentation_value=2.0)


# ============================================================================
# Tests
# ============================================================================

class TestSelfSupervisedEngineInitialization:
    """Test engine initialization."""
    
    def test_basic_initialization(self, mock_model):
        """Test basic engine initialization with minimal arguments."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        assert engine.model is mock_model
        assert engine.optimizer_class == torch.optim.Adam
        assert engine.optimizer_kwargs == {}
        assert engine.scheduler_class is None
        assert engine.scheduler_kwargs == {}
        assert engine.train_metrics is None
        assert engine.val_metrics is None
        assert engine.test_metrics is None
    
    def test_initialization_with_optimizer(self, mock_model):
        """Test initialization with custom optimizer."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.01, 'momentum': 0.9}
        )
        
        assert engine.optimizer_class == SGD
        assert engine.optimizer_kwargs == {'lr': 0.01, 'momentum': 0.9}
    
    def test_initialization_with_scheduler(self, mock_model):
        """Test initialization with scheduler."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=AdamW,
            optimizer_kwargs={'lr': 1e-3},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 10, 'gamma': 0.1}
        )
        
        assert engine.scheduler_class == StepLR
        assert engine.scheduler_kwargs == {'step_size': 10, 'gamma': 0.1}
    
    def test_initialization_with_metrics(self, mock_classification_model):
        """Test initialization with metrics."""
        metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=3)
        }
        
        engine = SelfSupervisedEngine(
            model=mock_classification_model,
            metrics=metrics
        )
        
        assert engine.train_metrics is not None
        assert engine.val_metrics is not None
        assert engine.test_metrics is not None
        assert 'accuracy' in engine.train_metrics
        assert 'accuracy' in engine.val_metrics
        assert 'accuracy' in engine.test_metrics


class TestSelfSupervisedEngineForward:
    """Test forward pass methods."""
    
    def test_forward(self, mock_model, simple_batch):
        """Test forward method."""
        engine = SelfSupervisedEngine(model=mock_model)
        output = engine.forward(simple_batch)
        
        assert 'output' in output
        assert 'latent' in output
        assert output['output'].shape == (4, 5)
    
    def test_predict(self, mock_model, simple_batch):
        """Test predict method."""
        engine = SelfSupervisedEngine(model=mock_model)
        output = engine.predict(simple_batch)
        
        assert 'output' in output
        assert output['output'].shape == (4, 5)


class TestSelfSupervisedEngineTraining:
    """Test training step functionality."""
    
    def test_training_step(self, mock_model, simple_batch):
        """Test training_step with model using loss method."""
        engine = SelfSupervisedEngine(model=mock_model)
        loss = engine.training_step(simple_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert not torch.isnan(loss)
    
    def test_validation_step(self, mock_model, simple_batch):
        """Test validation step."""
        engine = SelfSupervisedEngine(model=mock_model)
        loss = engine.validation_step(simple_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    def test_test_step(self, mock_model, simple_batch):
        """Test test step."""
        engine = SelfSupervisedEngine(model=mock_model)
        loss = engine.test_step(simple_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestSelfSupervisedEngineMetrics:
    """Test metrics functionality."""
    
    def test_metrics_update_and_log(self, mock_classification_model, classification_batch):
        """Test metrics are updated correctly."""
        metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=3)
        }
        
        engine = SelfSupervisedEngine(
            model=mock_classification_model,
            metrics=metrics
        )
        
        # Run training step
        loss = engine.training_step(classification_batch, batch_idx=0)
        
        # Check that metrics exist
        assert engine.train_metrics is not None
        assert 'accuracy' in engine.train_metrics
    
    def test_no_metrics(self, mock_model, simple_batch):
        """Test engine works without metrics."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        # Should not raise any errors
        loss = engine.training_step(simple_batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)


class TestSelfSupervisedEngineBatchProcessing:
    """Test batch processing hooks."""
    
    def test_on_after_batch_transfer_with_x_key(self, mock_model):
        """Test batch transfer hook adds batch_size."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        batch = {'x': torch.randn(8, 10)}
        processed_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        assert 'batch_size' in processed_batch
        assert processed_batch['batch_size'] == 8
    
    def test_on_after_batch_transfer_without_x_key(self, mock_model):
        """Test batch transfer hook with different key."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        batch = {'data': torch.randn(5, 10)}
        processed_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        assert 'batch_size' in processed_batch
        assert processed_batch['batch_size'] == 5
    
    def test_base_engine_no_augmentation(self, mock_model):
        """Test BaseSelfSupervisedEngine does not perform augmentation itself."""
        aug_pipeline = MockDataAugmentation(augmentation_value=2.0)
        engine = BaseSelfSupervisedEngine(
            model=mock_model,
            data_augmentation=aug_pipeline
        )
        
        # Create batch with x_ keys
        original_x_1 = torch.ones(4, 10)
        original_x_2 = torch.ones(4, 10) * 2
        original_x = torch.ones(4, 10) * 3
        
        batch = {
            'x_1': original_x_1.clone(),
            'x_2': original_x_2.clone(),
            'x': original_x.clone(),
        }
        
        processed_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # BaseSelfSupervisedEngine should NOT augment anything, just add batch_size
        assert torch.allclose(processed_batch['x_1'], original_x_1)
        assert torch.allclose(processed_batch['x_2'], original_x_2)
        assert torch.allclose(processed_batch['x'], original_x)
        assert 'batch_size' in processed_batch
    
    def test_no_data_augmentation(self, mock_model):
        """Test engine works without data augmentation."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        original_x_1 = torch.ones(4, 10)
        batch = {'x_1': original_x_1.clone(), 'x': torch.ones(4, 10)}
        
        processed_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Without augmentation, values should remain unchanged
        assert torch.allclose(processed_batch['x_1'], original_x_1)
    
    def test_data_augmentation_with_no_x_keys(self, mock_model):
        """Test data augmentation when no x_ keys are present."""
        aug_pipeline = MockDataAugmentation(augmentation_value=2.0)
        engine = SelfSupervisedEngine(
            model=mock_model,
            data_augmentation=aug_pipeline
        )
        
        original_x = torch.ones(4, 10)
        original_y = torch.ones(4, 5)
        
        batch = {
            'x': original_x.clone(),
            'y': original_y.clone()
        }
        
        processed_batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # No keys starting with x_, so nothing should be augmented
        assert torch.allclose(processed_batch['x'], original_x)
        assert torch.allclose(processed_batch['y'], original_y)


class TestSelfSupervisedEngineOptimizers:
    """Test optimizer configuration."""
    
    def test_configure_optimizers_default(self, mock_model):
        """Test default optimizer configuration."""
        engine = SelfSupervisedEngine(model=mock_model)
        config = engine.configure_optimizers()
        
        assert 'optimizer' in config
        assert isinstance(config['optimizer'], Adam)
    
    def test_configure_optimizers_custom(self, mock_model):
        """Test custom optimizer configuration."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.01, 'momentum': 0.9}
        )
        config = engine.configure_optimizers()
        
        assert isinstance(config['optimizer'], SGD)
        assert config['optimizer'].param_groups[0]['lr'] == 0.01
        assert config['optimizer'].param_groups[0]['momentum'] == 0.9
    
    def test_configure_optimizers_with_scheduler(self, mock_model):
        """Test optimizer with scheduler configuration."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=Adam,
            optimizer_kwargs={'lr': 1e-3},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 5, 'gamma': 0.5}
        )
        config = engine.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
        assert isinstance(config['lr_scheduler'], StepLR)
    
    def test_configure_optimizers_with_monitor(self, mock_model):
        """Test optimizer with scheduler and monitor metric."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=Adam,
            optimizer_kwargs={'lr': 1e-3},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 5, 'monitor': 'val_loss'}
        )
        config = engine.configure_optimizers()
        
        assert 'monitor' in config
        assert config['monitor'] == 'val_loss'


class TestSelfSupervisedEngineErrorHandling:
    """Test error handling."""
    
    def test_nan_loss_warning(self, simple_batch, capsys):
        """Test NaN loss warning is printed."""
        
        class ModelWithNaNLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 5)
            
            def forward(self, batch):
                return {'output': self.net(batch['x'])}
            
            def loss(self, output, batch, step='train'):
                return torch.tensor(float('nan'))
        
        model = ModelWithNaNLoss()
        engine = SelfSupervisedEngine(model=model)
        
        # Training step should still run but print warning
        loss = engine.training_step(simple_batch, batch_idx=5)
        
        captured = capsys.readouterr()
        assert 'Warning: NaN loss' in captured.out
        assert 'batch 5' in captured.out


class TestSelfSupervisedEngineIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_training_workflow(self, mock_model):
        """Test a complete mini training workflow."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=Adam,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        # Create multiple batches
        batches = [
            {'x': torch.randn(4, 10), 'target': torch.randn(4, 5)}
            for _ in range(3)
        ]
        
        # Run training steps
        losses = []
        for idx, batch in enumerate(batches):
            loss = engine.training_step(batch, batch_idx=idx)
            losses.append(loss.item())
        
        assert len(losses) == 3
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)
    
    def test_train_val_test_workflow(self, mock_model):
        """Test train/val/test workflow."""
        engine = SelfSupervisedEngine(model=mock_model)
        
        batch = {'x': torch.randn(4, 10), 'target': torch.randn(4, 5)}
        
        # All steps should work
        train_loss = engine.training_step(batch, batch_idx=0)
        val_loss = engine.validation_step(batch, batch_idx=0)
        test_loss = engine.test_step(batch, batch_idx=0)
        
        assert all(isinstance(l, torch.Tensor) for l in [train_loss, val_loss, test_loss])


class TestContrastiveEngine:
    """Test ContrastiveEngine augmentation behavior."""
    
    def test_contrastive_creates_x1_x2_with_augmentation(self, mock_data_augmentation):
        """Test ContrastiveEngine creates x_1 and x_2 from x with augmentation."""
        
        class MockContrastiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 5)
            
            def forward(self, batch):
                # Contrastive model expects x_1 and x_2
                return {
                    'z1': self.net(batch['x_1']),
                    'z2': self.net(batch['x_2'])
                }
            
            def loss(self, output, batch, step='train'):
                # Simple contrastive loss
                return torch.mean((output['z1'] - output['z2']) ** 2)
        
        model = MockContrastiveModel()
        engine = ContrastiveEngine(
            model=model,
            data_augmentation=mock_data_augmentation
        )
        
        batch = {'x': torch.randn(4, 10)}
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are created
        assert 'x_1' in batch
        assert 'x_2' in batch
        assert batch['x_1'].shape == (4, 10)
        assert batch['x_2'].shape == (4, 10)
        
        # Check that x_1 and x_2 are different (due to augmentation)
        assert not torch.equal(batch['x_1'], batch['x_2'])
        
        # Original x should still be there
        assert 'x' in batch
    
    def test_contrastive_without_augmentation(self):
        """Test ContrastiveEngine without augmentation copies x to x_1 and x_2."""
        
        class MockContrastiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 5)
            
            def forward(self, batch):
                return {
                    'z1': self.net(batch['x_1']),
                    'z2': self.net(batch['x_2'])
                }
            
            def loss(self, output, batch, step='train'):
                return torch.mean((output['z1'] - output['z2']) ** 2)
        
        model = MockContrastiveModel()
        engine = ContrastiveEngine(model=model, data_augmentation=None)
        
        batch = {'x': torch.randn(4, 10)}
        original_x = batch['x'].clone()
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_1 and x_2 are created and equal to original x
        assert 'x_1' in batch
        assert 'x_2' in batch
        assert torch.equal(batch['x_1'], original_x)
        assert torch.equal(batch['x_2'], original_x)
    
    def test_contrastive_training_step(self, mock_data_augmentation):
        """Test ContrastiveEngine full training step."""
        
        class MockContrastiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 5)
            
            def forward(self, batch):
                return {
                    'z1': self.net(batch['x_1']),
                    'z2': self.net(batch['x_2'])
                }
            
            def loss(self, output, batch, step='train'):
                return torch.mean((output['z1'] - output['z2']) ** 2)
        
        model = MockContrastiveModel()
        engine = ContrastiveEngine(
            model=model,
            data_augmentation=mock_data_augmentation
        )
        
        batch = {'x': torch.randn(4, 10)}
        # Manually call on_after_batch_transfer since we're not using a real trainer
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        loss = engine.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)


class TestReconstructionEngine:
    """Test ReconstructionEngine augmentation behavior."""
    
    def test_reconstruction_creates_x_aug_with_augmentation(self, mock_data_augmentation):
        """Test ReconstructionEngine creates x_aug from x with augmentation."""
        
        class MockReconstructionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 5)
                self.decoder = nn.Linear(5, 10)
            
            def forward(self, batch):
                # Reconstruction model expects x_aug
                z = self.encoder(batch['x_aug'])
                x_recon = self.decoder(z)
                return {'x_recon': x_recon}
            
            def loss(self, output, batch, step='train'):
                # Reconstruction loss
                return torch.mean((output['x_recon'] - batch['x']) ** 2)
        
        model = MockReconstructionModel()
        engine = ReconstructionEngine(
            model=model,
            data_augmentation=mock_data_augmentation
        )
        
        batch = {'x': torch.randn(4, 10)}
        original_x = batch['x'].clone()
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_aug is created
        assert 'x_aug' in batch
        assert batch['x_aug'].shape == (4, 10)
        
        # Check that x_aug is different from original x (due to augmentation)
        assert not torch.equal(batch['x_aug'], original_x)
        
        # Original x should still be there
        assert 'x' in batch
    
    def test_reconstruction_without_augmentation(self):
        """Test ReconstructionEngine without augmentation copies x to x_aug."""
        
        class MockReconstructionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 5)
                self.decoder = nn.Linear(5, 10)
            
            def forward(self, batch):
                z = self.encoder(batch['x_aug'])
                x_recon = self.decoder(z)
                return {'x_recon': x_recon}
            
            def loss(self, output, batch, step='train'):
                return torch.mean((output['x_recon'] - batch['x']) ** 2)
        
        model = MockReconstructionModel()
        engine = ReconstructionEngine(model=model, data_augmentation=None)
        
        batch = {'x': torch.randn(4, 10)}
        original_x = batch['x'].clone()
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        
        # Check that x_aug is created and equal to original x
        assert 'x_aug' in batch
        assert torch.equal(batch['x_aug'], original_x)
    
    def test_reconstruction_training_step(self, mock_data_augmentation):
        """Test ReconstructionEngine full training step."""
        
        class MockReconstructionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 5)
                self.decoder = nn.Linear(5, 10)
            
            def forward(self, batch):
                z = self.encoder(batch['x_aug'])
                x_recon = self.decoder(z)
                return {'x_recon': x_recon}
            
            def loss(self, output, batch, step='train'):
                return torch.mean((output['x_recon'] - batch['x']) ** 2)
        
        model = MockReconstructionModel()
        engine = ReconstructionEngine(
            model=model,
            data_augmentation=mock_data_augmentation
        )
        
        batch = {'x': torch.randn(4, 10)}
        # Manually call on_after_batch_transfer since we're not using a real trainer
        batch = engine.on_after_batch_transfer(batch, dataloader_idx=0)
        loss = engine.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)


class TestEngineBackwardCompatibility:
    """Test backward compatibility with SelfSupervisedEngine alias."""
    
    def test_self_supervised_engine_is_base(self, mock_model):
        """Test that SelfSupervisedEngine is an alias for BaseSelfSupervisedEngine."""
        engine = SelfSupervisedEngine(model=mock_model)
        assert isinstance(engine, BaseSelfSupervisedEngine)
    
    def test_self_supervised_engine_basic_functionality(self, mock_model):
        """Test that SelfSupervisedEngine works as before."""
        engine = SelfSupervisedEngine(
            model=mock_model,
            optimizer=Adam,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        batch = {'x': torch.randn(4, 10), 'target': torch.randn(4, 5)}
        loss = engine.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
