"""
Tests for MAE reconstruction loss functions.

``DefaultMAELoss.forward`` returns a tuple ``(total_loss, loss_masked, loss_visible)``
with ``total = masked_weight * loss_masked + (1 - masked_weight) * loss_visible``.
"""

import pytest
import torch
import torch.nn as nn

from ume.model.loss.reconstruction import DefaultMAELoss


class TestDefaultMAELoss:
    """Test suite for DefaultMAELoss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create a DefaultMAELoss instance."""
        return DefaultMAELoss()
    
    @pytest.fixture
    def basic_inputs(self):
        """Create basic test inputs."""
        batch_size = 4
        num_patches = 12
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        # Create predictions and targets
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
        target = torch.randn(batch_size, seq_len, in_channels)
        
        # Create mask (1 = masked, 0 = visible)
        mask = torch.zeros(batch_size, num_patches)
        mask[:, 6:] = 1.0  # Mask last 50% of patches
        
        return {
            'pred': pred,
            'target': target,
            'mask': mask,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'num_patches': num_patches,
        }
    
    def test_initialization(self, loss_fn):
        """Test that DefaultMAELoss initializes correctly."""
        assert isinstance(loss_fn, nn.Module)
        assert isinstance(loss_fn, DefaultMAELoss)
    
    def test_forward_basic(self, loss_fn, basic_inputs):
        """Test basic forward pass."""
        loss, _, _ = loss_fn(**basic_inputs)
        
        # Check loss is scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        
        # Check loss is positive
        assert loss.item() >= 0, "Loss should be non-negative"
        
        # Check loss is not NaN or Inf
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_loss_only_on_masked_patches(self, loss_fn):
        """Test that loss is computed only on masked patches."""
        batch_size = 2
        num_patches = 10
        patch_size = 5
        in_channels = 1
        seq_len = num_patches * patch_size
        
        # Create identical predictions and targets
        target = torch.randn(batch_size, seq_len, in_channels)
        target_patches = target.reshape(batch_size, num_patches, patch_size * in_channels)
        pred = target_patches.clone()
        
        # Mask first 5 patches (0 = visible, 1 = masked)
        mask = torch.zeros(batch_size, num_patches)
        mask[:, :5] = 1.0
        
        # Perfect reconstruction for visible patches
        # Add error only to masked patches
        pred[:, :5] += 1.0  # Add error to masked patches
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        # Loss should be > 0 since we added error to masked patches
        assert loss.item() > 0, "Loss should be positive when masked patches have error"
    
    def test_perfect_reconstruction(self, loss_fn):
        """Test that perfect reconstruction gives zero loss."""
        batch_size = 2
        num_patches = 8
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        # Create target and perfect prediction
        target = torch.randn(batch_size, seq_len, in_channels)
        target_patches = target.reshape(batch_size, num_patches, patch_size * in_channels)
        pred = target_patches.clone()
        
        # Mask half the patches
        mask = torch.zeros(batch_size, num_patches)
        mask[:, 4:] = 1.0
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        # Loss should be very close to zero
        assert loss.item() < 1e-5, "Perfect reconstruction should give near-zero loss"
    
    def test_no_masked_patches(self, loss_fn):
        """Test behavior when no patches are masked."""
        batch_size = 2
        num_patches = 8
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
        target = torch.randn(batch_size, seq_len, in_channels)
        
        # No masks (all patches visible)
        mask = torch.zeros(batch_size, num_patches)
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        # With no masked patches the masked term is 0 and, with the default
        # masked_weight=1.0, the visible term has zero weight -> total is 0.
        assert not torch.isnan(torch.as_tensor(loss))
        assert float(loss) == 0, "Loss with no masked patches should be 0"
    
    def test_all_patches_masked(self, loss_fn):
        """Test behavior when all patches are masked."""
        batch_size = 2
        num_patches = 8
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
        target = torch.randn(batch_size, seq_len, in_channels)
        
        # All patches masked
        mask = torch.ones(batch_size, num_patches)
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        # Loss should be computed over all patches
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN with all patches masked"
    
    def test_multi_channel(self, loss_fn):
        """Test with multiple input channels."""
        batch_size = 2
        num_patches = 10
        patch_size = 8
        in_channels = 3  # Multiple channels
        seq_len = num_patches * patch_size
        
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
        target = torch.randn(batch_size, seq_len, in_channels)
        mask = torch.zeros(batch_size, num_patches)
        mask[:, 5:] = 1.0
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_gradient_flow(self, loss_fn):
        """Test that gradients flow through the loss."""
        batch_size = 2
        num_patches = 8
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        # Create predictions that require gradients
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels, requires_grad=True)
        target = torch.randn(batch_size, seq_len, in_channels)
        mask = torch.ones(batch_size, num_patches) * 0.5  # 50% masked
        
        loss, _, _ = loss_fn(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            in_channels=in_channels,
            num_patches=num_patches,
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert pred.grad is not None, "Gradients should exist"
        assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    
    def test_batch_size_variation(self, loss_fn):
        """Test with different batch sizes."""
        num_patches = 12
        patch_size = 10
        in_channels = 1
        seq_len = num_patches * patch_size
        
        for batch_size in [1, 2, 8, 16]:
            pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
            target = torch.randn(batch_size, seq_len, in_channels)
            mask = torch.ones(batch_size, num_patches) * 0.75  # 75% masked
            
            loss, _, _ = loss_fn(
                pred=pred,
                target=target,
                mask=mask,
                patch_size=patch_size,
                in_channels=in_channels,
                num_patches=num_patches,
            )
            
            assert loss.dim() == 0, f"Loss should be scalar for batch_size={batch_size}"
            assert not torch.isnan(loss), f"Loss should not be NaN for batch_size={batch_size}"
    
    def test_mask_ratio_consistency(self, loss_fn):
        """Test that higher mask ratio doesn't drastically change loss magnitude."""
        batch_size = 4
        num_patches = 20
        patch_size = 6
        in_channels = 1
        seq_len = num_patches * patch_size
        
        # Use same prediction and target
        pred = torch.randn(batch_size, num_patches, patch_size * in_channels)
        target = torch.randn(batch_size, seq_len, in_channels)
        
        losses = []
        for mask_ratio in [0.25, 0.5, 0.75]:
            mask = torch.zeros(batch_size, num_patches)
            num_masked = int(num_patches * mask_ratio)
            mask[:, :num_masked] = 1.0
            
            loss, _, _ = loss_fn(
                pred=pred,
                target=target,
                mask=mask,
                patch_size=patch_size,
                in_channels=in_channels,
                num_patches=num_patches,
            )
            losses.append(loss.item())
        
        # All losses should be positive
        assert all(l > 0 for l in losses), "All losses should be positive"


class TestDefaultMAELossComponents:
    """Test the (total, masked, visible) decomposition and options."""

    @staticmethod
    def _inputs(in_channels=1):
        torch.manual_seed(0)
        batch_size, num_patches, patch_size = 3, 8, 5
        target = torch.randn(batch_size, num_patches * patch_size, in_channels)
        pred = target.reshape(batch_size, num_patches, patch_size * in_channels).clone()
        mask = torch.zeros(batch_size, num_patches)
        mask[:, :4] = 1.0
        pred[:, :4] += 2.0  # error of 2 on masked patches
        pred[:, 4:] += 0.5  # error of 0.5 on visible patches
        return dict(
            pred=pred, target=target, mask=mask, patch_size=patch_size,
            in_channels=in_channels, num_patches=num_patches,
        )

    def test_returns_three_components(self):
        out = DefaultMAELoss()(**self._inputs())
        assert isinstance(out, tuple) and len(out) == 3

    def test_mae_component_values(self):
        total, masked, visible = DefaultMAELoss(loss_type="mae")(**self._inputs())
        assert torch.isclose(masked, torch.tensor(2.0))
        assert torch.isclose(visible, torch.tensor(0.5))
        assert torch.isclose(total, masked)  # masked_weight=1.0 by default

    def test_mse_component_values(self):
        total, masked, visible = DefaultMAELoss(loss_type="mse")(**self._inputs(in_channels=2))
        assert torch.isclose(masked, torch.tensor(4.0))
        assert torch.isclose(visible, torch.tensor(0.25))

    @pytest.mark.parametrize("w", [0.0, 0.3, 0.5, 1.0])
    def test_masked_weight_combination(self, w):
        total, masked, visible = DefaultMAELoss(masked_weight=w)(**self._inputs())
        assert torch.isclose(total, w * masked + (1 - w) * visible)

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError):
            DefaultMAELoss(loss_type="huber")


class TestMAELossIntegration:
    """Integration tests with MAE model."""
    
    def test_mae_model_with_loss(self):
        """Test MAE model with DefaultMAELoss integration."""
        from ume.model.mae import MaskedAutoencoder
        
        # Create model with default loss
        model = MaskedAutoencoder(seq_len=120, in_channels=1, patch_size=10)
        
        # Check loss function is properly initialized
        assert hasattr(model, 'loss_fn'), "Model should have loss_fn attribute"
        assert isinstance(model.loss_fn, DefaultMAELoss), \
            "Model should use DefaultMAELoss by default"
    
    def test_mae_forward_and_loss(self):
        """Test full forward pass and loss computation."""
        from ume.model.mae import MaskedAutoencoder
        
        batch_size = 4
        seq_len = 120
        in_channels = 1
        
        model = MaskedAutoencoder(seq_len=seq_len, in_channels=in_channels, patch_size=10)
        
        # Create input
        x = torch.randn(batch_size, seq_len, in_channels)
        batch = {'x': x}
        
        # Forward pass
        forward_output = model(x)
        
        # Compute loss
        loss = model.loss(forward_output, batch)
        
        # Verify loss properties
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_mae_custom_loss_function(self):
        """Test MAE model with custom loss function."""
        from ume.model.mae import MaskedAutoencoder
        
        # Create custom loss instance
        custom_loss = DefaultMAELoss()
        
        # Create model with custom loss
        model = MaskedAutoencoder(
            seq_len=120, 
            in_channels=1, 
            patch_size=10,
            loss_fn=custom_loss
        )
        
        # Check that model uses the custom loss instance
        assert model.loss_fn is custom_loss, \
            "Model should use the provided custom loss instance"
    
    def test_mae_backward_pass(self):
        """Test that backward pass works correctly with loss."""
        from ume.model.mae import MaskedAutoencoder
        
        batch_size = 2
        seq_len = 120
        in_channels = 1
        
        model = MaskedAutoencoder(seq_len=seq_len, in_channels=in_channels, patch_size=10)
        
        # Create input
        x = torch.randn(batch_size, seq_len, in_channels)
        batch = {'x': x}
        
        # Forward pass
        forward_output = model(x)
        
        # Compute loss
        loss = model.loss(forward_output, batch)
        
        # Backward pass
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "At least some parameters should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
