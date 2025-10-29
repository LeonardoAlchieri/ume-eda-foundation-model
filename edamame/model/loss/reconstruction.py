"""
Loss functions for reconstruction learning (autoencoders, MAE, etc.).
"""

import torch
import torch.nn as nn


class DefaultMAELoss(nn.Module):
    """
    Weighted Masked Autoencoder Loss.
    
    Computes MSE loss on both masked and visible patches with configurable weighting.
    
    This implementation is inspired by:
    - MAE (He et al., 2022): Original approach using masked patches only
    - CAE (Chen et al., 2023): Context Autoencoder with weighted reconstruction
    - Partial Autoencoding (Chen et al., 2021): Balancing masked and visible losses
    
    The loss is computed as:
        L = alpha * L_masked + (1 - alpha) * L_visible
    
    Where:
    - alpha = 1.0: Standard MAE (masked patches only)
    - alpha = 0.5: Equal weight to both masked and visible
    - alpha = 0.0: Only visible patches (like standard autoencoder)
    
    For reconstruction tasks, alpha > 0.5 is recommended to emphasize the 
    challenging masked prediction task.
    """
    
    def __init__(self, masked_weight=1.0):
        """
        Initialize the MAE loss module with configurable weights.
        
        Args:
            masked_weight: Weight for loss on masked patches (default: 1.0)
            visible_weight: Weight for loss on visible patches (default: 0.0)
            
        Note:
            - masked_weight=1.0, visible_weight=0.0: Standard MAE
            - masked_weight=1.0, visible_weight=0.5: CAE-style weighted loss
            - masked_weight=1.0, visible_weight=1.0: Equal weighting
            
            The weights are normalized internally to alpha = masked_weight / (masked_weight + visible_weight)
        """
        super(DefaultMAELoss, self).__init__()
        self.masked_weight = masked_weight
    
    def forward(self, pred, target, mask, patch_size, in_channels, num_patches):
        """
        Compute weighted MAE loss on both masked and visible patches.
        
        Args:
            pred: Predicted patches of shape (batch_size, num_patches, patch_size * in_channels)
            target: Original input of shape (batch_size, seq_len, in_channels)
            mask: Binary mask where 1 = removed/masked, 0 = kept/visible
            patch_size: Size of each patch
            in_channels: Number of input channels
            num_patches: Number of patches
            
        Returns:
            Loss value (scalar)
        """
        # Patchify target
        batch_size = target.shape[0]
        target_patches = target.reshape(
            batch_size, num_patches, patch_size * in_channels
        )
        
        # Compute per-patch MSE loss
        loss_per_patch = (pred - target_patches) ** 2
        loss_per_patch = loss_per_patch.mean(dim=-1)  # Mean over patch dimension
        
        # Compute loss on masked patches
        if mask.sum() > 0:
            loss_masked = (loss_per_patch * mask).sum() / mask.sum()
        else:
            loss_masked = 0.0
        
        # Compute loss on visible patches (inverted mask: 1 - mask)
        visible_mask = 1 - mask
        if visible_mask.sum() > 0:
            loss_visible = (loss_per_patch * visible_mask).sum() / visible_mask.sum()
        else:
            loss_visible = 0.0
        
        # Weighted combination
        total_loss = self.masked_weight * loss_masked + (1 - self.masked_weight) * loss_visible
        
        return total_loss, loss_masked, loss_visible


class StandardizedMAELoss(DefaultMAELoss):
    """
    Standardized Masked Autoencoder Loss.
    
    Computes MAE loss on standardized versions of predictions and targets.
    The standardization is performed per-sample based on the input's mean and std.
    
    This approach normalizes the scale of different signals, making the loss
    more robust to amplitude variations and focusing on pattern reconstruction
    rather than absolute value matching.
    
    The standardization is computed as:
        z = (x - mean(x)) / std(x)
    
    where mean and std are computed from the original input signal.
    """
    
    def __init__(self, masked_weight=1.0, eps=1e-8):
        """
        Initialize the standardized MAE loss module.
        
        Args:
            masked_weight: Weight for loss on masked patches (default: 1.0)
            eps: Small constant for numerical stability in standardization (default: 1e-8)
        """
        super(StandardizedMAELoss, self).__init__(masked_weight=masked_weight)
        self.eps = eps
    
    def unpatchify(self, patches, batch_size, num_patches, patch_size, in_channels, seq_len):
        """
        Convert patches back to time series format.
        
        Args:
            patches: Patches of shape (batch_size, num_patches, patch_size * in_channels)
            batch_size: Batch size
            num_patches: Number of patches
            patch_size: Size of each patch
            in_channels: Number of input channels
            seq_len: Original sequence length
            
        Returns:
            Time series of shape (batch_size, seq_len, in_channels)
        """
        # Reshape patches to (batch_size, num_patches, patch_size, in_channels)
        x = patches.reshape(batch_size, num_patches, patch_size, in_channels)
        # Reshape to (batch_size, seq_len, in_channels)
        x = x.reshape(batch_size, seq_len, in_channels)
        return x
    
    def standardize(self, x, reference_mean, reference_std):
        """
        Standardize signal based on reference statistics.
        
        Args:
            x: Signal to standardize of shape (batch_size, seq_len, in_channels)
            reference_mean: Mean values of shape (batch_size, 1, in_channels)
            reference_std: Std values of shape (batch_size, 1, in_channels)
            
        Returns:
            Standardized signal of same shape as input
        """
        return (x - reference_mean) / (reference_std + self.eps)
    
    def forward(self, pred, target, mask, patch_size, in_channels, num_patches):
        """
        Compute weighted MAE loss on standardized patches.
        
        The process:
        1. Unpatchify predictions to get full reconstructed signal
        2. Compute mean and std from the original target signal
        3. Standardize both target and prediction using target's statistics
        4. Re-patchify the standardized signals
        5. Compute loss using parent class's forward method
        
        Args:
            pred: Predicted patches of shape (batch_size, num_patches, patch_size * in_channels)
            target: Original input of shape (batch_size, seq_len, in_channels)
            mask: Binary mask where 1 = removed/masked, 0 = kept/visible
            patch_size: Size of each patch
            in_channels: Number of input channels
            num_patches: Number of patches
            
        Returns:
            Tuple of (total_loss, loss_masked, loss_visible)
        """
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        
        # Step 1: Unpatchify predictions to get full signal
        # pred shape: (batch_size, num_patches, patch_size * in_channels)
        pred_unpatchified = self.unpatchify(
            pred, batch_size, num_patches, patch_size, in_channels, seq_len
        )
        # pred_unpatchified shape: (batch_size, seq_len, in_channels)
        
        # Step 2: Compute statistics from original target
        # Compute per-sample, per-channel statistics
        target_mean = target.mean(dim=1, keepdim=True)  # (batch_size, 1, in_channels)
        target_std = target.std(dim=1, keepdim=True)    # (batch_size, 1, in_channels)
        
        # Step 3: Standardize both signals using target statistics
        target_standardized = self.standardize(target, target_mean, target_std)
        pred_standardized = self.standardize(pred_unpatchified, target_mean, target_std)
        
        # Step 4: Re-patchify the standardized signals
        pred_patches_std = pred_standardized.reshape(
            batch_size, num_patches, patch_size * in_channels
        )
        
        # Step 5: Call parent's forward with standardized patches
        # Note: target_standardized is already in the correct shape for parent's forward
        return super(StandardizedMAELoss, self).forward(
            pred_patches_std, 
            target_standardized, 
            mask, 
            patch_size, 
            in_channels, 
            num_patches
        )

