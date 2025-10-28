"""
Loss functions for reconstruction learning (autoencoders, MAE, etc.).
"""

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
