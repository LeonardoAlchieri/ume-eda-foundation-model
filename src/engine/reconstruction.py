from typing import Any, Optional, Mapping, Type
import pickle

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric, MetricCollection
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


from src.engine.base import BaseSelfSupervisedEngine


class ReconstructionEngine(BaseSelfSupervisedEngine):
    """
    Engine for reconstruction learning (e.g., autoencoders, denoising).
    
    Creates one augmented view (x_aug) from batch['x'] using
    the data augmentation pipeline.
    """
    
    def __init__(self, *args, plot_n_samples: int = 1, **kwargs):
        """
        Args:
            plot_n_samples: Number of samples to plot at validation epoch end
            *args, **kwargs: Arguments passed to BaseSelfSupervisedEngine
        """
        super().__init__(*args, **kwargs)
        self.plot_n_samples = plot_n_samples
        self.validation_batch_for_plotting = None
    
    def forward(self, batch):
        """Forward pass through the model."""
        x_aug = batch['x_aug']
        return self.model(x_aug)

    def validation_step(self, batch, batch_idx):
        """Validation step with batch storage for plotting."""
        # Store first batch for plotting at epoch end
        if batch_idx == 0 and self.validation_batch_for_plotting is None:
            # Store a detached copy of the batch
            self.validation_batch_for_plotting = {
                'x': batch['x'].detach().cpu(),
                'x_aug': batch['x_aug'].detach().cpu(),
            }
        
        # Call parent validation_step
        return super().validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        """
        Hook called at the end of validation epoch.
        Creates visualization plots of original, augmented, and reconstructed signals.
        """
        # Only plot if we have a stored batch
        if self.validation_batch_for_plotting is None:
            return
        
        # Get the stored batch
        batch = self.validation_batch_for_plotting
        x_original = batch['x']
        x_augmented = batch['x_aug']
        
        # Limit to plot_n_samples
        n_samples = min(self.plot_n_samples, x_original.shape[0])
        x_original = x_original[:n_samples]
        x_augmented = x_augmented[:n_samples]
        
        # Move to device and get reconstructions
        device = next(self.model.parameters()).device
        x_augmented_device = x_augmented.to(device)
        
        # Get reconstructions from model
        with torch.no_grad():
            self.model.eval()
            reconstructed = self.model.reconstruct(x_augmented_device)
            reconstructed = reconstructed.detach().cpu()
            reconstructed = reconstructed.reshape(x_original.shape)
        
        # Create plots
        self._plot_reconstructions(x_original, x_augmented, reconstructed)
        
        # Clear the stored batch for next epoch
        self.validation_batch_for_plotting = None
    
    def _plot_reconstructions(self, x_original, x_augmented, x_reconstructed):
        """
        Create and log plots comparing original, augmented, and reconstructed signals.
        
        Args:
            x_original: Original signals, shape (n_samples, T, C)
            x_augmented: Augmented signals, shape (n_samples, T, C)
            x_reconstructed: Reconstructed signals, shape (n_samples, T, C)
        """
        n_samples = x_original.shape[0]
        n_channels = x_original.shape[-1] if len(x_original.shape) == 3 else 1
        
        # Handle 2D case (N, T) -> (N, T, 1)
        if len(x_original.shape) == 2:
            x_original = x_original.unsqueeze(-1)
            x_augmented = x_augmented.unsqueeze(-1)
            x_reconstructed = x_reconstructed.unsqueeze(-1)
        
        for sample_idx in range(n_samples):
            # Create figure with subplots for each channel
            fig, axes = plt.subplots(n_channels, 3, figsize=(15, 4 * n_channels))
            
            # Handle single channel case
            if n_channels == 1:
                axes = axes.reshape(1, -1)
            
            for channel_idx in range(n_channels):
                # Extract channel data
                original_channel = x_original[sample_idx, :, channel_idx].numpy()
                augmented_channel = x_augmented[sample_idx, :, channel_idx].numpy()
                reconstructed_channel = x_reconstructed[sample_idx, :, channel_idx].numpy()
                
                # Plot original
                axes[channel_idx, 0].plot(original_channel)
                axes[channel_idx, 0].set_title(f'Original Signal (Ch {channel_idx})')
                axes[channel_idx, 0].set_xlabel('Time')
                axes[channel_idx, 0].set_ylabel('Value')
                axes[channel_idx, 0].grid(True, alpha=0.3)
                
                # Plot augmented
                axes[channel_idx, 1].plot(augmented_channel)
                axes[channel_idx, 1].set_title(f'Augmented Signal (Ch {channel_idx})')
                axes[channel_idx, 1].set_xlabel('Time')
                axes[channel_idx, 1].set_ylabel('Value')
                axes[channel_idx, 1].grid(True, alpha=0.3)
                
                # Plot reconstructed
                axes[channel_idx, 2].plot(reconstructed_channel)
                axes[channel_idx, 2].set_title(f'Reconstructed Signal (Ch {channel_idx})')
                axes[channel_idx, 2].set_xlabel('Time')
                axes[channel_idx, 2].set_ylabel('Value')
                axes[channel_idx, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log figure using Lightning's logger
            self._log_figure(fig, f'reconstruction_sample_{sample_idx}')
            
            plt.close(fig)
    
    def _log_figure(self, fig, tag):
        """
        Log a matplotlib figure using Lightning's logger.
        
        Args:
            fig: Matplotlib figure to log
            tag: Tag/name for the figure
        """
        # Try different logging methods based on logger type
        if hasattr(self.logger, 'experiment'):
            experiment = self.logger.experiment
            
            # WandB logger
            if hasattr(experiment, 'log'):
                try:
                    import wandb
                    experiment.log({tag: wandb.Image(fig)}, step=self.current_epoch)
                    return
                except (ImportError, AttributeError):
                    pass
            
            # TensorBoard logger
            if hasattr(experiment, 'add_figure'):
                try:
                    experiment.add_figure(tag, fig, global_step=self.current_epoch)
                    return
                except AttributeError:
                    pass
        
        # Fallback: save to file
        if hasattr(self.logger, 'log_dir'):
            import os
            log_dir = self.logger.log_dir
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                filepath = os.path.join(log_dir, f'{tag}_epoch{self.current_epoch}.png')
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {filepath}")

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Creates x_aug from x using data augmentation for reconstruction learning.
        Data augmentation is only applied during training.
        """
        if isinstance(batch, dict) and 'x' in batch:
            # Apply data augmentation to create two views only during training
            if self.scaling_params is not None:
                batch['x'] = self.apply_scaling(batch['x'])
                    
            if self.data_augmentation is not None and self.training:
                # Convert to numpy for augmentation
                x = batch['x']
                x_np = x.cpu().numpy()
                device = x.device

                # Vectorized augmentation function
                augment_fn = np.vectorize(self.data_augmentation, signature='(t)->(t)' if len(x.shape) == 2 else '(t,c)->(t,c)')

                # Apply augmentation in a vectorized way
                batch_aug_1_np = augment_fn(x_np)

                # Convert back to tensor
                batch_aug_1 = torch.from_numpy(batch_aug_1_np).float().to(device)
                
                batch['x_aug'] = batch_aug_1
            else:
                # If no augmentation or not training, just copy x to x_aug
                batch['x_aug'] = batch['x']

            # Add batch size
            batch['batch_size'] = batch['x'].shape[0]
        
        return batch
