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


class BaseSelfSupervisedEngine(pl.LightningModule):
    """
    Base PyTorch Lightning engine for self-supervised learning.
    
    This engine requires models to implement a `loss(output, batch, step)` method
    for computing the loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        metrics: Optional[Mapping[str, Metric]] = None,
        optimizer: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: Optional[Mapping[str, Any]] = None,
        data_augmentation: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: The model to train (must implement loss method)
            metrics: Dictionary of metrics to track during training/validation/testing
            optimizer: Optimizer class (e.g., torch.optim.Adam)
            optimizer_kwargs: Keyword arguments for optimizer initialization
            scheduler: Learning rate scheduler class (e.g., torch.optim.lr_scheduler.CosineAnnealingLR)
            scheduler_kwargs: Keyword arguments for scheduler initialization
            data_augmentation: Data augmentation pipeline (nn.Module) to apply to batch
        """
        super(BaseSelfSupervisedEngine, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model", "data_augmentation"], logger=False)

        # Store optimizer and scheduler classes and configs
        self.optimizer_class = optimizer or torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or dict()
        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs or dict()
        
        # Store data augmentation pipeline
        self.data_augmentation = data_augmentation

        # Setup metrics
        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

    def forward(self, batch):
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement the forward method.")

    def predict(self, *args, **kwargs):
        """Prediction method."""
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric):
        """Clone and reset a metric."""
        if metric is None:
            return None
        metric = metric.clone()
        metric.reset()
        return metric
    
    def _set_metrics(self, metrics):
        """
        Setup metric collections for train/val/test splits.
        
        Args:
            metrics: Dictionary of metric name -> Metric objects
        """
        # Create metric collections for each split
        valid_metrics = {k: v for k, v in metrics.items() if v is not None}
        
        if valid_metrics:
            self.train_metrics = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in valid_metrics.items()},
                prefix="train/"
            )
            self.val_metrics = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in valid_metrics.items()},
                prefix="val/"
            )
            self.test_metrics = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in valid_metrics.items()},
                prefix="test/"
            )
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None      

    def log_metrics(self, metrics, **kwargs):
        """Log metrics to logger."""
        if metrics is not None:
            self.log_dict(
                metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True, **kwargs
            )

    def log_loss(self, name, loss, **kwargs):
        """Log loss value to logger."""
        self.log(
            name + "_loss",
            loss.detach() if isinstance(loss, torch.Tensor) else loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            **kwargs,
        )
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Applies data augmentation and adds batch_size.
        
        This method should be overridden by subclasses to implement
        specific augmentation strategies.
        """
        if isinstance(batch, dict):
            # Extract batch size from the batch
            if 'x' in batch:
                batch['batch_size'] = batch['x'].shape[0]
            elif len(batch) > 0:
                # Get batch size from first tensor in batch
                first_value = next(iter(batch.values()))
                if isinstance(first_value, torch.Tensor):
                    batch['batch_size'] = first_value.shape[0]
        return batch

    def update_and_log_metrics(self, step, output, batch):
        """
        Update and log metrics for a given step (train/val/test).
        
        Args:
            step: One of 'train', 'val', or 'test'
            output: Model output (dict or tensor)
            batch: Input batch dict
        """
        metric_collection = getattr(self, f"{step}_metrics", None)
        
        if metric_collection is None:
            return
        
        # If model provides method to extract predictions for metrics, use it
        if hasattr(self.model, 'get_predictions_for_metrics'):
            predictions, targets = self.model.get_predictions_for_metrics(output, batch)
            if predictions is not None and targets is not None:
                metric_collection.update(predictions, targets)
                batch_size = batch.get('batch_size', 1)
                self.log_metrics(metric_collection, batch_size=batch_size)

    def shared_step(self, batch, step):
        """
        Shared step for train/val/test.
        Computes forward pass and loss using model's loss method.
        
        Args:
            batch: Input batch
            step: One of 'train', 'val', or 'test'
            
        Returns:
            Tuple of (loss, output)
        """
        # Forward pass through model
        output = self.forward(batch)
        
        # Compute loss using model's loss method
        loss = self.model.loss(output, batch, step=step)
        
        return loss, output

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, output = self.shared_step(batch, step='train')
        
        if torch.isnan(loss).any():
            print(f'Warning: NaN loss at epoch {self.current_epoch}, batch {batch_idx}')
        
        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("train", output, batch)
        
        batch_size = batch.get('batch_size', 1)
        self.log_loss("train", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        val_loss, output = self.shared_step(batch, step='val')
        
        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("val", output, batch)
        
        batch_size = batch.get('batch_size', 1)
        self.log_loss("val", val_loss, batch_size=batch_size)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        test_loss, output = self.shared_step(batch, step='test')
        
        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("test", output, batch)
        
        batch_size = batch.get('batch_size', 1)
        self.log_loss("test", test_loss, batch_size=batch_size)
        return test_loss

    def on_test_epoch_end(self):
        """Hook called at the end of test epoch."""
        if self.test_metrics is not None:
            # Compute and log final test metrics
            final_metrics = self.test_metrics.compute()
            for metric_name, value in final_metrics.items():
                print(f"{metric_name}: {value.item() if isinstance(value, torch.Tensor) else value}")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        cfg = dict()
        
        # Create optimizer with stored class and kwargs
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        cfg["optimizer"] = optimizer
        
        # Setup scheduler if provided
        if self.scheduler_class is not None:
            scheduler_kwargs = self.scheduler_kwargs.copy()
            metric = scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        
        return cfg


class ContrastiveEngine(BaseSelfSupervisedEngine):
    """
    Engine for contrastive learning.
    
    Creates two augmented views (x_1 and x_2) from batch['x'] using
    the data augmentation pipeline.
    """
    
    def forward(self, batch):
        """Forward pass through the model."""
        x_1 = batch['x_1']
        x_2 = batch['x_2']
        return self.model(x_1, x_2)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Creates x_1 and x_2 from x using data augmentation for contrastive learning.
        Data augmentation is only applied during training.
        """
        if isinstance(batch, dict) and 'x' in batch:
            # Apply data augmentation to create two views only during training
            if self.data_augmentation is not None and self.training:
                # Convert to numpy for augmentation
                x = batch['x']
                x_np = x.cpu().numpy()
                device = x.device

                # Vectorized augmentation function
                augment_fn = np.vectorize(self.data_augmentation, signature='(t)->(t)' if len(x.shape) == 2 else '(t,c)->(t,c)')

                # Apply augmentation in a vectorized way
                batch_aug_1_np = augment_fn(x_np)
                batch_aug_2_np = augment_fn(x_np)

                # Convert back to tensor
                batch_aug_1 = torch.from_numpy(batch_aug_1_np).float().to(device)
                batch_aug_2 = torch.from_numpy(batch_aug_2_np).float().to(device)
                
                batch['x_1'] = batch_aug_1
                batch['x_2'] = batch_aug_2
            else:
                # If no augmentation or not training, just copy x to x_1 and x_2
                batch['x_1'] = batch['x']
                batch['x_2'] = batch['x']
            
            # Add batch size
            batch['batch_size'] = batch['x'].shape[0]
        
        return batch


class ReconstructionEngine(BaseSelfSupervisedEngine):
    """
    Engine for reconstruction learning (e.g., autoencoders, denoising).
    
    Creates one augmented view (x_aug) from batch['x'] using
    the data augmentation pipeline.
    """
    
    def __init__(self, *args, plot_n_samples: int = 5, **kwargs):
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
            reconstructed = self.model(x_augmented_device)
            reconstructed = reconstructed[0].detach().cpu()
        
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


# Backward compatibility alias
SelfSupervisedEngine = BaseSelfSupervisedEngine
