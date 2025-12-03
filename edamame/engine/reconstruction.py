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

matplotlib.use("Agg")  # Non-interactive backend for server environments


from edamame.engine.base import BaseSelfSupervisedEngine


class ReconstructionEngine(BaseSelfSupervisedEngine):
    """
    Engine for reconstruction learning (e.g., autoencoders, denoising).

    Creates one augmented view (x_aug) from batch['x'] using
    the data augmentation pipeline.
    """

    def __init__(self, *args, plot_n_samples: int = 3, **kwargs):
        """
        Args:
            plot_n_samples: Number of samples to plot at validation epoch end
            *args, **kwargs: Arguments passed to BaseSelfSupervisedEngine
        """
        super().__init__(*args, **kwargs)
        self.plot_n_samples = plot_n_samples
        self.validation_batch_for_plotting = []
        self.samples_batch_idxs = []
        self.samples_batch_within_batch_idx = []
        self.samples_idxs = []

    def forward(self, batch):
        """Forward pass through the model."""
        x_aug = batch["x_aug"]
        return self.model(x_aug)

    def encode(self, x, mask_ratio=0.0) -> tuple:
        """
        Extract encoder embeddings from input data.
        
        This method extracts latent representations from the model's encoder
        without going through the decoder. Useful for downstream tasks or
        feature extraction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
               Can also be a batch dict with 'x' key
            mask_ratio: Ratio of patches to mask (default: 0.0 for full encoding).
                       Set to 0.0 to encode all patches without masking.
        
        Returns:
            Encoder embeddings. The exact shape depends on the model architecture:
            - For MAE: (batch_size, num_visible_patches, embed_dim) if mask_ratio > 0
                      or (batch_size, num_patches, embed_dim) if mask_ratio = 0
            - For other models: model-specific embedding format
        
        Example:
            >>> engine = ReconstructionEngine(model, ...)
            >>> embeddings = engine.encode(batch_data)
            >>> # Or with masking
            >>> embeddings = engine.encode(batch_data, mask_ratio=0.75)
        """
        # Handle batch dict format
        if isinstance(x, dict):
            x = x["x"]
        
        # Ensure model is in eval mode
        self.model.eval()
        
        with torch.no_grad():
            # Check if model has an encode method
            if hasattr(self.model, "encode"):
                # For MAE-like models with explicit encode method
                if mask_ratio > 0:
                    # Return embeddings with mask info
                    embeddings, mask, ids_restore = self.model.encode(x, mask_ratio=mask_ratio)
                    return embeddings, mask, ids_restore
                else:
                    # Return just embeddings when no masking
                    embeddings, _, _ = self.model.encode(x, mask_ratio=mask_ratio)
                    return embeddings, None, None
            elif hasattr(self.model, "get_representation"):
                # For models with get_representation method (e.g., EfficientNet)
                return self.model.get_representation(x)
            else:
                raise NotImplementedError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "'encode' or 'get_representation' method for embedding extraction."
                )

    def validation_step(self, batch, batch_idx):
        """Validation step with batch storage for plotting."""
        # Store first batch for plotting at epoch end
        if batch_idx in self.samples_batch_idxs:
            within_batch_idx = self.samples_batch_within_batch_idx[self.samples_batch_idxs.index(batch_idx)]
            self.validation_batch_for_plotting.append(batch['x'][within_batch_idx].detach().cpu())
            

        output = self.forward(batch)

        # Compute loss using model's loss method
        val_loss, val_loss_masked, val_loss_visible = self.model.loss_components(
            output, batch, step="val"
        )

        if output is not None:
            self.update_and_log_metrics("val", output, batch)

        batch_size = batch["batch_size"]
        self.log_loss("val", val_loss, batch_size=batch_size)
        self.log_loss("val_masked", val_loss_masked, batch_size=batch_size)
        self.log_loss("val_visible", val_loss_visible, batch_size=batch_size)

        # Call parent validation_step
        return val_loss

    def on_validation_epoch_end(self):
        """
        Hook called at the end of validation epoch.
        Creates visualization plots of original, masked, and reconstructed signals.
        """
        # Only plot if we have a stored batch
        if len(self.validation_batch_for_plotting) == 0:
            return

        # Get the stored batch
        x_original = torch.stack(self.validation_batch_for_plotting, axis=0)
        epoch = self.current_epoch

        # Move to device and get reconstructions
        device = next(self.model.parameters()).device
        x_original_device = x_original.to(device)

        # Get reconstructions and mask from model
        with torch.no_grad():
            self.model.eval()

            # Check if model has reconstruct method that returns mask
            output = self.model.reconstruct(x_original_device)
    
                # Handle different return types
            if isinstance(output, tuple) and len(output) == 2:
                reconstructed, mask = output
                mask = mask.detach().cpu()
            else:
                reconstructed = output
                mask = None

            reconstructed = reconstructed.detach().cpu()

            # Reshape if needed
            if reconstructed.shape != x_original.shape:
                reconstructed = reconstructed.reshape(x_original.shape)

        # Create plots
        self._plot_reconstructions(x_original, reconstructed, epoch, mask)

        # Clear the stored batch for next epoch
        self.validation_batch_for_plotting = []

    def _plot_reconstructions(
        self, x_original, x_reconstructed, epoch, mask=None
    ):
        """
        Create and log overlay plots of original signal with masked regions and reconstruction.

        Args:
            x_original: Original signals, shape (n_samples, T) or (n_samples, T, C)
            x_reconstructed: Reconstructed signals, shape (n_samples, T) or (n_samples, T, C)
            mask: Optional binary mask, shape (n_samples, num_patches) where 1 = masked, 0 = visible
            masked_input: Optional masked input showing which patches were masked
        """
        
        n_samples = x_original.shape[0]

        # Handle 2D case (N, T) -> (N, T, 1)
        if len(x_original.shape) == 2:
            x_original = x_original.unsqueeze(-1)
            x_reconstructed = x_reconstructed.unsqueeze(-1)

        n_channels = x_original.shape[-1]

        for sample_idx in range(n_samples):
            # Create figure with two columns: original only (left) and overlay (right)
            fig, axes = plt.subplots(n_channels, 2, figsize=(20, 4 * n_channels))

            # Handle single channel case
            if n_channels == 1:
                axes = axes.reshape(1, -1)

            for channel_idx in range(n_channels):
                # Extract channel data
                original_channel = x_original[sample_idx, :, channel_idx].numpy()
                reconstructed_channel = x_reconstructed[
                    sample_idx, :, channel_idx
                ].numpy()

                # Create time axis
                time_steps = np.arange(len(original_channel))

                # === LEFT PLOT: Original Signal Only ===
                ax_orig = axes[channel_idx, 0]

                ax_orig.plot(
                    time_steps, original_channel, color="blue", linewidth=2, alpha=0.8
                )

                title_orig = f"Original Signal"
                if n_channels > 1:
                    title_orig += f" (Channel {channel_idx})"
                ax_orig.set_title(title_orig, fontsize=13, fontweight="bold")
                ax_orig.set_xlabel("Time Step", fontsize=11)
                ax_orig.set_ylabel("Value", fontsize=11)
                ax_orig.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

                # === RIGHT PLOT: Overlay with Masking and Reconstruction ===
                ax_overlay = axes[channel_idx, 1]

                # Highlight masked regions if mask is available
                if mask is not None and hasattr(self.model, "patch_size"):
                    patch_size = self.model.patch_size
                    num_patches = self.model.num_patches
                    sample_mask = mask[sample_idx].numpy()  # (num_patches,)

                    # Shade masked patches
                    for patch_idx in range(num_patches):
                        if sample_mask[patch_idx] == 1:  # Masked patch
                            start_idx = patch_idx * patch_size
                            end_idx = (patch_idx + 1) * patch_size
                            ax_overlay.axvspan(
                                start_idx,
                                end_idx,
                                alpha=0.25,
                                color="red",
                                label=(
                                    "Masked Region"
                                    if patch_idx == sample_mask.argmax()
                                    else ""
                                ),
                            )

                # Plot original signal (semi-transparent blue)
                ax_overlay.plot(
                    time_steps,
                    original_channel,
                    label="Original",
                    color="blue",
                    linewidth=2.5,
                    alpha=0.6,
                    zorder=1,
                )


                # Plot reconstruction on top (solid green)
                ax_overlay.plot(
                    time_steps,
                    reconstructed_channel,
                    label="Reconstruction",
                    color="green",
                    linewidth=2.5,
                    alpha=0.9,
                    zorder=3,
                )

                # Formatting for overlay plot
                title_overlay = f"Reconstruction Overlay"
                if n_channels > 1:
                    title_overlay += f" (Channel {channel_idx})"
                ax_overlay.set_title(title_overlay, fontsize=13, fontweight="bold")
                ax_overlay.set_xlabel("Time Step", fontsize=11)
                ax_overlay.set_ylabel("Value", fontsize=11)

                # Create legend with unique labels only
                handles, labels = ax_overlay.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_overlay.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="best",
                    fontsize=10,
                    framealpha=0.9,
                )

                ax_overlay.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

                # Add text annotation with reconstruction quality
                mse = np.mean((original_channel - reconstructed_channel) ** 2)
                ax_overlay.text(
                    0.02,
                    0.98,
                    f"MSE: {mse:.4f}",
                    transform=ax_overlay.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            plt.tight_layout()

            
            # Log figure using Lightning's logger
            self.log_figure(fig, f"reconstruction_overlay_sample_{sample_idx}", epoch)

            plt.close(fig)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Creates x_aug from x using data augmentation for reconstruction learning.
        Data augmentation is only applied during training.
        """
        if isinstance(batch, dict) and "x" in batch:
            # Apply data augmentation to create two views only during training
            if self.scaling_params is not None:
                batch["x"] = self.apply_scaling(batch["x"])

            if self.data_augmentation is not None and self.training:
                # Convert to numpy for augmentation
                x = batch["x"]
                x_np = x.cpu().numpy()
                device = x.device

                # Vectorized augmentation function
                augment_fn = np.vectorize(
                    self.data_augmentation,
                    signature="(t)->(t)" if len(x.shape) == 2 else "(t,c)->(t,c)",
                )

                # Apply augmentation in a vectorized way
                batch_aug_1_np = augment_fn(x_np)

                # Convert back to tensor
                batch_aug_1 = torch.from_numpy(batch_aug_1_np).float().to(device)

                batch["x_aug"] = batch_aug_1
            else:
                # If no augmentation or not training, just copy x to x_aug
                batch["x_aug"] = batch["x"]

            # Add batch size
            batch["batch_size"] = batch["x"].shape[0]

        return batch

    def on_train_start(self) -> None:
        """Hook called at the start of training (after dataloaders are set up)."""
        # Call parent hook if it exists
        if hasattr(super(), 'on_train_start'):
            super().on_train_start()
        
        # Prepare random indices for plotting samples
        if self.plot_n_samples > 0:
            # Access validation dataloader (guaranteed to exist at this point)
            val_dataloader = self.trainer.val_dataloaders
            
            # Handle list of dataloaders
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]
            
            # Get dataset size and batch size
            val_dataset = val_dataloader.dataset
            dataset_size = len(val_dataset)
            batch_size = val_dataloader.batch_size
            
            n_samples = min(self.plot_n_samples, dataset_size)
            self.samples_random_idxs = torch.randperm(dataset_size)[:n_samples]
            
            # Calculate which batch each sample belongs to
            # batch_idx = sample_idx // batch_size
            # within_batch_idx = sample_idx % batch_size
            for sample_idx in self.samples_random_idxs:
                batch_idx = sample_idx.item() // batch_size
                within_batch_idx = sample_idx.item() % batch_size
                
                self.samples_batch_idxs.append(batch_idx)
                self.samples_batch_within_batch_idx.append(within_batch_idx)
                self.samples_idxs.append(sample_idx.item())
        else:
            self.samples_random_idxs = torch.tensor([], dtype=torch.long)
            self.samples_batch_mapping = []
