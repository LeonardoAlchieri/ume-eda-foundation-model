from typing import Any, Mapping, Optional, Type

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric, MetricCollection


class BaseSelfSupervisedEngine(pl.LightningModule):
    """
    Base PyTorch Lightning engine for self-supervised learning.

    This engine requires models to implement a `loss(output, batch, step)` method
    for computing the loss.
    """

    def __init__(
        self,
        model: nn.Module,
        seed: int,
        metrics: Optional[Mapping[str, Metric]] = None,
        optimizer: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: Optional[Mapping[str, Any]] = None,
        data_augmentation: Optional[nn.Module] = None,
        scaling_params: Optional[dict] = None,
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
        self.seed = seed
        self.model = model
        self.save_hyperparameters(ignore=["model", "data_augmentation"], logger=False)

        # Store optimizer and scheduler classes and configs
        self.optimizer_class = optimizer or torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or dict()
        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs or dict()

        # Store data augmentation pipeline
        self.data_augmentation = data_augmentation
        self.scaling_params = scaling_params

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
                prefix="train/",
            )
            self.val_metrics = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in valid_metrics.items()},
                prefix="val/",
            )
            self.test_metrics = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in valid_metrics.items()},
                prefix="test/",
            )
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

    def log_metrics(self, metrics, **kwargs):
        """Log metrics to logger."""
        if metrics is not None:
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                **kwargs,
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

    def log_figure(self, fig, tag: str, step: Optional[int] = None):
        """
        Log a matplotlib figure using Lightning's logger.
        
        Supports multiple logger types (WandB, TensorBoard, etc.) and falls back
        to saving to file if no compatible logger is found.

        Args:
            fig: Matplotlib figure to log
            tag: Tag/name for the figure
            step: Optional step/epoch number (defaults to self.current_epoch)
        """
        if step is None:
            step = self.current_epoch
            
        # Try different logging methods based on logger type
        if self.logger is None:
            # No logger configured, save to file as fallback
            self._save_figure_to_file(fig, tag, step)
            return
            
        if hasattr(self.logger, "experiment"):
            experiment = self.logger.experiment

            # WandB logger
            if hasattr(experiment, "log"):
                try:
                    import wandb
                    experiment.log({tag: wandb.Image(fig)}, step=step)
                    return
                except (ImportError, AttributeError):
                    pass

            # TensorBoard logger
            if hasattr(experiment, "add_figure"):
                try:
                    experiment.add_figure(tag, fig, global_step=step)
                    return
                except AttributeError:
                    pass

        # Other loggers with log_figure method
        if hasattr(self.logger, "log_figure"):
            try:
                self.logger.log_figure(tag, fig, step)
                return
            except (AttributeError, TypeError):
                pass

        # Fallback: save to file
        self._save_figure_to_file(fig, tag, step)

    def _save_figure_to_file(self, fig, tag: str, step: int):
        """
        Fallback method to save figure to file when logger doesn't support figures.
        
        Args:
            fig: Matplotlib figure to save
            tag: Tag/name for the figure
            step: Step/epoch number
        """
        import os
        
        # Try to get log directory from logger
        log_dir = None
        if hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
            log_dir = self.logger.log_dir
        elif hasattr(self.logger, "save_dir") and self.logger.save_dir is not None:
            log_dir = self.logger.save_dir
        else:
            # Default to current directory
            log_dir = "."
            
        os.makedirs(log_dir, exist_ok=True)
        
        # Create filename with tag and step
        filename = f"{tag.replace('/', '_')}_step{step}.png"
        filepath = os.path.join(log_dir, filename)
        
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {filepath}")

    def apply_scaling(self, x: torch.Tensor) -> torch.Tensor:
        if "mean" not in self.scaling_params or "std" not in self.scaling_params:
            raise KeyError("scaling_params must contain 'mean' and 'std' keys.")
        mean = self.scaling_params["mean"]
        std = self.scaling_params["std"]
        return (x - mean) / std

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Applies data augmentation and adds batch_size.

        This method should be overridden by subclasses to implement
        specific augmentation strategies.
        """
        if isinstance(batch, dict):
            # Extract batch size from the batch
            if "x" in batch:
                batch["batch_size"] = batch["x"].shape[0]
                if self.scaling_params is not None:
                    batch["x"] = self.apply_scaling(batch["x"])
            else:
                raise KeyError("Batch dictionary must contain key 'x' for input data.")
        else:
            raise TypeError(
                "Batch must be a dictionary, with at least the 'x' key; got {}".format(
                    type(batch)
                )
            )

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
        
        predictions, targets = self.model.get_predictions_for_metrics(output, batch)
        if predictions is not None and targets is not None:
            metric_collection.update(predictions, targets)
            batch_size = batch["batch_size"]
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
        loss, output = self.shared_step(batch, step="train")

        if torch.isnan(loss).any():
            print(f"Warning: NaN loss at epoch {self.current_epoch}, batch {batch_idx}")

        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("train", output, batch)

        batch_size = batch["batch_size"]
        self.log_loss("train", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        val_loss, output = self.shared_step(batch, step="val")

        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("val", output, batch)

        batch_size = batch["batch_size"]
        self.log_loss("val", val_loss, batch_size=batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        test_loss, output = self.shared_step(batch, step="test")

        # Update metrics and log
        if output is not None:
            self.update_and_log_metrics("test", output, batch)

        batch_size = batch["batch_size"]
        self.log_loss("test", test_loss, batch_size=batch_size)
        return test_loss

    def on_test_epoch_end(self):
        """Hook called at the end of test epoch."""
        if self.test_metrics is not None:
            # Compute and log final test metrics
            final_metrics = self.test_metrics.compute()
            for metric_name, value in final_metrics.items():
                print(
                    f"{metric_name}: {value.item() if isinstance(value, torch.Tensor) else value}"
                )

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
