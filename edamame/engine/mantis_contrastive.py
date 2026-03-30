from __future__ import annotations

from copy import deepcopy

import torch

from edamame.engine.contrastive import ContrastiveEngine


class _LoaderLengthProxy:
    """Small proxy to provide `len(loader)` for Mantis LR scheduling."""

    def __init__(self, length: int):
        self.length = max(1, int(length))

    def __len__(self) -> int:
        return self.length


class MantisContrastiveEngine(ContrastiveEngine):
    """
    Contrastive engine variant that mirrors the Mantis pretraining loop.

    Key alignments with Mantis trainer:
    - Supports two explicit augmentation functions (augmentation_1, augmentation_2)
    - Applies augmentations in torch on batched tensors
    - Optional per-step Mantis cosine+warmup LR schedule
    """

    def __init__(
        self,
        *args,
        augmentation_1=None,
        augmentation_2=None,
        learning_rate_adjusting: bool = True,
        base_learning_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if augmentation_1 is None and self.data_augmentation is not None:
            augmentation_1 = deepcopy(self.data_augmentation)
        if augmentation_2 is None and self.data_augmentation is not None:
            augmentation_2 = deepcopy(self.data_augmentation)

        self.augmentation_1 = augmentation_1
        self.augmentation_2 = augmentation_2
        self.learning_rate_adjusting = learning_rate_adjusting
        self.base_learning_rate = (
            base_learning_rate
            if base_learning_rate is not None
            else float(self.optimizer_kwargs.get("lr", 2e-3))
        )
        self._mantis_step = 0

    @staticmethod
    def _to_channel_first(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Convert (N, T, C) to (N, C, T) when needed."""
        if x.ndim == 3 and x.shape[-1] <= 16 and x.shape[1] > x.shape[-1]:
            return x.transpose(1, 2), True
        return x, False

    @staticmethod
    def _to_time_first(x: torch.Tensor, was_transposed: bool) -> torch.Tensor:
        if was_transposed:
            return x.transpose(1, 2)
        return x

    def _apply_augmentation_pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the two Mantis-style augmentation functions.

        Expects x in torch tensor format.
        """
        x_aug_input, transposed = self._to_channel_first(x)

        if self.augmentation_1 is None:
            x_1 = x_aug_input
        else:
            x_1 = self.augmentation_1(x_aug_input)

        if self.augmentation_2 is None:
            x_2 = x_aug_input
        else:
            x_2 = self.augmentation_2(x_aug_input)

        x_1 = self._to_time_first(x_1, transposed)
        x_2 = self._to_time_first(x_2, transposed)

        return x_1.float().to(x.device), x_2.float().to(x.device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Build contrastive views with Mantis-compatible torch augmentations.
        """
        if isinstance(batch, dict) and "x" in batch:
            x = batch["x"]

            if self.scaling_params is not None:
                x = self.apply_scaling(x)
                batch["x"] = x

            if self.training:
                x_1, x_2 = self._apply_augmentation_pair(x)
                batch["x_1"] = x_1
                batch["x_2"] = x_2
            else:
                batch["x_1"] = x
                batch["x_2"] = x

            batch["batch_size"] = x.shape[0]

        return batch

    def on_train_batch_start(self, batch, batch_idx):
        """
        Reuse Mantis per-step LR scheduling when available.
        """
        super().on_train_batch_start(batch, batch_idx)

        if not self.learning_rate_adjusting:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return

        try:
            from mantis.trainer.trainer_utils.scheduling import adjust_learning_rate
        except ImportError as exc:
            raise ImportError(
                "Mantis LR scheduling requires mantis-tsfm to be installed."
            ) from exc

        num_batches = trainer.num_training_batches
        if isinstance(num_batches, float):
            num_batches = int(num_batches)
        loader_proxy = _LoaderLengthProxy(num_batches)

        optimizer = optimizers[0]
        adjust_learning_rate(
            trainer.max_epochs,
            optimizer,
            loader_proxy,
            self._mantis_step,
            self.base_learning_rate,
        )
        self._mantis_step += 1
