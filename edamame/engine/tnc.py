"""PyTorch Lightning engine for Temporal Neighborhood Coding (TNC).

Unlike :class:`~edamame.engine.contrastive.ContrastiveEngine`, TNC uses **no data
augmentation**. The contrastive signal comes entirely from temporal proximity, so
``on_after_batch_transfer`` only stamps the batch size and (optionally) applies
the dataset-level scaling to the reference and the sampled windows. The actual
encoding of reference / neighbor / non-neighbor windows happens inside
:class:`~edamame.model.tnc.TNCModel`.
"""

import torch

from edamame.engine.base import BaseSelfSupervisedEngine


class TNCEngine(BaseSelfSupervisedEngine):
    """Self-supervised engine for the TNC objective (augmentation-free)."""

    def forward(self, batch):
        """Forward pass: the model encodes all windows and returns embeddings."""
        return self.model(batch)

    def get_representation(self, x):
        """Extract a frozen embedding for downstream feature extraction.

        Args:
            x: Input tensor ``(B, T, C)`` or a batch dict with key ``"x"``.
        """
        if isinstance(x, dict):
            x = x["x"]
        self.model.eval()
        with torch.no_grad():
            return self.model.get_representation(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Add batch size and (optionally) scale all window tensors. No augmentation."""
        if not (isinstance(batch, dict) and "x" in batch):
            raise KeyError("Batch dictionary must contain key 'x' for input data.")

        if self.scaling_params is not None:
            batch["x"] = self.apply_scaling(batch["x"])
            if "x_neighbors" in batch:
                batch["x_neighbors"] = self.apply_scaling(batch["x_neighbors"])
            if "x_non_neighbors" in batch:
                batch["x_non_neighbors"] = self.apply_scaling(batch["x_non_neighbors"])

        batch["batch_size"] = batch["x"].shape[0]
        return batch
