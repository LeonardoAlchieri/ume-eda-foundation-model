"""Temporal Neighborhood Coding (TNC) model.

Wraps an arbitrary single-view encoder (by default the
:class:`~edamame.model.efficientnet_contrastive.EfficientNetContrastive` used by
the rest of the UME family) and pairs it with a
:class:`~edamame.model.loss.tnc.TNCLoss` discriminator. Unlike the augmentation
based contrastive model, TNC builds no augmented views: it encodes a reference
window together with temporal neighbor / non-neighbor windows supplied by
:class:`edamame.data.dataset_online.EDAtaTNCOnlineSegmentation` and contrasts them
by temporal proximity.

The encoder is reused verbatim, so a TNC-pretrained encoder is directly
comparable to (and loadable like) the contrastive UME encoder for downstream
feature extraction via ``get_representation``.
"""

import torch
import torch.nn as nn


class TNCModel(nn.Module):
    """Encoder + neighborhood-discriminator model implementing the engine contract.

    Args:
        encoder: Single-view encoder exposing ``get_representation(x) -> (B, d)``
            for ``x`` of shape ``(B, T, C)`` (or ``(B, T)`` for a single channel).
        loss_fn: A :class:`~edamame.model.loss.tnc.TNCLoss` instance (holds the
            learnable discriminator).
    """

    def __init__(self, encoder: nn.Module, loss_fn: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.loss_fn = loss_fn
        # Mirror the encoder's representation size for downstream consumers.
        self.embedding_dim = getattr(encoder, "embedding_dim", None)
        self.output_dim = getattr(encoder, "output_dim", self.embedding_dim)

    def _encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a ``(N, T, C)`` (or ``(N, T)``) batch into ``(N, d)``."""
        return self.encoder.get_representation(x)

    def forward(self, batch: dict) -> dict:
        """Encode reference + neighbor + non-neighbor windows in one pass.

        Encoding everything through a single forward keeps BatchNorm statistics
        consistent across the reference and the sampled windows.
        """
        x = batch["x"]  # (B, T, C) or (B, T)
        neighbors = batch["x_neighbors"]  # (B, M, T, C) or (B, M, T)
        non_neighbors = batch["x_non_neighbors"]  # same shape as neighbors

        b, m = neighbors.shape[0], neighbors.shape[1]
        window_shape = x.shape[1:]  # (T, C) or (T,)

        # Stack as (B, 1 + 2M, *window) then flatten the leading dims for one encode.
        stacked = torch.cat(
            [x.unsqueeze(1), neighbors, non_neighbors], dim=1
        )  # (B, 1 + 2M, *window)
        z = self._encode_flat(stacked.reshape(b * (1 + 2 * m), *window_shape))
        z = z.reshape(b, 1 + 2 * m, -1)

        return {
            "z_ref": z[:, 0],  # (B, d)
            "z_neighbors": z[:, 1 : 1 + m],  # (B, M, d)
            "z_non_neighbors": z[:, 1 + m :],  # (B, M, d)
        }

    def loss(self, output: dict, batch: dict, step: str = "train") -> torch.Tensor:
        return self.loss_fn(
            output["z_ref"],
            output["z_neighbors"],
            output["z_non_neighbors"],
            batch.get("non_neighbor_mask"),
        )

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single window batch ``(B, T, C)`` -> ``(B, d)`` (for inference)."""
        return self.encoder.get_representation(x)

    def get_predictions_for_metrics(self, output, batch):
        """No supervised targets for TNC pretraining."""
        return None, None
