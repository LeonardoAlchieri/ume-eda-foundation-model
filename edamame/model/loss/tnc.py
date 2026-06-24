"""Temporal Neighborhood Coding (TNC) loss.

Reference: Tonekaboni, Eytan & Goldenberg, "Unsupervised Representation Learning
for Time Series with Temporal Neighborhood Coding", ICLR 2021
(https://arxiv.org/abs/2106.00750).

TNC trains an encoder together with a small **discriminator** ``D(z_ref, z_other)``
that predicts whether two windows belong to the same temporal neighborhood. The
objective is a Positive-Unlabeled (PU) contrastive loss: neighbors are positives,
while non-neighbors are *unlabeled* (they may still be similar under
non-stationarity), so they are weighted by ``w``::

    L = - E_{neighbor}[ log D(z_ref, z_n) ]
        - E_{non-neighbor}[ (1-w) log(1 - D(z_ref, z_l)) + w log D(z_ref, z_l) ]

Here the non-neighbor term uses the *same-recording* far windows produced by
:class:`edamame.data.dataset_online.EDAtaTNCOnlineSegmentation`. Optionally, extra
*cross-recording* negatives are formed from other samples in the batch; these come
from different recordings/users and are treated as confident negatives (label 0,
no PU weighting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNCLoss(nn.Module):
    """PU contrastive loss with a learnable neighborhood discriminator.

    Args:
        embedding_dim: Dimension ``d`` of the encoder embeddings. The discriminator
            consumes a concatenated pair, i.e. an input of size ``2*d``.
        w: PU weight in ``[0, 1)`` for same-recording non-neighbors (the chance a
            non-neighbor is actually a positive). Default ``0.05`` as in the paper.
        hidden_multiplier: Hidden width of the discriminator as a multiple of ``d``.
        discriminator_dropout: Dropout inside the discriminator MLP.
        cross_recording_negatives: If ``True``, add batch (cross-recording) negatives.
        cross_neg_per_sample: Number of cross-recording negatives per reference.
            ``None`` (default) uses the per-sample neighbor count ``M``.
    """

    def __init__(
        self,
        embedding_dim: int,
        w: float = 0.05,
        hidden_multiplier: int = 4,
        discriminator_dropout: float = 0.5,
        cross_recording_negatives: bool = True,
        cross_neg_per_sample: int | None = None,
    ):
        super().__init__()
        if not (0.0 <= w < 1.0):
            raise ValueError("w must be in [0, 1).")
        self.w = w
        self.cross_recording_negatives = cross_recording_negatives
        self.cross_neg_per_sample = cross_neg_per_sample

        hidden = max(1, hidden_multiplier * embedding_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(discriminator_dropout),
            nn.Linear(hidden, 1),
        )

    def _score(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Discriminator logit for paired embeddings. Shapes broadcast on ``(..., d)``."""
        return self.discriminator(torch.cat([z_a, z_b], dim=-1)).squeeze(-1)

    def forward(
        self,
        z_ref: torch.Tensor,
        z_neighbors: torch.Tensor,
        z_non_neighbors: torch.Tensor,
        non_neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the TNC PU loss.

        Args:
            z_ref: Reference embeddings ``(B, d)``.
            z_neighbors: Neighbor embeddings ``(B, M, d)``.
            z_non_neighbors: Same-recording non-neighbor embeddings ``(B, M, d)``.
            non_neighbor_mask: ``(B, M)`` float mask; ``1`` where the non-neighbor
                window is valid (recording long enough), ``0`` otherwise.
        """
        b, m, _ = z_neighbors.shape
        z_ref_exp = z_ref.unsqueeze(1).expand(-1, m, -1)  # (B, M, d)

        # --- positives: neighbors should be classified as neighbors (label 1) ---
        logit_pos = self._score(z_ref_exp, z_neighbors)  # (B, M)
        loss_pos = F.binary_cross_entropy_with_logits(
            logit_pos, torch.ones_like(logit_pos)
        )

        # --- same-recording non-neighbors: PU-weighted ---
        logit_neg = self._score(z_ref_exp, z_non_neighbors)  # (B, M)
        bce_to_0 = F.binary_cross_entropy_with_logits(
            logit_neg, torch.zeros_like(logit_neg), reduction="none"
        )
        bce_to_1 = F.binary_cross_entropy_with_logits(
            logit_neg, torch.ones_like(logit_neg), reduction="none"
        )
        pu = (1.0 - self.w) * bce_to_0 + self.w * bce_to_1  # (B, M)
        if non_neighbor_mask is not None:
            denom = non_neighbor_mask.sum().clamp(min=1.0)
            loss_neg = (pu * non_neighbor_mask).sum() / denom
        else:
            loss_neg = pu.mean()

        loss = loss_pos + loss_neg

        # --- cross-recording negatives drawn from other batch samples (label 0) ---
        if self.cross_recording_negatives and b > 1:
            k = self.cross_neg_per_sample or m
            k = min(k, b - 1)
            # k distinct non-zero cyclic shifts -> k distinct other samples per ref
            shifts = (torch.randperm(b - 1, device=z_ref.device)[:k] + 1)
            idx = (torch.arange(b, device=z_ref.device).unsqueeze(1) + shifts.unsqueeze(0)) % b
            z_other = z_ref[idx]  # (B, k, d)
            logit_cross = self._score(z_ref.unsqueeze(1).expand(-1, k, -1), z_other)
            loss_cross = F.binary_cross_entropy_with_logits(
                logit_cross, torch.zeros_like(logit_cross)
            )
            loss = loss + loss_cross

        return loss
