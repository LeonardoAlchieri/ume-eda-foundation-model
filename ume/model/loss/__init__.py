"""
Loss functions for self-supervised learning.

This module provides loss functions for different SSL paradigms:
- Reconstruction losses (MAE, VAE, etc.)
- Contrastive losses (NCE, NT-Xent, etc.)
"""

from ume.model.loss.reconstruction import DefaultMAELoss, StandardizedMAELoss
from ume.model.loss.contrastive import NCELoss

__all__ = [
    "DefaultMAELoss",
    "StandardizedMAELoss",
    "NCELoss",
]
