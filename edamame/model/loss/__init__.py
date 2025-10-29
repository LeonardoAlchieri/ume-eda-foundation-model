"""
Loss functions for self-supervised learning.

This module provides loss functions for different SSL paradigms:
- Reconstruction losses (MAE, VAE, etc.)
- Contrastive losses (NCE, NT-Xent, etc.)
"""

from edamame.model.loss.reconstruction import DefaultMAELoss
from edamame.model.loss.contrastive import NCELoss, NCELossCorrected

__all__ = [
    "DefaultMAELoss",
    "NCELoss",
    "NCELossCorrected",
]
