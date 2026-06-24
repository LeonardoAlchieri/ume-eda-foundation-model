"""
Loss functions for self-supervised learning.

This module provides loss functions for different SSL paradigms:
- Reconstruction losses (MAE, VAE, etc.)
- Contrastive losses (NCE, NT-Xent, etc.)
"""

from edamame.model.loss.reconstruction import DefaultMAELoss, StandardizedMAELoss
from edamame.model.loss.contrastive import NCELoss
from edamame.model.loss.tnc import TNCLoss

__all__ = [
    "DefaultMAELoss",
    "StandardizedMAELoss",
    "NCELoss",
    "TNCLoss",
]
