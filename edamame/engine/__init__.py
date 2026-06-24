"""
Self-supervised learning engines for training foundation models.

This module provides PyTorch Lightning engines for different SSL paradigms:
- BaseSelfSupervisedEngine: Base class for all engines
- ReconstructionEngine: For reconstruction-based learning (VAE, MAE, etc.)
- ContrastiveEngine: For contrastive learning (SimCLR, etc.)
"""

from edamame.engine.base import BaseSelfSupervisedEngine
from edamame.engine.reconstruction import ReconstructionEngine
from edamame.engine.contrastive import ContrastiveEngine
from edamame.engine.mantis_contrastive import MantisContrastiveEngine
from edamame.engine.tnc import TNCEngine

__all__ = [
    "BaseSelfSupervisedEngine",
    "ReconstructionEngine",
    "ContrastiveEngine",
    "MantisContrastiveEngine",
    "TNCEngine",
]
