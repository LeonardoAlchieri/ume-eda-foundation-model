"""
Self-supervised learning engines for training foundation models.

This module provides PyTorch Lightning engines for different SSL paradigms:
- BaseSelfSupervisedEngine: Base class for all engines
- ReconstructionEngine: For reconstruction-based learning (VAE, MAE, etc.)
- ContrastiveEngine: For contrastive learning (SimCLR, etc.)
"""

from ume.engine.base import BaseSelfSupervisedEngine
from ume.engine.reconstruction import ReconstructionEngine
from ume.engine.contrastive import ContrastiveEngine
from ume.engine.mantis_contrastive import MantisContrastiveEngine

__all__ = [
    "BaseSelfSupervisedEngine",
    "ReconstructionEngine",
    "ContrastiveEngine",
    "MantisContrastiveEngine",
]
