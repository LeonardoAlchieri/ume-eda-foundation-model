"""
Self-supervised learning engines for training foundation models.

This module provides PyTorch Lightning engines for different SSL paradigms:
- BaseSelfSupervisedEngine: Base class for all engines
- ReconstructionEngine: For reconstruction-based learning (VAE, MAE, etc.)
- ContrastiveEngine: For contrastive learning (SimCLR, etc.)
"""

from src.engine.base import BaseSelfSupervisedEngine
from src.engine.reconstruction import ReconstructionEngine
from src.engine.contrastive import ContrastiveEngine

__all__ = [
    "BaseSelfSupervisedEngine",
    "ReconstructionEngine",
    "ContrastiveEngine",
]
