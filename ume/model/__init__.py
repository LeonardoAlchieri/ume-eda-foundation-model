"""
Model implementations for EDA foundation models.
"""

from ume.model.vae import VariationalAutoencoder, VAEMedium
from ume.model.mae import MaskedAutoencoder
from ume.model.encoder_decoder import EncoderDecoderModel
from ume.model.matton_constrastive import ContrastiveModel
from ume.model.efficientnet_contrastive import EfficientNetContrastive, MBConv1D
from ume.model.mantis_contrastive import MantisContrastive
from ume.model.resnet_contrastive import (
    ResNetContrastive,
    BasicBlock1D,
    BottleneckBlock1D,
    resnet18_contrastive,
    resnet34_contrastive,
    resnet50_contrastive,
)

__all__ = [
    'VariationalAutoencoder',
    'VAEMedium',
    'MaskedAutoencoder',
    'EncoderDecoderModel',
    'ContrastiveModel',
    'EfficientNetContrastive',
    'MBConv1D',
    'MantisContrastive',
    'ResNetContrastive',
    'BasicBlock1D',
    'BottleneckBlock1D',
    'resnet18_contrastive',
    'resnet34_contrastive',
    'resnet50_contrastive',
]
