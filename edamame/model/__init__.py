"""
Model implementations for EDA foundation models.
"""

from edamame.model.vae import VariationalAutoencoder, VAEMedium
from edamame.model.mae import MaskedAutoencoder
from edamame.model.encoder_decoder import EncoderDecoderModel
from edamame.model.matton_constrastive import ContrastiveModel
from edamame.model.efficientnet_contrastive import EfficientNetContrastive, MBConv1D
from edamame.model.mantis_contrastive import MantisContrastive
from edamame.model.resnet_contrastive import (
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
