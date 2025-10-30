"""
Model implementations for EDA foundation models.
"""

from edamame.model.vae import VariationalAutoencoder, VAEMedium
from edamame.model.mae import MaskedAutoencoder
from edamame.model.encoder_decoder import EncoderDecoderModel
from edamame.model.matton_constrastive import ContrastiveModel
from edamame.model.efficientnet_contrastive import EfficientNetContrastive, MBConv1D

__all__ = [
    'VariationalAutoencoder',
    'VAEMedium',
    'MaskedAutoencoder',
    'EncoderDecoderModel',
    'ContrastiveModel',
    'EfficientNetContrastive',
    'MBConv1D',
]
