"""
Generic encoder-decoder architecture for reconstruction and generation tasks.
"""

import torch.nn as nn


class EncoderDecoderModel(nn.Module):
    """
    Class for basic encoder-decoder model.
    
    This architecture maps input data through an encoder to a latent representation,
    then reconstructs or generates output through a decoder network.
    """

    def __init__(
        self,
        encoder_net: nn.Module,
        decoder_net: nn.Module,
        loss_fn: nn.Module,
        bottleneck: nn.Module | None = None,
    ):
        """
        Args:
            encoder_net: Network for mapping from raw data to learned representations (latent space)
            decoder_net: Network for mapping from latent representations back to data space
            loss_fn: Loss function for comparing reconstructed output with target
            bottleneck (optional): Network for transforming encoder output (e.g., VAE reparameterization)
            data_augmentation (optional): Data augmentation module to apply to inputs
        """
        super().__init__()
        
        # Initialize encoder and decoder
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.bottleneck = bottleneck
        
        # Store networks in ModuleDict for proper registration
        self.nn_dict = nn.ModuleDict({
            "encoder_net": self.encoder_net,
            "decoder_net": self.decoder_net
        })
        
        # Track dimensions
        self.latent_dim = self.encoder_net.output_dim
        self.output_dim = self.decoder_net.output_dim
        
        if self.bottleneck is not None:
            self.nn_dict["bottleneck"] = self.bottleneck
            self.latent_dim = self.bottleneck.output_dim
        
        self.loss_fn = loss_fn

    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input data
            
        Returns:
            Latent representation
        """
        h = self.encoder_net(x)
        if self.bottleneck is not None:
            h = self.bottleneck(h)
        return h
    
    def decode(self, h):
        """
        Decode latent representation to output space.
        
        Args:
            h: Latent representation
            
        Returns:
            Reconstructed/generated output
        """
        return self.decoder_net(h)

    def forward(self, batch):
        """
        Forward pass through encoder-decoder architecture.
        
        Args:
            batch: Dictionary containing input data with key 'x' (and optionally 'target')
            
        Returns:
            Dictionary with 'reconstruction' and 'latent' representation
        """
        
        # Get input
        x = batch["x_aug"]
        
        # Encode
        h = self.encode(x)
        
        # Decode
        reconstruction = self.decode(h)
        
        return {
            "reconstruction": reconstruction,
            "latent": h
        }

    def loss(self, reconstruction, target, **kwargs):
        """
        Compute loss between reconstruction and target.
        
        Args:
            reconstruction: Output from decoder
            target: Ground truth target
            **kwargs: Additional arguments for loss function
            
        Returns:
            Loss value
        """
        loss = self.loss_fn(reconstruction, target, **kwargs)
        return loss
    
    def get_representation(self, x):
        """
        Get latent representation for input data.
        
        Args:
            x: Input data
            
        Returns:
            Latent representation
        """
        return self.encode(x)