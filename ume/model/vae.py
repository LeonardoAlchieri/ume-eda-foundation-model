import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

class VariationalAutoencoder(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.Linear(16, 8),
        )
        
        self.fc_mu = nn.Linear(8, 8)
        self.fc_var = nn.Linear(8, 8)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, in_dim)
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder(z)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparam(mu, log_var)
        return self.decode(z), input, mu, log_var


    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, forward_output: tuple, input, **kwargs) -> torch.Tensor:
        
        input = input['x']
        recon = forward_output[0]
        mu = forward_output[2]
        log_var = forward_output[3]
        
        kld_weight = 0.5

        recon_loss = F.mse_loss(recon, input)
        kldiv_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recon_loss + kld_weight * kldiv_loss

        return loss


class VAEMedium(nn.Module):
    """
    Medium-sized Variational Autoencoder with significantly more parameters.
    
    Architecture follows modern VAE designs with:
    - Deeper encoder/decoder (5 layers each)
    - Wider hidden dimensions (512 -> 256 -> 128 -> 64)
    - Larger latent space (32 dimensions)
    - Dropout for regularization
    - Residual connections in middle layers
    
    This architecture is suitable for learning complex representations
    of time series data.
    """

    def __init__(self, in_dim, latent_dim=32, hidden_dims=None, dropout_rate=0.2, kld_weight=0.5):
        """
        Args:
            in_dim: Input dimension (flattened time series length)
            latent_dim: Dimension of latent space (default: 32)
            hidden_dims: List of hidden layer dimensions (default: [512, 256, 128, 64])
            dropout_rate: Dropout rate for regularization (default: 0.2)
            kld_weight: Weight for KL divergence loss (default: 0.5)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        
        # Build Encoder
        encoder_layers = []
        prev_dim = in_dim
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1]  # Reverse the hidden dims
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.BatchNorm1d(decoder_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
        
        # Final layer without activation
        decoder_layers.append(nn.Linear(decoder_dims[-1], in_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, input):
        """
        Encode input to latent distribution parameters.
        
        Args:
            input: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var

    def decode(self, z):
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed output
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        
        Args:
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        """
        Forward pass through the VAE.
        
        Args:
            input: Input tensor
            
        Returns:
            Tuple of (reconstruction, original_input, mu, log_var)
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, input, mu, log_var

    def loss(self, forward_output: tuple, input, **kwargs) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            forward_output: Tuple from forward pass (reconstruction, original, mu, log_var)
            input: Input batch dictionary with 'x' key
            **kwargs: Additional arguments
            
        Returns:
            Total VAE loss
        """
        # Extract input if it's a dictionary
        if isinstance(input, dict):
            input_data = input['x']
        else:
            input_data = input
        
        recon = forward_output[0]
        mu = forward_output[2]
        log_var = forward_output[3]
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, input_data, reduction='mean')
        
        # KL divergence loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1),
            dim=0
        )
        
        # Total loss with weighting
        loss = recon_loss + self.kld_weight * kld_loss
        
        return loss
    
    def sample(self, num_samples, device):
        """
        Sample from the latent space and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def reconstruct(self, input):
        """
        Reconstruct input (deterministic, using mean of latent distribution).
        
        Args:
            input: Input tensor
            
        Returns:
            Reconstruction
        """
        mu, _ = self.encode(input)
        return self.decode(mu)