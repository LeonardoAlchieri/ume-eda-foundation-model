"""
Masked Autoencoder for Time Series Data.

Inspired by MAE (He et al., 2022) but adapted for 1D time series:
- Patches the time series into segments
- Randomly masks a portion of patches
- Encodes visible patches with transformer encoder
- Decodes full sequence (visible + masked) with transformer decoder
- Reconstructs only the masked patches

This approach learns robust representations by forcing the model to reconstruct
missing temporal information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbedding(nn.Module):
    """
    Convert time series into patches and embed them.
    
    For a time series of length T with C channels, creates patches of size P,
    resulting in N = T/P patches.
    """
    
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        """
        Args:
            seq_len: Length of input time series
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = seq_len // patch_size
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            
        Returns:
            Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape into patches: (B, N, P*C) where N = num_patches, P = patch_size
        x = x.reshape(batch_size, self.num_patches, self.patch_size * self.in_channels)
        
        # Project to embedding dimension
        x = self.projection(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with multi-head attention."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block (similar to encoder for MAE)."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for time series data.
    
    Architecture:
    1. Patch embedding: Convert time series into patches
    2. Random masking: Keep only a subset of patches
    3. Encoder: Process visible patches with transformer
    4. Decoder: Reconstruct full sequence including masked patches
    5. Loss: MSE on masked patches only
    
    Key hyperparameters:
    - mask_ratio: Proportion of patches to mask (default: 0.75, following MAE paper)
    - patch_size: Size of each temporal patch
    - encoder_depth: Number of transformer encoder layers
    - decoder_depth: Number of transformer decoder layers
    """
    
    def __init__(
        self,
        seq_len,
        in_channels=1,
        patch_size=10,
        embed_dim=256,
        encoder_depth=6,
        encoder_num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        dropout=0.1
    ):
        """
        Args:
            seq_len: Length of input time series (must be divisible by patch_size)
            in_channels: Number of input channels
            patch_size: Size of each patch
            embed_dim: Encoder embedding dimension
            encoder_depth: Number of encoder layers
            encoder_num_heads: Number of attention heads in encoder
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder layers
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            mask_ratio: Ratio of patches to mask (0.0 to 1.0)
            dropout: Dropout rate
        """
        super().__init__()
        
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(seq_len, patch_size, in_channels, embed_dim)
        
        # Positional embedding for encoder
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=True
        )
        
        # Encoder (only processes visible patches)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, encoder_num_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=True
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head (reconstruct patches)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_channels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following MAE paper."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize mask token
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize patch embedding projection
        nn.init.xavier_uniform_(self.patch_embed.projection.weight)
        nn.init.zeros_(self.patch_embed.projection.bias)
        
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by per-sample shuffling.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
            mask_ratio: Ratio of patches to mask
            
        Returns:
            x_masked: Visible patches only
            mask: Binary mask (0 = keep, 1 = remove)
            ids_restore: Indices to restore original order
        """
        batch_size, num_patches, embed_dim = x.shape
        num_keep = int(num_patches * (1 - mask_ratio))
        
        # Random shuffle
        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only the first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim)
        )
        
        # Generate binary mask: 0 = keep, 1 = remove
        mask = torch.ones([batch_size, num_patches], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def encode(self, x, mask_ratio=None):
        """
        Encode visible patches.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            mask_ratio: Override default mask ratio (optional)
            
        Returns:
            Encoded visible patches, mask, restore indices
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Masking: remove random patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def decode(self, x, ids_restore):
        """
        Decode to reconstruct masked patches.
        
        Args:
            x: Encoded visible patches
            ids_restore: Indices to restore original patch order
            
        Returns:
            Reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        batch_size = x.shape[0]
        mask_tokens = self.mask_token.repeat(
            batch_size, ids_restore.shape[1] - x.shape[1], 1
        )
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2])
        )
        
        # Add positional embedding
        x = x_full + self.decoder_pos_embed
        
        # Apply decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict patches
        x = self.decoder_pred(x)
        
        return x
    
    def forward(self, x, mask_ratio=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            mask_ratio: Override default mask ratio (optional)
            
        Returns:
            Reconstructed patches, mask
        """
        # Encode
        latent, mask, ids_restore = self.encode(x, mask_ratio)
        
        # Decode
        pred = self.decode(latent, ids_restore)
        
        return pred, mask
    
    def loss(self, forward_output, batch, **kwargs):
        """
        Compute MAE loss (MSE on masked patches only).
        
        Args:
            forward_output: Tuple of (predictions, mask)
            batch: Input batch dictionary with 'x' key
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        # Extract input
        if isinstance(batch, dict):
            x = batch['x']
        else:
            x = batch
            
        pred, mask = forward_output
        
        # Patchify target
        batch_size = x.shape[0]
        target = x.reshape(
            batch_size, self.num_patches, self.patch_size * self.in_channels
        )
        
        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch dimension
        
        # Average loss only on removed (masked) patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def unpatchify(self, x):
        """
        Convert patches back to time series.
        
        Args:
            x: Patches of shape (batch_size, num_patches, patch_size * in_channels)
            
        Returns:
            Time series of shape (batch_size, seq_len, in_channels)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches, self.patch_size, self.in_channels)
        x = x.reshape(batch_size, self.seq_len, self.in_channels)
        return x
    
    def reconstruct(self, x, mask_ratio=None):
        """
        Reconstruct input (for visualization).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            mask_ratio: Override default mask ratio (optional)
            
        Returns:
            Reconstruction of shape (batch_size, seq_len, in_channels)
        """
        pred, _ = self.forward(x, mask_ratio)
        return self.unpatchify(pred)
