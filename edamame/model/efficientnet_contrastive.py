"""
EfficientNet-inspired contrastive learning model for time series data.

Based on EfficientNet architecture adapted for 1D signals with MBConv blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish activation function (also known as SiLU)."""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConv1D(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block for 1D signals.
    
    Architecture:
    1. First part: Conv1D -> BatchNorm -> Swish -> Conv1D -> BatchNorm -> Swish
    2. Second part (Squeeze-Excitation): AvgPool -> Conv1D -> Swish -> Conv1D -> Sigmoid
    3. Third part: (First part ⊙ Second part) -> Conv1D -> BatchNorm
    
    Where ⊙ denotes element-wise multiplication.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        se_ratio=0.25,
        dropout=0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            stride: Stride for the first convolution
            expand_ratio: Expansion ratio for hidden dimension
            se_ratio: Squeeze-excitation ratio for SE block
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Calculate expanded channels
        hidden_dim = int(in_channels * expand_ratio)
        
        # Calculate SE channels (squeeze-excitation)
        se_channels = max(1, int(in_channels * se_ratio))
        
        # First part: Expansion and depthwise convolution
        self.expand_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm1d(hidden_dim)
        self.expand_activation = Swish()
        
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=hidden_dim,  # Depthwise convolution
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(hidden_dim)
        self.depthwise_activation = Swish()
        
        # Second part: Squeeze-Excitation block
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_conv1 = nn.Conv1d(hidden_dim, se_channels, kernel_size=1)
        self.se_activation = Swish()
        self.se_conv2 = nn.Conv1d(se_channels, hidden_dim, kernel_size=1)
        self.se_sigmoid = nn.Sigmoid()
        
        # Third part: Project back to output channels
        self.project_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (if input and output dimensions match)
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        
    def forward(self, x):
        """
        Forward pass through MBConv1D block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len // stride)
        """
        identity = x
        
        # First part: Expansion and depthwise convolution
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_activation(out)
        
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_activation(out)
        
        # Second part: Squeeze-Excitation
        se = self.se_pool(out)
        se = self.se_conv1(se)
        se = self.se_activation(se)
        se = self.se_conv2(se)
        se = self.se_sigmoid(se)
        
        # Element-wise multiplication
        out = out * se
        
        # Third part: Project back
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Skip connection
        if self.use_skip:
            out = out + identity
            
        return out


class EfficientNetContrastive(nn.Module):
    """
    EfficientNet-inspired encoder for contrastive learning on 1D signals.
    
    Architecture:
    1. Stem: Conv1D -> BatchNorm -> Swish
    2. MBConv1D blocks (repeated M times)
    3. Head: Conv1D -> BatchNorm -> Swish -> AvgPool1D
    4. Embedding layer
    """
    
    def __init__(
        self,
        in_channels=1,
        seq_len=240,
        stem_channels=32,
        mbconv_channels=64,
        num_mbconv_blocks=4,
        head_channels=128,
        embedding_dim=256,
        loss_fn=None,
        dropout=0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            seq_len: Length of input sequence
            stem_channels: Number of channels after stem convolution
            mbconv_channels: Number of channels in MBConv blocks
            num_mbconv_blocks: Number of MBConv1D blocks (M parameter)
            head_channels: Number of channels in head convolution
            embedding_dim: Dimension of final embedding
            loss_fn: Contrastive loss function (e.g., InfoNCE)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # Stem: Initial convolution
        self.stem_conv = nn.Conv1d(
            in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.stem_activation = Swish()
        
        # MBConv blocks
        self.mbconv_blocks = nn.ModuleList()
        current_channels = stem_channels
        
        for i in range(num_mbconv_blocks):
            # First block may change channels, others maintain the same
            out_channels = mbconv_channels if i == 0 else current_channels
            stride = 2 if i == 0 else 1  # Downsample in first MBConv block
            
            self.mbconv_blocks.append(
                MBConv1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    expand_ratio=4,
                    se_ratio=0.25,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        
        # Head: Final convolution and pooling
        self.head_conv = nn.Conv1d(
            current_channels, head_channels, kernel_size=1, bias=False
        )
        self.head_bn = nn.BatchNorm1d(head_channels)
        self.head_activation = Swish()
        self.head_pool = nn.AdaptiveAvgPool1d(1)
        
        # Embedding layer
        self.embedding = nn.Linear(head_channels, embedding_dim)
        
        # Loss function
        self.loss_fn = loss_fn
        
    def forward_single(self, x):
        """
        Forward pass for a single view.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels), 
            to be changed to (batch_size, in_channels, seq_len)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        if x.dim() != 3:
            # assuming input is (batch_size, seq_len), with a single channel. Adding channel dimension.
            x = x.unsqueeze(2)  # (batch_size, 1, seq_len
        x = x.permute(0, 2, 1)  # Change to (batch_size, in_channels, seq_len)
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        
        # MBConv blocks
        for mbconv in self.mbconv_blocks:
            x = mbconv(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_activation(x)
        x = self.head_pool(x)
        
        # Flatten
        x = x.squeeze(-1)  # (batch_size, head_channels)
        
        # Embedding
        x = self.embedding(x)
        
        return x
    
    def forward(self, x_1, x_2):
        """
        Forward pass for contrastive learning with two views.
        
        Args:
            x_1: First view of shape (batch_size, in_channels, seq_len)
            x_2: Second view of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Tuple of (embedding_1, embedding_2)
        """
        h_1 = self.forward_single(x_1)
        h_2 = self.forward_single(x_2)
        
        return h_1, h_2
    
    def loss(self, output, batch, step='train'):
        """
        Compute contrastive loss.
        
        Args:
            output: Tuple of (h_1, h_2) embeddings
            batch: Input batch (not used for contrastive loss)
            step: Training step indicator
            
        Returns:
            Loss value
        """
        if self.loss_fn is None:
            raise ValueError("Loss function not provided. Please set loss_fn.")
        
        h_1, h_2 = output
        # TODO: Implement ids using user assignment!
        return self.loss_fn(h_1, h_2)
    
    def get_representation(self, x):
        """
        Get representation for a single input (for inference/evaluation).
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.forward_single(x)
