"""
ResNet-inspired contrastive learning model for time series data.

Based on ResNet architecture adapted for 1D signals with residual blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """
    Basic residual block for 1D signals.
    
    Architecture:
    Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm -> (+ skip) -> ReLU
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            stride: Stride for the first convolution (for downsampling)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BasicBlock1D.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len // stride)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class BottleneckBlock1D(nn.Module):
    """
    Bottleneck residual block for 1D signals.
    
    Architecture:
    Conv1D(1x1) -> BatchNorm -> ReLU -> Conv1D(kx1) -> BatchNorm -> ReLU -> Conv1D(1x1) -> BatchNorm -> (+ skip) -> ReLU
    
    Uses 1x1 convolutions to reduce and expand dimensions for efficiency.
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            kernel_size: Kernel size for the middle convolution
            stride: Stride for the middle convolution (for downsampling)
            dropout: Dropout rate
        """
        super().__init__()
        
        expanded_channels = out_channels * self.expansion
        
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Main convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv1d(out_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(expanded_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(expanded_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BottleneckBlock1D.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels * expansion, seq_len // stride)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNetContrastive(nn.Module):
    """
    ResNet-inspired encoder for contrastive learning on 1D signals.
    
    Architecture:
    1. Stem: Conv1D -> BatchNorm -> ReLU -> MaxPool1D
    2. Residual stages (each with multiple blocks)
    3. Global Average Pooling
    4. Embedding layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 240,
        stem_channels: int = 64,
        kernel_size: int = 7,
        stem_kernel_size: int = 15,
        stem_stride: int = 2,
        block_type: str = "basic",
        num_blocks: list = None,
        channels: list = None,
        embedding_dim: int = 256,
        loss_fn = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            seq_len: Length of input sequence
            stem_channels: Number of channels after stem convolution
            kernel_size: Kernel size for residual block convolutions
            stem_kernel_size: Kernel size for stem convolution
            stem_stride: Stride for stem convolution
            block_type: Type of residual block ('basic' or 'bottleneck')
            num_blocks: List specifying number of blocks in each stage (e.g., [2, 2, 2, 2] for ResNet-18)
            channels: List specifying output channels for each stage (e.g., [64, 128, 256, 512])
            embedding_dim: Dimension of final embedding
            loss_fn: Contrastive loss function (e.g., NCELoss)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # Default configurations (ResNet-18 style)
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if channels is None:
            channels = [64, 128, 256, 512]
        
        # Choose block type
        if block_type == "basic":
            block_class = BasicBlock1D
        elif block_type == "bottleneck":
            block_class = BottleneckBlock1D
        else:
            raise ValueError(f"Unknown block type: {block_type}. Use 'basic' or 'bottleneck'.")
        
        self.block_class = block_class
        
        # Stem: Initial convolution
        self.stem_conv = nn.Conv1d(
            in_channels,
            stem_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_kernel_size // 2,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Build residual stages
        self.stages = nn.ModuleList()
        current_channels = stem_channels
        
        for stage_idx, (n_blocks, out_channels) in enumerate(zip(num_blocks, channels)):
            # First block of each stage (except first) does downsampling
            stride = 2 if stage_idx > 0 else 1
            
            stage = self._make_stage(
                block_class=block_class,
                in_channels=current_channels,
                out_channels=out_channels,
                num_blocks=n_blocks,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            )
            self.stages.append(stage)
            
            # Update current channels based on block type
            if block_class == BottleneckBlock1D:
                current_channels = out_channels * block_class.expansion
            else:
                current_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Embedding layer
        self.embedding = nn.Linear(current_channels, embedding_dim)
        
        # Loss function
        self.loss_fn = loss_fn
        
        # Initialize weights
        self._init_weights()
    
    def _make_stage(
        self,
        block_class,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        """
        Create a stage with multiple residual blocks.
        
        Args:
            block_class: Block class to use (BasicBlock1D or BottleneckBlock1D)
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks in this stage
            kernel_size: Kernel size for convolutions
            stride: Stride for the first block (for downsampling)
            dropout: Dropout rate
            
        Returns:
            Sequential module containing all blocks
        """
        layers = []
        
        # First block (may downsample)
        layers.append(
            block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            )
        )
        
        # Remaining blocks
        # For bottleneck, input channels are out_channels * expansion
        if block_class == BottleneckBlock1D:
            in_channels = out_channels * block_class.expansion
        else:
            in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dropout=dropout,
                )
            )
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single view.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels),
               to be changed to (batch_size, in_channels, seq_len)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        if x.dim() != 3:
            # Assuming input is (batch_size, seq_len), with a single channel. Adding channel dimension.
            x = x.unsqueeze(2)  # (batch_size, seq_len, 1)
        x = x.permute(0, 2, 1)  # Change to (batch_size, in_channels, seq_len)
        
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)
        
        # Residual stages
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Embedding
        x = self.embedding(x)
        
        return x
    
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """
        Forward pass for contrastive learning with two views.
        
        Args:
            x_1: First view of shape (batch_size, seq_len, in_channels)
            x_2: Second view of shape (batch_size, seq_len, in_channels)
            
        Returns:
            Tuple of (embedding_1, embedding_2)
        """
        h_1 = self.forward_single(x_1)
        h_2 = self.forward_single(x_2)
        
        return h_1, h_2
    
    def loss(self, output, batch, step: str = 'train'):
        """
        Compute contrastive loss.
        
        Args:
            output: Tuple of (h_1, h_2) embeddings
            batch: Input batch
            step: Training step indicator
            
        Returns:
            Loss value
        """
        if self.loss_fn is None:
            raise ValueError("Loss function not provided. Please set loss_fn.")
        
        h_1, h_2 = output
        
        if "user" in batch.keys():
            user_ids = batch["user"]
            return self.loss_fn(h_1, h_2, key_ids=user_ids)
        else:
            return self.loss_fn(h_1, h_2)
    
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get representation for a single input (for inference/evaluation).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.forward_single(x)


# Predefined ResNet configurations
def resnet18_contrastive(**kwargs) -> ResNetContrastive:
    """ResNet-18 configuration for contrastive learning."""
    return ResNetContrastive(
        block_type="basic",
        num_blocks=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
        **kwargs
    )


def resnet34_contrastive(**kwargs) -> ResNetContrastive:
    """ResNet-34 configuration for contrastive learning."""
    return ResNetContrastive(
        block_type="basic",
        num_blocks=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        **kwargs
    )


def resnet50_contrastive(**kwargs) -> ResNetContrastive:
    """ResNet-50 configuration for contrastive learning."""
    return ResNetContrastive(
        block_type="bottleneck",
        num_blocks=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        **kwargs
    )
