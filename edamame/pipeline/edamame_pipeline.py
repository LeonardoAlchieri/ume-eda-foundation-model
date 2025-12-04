"""
EdamamePipeline: Inference API for pretrained models.

This module provides a unified interface for loading and using pretrained models
for embedding extraction and inference tasks.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any, Callable
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import yaml


class EdamamePipeline:
    """
    Pipeline for inference with pretrained models.

    This class provides a convenient interface to load pretrained models
    and extract embeddings from input time series data.

    Example:
        >>> # Example with default (no normalization)
        >>> pipeline = EdamamePipeline.from_pretrained(
        ...     model_name="mae",
        ...     weights_path="checkpoints/best.ckpt",
        ...     model_config="configs/train/model/mae.yaml",
        ...     device_map="auto",
        ... )
        >>> embeddings = pipeline.embed(batch_data)
        >>> 
        >>> # Example with custom normalization function
        >>> def z_normalize(x):
        ...     mean = x.mean(dim=1, keepdim=True)
        ...     std = x.std(dim=1, unbiased=False, keepdim=True)
        ...     return (x - mean) / (std + 1e-8)
        >>> 
        >>> pipeline = EdamamePipeline.from_pretrained(
        ...     model_name="mae",
        ...     weights_path="checkpoints/best.ckpt",
        ...     normalization_fn=z_normalize,
        ...     device_map="auto",
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str | None,
        device: torch.device,
        normalization_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            model: The loaded model
            model_name: Name of the model architecture
            device: Device to run inference on
            normalization_fn: Optional normalization function to apply to inputs.
                            Should accept a torch.Tensor and return a torch.Tensor.
                            If None, no normalization is applied.
            model_config: Model configuration dictionary
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model_config = model_config or {}
        # self.normalization_fn = normalization_fn

        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        weights_path: Union[str, Path],
        model_config: Optional[Union[str, Path, Dict, DictConfig]] = None,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        normalization_fn: Optional[Callable] = None,
    ) -> "EdamamePipeline":
        """
        Load a pretrained model from checkpoint.

        Args:
            model_name: Name of the model architecture (e.g., "mae", "efficientnet")
            weights_path: Path to the checkpoint file (.ckpt)
            model_config: Path to model config file or config dict.
                         If None, will try to extract from checkpoint.
            device_map: Device mapping strategy:
                       - "auto": automatically select device (MPS > CUDA > CPU)
                       - "cpu": use CPU
                       - "cuda": use CUDA GPU
                       - "cuda:0": use specific CUDA GPU
                       - "mps": use Apple Silicon GPU
            torch_dtype: Data type for model parameters (e.g., torch.float32, torch.float16)
            normalization_fn: Optional normalization function to apply to inputs.
                            Should accept a torch.Tensor and return a torch.Tensor.
                            If None, no normalization is applied.
                            Example: lambda x: (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        Returns:
            EdamamePipeline instance ready for inference

        Raises:
            ValueError: If model_name is not supported
            FileNotFoundError: If weights_path or model_config path doesn't exist
        """
        # Convert paths to Path objects
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        # Determine device
        device = cls._get_device(device_map)

        # Load model configuration
        config_dict = cls._load_model_config(model_config, model_name)

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Build model based on model_name
        if model_name.lower() == "mae":
            model = cls._load_mae_model(config_dict, checkpoint, torch_dtype)
        elif model_name.lower() == "efficientnet":
            model = cls._load_efficientnet_model(config_dict, checkpoint, torch_dtype)
        else:
            raise ValueError(
                f"Model '{model_name}' not supported. Currently supported: 'mae', 'efficientnet'."
            )

        # Create pipeline instance
        return cls(
            model=model,
            model_name=model_name,
            device=device,
            model_config=config_dict,
            normalization_fn=normalization_fn,
        )

    @staticmethod
    def _get_device(device_map: str) -> torch.device:
        """
        Determine the device based on device_map string.

        Args:
            device_map: Device mapping string

        Returns:
            torch.device object
        """
        if device_map == "auto":
            # Priority: MPS (Apple Silicon) > CUDA > CPU
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif device_map == "cpu":
            return torch.device("cpu")
        elif device_map == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            else:
                raise RuntimeError(
                    "MPS device requested but not available. "
                    "Make sure you're running on Apple Silicon with PyTorch >= 1.12"
                )
        elif device_map.startswith("cuda"):
            return torch.device(device_map)
        else:
            raise ValueError(f"Invalid device_map: {device_map}")

    @staticmethod
    def _load_model_config(
        model_config: Optional[Union[str, Path, Dict, DictConfig]],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Load model configuration from various sources.

        Args:
            model_config: Configuration source
            model_name: Name of the model

        Returns:
            Configuration dictionary
        """
        if model_config is None:
            # Return empty dict, will use defaults
            return {}

        if isinstance(model_config, (dict, DictConfig)):
            # Already a dictionary
            if isinstance(model_config, DictConfig):
                return OmegaConf.to_container(model_config, resolve=True)
            return model_config

        # Load from file
        config_path = Path(model_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Remove hydra-specific keys
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        return config_dict

    @staticmethod
    def _load_mae_model(
        config_dict: Dict[str, Any],
        checkpoint: Dict[str, Any],
        torch_dtype: Optional[torch.dtype] = None,
    ) -> nn.Module:
        """
        Load Masked Autoencoder model.

        Args:
            config_dict: Model configuration
            checkpoint: Loaded checkpoint
            torch_dtype: Data type for model

        Returns:
            Loaded MAE model
        """
        from edamame.model.mae import MaskedAutoencoder

        # Extract state dict - handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present (from Lightning checkpoints)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Infer model parameters from state dict if config is empty
        if not config_dict:
            config_dict = EdamamePipeline._infer_mae_config_from_state_dict(state_dict)

        # Create model
        model = MaskedAutoencoder(**config_dict)

        # Load weights
        model.load_state_dict(state_dict, strict=False)

        # Convert dtype if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model

    @staticmethod
    def _load_efficientnet_model(
        config_dict: Dict[str, Any],
        checkpoint: Dict[str, Any],
        torch_dtype: Optional[torch.dtype] = None,
    ) -> nn.Module:
        """
        Load EfficientNet contrastive model.

        Args:
            config_dict: Model configuration
            checkpoint: Loaded checkpoint
            torch_dtype: Data type for model

        Returns:
            Loaded EfficientNet model
        """
        from edamame.model.efficientnet_contrastive import EfficientNetContrastive

        # Extract state dict - handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present (from Lightning checkpoints)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Infer model parameters from state dict if config is empty
        if not config_dict:
            config_dict = EdamamePipeline._infer_efficientnet_config_from_state_dict(state_dict)

        # Create model
        model = EfficientNetContrastive(**config_dict)

        # Load weights
        model.load_state_dict(state_dict, strict=False)

        # Convert dtype if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model

    @staticmethod
    def _infer_efficientnet_config_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Infer EfficientNet configuration from state dictionary.

        Args:
            state_dict: Model state dictionary

        Returns:
            Inferred configuration dictionary
        """
        config = {}

        # Infer stem channels
        if "stem_conv.weight" in state_dict:
            config["in_channels"] = state_dict["stem_conv.weight"].shape[1]
            config["stem_channels"] = state_dict["stem_conv.weight"].shape[0]

        # Infer head channels
        if "head_conv.weight" in state_dict:
            config["head_channels"] = state_dict["head_conv.weight"].shape[0]

        # Infer embedding dimension
        if "embedding.weight" in state_dict:
            config["embedding_dim"] = state_dict["embedding.weight"].shape[0]

        # Count MBConv blocks
        num_mbconv_blocks = 0
        for key in state_dict.keys():
            if key.startswith("mbconv_blocks."):
                layer_idx = int(key.split(".")[1])
                num_mbconv_blocks = max(num_mbconv_blocks, layer_idx + 1)

        if num_mbconv_blocks > 0:
            config["num_mbconv_blocks"] = num_mbconv_blocks

        # Infer mbconv_channels from first MBConv block
        if "mbconv_blocks.0.expand_conv.weight" in state_dict:
            config["mbconv_channels"] = state_dict["mbconv_blocks.0.project_conv.weight"].shape[0]

        # Set default values
        config.setdefault("seq_len", 240)
        config.setdefault("dropout", 0.1)

        return config

    @staticmethod
    def _infer_mae_config_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Infer MAE configuration from state dictionary.

        Args:
            state_dict: Model state dictionary

        Returns:
            Inferred configuration dictionary
        """
        config = {}

        # Infer from positional embeddings
        if "pos_embed" in state_dict:
            num_patches = state_dict["pos_embed"].shape[1]
            embed_dim = state_dict["pos_embed"].shape[2]
            config["embed_dim"] = embed_dim

        # Infer decoder dimensions
        if "decoder_pos_embed" in state_dict:
            decoder_embed_dim = state_dict["decoder_pos_embed"].shape[2]
            config["decoder_embed_dim"] = decoder_embed_dim

        # Infer patch size and channels from decoder prediction layer
        if "decoder_pred.weight" in state_dict:
            patch_channels_product = state_dict["decoder_pred.weight"].shape[0]
            # Assume default values if we can't infer
            config["patch_size"] = 10  # default
            config["in_channels"] = 1  # default

        # Infer sequence length from number of patches (assuming patch_size=10)
        if "pos_embed" in state_dict and "patch_size" in config:
            num_patches = state_dict["pos_embed"].shape[1]
            config["seq_len"] = num_patches * config["patch_size"]

        # Count encoder and decoder layers
        encoder_depth = 0
        decoder_depth = 0
        for key in state_dict.keys():
            if key.startswith("encoder_blocks."):
                layer_idx = int(key.split(".")[1])
                encoder_depth = max(encoder_depth, layer_idx + 1)
            elif key.startswith("decoder_blocks."):
                layer_idx = int(key.split(".")[1])
                decoder_depth = max(decoder_depth, layer_idx + 1)

        if encoder_depth > 0:
            config["encoder_depth"] = encoder_depth
        if decoder_depth > 0:
            config["decoder_depth"] = decoder_depth

        # Set default values for other parameters
        config.setdefault("encoder_num_heads", 8)
        config.setdefault("decoder_num_heads", 8)
        config.setdefault("mlp_ratio", 4.0)
        config.setdefault("mask_ratio", 0.75)
        config.setdefault("dropout", 0.1)

        return config

    def embed(
        self,
        x: Union[torch.Tensor, Any],
        mask_ratio: float = 0.0,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Extract embeddings from input time series.

        For MAE models, this returns the encoded representation from the
        encoder (before decoding). By default, no masking is applied during
        inference to get embeddings of the full input.
        
        For EfficientNet models, this returns the embedding from the encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
               or any format that can be converted to this shape
            mask_ratio: Ratio of patches to mask (default: 0.0 for full encoding)
                       Only used for MAE models.
            return_mask: If True, return tuple of (embeddings, mask, ids_restore)
                        Only used for MAE models.

        Returns:
            Embeddings tensor of shape (batch_size, num_visible_patches, embed_dim) for MAE
            or (batch_size, embedding_dim) for EfficientNet
            or tuple of (embeddings, mask, ids_restore) if return_mask=True (MAE only)

        Example:
            >>> embeddings = pipeline.embed(batch_data)
            >>> # Get shape: (batch_size, num_patches, embed_dim) for MAE
            >>> # Get shape: (batch_size, embedding_dim) for EfficientNet
        """
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Move to device
        x = x.to(self.device)

        # Apply normalization if provided
        # if self.normalization_fn is not None:
        #     x = torch.from_numpy(self.normalization_fn(x.cpu().numpy()))

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():
            if self.model_name.lower() == "mae" or self.model_name.lower() == "maskedautoencoder":
                # Use MAE's encode method
                embeddings, mask, ids_restore = self.model.encode(
                    x, mask_ratio=mask_ratio
                )

                if return_mask:
                    return embeddings, mask, ids_restore
                return embeddings
            elif self.model_name.lower() == "efficientnet" or self.model_name.lower() == "efficientnetcontrastive":
                # Use EfficientNet's get_representation method
                embeddings = self.model.get_representation(x)
                return embeddings
            else:
                raise NotImplementedError(
                    f"Embed method not implemented for model: {self.model_name}"
                )

    def reconstruct(
        self,
        x: Union[torch.Tensor, Any],
        mask_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Reconstruct input time series.

        For MAE models, this performs the full forward pass: encode with masking,
        then decode to reconstruct the original signal.
        
        Note: This method is only available for MAE models. EfficientNet models
        do not support reconstruction.
        
        Important: If a normalization function is provided, the reconstruction
        will be in the normalized space. For inverse normalization (e.g., z-score),
        you need to manually apply the inverse transformation to the output.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            mask_ratio: Ratio of patches to mask (uses model's default if None)

        Returns:
            Reconstructed time series of shape (batch_size, seq_len, in_channels)
            
        Raises:
            NotImplementedError: If called on non-MAE models

        Example:
            >>> reconstructed = pipeline.reconstruct(batch_data, mask_ratio=0.75)
        """
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Move to device
        x = x.to(self.device)
        
        # Apply normalization if provided
        # if self.normalization_fn is not None:
        #     x = self.normalization_fn(x)

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():
            if self.model_name.lower() == "mae":
                # Use MAE's reconstruct method
                reconstruction, _ = self.model.reconstruct(x, mask_ratio=mask_ratio)
                return reconstruction
            else:
                raise NotImplementedError(
                    f"Reconstruct method not implemented for model: {self.model_name}. "
                    f"Only MAE models support reconstruction."
                )

    def __call__(
        self,
        x: Union[torch.Tensor, Any],
        task: str = "embed",
        **kwargs,
    ) -> torch.Tensor:
        """
        Convenience method for calling the pipeline.

        Args:
            x: Input time series
            task: Task to perform ("embed" or "reconstruct")
            **kwargs: Additional arguments passed to the task method

        Returns:
            Output tensor based on the task
        """
        if task == "embed":
            
            return self.embed(x, **kwargs)
        elif task == "reconstruct":
            return self.reconstruct(x, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'embed' or 'reconstruct'.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pipeline configuration to a dictionary.
        
        This method serializes the pipeline's configuration into a dictionary
        format that can be used for saving, logging, or reconstruction.
        
        Note: The model weights are not included in this dictionary. Use
        torch.save() to save the actual model weights separately.
        
        Returns:
            Dictionary containing:
                - model_name: Name of the model architecture
                - model_class: Class name of the model
                - device: Device type (cpu, cuda, mps)
                - device_type: String representation of device
                - model_config: Model configuration parameters
                - dtype: Data type of model parameters
        
        Example:
            >>> config_dict = pipeline.to_dict()
            >>> print(config_dict['model_name'])
            'mae'
        """
        # Get model dtype
        try:
            model_dtype = next(self.model.parameters()).dtype
            dtype_str = str(model_dtype).replace('torch.', '')
        except StopIteration:
            dtype_str = 'float32'
        
        return {
            'model_name': self.model_name,
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'model_config': self.model_config,
            'dtype': dtype_str,
        }

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (
            f"EdamamePipeline(\n"
            f"  model_name={self.model_name},\n"
            f"  device={self.device},\n"
            f"  model={self.model.__class__.__name__}\n"
            f")"
        )
