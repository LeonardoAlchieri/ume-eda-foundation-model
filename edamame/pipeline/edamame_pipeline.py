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
            model_name: Name of the model architecture (e.g., "mae", "efficientnet", "mantis", "tnc")
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
        elif model_name.lower() == "mantis":
            model = cls._load_mantis_model(config_dict, checkpoint, torch_dtype)
        elif model_name.lower() in ("tnc", "tncmodel"):
            model = cls._load_tnc_model(config_dict, checkpoint, torch_dtype)
        else:
            raise ValueError(
                f"Model '{model_name}' not supported. Currently supported: 'mae', 'efficientnet', 'mantis', 'tnc'."
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
    def _load_mantis_model(
        config_dict: Dict[str, Any],
        checkpoint: Dict[str, Any],
        torch_dtype: Optional[torch.dtype] = None,
    ) -> nn.Module:
        """
        Load Mantis contrastive model.

        Args:
            config_dict: Model configuration
            checkpoint: Loaded checkpoint
            torch_dtype: Data type for model

        Returns:
            Loaded Mantis model
        """
        from edamame.model.mantis_contrastive import MantisContrastive
        import torch.nn as nn

        # Extract state dict - handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Infer model parameters from state dict if config is empty
        if not config_dict:
            config_dict = EdamamePipeline._infer_mantis_config_from_state_dict(state_dict)

        # Ensure loss_fn exists (required by init but not used in eval)
        if "loss_fn" not in config_dict:
            config_dict["loss_fn"] = nn.Identity()

        if "pre_training" not in config_dict:
            config_dict["pre_training"] = False

        if "load_pretrained" not in config_dict:
            config_dict["load_pretrained"] = False

        # Create model
        model = MantisContrastive(**config_dict)

        # Load weights
        model.load_state_dict(state_dict, strict=False)

        # Convert dtype if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model

    @staticmethod
    def _load_tnc_model(
        config_dict: Dict[str, Any],
        checkpoint: Dict[str, Any],
        torch_dtype: Optional[torch.dtype] = None,
    ) -> nn.Module:
        """
        Load Temporal Neighborhood Coding (TNC) model.

        TNC wraps a single-view encoder (an EfficientNetContrastive by default) with
        a learnable neighborhood discriminator (TNCLoss). For downstream feature
        extraction only the encoder is used (via ``get_representation``), but the
        full model is rebuilt so the checkpoint's encoder weights load cleanly.

        Args:
            config_dict: Model configuration. If it contains an ``encoder`` (and
                optional ``loss_fn``) sub-config those are used directly; otherwise
                the architecture is inferred from the state dict.
            checkpoint: Loaded checkpoint
            torch_dtype: Data type for model

        Returns:
            Loaded TNC model
        """
        from edamame.model.tnc import TNCModel
        from edamame.model.efficientnet_contrastive import EfficientNetContrastive
        from edamame.model.loss.tnc import TNCLoss

        # Extract state dict - handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove the Lightning 'model.' prefix (only the leading one, so that
            # nested keys like 'encoder.stem_conv...' / 'loss_fn.discriminator...'
            # are preserved).
            state_dict = {
                (k[len("model.") :] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
        else:
            state_dict = checkpoint

        # Resolve encoder / loss_fn configs (explicit config wins, else infer).
        if config_dict and "encoder" in config_dict:
            encoder_cfg = dict(config_dict["encoder"])
            # The encoder carries no objective of its own under TNC.
            encoder_cfg.setdefault("loss_fn", None)
            loss_cfg = dict(config_dict.get("loss_fn", {}) or {})
        else:
            encoder_cfg, loss_cfg = EdamamePipeline._infer_tnc_config_from_state_dict(
                state_dict
            )

        encoder = EfficientNetContrastive(**encoder_cfg)
        loss_fn = TNCLoss(**loss_cfg)
        model = TNCModel(encoder=encoder, loss_fn=loss_fn)

        # Load weights (non-strict: discriminator keys may be absent/extra)
        model.load_state_dict(state_dict, strict=False)

        # Convert dtype if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model

    @staticmethod
    def _infer_tnc_config_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Infer TNC encoder and loss configurations from a state dictionary.

        The encoder config is recovered by reusing the EfficientNet inference on the
        ``encoder.``-prefixed keys; the loss (discriminator) config is recovered from
        the ``loss_fn.discriminator.`` linear layers.

        Args:
            state_dict: Model state dictionary

        Returns:
            Tuple ``(encoder_cfg, loss_cfg)`` of keyword-argument dicts.
        """
        # Strip the 'encoder.' prefix and reuse the EfficientNet inference.
        encoder_state = {
            k[len("encoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        encoder_cfg = EdamamePipeline._infer_efficientnet_config_from_state_dict(
            encoder_state
        )
        # The encoder carries no objective of its own under TNC.
        encoder_cfg["loss_fn"] = None

        # Infer discriminator (TNCLoss) sizing from its first linear layer.
        # discriminator.0.weight has shape (hidden, 2 * embedding_dim).
        loss_cfg: Dict[str, Any] = {}
        first_linear = state_dict.get("loss_fn.discriminator.0.weight")
        if first_linear is not None:
            hidden, two_d = first_linear.shape
            embedding_dim = two_d // 2
            loss_cfg["embedding_dim"] = embedding_dim
            if embedding_dim > 0:
                loss_cfg["hidden_multiplier"] = max(1, hidden // embedding_dim)
        else:
            # Fall back to the encoder's embedding dimension.
            loss_cfg["embedding_dim"] = encoder_cfg.get("embedding_dim", 64)

        return encoder_cfg, loss_cfg

    @staticmethod
    def _infer_mantis_config_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Infer Mantis configuration from state dictionary.

        Args:
            state_dict: Model state dictionary

        Returns:
            Inferred configuration dictionary
        """
        config = {}
        
        # Look for architecture specific keys
        if any("mantis.V2" in k for k in state_dict.keys()):
            config["architecture"] = "MantisV2"
        elif any("mantis.V1" in k for k in state_dict.keys()):
            config["architecture"] = "MantisV1"
        else:
            config["architecture"] = "Mantis8M"
            
        return config

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

        # Infer stem channels and kernel size
        if "stem_conv.weight" in state_dict:
            config["in_channels"] = state_dict["stem_conv.weight"].shape[1]
            config["stem_channels"] = state_dict["stem_conv.weight"].shape[0]
            config["kernel_size"] = state_dict["stem_conv.weight"].shape[2]

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

        # Infer mbconv_channels and kernel size from first MBConv block
        if "mbconv_blocks.0.project_conv.weight" in state_dict:
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

        # Infer patch size and in_channels from patch embedding layer
        # patch_embed.projection.weight has shape (embed_dim, patch_size * in_channels)
        if "patch_embed.projection.weight" in state_dict:
            patch_channels_product = state_dict["patch_embed.projection.weight"].shape[1]
            
            # Try to infer from decoder_pred as well (has shape: patch_size * in_channels, decoder_embed_dim)
            if "decoder_pred.weight" in state_dict:
                decoder_pred_input = state_dict["decoder_pred.weight"].shape[0]
                # Both should give us patch_size * in_channels
                patch_channels_product = decoder_pred_input
            
            # Common patch sizes to try (in order of preference)
            for patch_size in [10, 8, 5, 4, 2, 1, 16, 20]:
                if patch_channels_product % patch_size == 0:
                    in_channels = patch_channels_product // patch_size
                    config["patch_size"] = patch_size
                    config["in_channels"] = in_channels
                    break
            else:
                # Fallback: assume patch_size=10 and calculate in_channels
                config["patch_size"] = 10
                config["in_channels"] = max(1, patch_channels_product // 10)
        else:
            # Fallback to defaults if we can't infer
            config["patch_size"] = 10
            config["in_channels"] = 1

        # Infer sequence length from number of patches
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
        For Mantis models, this returns the embedding from the encoder.
        For TNC models, this returns the embedding from the wrapped encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
               or any format that can be converted to this shape
            mask_ratio: Ratio of patches to mask (default: 0.0 for full encoding)
                       Only used for MAE models.
            return_mask: If True, return tuple of (embeddings, mask, ids_restore)
                        Only used for MAE models.

        Returns:
            Embeddings tensor of shape (batch_size, num_visible_patches, embed_dim) for MAE
            or (batch_size, embedding_dim) for EfficientNet and Mantis
            or tuple of (embeddings, mask, ids_restore) if return_mask=True (MAE only)

        Example:
            >>> embeddings = pipeline.embed(batch_data)
            >>> # Get shape: (batch_size, num_patches, embed_dim) for MAE
            >>> # Get shape: (batch_size, embedding_dim) for EfficientNet and Mantis
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
            elif self.model_name.lower() in [
                "efficientnet",
                "efficientnetcontrastive",
                "mantis",
                "mantiscontrastive",
                "tnc",
                "tncmodel",
            ]:
                # Use get_representation method for contrastive models
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
        
        Note: This method is only available for MAE models. EfficientNet and Mantis models
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
