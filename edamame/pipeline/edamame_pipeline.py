"""
EdamamePipeline: Inference API for pretrained models.

This module provides a unified interface for loading and using pretrained models
for embedding extraction and inference tasks.
"""

import copy
from pathlib import Path
from typing import Dict, Optional, Union, Any, Callable, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

        # Fine-tuning state (populated by `fit`)
        self.head: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.num_classes: Optional[int] = None

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
            model_name: Name of the model architecture (e.g., "mae", "efficientnet", "mantis")
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
        else:
            raise ValueError(
                f"Model '{model_name}' not supported. Currently supported: 'mae', 'efficientnet', 'mantis'."
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
            elif self.model_name.lower() in ["efficientnet", "efficientnetcontrastive", "mantis", "mantiscontrastive"]:
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

    # ------------------------------------------------------------------ #
    # Supervised fine-tuning (scikit-learn / Mantis-style API)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_tensor(X: Union[torch.Tensor, np.ndarray, Any]) -> torch.Tensor:
        """Convert input to a float32 tensor without moving it to a device."""
        if isinstance(X, torch.Tensor):
            return X.float()
        return torch.as_tensor(np.asarray(X), dtype=torch.float32)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone representation -> classification head logits."""
        representation = self.model.get_representation(x)
        if representation.dim() != 2:
            raise ValueError(
                f"Fine-tuning expects the backbone to return a 2D "
                f"(batch, features) representation, got shape "
                f"{tuple(representation.shape)}. This is supported for the "
                f"EfficientNet (UME) encoder; MAE returns per-patch tokens "
                f"and is not supported."
            )
        return self.head(representation)

    def _evaluate(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray, List],
        criterion: nn.Module,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Evaluate the current backbone + head on ``(X, y)`` in a single pass.

        Labels are expected in the original (pre-encoding) space and are mapped
        back to the integer ids learned in ``fit`` via ``self.classes_``.

        Returns:
            Dict with keys ``loss`` (mean cross-entropy under ``criterion``),
            ``acc`` (accuracy) and ``macro_f1`` (macro-averaged F1).
        """
        from sklearn.metrics import accuracy_score, f1_score

        X_t = self._to_tensor(X)
        y_arr = np.asarray(y).reshape(-1)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_enc = np.array([class_to_idx[v] for v in y_arr])

        self.model.eval()
        self.head.eval()
        total_loss = 0.0
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, X_t.shape[0], batch_size):
                xb = X_t[start : start + batch_size].to(self.device)
                yb = torch.as_tensor(
                    y_enc[start : start + batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self._forward_logits(xb)
                total_loss += criterion(logits, yb).item() * xb.size(0)
                preds.append(logits.argmax(dim=1).cpu())

        y_pred = torch.cat(preds).numpy()
        return {
            "loss": total_loss / len(y_enc),
            "acc": float(accuracy_score(y_enc, y_pred)),
            "macro_f1": float(f1_score(y_enc, y_pred, average="macro")),
        }

    def fit(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray, List],
        fine_tuning_type: str = "full",
        learning_rate: float = 1e-3,
        backbone_lr: Optional[float] = None,
        batch_size: int = 64,
        num_epochs: int = 100,
        weight_decay: float = 0.0,
        validation_data: Optional[Tuple[Any, Any]] = None,
        validation_split: float = 0.0,
        class_weight: Optional[Union[str, torch.Tensor]] = None,
        early_stopping: bool = False,
        patience: int = 10,
        monitor: str = "val_macro_f1",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        refit_on_full_data: bool = False,
        use_layer_norm: bool = True,
        verbose: bool = True,
    ) -> "EdamamePipeline":
        """
        Fine-tune the pretrained UME (EfficientNet) encoder for classification.

        A linear classification head is attached on top of the encoder's
        embedding and trained with cross-entropy. Mirrors the Mantis
        ``MantisTrainer`` API: call ``fit(X, y)`` then ``predict`` /
        ``predict_proba``.

        Args:
            X: Input signals of shape (n_samples, seq_len, in_channels)
               (e.g. (n, 240, 3)). A 2D array (n, seq_len) is treated as a
               single channel.
            y: Class labels of shape (n_samples,). Labels may be arbitrary
               (ints or strings); they are encoded internally and the
               original values are restored by ``predict``.
            fine_tuning_type: One of
                - "full" (default): fine-tune the whole encoder + head.
                - "head": freeze the encoder (linear probe), train head only.
                - "scratch": re-initialise the encoder and train everything
                  from random weights (ignores the pretrained checkpoint).
            learning_rate: LR for the head (and for the backbone when
                ``backbone_lr`` is None).
            backbone_lr: Optional separate (typically smaller) LR for the
                encoder. Only used for "full"/"scratch". Defaults to
                ``learning_rate``.
            batch_size: Mini-batch size.
            num_epochs: Number of training epochs.
            weight_decay: AdamW weight decay.
            validation_data: Optional ``(X_val, y_val)`` for per-epoch
                validation accuracy reporting. Takes precedence over
                ``validation_split`` if both are given.
            validation_split: Fraction of the training data (in ``(0, 1)``)
                to hold out for per-epoch validation accuracy reporting. The
                split is stratified by label and only used when
                ``validation_data`` is None. No early stopping is performed;
                the held-out set is only used for reporting. Defaults to 0.0
                (no validation).
            class_weight: Optional cross-entropy class weights. Pass
                "balanced" for inverse-frequency weighting, or a tensor of
                shape (num_classes,).
            early_stopping: If True, stop training once the monitored
                validation metric stops improving. Requires a validation set
                (via ``validation_data`` or ``validation_split``); raises
                otherwise. Off by default (all ``num_epochs`` are run).
            patience: Number of epochs to wait with no improvement in the
                monitored metric before stopping. Only used when
                ``early_stopping`` is True.
            monitor: Validation metric that early stopping watches. One of
                "val_macro_f1" (default, maximized; matches Pulse-PPG and is
                robust to class imbalance), "val_acc" (maximized), or
                "val_loss" (minimized).
            min_delta: Minimum change in the monitored metric to qualify as an
                improvement. Improvements smaller than this are ignored.
            restore_best_weights: If True (default), roll the encoder + head
                back to the epoch with the best monitored metric when training
                ends (whether by early stopping or by exhausting ``num_epochs``).
                This is the "best checkpoint" selection used by Pulse-PPG.
                Ignored when ``refit_on_full_data`` is True (the refit
                replaces the phase-1 model entirely).
            refit_on_full_data: If True, training runs in two phases.
                Phase 1 trains with the validation hold-out and early
                stopping, only to determine the optimal training budget
                (the optimizer-step count of the best epoch — steps, not
                epochs, because the full dataset has more batches per
                epoch). Phase 2 then resets the encoder + head to their
                initial weights, rebuilds the optimizer, and retrains on
                100% of the data (training + validation combined) for that
                step budget, with no validation set. Use this on small
                datasets where permanently holding out labels hurts the
                final model. Requires ``early_stopping=True``. Note the
                refit model cannot be validated internally — rely on an
                outer evaluation loop (e.g. LOPO) for unbiased estimates.
            use_layer_norm: If True (default), prepend a LayerNorm over the
                encoder's embedding before the linear classification layer
                (LayerNorm -> Linear head), matching the head used by Mantis
                and Pulse-PPG. Set to False for a bare linear head.
            verbose: Print per-epoch loss/accuracy.

        Returns:
            self (fitted pipeline).

        Example:
            >>> pipeline = EdamamePipeline.from_pretrained(
            ...     model_name="efficientnet", weights_path="artifacts/ume.ckpt")
            >>> pipeline.fit(X, y)              # y = vector of class labels
            >>> probs = pipeline.predict_proba(X)
            >>> y_pred = pipeline.predict(X)
        """
        if self.model_name.lower() not in (
            "efficientnet",
            "efficientnetcontrastive",
        ):
            raise NotImplementedError(
                f"Fine-tuning is only supported for the EfficientNet (UME) "
                f"encoder, got model_name='{self.model_name}'."
            )
        if fine_tuning_type not in ("full", "head", "scratch"):
            raise ValueError(
                f"fine_tuning_type must be one of 'full', 'head', 'scratch', "
                f"got '{fine_tuning_type}'."
            )
        if refit_on_full_data and not early_stopping:
            raise ValueError(
                "refit_on_full_data=True requires early_stopping=True: the "
                "validated stopping point is what determines the refit's "
                "training budget."
            )

        X_t = self._to_tensor(X)
        y_arr = np.asarray(y).reshape(-1)
        if X_t.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y have mismatched number of samples: "
                f"{X_t.shape[0]} vs {y_arr.shape[0]}."
            )

        # Encode labels -> contiguous integer ids, remember originals.
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.num_classes = len(self.classes_)
        y_t = torch.as_tensor(y_encoded, dtype=torch.long)

        # Keep the untouched full training set around for the refit phase.
        X_full: Optional[torch.Tensor] = None
        y_full: Optional[torch.Tensor] = None
        if refit_on_full_data:
            if validation_data is not None:
                Xv_full = self._to_tensor(validation_data[0])
                yv_arr = np.asarray(validation_data[1]).reshape(-1)
                unknown = np.setdiff1d(yv_arr, self.classes_)
                if unknown.size:
                    raise ValueError(
                        f"validation_data contains labels not present in the "
                        f"training labels: {unknown.tolist()}. All classes "
                        f"must appear in y when refit_on_full_data=True."
                    )
                yv_encoded = np.searchsorted(self.classes_, yv_arr)
                X_full = torch.cat([X_t, Xv_full], dim=0)
                y_full = torch.cat(
                    [y_t, torch.as_tensor(yv_encoded, dtype=torch.long)]
                )
            else:
                # validation_split carves the hold-out out of X_t below;
                # these references keep the pre-split tensors.
                X_full = X_t
                y_full = y_t

        # Optionally hold out a stratified fraction for validation reporting.
        # Only used when no explicit validation_data is provided.
        if validation_data is None and validation_split > 0.0:
            if not 0.0 < validation_split < 1.0:
                raise ValueError(
                    f"validation_split must be in (0, 1), got {validation_split}."
                )
            rng = np.random.default_rng(0)
            val_idx_parts: List[np.ndarray] = []
            for cls in range(self.num_classes):
                cls_idx = np.flatnonzero(y_encoded == cls)
                rng.shuffle(cls_idx)
                n_val = int(round(len(cls_idx) * validation_split))
                val_idx_parts.append(cls_idx[:n_val])
            val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)
            train_mask = np.ones(X_t.shape[0], dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.flatnonzero(train_mask)

            validation_data = (X_t[val_idx], self.classes_[y_encoded[val_idx]])
            X_t = X_t[train_idx]
            y_t = y_t[train_idx]
            y_encoded = y_encoded[train_idx]
            if verbose:
                print(
                    f"Holding out {len(val_idx)} samples "
                    f"({validation_split:.0%}) for validation."
                )

        # Optionally reset the backbone to random weights ("scratch").
        if fine_tuning_type == "scratch":
            def _reset(module: nn.Module) -> None:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self.model.apply(_reset)

        # Infer feature dimension from a real forward pass and build the head.
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            feat = self.model.get_representation(X_t[:1].to(self.device))
        if feat.dim() != 2:
            raise ValueError(
                f"Backbone returned a {feat.dim()}D representation; a 2D "
                f"(batch, features) output is required for classification."
            )
        feat_dim = feat.shape[1]
        if use_layer_norm:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, self.num_classes),
            ).to(self.device)
        else:
            self.head = nn.Linear(feat_dim, self.num_classes).to(self.device)

        # Freeze / unfreeze backbone according to the strategy.
        train_backbone = fine_tuning_type in ("full", "scratch")
        for p in self.model.parameters():
            p.requires_grad = train_backbone

        # Snapshot the starting weights so the refit phase can replay
        # training from the same initialization on the full dataset.
        initial_state: Optional[Dict[str, Any]] = None
        if refit_on_full_data:
            initial_state = {
                "model": copy.deepcopy(self.model.state_dict()),
                "head": copy.deepcopy(self.head.state_dict()),
            }

        # Optimizer parameter groups.
        param_groups: List[dict] = [
            {"params": self.head.parameters(), "lr": learning_rate}
        ]
        if train_backbone:
            param_groups.append(
                {
                    "params": self.model.parameters(),
                    "lr": backbone_lr if backbone_lr is not None else learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Class weights.
        weight_tensor: Optional[torch.Tensor] = None
        if class_weight == "balanced":
            counts = np.bincount(y_encoded, minlength=self.num_classes)
            inv = len(y_encoded) / (self.num_classes * np.maximum(counts, 1))
            weight_tensor = torch.as_tensor(inv, dtype=torch.float32, device=self.device)
        elif isinstance(class_weight, torch.Tensor):
            weight_tensor = class_weight.to(self.device).float()
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )

        # ---- Early-stopping setup ----
        from sklearn.metrics import f1_score

        # metric name -> ("min"/"max" direction, key returned by _evaluate)
        monitor_spec = {
            "val_loss": ("min", "loss"),
            "val_acc": ("max", "acc"),
            "val_macro_f1": ("max", "macro_f1"),
        }
        if early_stopping:
            if monitor not in monitor_spec:
                raise ValueError(
                    f"monitor must be one of {sorted(monitor_spec)}, got "
                    f"'{monitor}'."
                )
            if validation_data is None:
                raise ValueError(
                    "early_stopping=True requires a validation set. Pass "
                    "validation_data=(X_val, y_val) or set validation_split > 0."
                )
        mode, metric_key = monitor_spec.get(monitor, ("max", "macro_f1"))
        best_score: Optional[float] = None
        best_state: Optional[Dict[str, Any]] = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train(train_backbone)  # keep BN frozen for linear probe
            self.head.train()
            running_loss, total = 0.0, 0
            train_preds: List[torch.Tensor] = []
            train_targets: List[torch.Tensor] = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self._forward_logits(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                total += xb.size(0)
                train_preds.append(logits.argmax(dim=1).detach().cpu())
                train_targets.append(yb.detach().cpu())

            train_f1 = float(
                f1_score(
                    torch.cat(train_targets).numpy(),
                    torch.cat(train_preds).numpy(),
                    average="macro",
                )
            )

            # Validation metrics (needed for reporting and/or early stopping).
            val_metrics: Optional[Dict[str, float]] = None
            if validation_data is not None:
                Xv, yv = validation_data
                val_metrics = self._evaluate(
                    Xv, yv, criterion, batch_size=batch_size
                )

            if verbose:
                msg = (
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"loss: {running_loss / total:.4f} - "
                    f"macro_f1: {train_f1:.4f}"
                )
                if val_metrics is not None:
                    msg += f" - val_macro_f1: {val_metrics['macro_f1']:.4f}"
                print(msg)

            # ---- Early stopping / best-checkpoint tracking ----
            if early_stopping:
                score = val_metrics[metric_key]  # type: ignore[index]
                if best_score is None:
                    improved = True
                elif mode == "max":
                    improved = score > best_score + min_delta
                else:
                    improved = score < best_score - min_delta

                if improved:
                    best_score = score
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    if restore_best_weights and not refit_on_full_data:
                        best_state = {
                            "model": copy.deepcopy(self.model.state_dict()),
                            "head": copy.deepcopy(self.head.state_dict()),
                        }
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch + 1}: no "
                                f"{monitor} improvement for {patience} epochs "
                                f"(best {monitor}: {best_score:.4f} "
                                f"@ epoch {best_epoch})."
                            )
                        break

        # Restore the best checkpoint seen during training.
        if early_stopping and restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.head.load_state_dict(best_state["head"])
            if verbose:
                print(
                    f"Restored best weights from epoch {best_epoch} "
                    f"(best {monitor}: {best_score:.4f})."
                )

        # ---- Refit phase: replay training on 100% of the data ----
        # Phase 1 above only served to find the stopping point. Reset to the
        # initial weights and retrain on the full dataset for the same number
        # of optimizer *steps* as the best epoch (steps rather than epochs,
        # since the full loader has more batches per epoch).
        if refit_on_full_data:
            assert (
                initial_state is not None
                and X_full is not None
                and y_full is not None
            )
            best_step = best_epoch * len(loader)
            if verbose:
                print(
                    f"Refitting on the full dataset ({X_full.shape[0]} samples) "
                    f"for {best_step} optimizer steps "
                    f"(best epoch {best_epoch} × {len(loader)} steps/epoch)."
                )
            self.model.load_state_dict(initial_state["model"])
            self.head.load_state_dict(initial_state["head"])

            param_groups = [
                {"params": self.head.parameters(), "lr": learning_rate}
            ]
            if train_backbone:
                param_groups.append(
                    {
                        "params": self.model.parameters(),
                        "lr": backbone_lr
                        if backbone_lr is not None
                        else learning_rate,
                    }
                )
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=weight_decay
            )

            # Recompute "balanced" class weights over the full label set.
            if class_weight == "balanced":
                counts = np.bincount(
                    y_full.numpy(), minlength=self.num_classes
                )
                inv = len(y_full) / (self.num_classes * np.maximum(counts, 1))
                criterion = nn.CrossEntropyLoss(
                    weight=torch.as_tensor(
                        inv, dtype=torch.float32, device=self.device
                    )
                )

            full_loader = DataLoader(
                TensorDataset(X_full, y_full),
                batch_size=batch_size,
                shuffle=True,
            )
            step = 0
            while step < best_step:
                self.model.train(train_backbone)
                self.head.train()
                running_loss, total = 0.0, 0
                for xb, yb in full_loader:
                    if step >= best_step:
                        break
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self._forward_logits(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)
                    total += xb.size(0)
                    step += 1
                if verbose and total:
                    print(
                        f"Refit step {step}/{best_step} - "
                        f"loss: {running_loss / total:.4f}"
                    )

        # Leave everything in eval mode for inference.
        self.model.eval()
        self.head.eval()
        return self

    def predict_proba(
        self,
        X: Union[torch.Tensor, np.ndarray],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Predict class probabilities for ``X``.

        Args:
            X: Input signals of shape (n_samples, seq_len, in_channels).
            batch_size: Inference batch size.

        Returns:
            Array of shape (n_samples, num_classes) with softmax probabilities.
            Column ``j`` corresponds to ``self.classes_[j]``.
        """
        if self.head is None:
            raise RuntimeError("Pipeline has not been fine-tuned. Call `fit` first.")

        X_t = self._to_tensor(X)
        self.model.eval()
        self.head.eval()

        probs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, X_t.shape[0], batch_size):
                xb = X_t[start : start + batch_size].to(self.device)
                logits = self._forward_logits(xb)
                probs.append(F.softmax(logits, dim=1).cpu())
        return torch.cat(probs, dim=0).numpy()

    def predict(
        self,
        X: Union[torch.Tensor, np.ndarray],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Predict class labels for ``X``.

        Args:
            X: Input signals of shape (n_samples, seq_len, in_channels).
            batch_size: Inference batch size.

        Returns:
            Array of shape (n_samples,) with predicted labels, in the original
            label space passed to ``fit``.
        """
        proba = self.predict_proba(X, batch_size=batch_size)
        indices = proba.argmax(axis=1)
        return self.classes_[indices]

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
