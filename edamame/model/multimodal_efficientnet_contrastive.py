"""Multimodal EfficientNet contrastive model with late fusion.

This module provides two building blocks:

1. :class:`EfficientNet1DEncoder`
     A 1D EfficientNet-inspired encoder that maps one sensor stream to an
     embedding vector.
2. :class:`MultiModalEfficientNetContrastive`
     A multimodal wrapper that creates one encoder per modality, concatenates
     modality embeddings (late fusion), and exposes a contrastive interface
     compatible with the training engines in this repository.

Key design choices:
- Same encoder architecture across modalities, but independent weights.
- No automatic channel dropping/selection. The provided tensor must match the
    configured channel count for that modality.
- No automatic temporal resampling across modalities. Different sequence
    lengths are allowed and encoded independently before fusion.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from edamame.model.efficientnet_contrastive import MBConv1D, Swish


class EfficientNet1DEncoder(nn.Module):
    """EfficientNet-inspired 1D encoder for a single modality.

    The encoder expects one modality tensor and returns an embedding of shape
    ``(batch_size, embedding_dim)``.

    Accepted input layouts:
    - ``(batch, seq_len)`` for single-channel inputs
    - ``(batch, seq_len, channels)``
    - ``(batch, channels, seq_len)``

    Notes:
    - The input channel count must match ``in_channels``.
    - If channel layout cannot be inferred safely, a :class:`ValueError`
      is raised with a modality-aware message.
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 32,
        kernel_size: int = 10,
        stride: int = 2,
        mbconv_channels: int = 64,
        num_mbconv_blocks: int = 4,
        head_channels: int = 128,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        modality_name: str = "unknown",
    ):
        super().__init__()

        self._validate_init_args(
            in_channels=in_channels,
            stem_channels=stem_channels,
            kernel_size=kernel_size,
            stride=stride,
            mbconv_channels=mbconv_channels,
            num_mbconv_blocks=num_mbconv_blocks,
            head_channels=head_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            modality_name=modality_name,
        )

        self.in_channels = in_channels
        self.output_dim = embedding_dim
        self.modality_name = modality_name

        self.stem_conv = nn.Conv1d(
            in_channels,
            stem_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.stem_activation = Swish()

        self.mbconv_blocks = nn.ModuleList()
        current_channels = stem_channels
        for block_idx in range(num_mbconv_blocks):
            out_channels = mbconv_channels if block_idx == 0 else current_channels
            current_stride = stride if block_idx == 0 else 1
            self.mbconv_blocks.append(
                MBConv1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=current_stride,
                    expand_ratio=4,
                    se_ratio=0.25,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.head_conv = nn.Conv1d(current_channels, head_channels, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm1d(head_channels)
        self.head_activation = Swish()
        self.head_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(head_channels, embedding_dim)

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")

    @classmethod
    def _validate_init_args(
        cls,
        in_channels: int,
        stem_channels: int,
        kernel_size: int,
        stride: int,
        mbconv_channels: int,
        num_mbconv_blocks: int,
        head_channels: int,
        embedding_dim: int,
        dropout: float,
        modality_name: str,
    ) -> None:
        cls._require_positive_int("in_channels", in_channels)
        cls._require_positive_int("stem_channels", stem_channels)
        cls._require_positive_int("kernel_size", kernel_size)
        cls._require_positive_int("stride", stride)
        cls._require_positive_int("mbconv_channels", mbconv_channels)
        cls._require_positive_int("num_mbconv_blocks", num_mbconv_blocks)
        cls._require_positive_int("head_channels", head_channels)
        cls._require_positive_int("embedding_dim", embedding_dim)

        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}.")

        if not isinstance(modality_name, str) or modality_name.strip() == "":
            raise ValueError("modality_name must be a non-empty string.")

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Convert supported layouts to ``(batch, channels, seq_len)``.

        Args:
            x: Input tensor in one of the supported layouts.

        Returns:
            Tensor in channels-first format with dtype ``float``.

        Raises:
            ValueError: If shape rank is not 2 or 3, or if channels cannot be
                inferred to match ``self.in_channels``.
            TypeError: If ``x`` is not a torch tensor.
        """
        if not torch.is_tensor(x):
            raise TypeError(
                f"Modality '{self.modality_name}' expects a torch.Tensor, got "
                f"{type(x)}."
            )

        if x.dim() == 2:
            if self.in_channels != 1:
                raise ValueError(
                    f"Modality '{self.modality_name}' expected {self.in_channels} channels "
                    f"but received a 2D tensor with shape {tuple(x.shape)}. "
                    "Use 3D input with explicit channel dimension."
                )
            # (batch, seq_len) -> (batch, 1, seq_len)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # Prefer keeping channel-first if already matching configured channels.
            if x.shape[1] == self.in_channels:
                pass
            elif x.shape[-1] == self.in_channels:
                # (batch, seq_len, channels) -> (batch, channels, seq_len)
                x = x.transpose(1, 2)
            elif self.in_channels == 1:
                raise ValueError(
                    f"Modality '{self.modality_name}' expects a single channel, but input "
                    f"shape {tuple(x.shape)} has no axis with size 1 for channels."
                )
            else:
                raise ValueError(
                    f"Modality '{self.modality_name}' cannot infer channel dimension for "
                    f"tensor shape {tuple(x.shape)} with expected channels={self.in_channels}."
                )
        else:
            raise ValueError(
                f"Modality '{self.modality_name}' expected 2D or 3D tensor input, "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[0] == 0:
            raise ValueError(f"Modality '{self.modality_name}' received an empty batch.")

        return x.contiguous().float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-modality batch into embeddings.

        Args:
            x: Tensor in one of the supported layouts.

        Returns:
            Embedding tensor of shape ``(batch_size, embedding_dim)``.
        """
        try:
            x = self._to_channels_first(x)

            x = self.stem_conv(x)
            x = self.stem_bn(x)
            x = self.stem_activation(x)

            for mbconv in self.mbconv_blocks:
                x = mbconv(x)

            x = self.head_conv(x)
            x = self.head_bn(x)
            x = self.head_activation(x)
            x = self.head_pool(x).squeeze(-1)
            x = self.embedding(x)
            return x
        except RuntimeError as exc:
            raise RuntimeError(
                f"Encoder forward failed for modality '{self.modality_name}' with "
                f"input shape {tuple(x.shape) if torch.is_tensor(x) else type(x)}."
            ) from exc


class MultiModalEfficientNetContrastive(nn.Module):
    """Multimodal contrastive model with late fusion.

    One encoder is instantiated per modality using the same architecture but
    independent parameters. In the forward path, each modality is encoded
    separately and modality embeddings are concatenated.

    Input conventions:
    - Preferred format: ``Mapping[str, Tensor]`` where each key is a modality.
    - Also supported: ``list`` / ``tuple`` with one tensor per modality in the
      same order as ``modalities``.
    - Also supported: single 3D tensor for auto-splitting by modality axis when
      ``num_modalities > 1`` and each modality expects exactly one channel.

    Contrastive interface:
    - ``forward(x_1, x_2) -> (h_1, h_2)``
    - ``loss(output, batch, step)``
    - ``get_representation(x)``
    """

    def __init__(
        self,
        modalities: Sequence[str],
        in_channels_per_modality: Mapping[str, int] | None = None,
        stem_channels: int = 32,
        kernel_size: int = 10,
        stride: int = 2,
        mbconv_channels: int = 64,
        num_mbconv_blocks: int = 4,
        head_channels: int = 128,
        embedding_dim: int = 256,
        fused_embedding_dim: int | None = None,
        loss_fn=None,
        dropout: float = 0.1,
        normalize_embeddings: bool = False,
    ):
        """Initialize multimodal late-fusion contrastive model.

        Args:
            modalities: Ordered list of modality names.
            in_channels_per_modality: Mapping modality -> number of channels
                expected by that modality encoder. If omitted, all modalities
                default to 1 channel.
            stem_channels: Number of channels in stem convolution.
            kernel_size: Convolution kernel size used in MBConv blocks.
            stride: Stride used in stem and first MBConv block.
            mbconv_channels: Number of channels for MBConv stack output.
            num_mbconv_blocks: Number of MBConv blocks per modality encoder.
            head_channels: Number of channels in encoder head conv.
            embedding_dim: Per-modality embedding dimensionality.
            fused_embedding_dim: Optional projection size after concatenation.
                If ``None``, fused output is the raw concatenation.
            loss_fn: Contrastive loss callable with signature compatible with
                ``loss_fn(h_1, h_2, key_ids=...)``.
            dropout: Dropout used inside modality encoders.
            normalize_embeddings: If ``True``, L2-normalize fused embeddings.

        Raises:
            ValueError: If no modalities are provided or channel specs are
                missing for any modality.
        """
        super().__init__()

        self.modalities = self._validate_modalities(modalities)
        self._validate_model_hparams(
            stem_channels=stem_channels,
            kernel_size=kernel_size,
            stride=stride,
            mbconv_channels=mbconv_channels,
            num_mbconv_blocks=num_mbconv_blocks,
            head_channels=head_channels,
            embedding_dim=embedding_dim,
            fused_embedding_dim=fused_embedding_dim,
            dropout=dropout,
            normalize_embeddings=normalize_embeddings,
        )

        self.num_modalities = len(self.modalities)
        self.loss_fn = loss_fn
        self.normalize_embeddings = normalize_embeddings

        self.in_channels_per_modality = self._validate_in_channels_per_modality(
            in_channels_per_modality
        )

        self.encoders = nn.ModuleDict(
            {
                modality: EfficientNet1DEncoder(
                    in_channels=self.in_channels_per_modality[modality],
                    stem_channels=stem_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    mbconv_channels=mbconv_channels,
                    num_mbconv_blocks=num_mbconv_blocks,
                    head_channels=head_channels,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    modality_name=modality,
                )
                for modality in self.modalities
            }
        )

        concatenated_dim = embedding_dim * self.num_modalities
        if fused_embedding_dim is None:
            self.fusion_projection = nn.Identity()
            self.output_dim = concatenated_dim
        else:
            self.fusion_projection = nn.Linear(concatenated_dim, fused_embedding_dim)
            self.output_dim = fused_embedding_dim

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")

    @staticmethod
    def _validate_modalities(modalities: Sequence[str]) -> list[str]:
        if not modalities:
            raise ValueError("'modalities' must contain at least one modality name.")

        if any(not isinstance(modality, str) for modality in modalities):
            raise TypeError("All entries in 'modalities' must be strings.")

        if any(modality.strip() == "" for modality in modalities):
            raise ValueError("Modality names must be non-empty strings.")

        if len(set(modalities)) != len(modalities):
            raise ValueError(f"'modalities' contains duplicates: {list(modalities)}.")

        return list(modalities)

    @classmethod
    def _validate_model_hparams(
        cls,
        stem_channels: int,
        kernel_size: int,
        stride: int,
        mbconv_channels: int,
        num_mbconv_blocks: int,
        head_channels: int,
        embedding_dim: int,
        fused_embedding_dim: int | None,
        dropout: float,
        normalize_embeddings: bool,
    ) -> None:
        cls._require_positive_int("stem_channels", stem_channels)
        cls._require_positive_int("kernel_size", kernel_size)
        cls._require_positive_int("stride", stride)
        cls._require_positive_int("mbconv_channels", mbconv_channels)
        cls._require_positive_int("num_mbconv_blocks", num_mbconv_blocks)
        cls._require_positive_int("head_channels", head_channels)
        cls._require_positive_int("embedding_dim", embedding_dim)

        if fused_embedding_dim is not None:
            cls._require_positive_int("fused_embedding_dim", fused_embedding_dim)

        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}.")

        if not isinstance(normalize_embeddings, bool):
            raise TypeError(
                f"normalize_embeddings must be bool, got {type(normalize_embeddings)}."
            )

    def _validate_in_channels_per_modality(
        self, in_channels_per_modality: Mapping[str, int] | None
    ) -> dict[str, int]:
        if in_channels_per_modality is None:
            channel_map = {modality: 1 for modality in self.modalities}
        else:
            channel_map = dict(in_channels_per_modality)

        unexpected = [m for m in channel_map if m not in self.modalities]
        if unexpected:
            raise ValueError(
                "Unexpected entries in 'in_channels_per_modality': "
                f"{unexpected}. Expected only {self.modalities}."
            )

        missing = [m for m in self.modalities if m not in channel_map]
        if missing:
            raise ValueError(
                "Missing channel definitions in 'in_channels_per_modality' for: "
                f"{missing}."
            )

        invalid = [
            m for m, ch in channel_map.items() if not isinstance(ch, int) or ch < 1
        ]
        if invalid:
            details = {m: channel_map[m] for m in invalid}
            raise ValueError(
                "Each modality channel count must be a positive int. "
                f"Invalid entries: {details}."
            )

        return channel_map

    @staticmethod
    def _ensure_modality_tensor(modality: str, tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Input for modality '{modality}' must be a torch.Tensor, "
                f"got {type(tensor)}."
            )
        return tensor

    @staticmethod
    def _validate_loss_output(output) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError(
                "'output' must be a tuple/list of (h_1, h_2) embeddings. "
                f"Got type {type(output)} with value {output!r}."
            )

        h_1, h_2 = output
        if not torch.is_tensor(h_1) or not torch.is_tensor(h_2):
            raise TypeError("Both h_1 and h_2 must be torch.Tensor objects.")
        return h_1, h_2

    def _split_tensor_into_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a single tensor into one tensor per modality.

        This is mainly a convenience path for simple cases where modalities are
        stacked in one axis.

        Supported formats when ``num_modalities > 1``:
        - ``(batch, seq_len, num_modalities)``
        - ``(batch, num_modalities, seq_len)``

        Args:
            x: Input tensor containing all modalities.

        Returns:
            Dictionary mapping modality name to tensor.

        Raises:
            ValueError: If tensor rank/shape is incompatible with splitting.
        """
        if self.num_modalities == 1:
            return {self.modalities[0]: x}

        multi_channel_modalities = [
            modality
            for modality, channels in self.in_channels_per_modality.items()
            if channels != 1
        ]
        if multi_channel_modalities:
            raise ValueError(
                "Single-tensor modality splitting is only supported when every "
                "modality expects one channel. Found multi-channel modalities: "
                f"{multi_channel_modalities}. Use mapping/list input instead."
            )

        if x.dim() != 3:
            raise ValueError(
                "When more than one modality is configured, tensor input must be 3D "
                f"for automatic splitting. Got shape {tuple(x.shape)}."
            )

        if x.shape[-1] == self.num_modalities:
            return {
                modality: x[..., idx]
                for idx, modality in enumerate(self.modalities)
            }

        if x.shape[1] == self.num_modalities:
            return {
                modality: x[:, idx, :]
                for idx, modality in enumerate(self.modalities)
            }

        raise ValueError(
            "Unable to split modalities from tensor shape "
            f"{tuple(x.shape)} for {self.num_modalities} modalities."
        )

    def _normalize_modality_input(self, x) -> dict[str, torch.Tensor]:
        """Normalize multimodal input to ``dict[modality, tensor]``.

        Args:
            x: One of the supported multimodal representations:
                mapping, list/tuple, or tensor.

        Returns:
            Modality dictionary aligned with ``self.modalities`` order.

        Raises:
            ValueError: If modalities are missing or list/tuple length is wrong.
            TypeError: If input type is unsupported.
        """
        if isinstance(x, Mapping):
            unexpected = [m for m in x if m not in self.modalities]
            if unexpected:
                raise ValueError(
                    f"Unexpected modalities in input mapping: {unexpected}. "
                    f"Expected modalities: {self.modalities}."
                )
            missing = [m for m in self.modalities if m not in x]
            if missing:
                raise ValueError(f"Missing modalities in input mapping: {missing}.")
            return {
                modality: self._ensure_modality_tensor(modality, x[modality])
                for modality in self.modalities
            }

        if isinstance(x, (list, tuple)):
            if len(x) != self.num_modalities:
                raise ValueError(
                    f"Expected {self.num_modalities} modality tensors, got {len(x)}."
                )
            return {
                modality: self._ensure_modality_tensor(modality, modality_tensor)
                for modality, modality_tensor in zip(self.modalities, x)
            }

        if torch.is_tensor(x):
            return self._split_tensor_into_modalities(x)

        raise TypeError(
            "Unsupported multimodal input type. Use dict, list/tuple, or tensor."
        )

    def _validate_modality_tensors(
        self, modality_tensors: Mapping[str, torch.Tensor]
    ) -> None:
        """Validate per-modality tensor shape and batch consistency."""
        batch_sizes = {}
        for modality in self.modalities:
            tensor = modality_tensors[modality]
            if tensor.dim() not in (2, 3):
                raise ValueError(
                    f"Modality '{modality}' must be 2D or 3D, got shape {tuple(tensor.shape)}."
                )
            if tensor.shape[0] == 0:
                raise ValueError(f"Modality '{modality}' received an empty batch.")
            batch_sizes[modality] = int(tensor.shape[0])

        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) != 1:
            raise ValueError(
                "All modalities must have the same batch size. "
                f"Received batch sizes: {batch_sizes}."
            )

    def _encode(self, x) -> torch.Tensor:
        """Encode multimodal input into a fused embedding.

        Args:
            x: Multimodal input in any supported format.

        Returns:
            Fused embedding tensor of shape ``(batch_size, output_dim)``.
        """
        modality_tensors = self._normalize_modality_input(x)
        self._validate_modality_tensors(modality_tensors)

        embeddings = []
        for modality in self.modalities:
            try:
                embeddings.append(self.encoders[modality](modality_tensors[modality]))
            except (ValueError, TypeError, RuntimeError) as exc:
                raise type(exc)(
                    f"Encoding failed for modality '{modality}': {exc}"
                ) from exc

        fused = torch.cat(embeddings, dim=-1)
        fused = self.fusion_projection(fused)
        if self.normalize_embeddings:
            fused = F.normalize(fused, dim=-1)
        return fused

    def forward(self, x_1, x_2):
        """Encode two augmented views for contrastive learning.

        Args:
            x_1: First view in any supported multimodal format.
            x_2: Second view in any supported multimodal format.

        Returns:
            Tuple ``(h_1, h_2)`` of fused embeddings.
        """
        h_1 = self._encode(x_1)
        h_2 = self._encode(x_2)

        if h_1.shape[0] != h_2.shape[0]:
            raise ValueError(
                "Batch size mismatch between contrastive views: "
                f"h_1 has {h_1.shape[0]}, h_2 has {h_2.shape[0]}."
            )
        return h_1, h_2

    def loss(self, output, batch, step: str = "train"):
        """Compute contrastive loss.

        Args:
            output: Tuple ``(h_1, h_2)`` from :meth:`forward`.
            batch: Batch dictionary. If ``"user"`` exists, it is forwarded as
                ``key_ids`` to the loss function.
            step: Unused step label kept for engine compatibility.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If ``loss_fn`` was not provided.
        """
        del step
        if self.loss_fn is None:
            raise ValueError("Loss function not provided. Please set 'loss_fn'.")

        h_1, h_2 = self._validate_loss_output(output)
        if h_1.shape[0] != h_2.shape[0]:
            raise ValueError(
                f"h_1 and h_2 must have same batch size, got {h_1.shape[0]} and {h_2.shape[0]}."
            )

        if "user" in batch:
            if len(batch["user"]) != h_1.shape[0]:
                raise ValueError(
                    "Length of batch['user'] must match embedding batch size. "
                    f"Got {len(batch['user'])} and {h_1.shape[0]}."
                )
            return self.loss_fn(h_1, h_2, key_ids=batch["user"])
        return self.loss_fn(h_1, h_2)

    def get_representation(self, x):
        """Return fused embedding for one multimodal input view."""
        return self._encode(x)
