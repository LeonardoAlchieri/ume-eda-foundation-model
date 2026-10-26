"""
Mantis-based contrastive model wrapper.

This module adapts Mantis foundation models to UME's contrastive interface:
- forward(x_1, x_2)
- loss(output, batch, step)
- get_representation(x)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MantisContrastive(nn.Module):
    """
    Wrapper around Mantis foundation models for self-supervised contrastive pretraining.

    The wrapped model is expected to map tensors of shape (batch, channels, seq_len)
    to embedding tensors of shape (batch, embedding_dim).
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        model_name: str = "paris-noah/Mantis-8M",
        architecture: str = "Mantis8M",
        load_pretrained: bool = True,
        pre_training: bool = True,
        seq_len: int = 512,
        output_token: str = "cls_token",
        return_transf_layer: int = -1,
        freeze_backbone: bool = False,
        channels_strategy: str = "concat",
        mantis_kwargs: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.seq_len = seq_len
        self.channels_strategy = channels_strategy

        self.backbone = self._build_backbone(
            architecture=architecture,
            model_name=model_name,
            load_pretrained=load_pretrained,
            pre_training=pre_training,
            seq_len=seq_len,
            output_token=output_token,
            return_transf_layer=return_transf_layer,
            mantis_kwargs=mantis_kwargs or {},
            device=device,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = getattr(self.backbone, "hidden_dim", None)
        if hidden_dim is None:
            raise AttributeError(
                "Mantis backbone must expose 'hidden_dim' to determine output_dim."
            )
        self.output_dim = hidden_dim

    @staticmethod
    def _build_backbone(
        architecture: str,
        model_name: str,
        load_pretrained: bool,
        pre_training: bool,
        seq_len: int,
        output_token: str,
        return_transf_layer: int,
        mantis_kwargs: dict[str, Any],
        device: torch.device | str | None,
    ) -> nn.Module:
        try:
            from mantis.architecture import Mantis8M, MantisV1, MantisV2
        except ImportError as exc:
            raise ImportError(
                "Mantis is required for MantisContrastive. Install with: pip install mantis-tsfm"
            ) from exc

        cls_map = {
            "Mantis8M": Mantis8M,
            "MantisV1": MantisV1,
            "MantisV2": MantisV2,
        }
        if architecture not in cls_map:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Expected one of {list(cls_map)}."
            )

        model_cls = cls_map[architecture]

        model_init_kwargs = dict(mantis_kwargs)
        model_init_kwargs["pre_training"] = pre_training

        # MantisV1/V2 support explicit output-token controls; Mantis8M keeps legacy API.
        if architecture in {"MantisV1", "MantisV2"}:
            model_init_kwargs["output_token"] = output_token
            model_init_kwargs["return_transf_layer"] = return_transf_layer

        # MantisV2 does not use seq_len, while MantisV1/Mantis8M do.
        if architecture in {"Mantis8M", "MantisV1"}:
            model_init_kwargs["seq_len"] = seq_len
            
        model_init_kwargs['device'] = device

        model = model_cls(**model_init_kwargs)
        if load_pretrained:
            model = model.from_pretrained(model_name)
        return model

    def _to_mantis_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input into Mantis format (batch, channels, seq_len).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # Common UME format: (batch, seq_len, channels)
            if x.shape[-1] <= 16 and x.shape[1] > x.shape[-1]:
                x = x.transpose(1, 2)
            # If already (batch, channels, seq_len), keep as is.
        else:
            raise ValueError(
                f"Expected 2D/3D input tensor, got shape {tuple(x.shape)}."
            )

        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)

        return x.contiguous().float()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_mantis_input(x)

        if x.shape[1] == 1:
            return self.backbone(x)

        if self.channels_strategy == "first":
            return self.backbone(x[:, :1, :])

        if self.channels_strategy != "concat":
            raise ValueError(
                f"Unknown channels_strategy '{self.channels_strategy}'. Use 'concat' or 'first'."
            )

        per_channel_embeddings = [
            self.backbone(x[:, [channel_idx], :]) for channel_idx in range(x.shape[1])
        ]
        return torch.cat(per_channel_embeddings, dim=-1)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        h_1 = self._encode(x_1)
        h_2 = self._encode(x_2)
        return h_1, h_2

    def loss(self, output, batch, step: str = "train"):
        del step
        h_1, h_2 = output

        if "user" in batch:
            return self.loss_fn(h_1, h_2, key_ids=batch["user"])
        return self.loss_fn(h_1, h_2)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)
