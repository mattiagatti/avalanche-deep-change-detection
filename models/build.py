"""Unified model construction for SwinUNet and the baseline change-detection models.

Centralizes the ``swinunet`` vs. baseline+adapter branch that was previously
duplicated across ``train.py``, ``test.py`` and the inference scripts.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from models.baselines.adapter import CDModelAdapter
from models.baselines.factory import BuildArgs, available_models, build_baseline
from models.swinunet import ChangeDetectionSwinUNet


def model_choices() -> List[str]:
    """All valid ``--model`` values (baselines + swinunet)."""
    return available_models() + ["swinunet"]


def build_model(
    model_name: str,
    patch_size: int,
    *,
    use_aux: bool = False,
    model_size: str = "tiny",
    fusion_type: str = "diff",
    device: Optional[torch.device] = None,
    sar_in_channels: int = 2,
    out_ch: int = 1,
) -> nn.Module:
    """Build a change-detection model and move it to ``device``.

    Args:
        model_name: "swinunet" or one of the registered baselines.
        patch_size: input patch size (square).
        use_aux: enable auxiliary input (SwinUNet only).
        model_size: SwinUNet size ("tiny"/"small"/"base").
        fusion_type: SwinUNet encoder fusion ("diff"/"agm"/"cross").
        device: target device (defaults to CUDA if available, else CPU).
        sar_in_channels: per-image SAR channels (baselines receive this as ``in_ch``).
        out_ch: number of output channels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "swinunet":
        model = ChangeDetectionSwinUNet(
            model_size=model_size,
            img_size=patch_size,
            use_aux=use_aux,
            fusion_type=fusion_type,
            sar_in_channels=sar_in_channels,
            num_classes=out_ch,
        )
    else:
        core = build_baseline(
            model_name,
            BuildArgs(
                device=device,
                patch_size=patch_size,
                in_ch=sar_in_channels,
                out_ch=out_ch,
            ),
        )
        model = CDModelAdapter(core, model_name=model_name)

    return model.to(device)
