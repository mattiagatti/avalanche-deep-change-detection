from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import torch
import torch.nn as nn

# ---- Public API -------------------------------------------------------------


@dataclass
class BuildArgs:
    device: torch.device
    patch_size: int
    in_ch: int = 2               # per-image channels or total (model-dependent)
    out_ch: int = 1


# Registry maps name -> builder(BuildArgs) -> nn.Module (the *core* model)
Builder = Callable[[BuildArgs], nn.Module]
_REGISTRY: Dict[str, Builder] = {}


def register(name: str):
    def deco(fn: Builder):
        _REGISTRY[name] = fn
        return fn
    return deco


def available_models() -> List[str]:
    return sorted(_REGISTRY.keys())


def build_baseline(name: str, args: BuildArgs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Available: {available_models()}")
    return _REGISTRY[name](args)

# ---- Builders ---------------------------------------------------------------


@register("bit")
def _build_bit(a: BuildArgs) -> nn.Module:
    from . import BIT
    return BIT(
        variant="base_transformer_pos_s4_dd8",
        input_nc=a.in_ch,
        output_nc=a.out_ch,
        backbone="resnet18",
        pretrained_backbone=True
    ).to(a.device)


@register("changeformer")
def _build_changeformer(a: BuildArgs) -> nn.Module:
    # Uses your previous defaults
    from . import ChangeFormerV5
    return ChangeFormerV5(
        img_size=a.patch_size, input_nc=a.in_ch, output_nc=a.out_ch, decoder_softmax=False
    ).to(a.device)


@register("siamunet_conc")
def _build_siamunet_conc(a: BuildArgs) -> nn.Module:
    from . import SiamUnet_conc
    return SiamUnet_conc(input_nbr=a.in_ch, label_nbr=a.out_ch).to(a.device)


@register("siamunet_diff")
def _build_siamunet_diff(a: BuildArgs) -> nn.Module:
    from . import SiamUnet_diff
    return SiamUnet_diff(input_nbr=a.in_ch, label_nbr=a.out_ch).to(a.device)


@register("stanet")
def _build_stanet(a: BuildArgs) -> nn.Module:
    from . import STANet
    # your previous code used in_ch=2 (two images stacked or diff’d)
    return STANet(in_c=a.in_ch, num_classes=a.out_ch).to(a.device)

@register("stnet")
def _build_stnet(a: BuildArgs) -> nn.Module:
    from . import STNet
    # your previous code used in_ch=2 (two images stacked or diff’d)
    return STNet(in_channels=a.in_ch, num_class=a.out_ch).to(a.device)


@register("snunet")
def _build_snunet(a: BuildArgs) -> nn.Module:
    from . import SNUNet_ECAM
    # your previous code used in_ch=2 (two images stacked or diff’d)
    return SNUNet_ECAM(in_ch=a.in_ch, out_ch=a.out_ch).to(a.device)


@register("tinycd")
def _build_tinycd(a: BuildArgs) -> nn.Module:
    from . import TinyCD
    return TinyCD(in_ch=a.in_ch, out_ch=a.out_ch, out_activation=None).to(a.device)