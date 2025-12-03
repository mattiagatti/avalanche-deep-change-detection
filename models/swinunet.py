import logging
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.swin_transformer_v2 import SwinTransformerV2Block

# --------------------------------------------------------------------------- #
# Model registry & heads per stage
# --------------------------------------------------------------------------- #

SWINV2_NAME = {
    "tiny": "swinv2_tiny_window8_256.ms_in1k",
    "small": "swinv2_small_window8_256.ms_in1k",
    "base": "swinv2_base_window8_256.ms_in1k",
}

SWINV2_NUM_HEADS = {
    "swinv2_tiny_window8_256.ms_in1k": [3, 6, 12, 24],
    "swinv2_small_window8_256.ms_in1k": [3, 6, 12, 24],
    "swinv2_base_window8_256.ms_in1k": [4, 8, 16, 32],
}

DEFAULT_TIMM_MODEL = "swinv2_tiny_window8_256.ms_in1k"


# --------------------------------------------------------------------------- #
# Fusion modules
# --------------------------------------------------------------------------- #

class DiffFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_aux: bool = True) -> None:
        super().__init__()
        self.use_aux = use_aux
        self.relu = nn.ReLU(inplace=True)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        fused_feat = x2 - x1
        fused = torch.cat([fused_feat, aux], dim=1) if (self.use_aux and aux is not None) else fused_feat
        fused_out = self.fusion(fused)
        return self.relu(fused_out + fused_feat)


class AGMFFusion(nn.Module):
    """
    Attention-Gated Multi-cue Fusion with explicit channel sizes.

    Args
    ----
    sar_channels : int      # channels of x1/x2
    out_channels: int       # channels after fusion (usually == sar_channels)
    aux_channels: int       # channels of aux (0 if unused)
    use_aux     : bool
    reduction   : int       # channel MLP reduction ratio
    """

    def __init__(
        self,
        sar_channels: int,
        out_channels: int,
        aux_channels: int = 0,
        use_aux: bool = True,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.use_aux = use_aux
        self.sar_channels = sar_channels
        self.aux_channels = aux_channels if use_aux else 0
        self.reduction = reduction

        cue_channels = 4 * sar_channels + self.aux_channels

        mid = max(1, cue_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(cue_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, cue_channels, 1, bias=False),
        )

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.reduce = nn.Conv2d(cue_channels, out_channels, 1, bias=True)
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        assert x1.shape[1] == self.sar_channels and x2.shape[1] == self.sar_channels, (
            f"SAR channels mismatch: expected {self.sar_channels}, got {x1.shape[1]} and {x2.shape[1]}"
        )
        if self.use_aux:
            assert aux is not None and aux.shape[1] == self.aux_channels, (
                f"AUX channels mismatch: expected {self.aux_channels}, got {0 if aux is None else aux.shape[1]}"
            )

        diff = x2 - x1
        feats = [diff, diff.abs(), x1 + x2, x1 * x2]
        if self.use_aux and aux is not None:
            feats.append(aux)
        f = torch.cat(feats, dim=1)

        w_c = 2 * torch.sigmoid(self.mlp(f).mean([2, 3], keepdim=True)) - 1
        f = f + w_c * f

        avg = f.mean(dim=1, keepdim=True)
        mx = f.amax(dim=1, keepdim=True)
        w_s = 2 * torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1))) - 1
        f = f + w_s * f

        f = self.reduce(f)
        f = self.mix(f)
        return self.relu(f + diff)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of two feature maps (time-1, time-2) plus optional aux.

    Parameters
    ----------
    in_channels : int
    num_heads   : int
    use_aux     : bool
    windowed    : bool
    window_size : int
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        use_aux: bool = True,
        windowed: bool = True,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.use_aux = use_aux
        self.windowed = windowed
        self.window_size = window_size

        self.q_proj = nn.Conv1d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv1d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv1d(in_channels, in_channels, 1)
        if use_aux:
            self.k_aux = nn.Conv1d(in_channels, in_channels, 1)
            self.v_aux = nn.Conv1d(in_channels, in_channels, 1)

        self.attn = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.out_fc = nn.Conv1d(in_channels, in_channels, 1)

        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _to_windows(x: torch.Tensor, ws: int) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h % ws == 0 and w % ws == 0
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(-1, c, ws, ws)

    @staticmethod
    def _from_windows(x: torch.Tensor, b: int, c: int, h: int, w: int, ws: int) -> torch.Tensor:
        n_h, n_w = h // ws, w // ws
        x = x.view(b, n_h, n_w, c, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, h, w)

    def _fuse_flat(self, x1: torch.Tensor, x2: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        b, c, h, w = x1.shape

        def flat(z: torch.Tensor) -> torch.Tensor:
            return rearrange(z, "b c h w -> b (h w) c")

        q = self.q_proj(flat(x2).transpose(1, 2)).transpose(1, 2)  # B, L, C
        k = self.k_proj(flat(x1).transpose(1, 2)).transpose(1, 2)
        v = self.v_proj(flat(x1).transpose(1, 2)).transpose(1, 2)

        if self.use_aux and aux is not None:
            k_aux = self.k_aux(flat(aux).transpose(1, 2)).transpose(1, 2)
            v_aux = self.v_aux(flat(aux).transpose(1, 2)).transpose(1, 2)
            k = torch.cat([k, k_aux], dim=1)
            v = torch.cat([v, v_aux], dim=1)

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        out = self.out_fc(attn_out.transpose(1, 2))
        return out.view(b, c, h, w)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x1.shape
        ws = self.window_size

        if self.windowed and (h > ws or w > ws):
            x1_w = self._to_windows(x1, ws)
            x2_w = self._to_windows(x2, ws)
            aux_w = self._to_windows(aux, ws) if (self.use_aux and aux is not None) else None
            fused_w = self._fuse_flat(x1_w, x2_w, aux_w)
            fused = self._from_windows(fused_w, b, c, h, w, ws)
        else:
            fused = self._fuse_flat(x1, x2, aux)

        fused = self.mix(fused) + (x2 - x1)
        return fused


# --------------------------------------------------------------------------- #
# Decoder blocks
# --------------------------------------------------------------------------- #

class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        x = self.expand(x)
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"
        x = x.view(b, h, w, c)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=c // 4)
        x = x.view(b, -1, c // 4)
        return self.norm(x)


class FinalPatchExpandXN(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 4) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2) * dim, bias=False)
        self.output_dim = dim
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        x = self.expand(x)
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"
        x = x.view(b, h, w, c)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=c // (self.dim_scale**2),
        )
        x = x.view(b, -1, self.output_dim)
        return self.norm(x)


class BasicLayerUp(nn.Module):
    """A basic Swin Transformer layer for one decoder stage."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        upsample: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution

        block = SwinTransformerV2Block
        self.blocks = nn.ModuleList(
            [
                block(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                )
                for i in range(depth)
            ]
        )
        self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2) if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, l, c = x.shape
        x = x.view(b, h, w, c)
        for blk in self.blocks:
            x = blk(x)
        x = x.view(b, l, c)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        enc_channels: Tuple[int, ...] = (96, 192, 384, 768),
        depths_decoder: List[int] | Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: List[int] | Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.encoder_channels = enc_channels
        self.num_layers = len(depths_decoder)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i in range(self.num_layers):
            curr_dim = enc_channels[-(i + 1)]
            concat_linear = nn.Linear(2 * curr_dim, curr_dim) if i > 0 else nn.Identity()

            stage_h = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i))
            stage_w = self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i))
            safe_ws = min(window_size, stage_h, stage_w)

            if i == 0:
                layer_up = PatchExpand(input_resolution=(stage_h, stage_w), dim=curr_dim, dim_scale=2)
            else:
                layer_up = BasicLayerUp(
                    dim=curr_dim,
                    input_resolution=(stage_h, stage_w),
                    depth=depths_decoder[-(i + 1)],
                    num_heads=num_heads[-(i + 1)],
                    window_size=safe_ws,
                    upsample=PatchExpand if (i < self.num_layers - 1) else None,
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = nn.LayerNorm(self.encoder_channels[0])
        self.up = FinalPatchExpandXN(input_resolution=self.patches_resolution, dim=self.encoder_channels[0], dim_scale=patch_size)
        self.output = nn.Conv2d(in_channels=self.encoder_channels[0], out_channels=num_classes, kernel_size=1)

    def forward_up_features(self, x: torch.Tensor, x_downsample: List[torch.Tensor]) -> torch.Tensor:
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[len(x_downsample) - i]], dim=-1)
                x = self.concat_back_dim[i](x)
                x = layer_up(x)
        return self.norm_up(x)

    def up_x(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.patches_resolution
        b, l, c = x.shape
        assert l == h * w, f"Input features have wrong size: {l} != {h * w}"
        x = self.up(x)                                   # FinalPatchExpandXN
        s = self.up.dim_scale                            # <-- use actual scale (== patch_size)
        x = x.view(b, s * h, s * w, -1)                  # <-- no hard-coded 4
        x = x.permute(0, 3, 1, 2)                        # (B, C, H, W)
        return self.output(x)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        feats = [xi.flatten(2).transpose(1, 2) for xi in feats]
        x, x_down = feats[-1], feats[:-1]
        x = self.forward_up_features(x, x_down)
        return self.up_x(x)                               # <-- was self.up_x4(x)



# --------------------------------------------------------------------------- #
# Swin Encoder (features_only) with configurable patch size
# --------------------------------------------------------------------------- #

class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder (timm, features_only). Returns a list of feature maps (for skips).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TIMM_MODEL,
        img_size: int = 128,
        in_chans: int = 2,
        pretrained: bool = False,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.out_channels = self.model.feature_info.channels()
        self.reductions = self.model.feature_info.reduction()  # e.g. [2, 4, 8, 16] for patch_size=2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.model(x)  # list of (B, H, W, C)
        return [f.permute(0, 3, 1, 2) for f in features]  # → (B, C, H, W)


# --------------------------------------------------------------------------- #
# Change Detection Swin-UNet
# --------------------------------------------------------------------------- #

class ChangeDetectionSwinUNet(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        sar_in_channels: int = 2,
        aux_in_channels: int = 4,
        num_classes: int = 1,
        use_aux: bool = False,
        model_size: str = "tiny",
        window_size: int = 4,            # smaller windows work better at high res token grids
        fusion_type: str = "diff",      # "diff" | "agm" | "cross"
        target_stride: int = 8,          # cap encoder downsampling (keep ≤8 to preserve detail)
        encoder_patch_size: int = 2,     # finer first tokenization (we do NOT use pretrained)
    ) -> None:
        """
        For 128×128 inputs we keep detail by:
          - using a smaller Swin patch size (2) so the stem stride is 2 not 4;
          - capping the deepest encoder stride (≤ target_stride, e.g. 8).
        """
        super().__init__()
        self.img_size = img_size
        self.use_aux = use_aux
        self.encoder_patch_size = encoder_patch_size

        sar_model_name = SWINV2_NAME[model_size]
        aux_model_name = SWINV2_NAME[model_size]
        all_heads = SWINV2_NUM_HEADS[sar_model_name]

        # Probe reductions with the desired patch size, then pick stages ≤ target_stride.
        probe = SwinEncoder(
            model_name=sar_model_name,
            img_size=img_size,
            in_chans=sar_in_channels,
            pretrained=False,
            out_indices=(0, 1, 2, 3),
            patch_size=encoder_patch_size,
        )
        reductions = probe.reductions  # e.g. [2, 4, 8, 16]
        out_indices = tuple(i for i, r in enumerate(reductions) if r <= target_stride) or (0,)

        # --- Encoders (no pretrained) ---
        self.sar_encoder = SwinEncoder(
            model_name=sar_model_name,
            img_size=img_size,
            in_chans=sar_in_channels,
            pretrained=False,
            out_indices=out_indices,
            patch_size=encoder_patch_size,
        )
        if use_aux:
            self.aux_encoder = SwinEncoder(
                model_name=aux_model_name,
                img_size=img_size,
                in_chans=aux_in_channels,
                pretrained=False,
                out_indices=out_indices,
                patch_size=encoder_patch_size,
            )
        else:
            self.aux_encoder = None

        encoder_channels = self.sar_encoder.out_channels
        stage_heads = all_heads[: len(out_indices)]

        # Helpers for stage geometry
        def stage_hw_from_reduction(h: int, w: int, idx: int) -> Tuple[int, int]:
            r = reductions[idx]
            return h // r, w // r

        def safe_window_size(h: int, w: int, want: int) -> int:
            cap = min(want, h, w)
            for s in range(cap, 0, -1):
                if (h % s == 0) and (w % s == 0):
                    return s
            return 1

        # --- Fusion modules (one per stage kept) ---
        self.fusion_stages = nn.ModuleList()
        for i, ch in enumerate(encoder_channels[: len(out_indices)]):
            if fusion_type == "diff":
                in_ch = ch + (ch if use_aux else 0)
                module = DiffFusionModule(in_channels=in_ch, out_channels=ch, use_aux=use_aux)
            elif fusion_type == "agm":
                module = AGMFFusion(
                    sar_channels=ch,
                    aux_channels=(ch if use_aux else 0),
                    out_channels=ch,
                    use_aux=use_aux,
                    reduction=4,
                )
            elif fusion_type == "cross":
                heads = stage_heads[i]
                h_i, w_i = stage_hw_from_reduction(self.img_size, self.img_size, i)
                ws = safe_window_size(h_i, w_i, window_size)
                module = CrossAttentionFusion(
                    in_channels=ch, num_heads=heads, use_aux=use_aux, windowed=True, window_size=ws
                )
            else:
                raise ValueError("fusion_type must be one of {'diff','agm','cross'}")
            self.fusion_stages.append(module)

        # --- Decoder (grid driven by encoder_patch_size) ---
        self.decoder = TransformerDecoder(
            img_size=self.img_size,
            patch_size=self.encoder_patch_size,
            enc_channels=tuple(encoder_channels[: len(out_indices)]),
            depths_decoder=tuple([2] * len(out_indices)),
            num_heads=tuple(stage_heads),
            window_size=window_size,
            num_classes=num_classes,
        )

    @property
    def domain_feat_dim(self) -> int:
        return self.sar_encoder.out_channels[-1]

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        aux: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        f1_list = self.sar_encoder(x1)
        f2_list = self.sar_encoder(x2)
        faux_list = self.aux_encoder(aux) if self.use_aux else None

        fused_features: List[torch.Tensor] = []
        for i in range(len(f1_list)):
            if self.use_aux:
                fused_i = self.fusion_stages[i](f1_list[i], f2_list[i], faux_list[i])
            else:
                fused_i = self.fusion_stages[i](f1_list[i], f2_list[i])
            fused_features.append(fused_i)

        logits = self.decoder(fused_features)

        if not return_features:
            return logits

        feat = fused_features[-1].mean(dim=(2, 3))
        return logits, feat

    @torch.no_grad()
    def predict(self, x1: torch.Tensor, x2: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        return self.forward(x1, x2, aux)  # type: ignore[return-value]