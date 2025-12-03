# Hao Chen, Zipeng Qi & Zhenwei Shi
# https://github.com/justchenhao/BIT_CD
# Chen, H., Qi, Z., & Shi, Z.
# "Remote Sensing Image Change Detection with Transformers."
# IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2021. doi:10.1109/TGRS.2021.3095166.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# -----------------------------
# Small utility head
# -----------------------------
class TwoLayerConv2d(nn.Module):
    """Lightweight head to map features -> logits."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Backbone: ResNet to ~1/8 resolution 32-ch map
# -----------------------------
class ResNetBackbone(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        stages: int = 5,
        if_upsample_2x: bool = True,
        out_ch: int = 32,
        pretrained: bool = True,
        in_ch: int = 3,                          # <— new
    ):
        super().__init__()
        self.if_upsample_2x = if_upsample_2x
        self.stages = stages

        expand = 1
        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet50(weights=weights)
            expand = 4
        else:
            raise NotImplementedError(f"Unknown backbone: {backbone}")

        # --- adapt first conv to arbitrary input channels ---
        if in_ch != 3:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_ch, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )
            with torch.no_grad():
                if pretrained:
                    # Start from RGB weights and project to in_ch.
                    # For in_ch==1: luminance-like; for in_ch==2+: average and repeat.
                    base = old.weight   # [64,3,7,7]
                    w = base.mean(dim=1, keepdim=True)  # [64,1,7,7]
                    w = w.repeat(1, in_ch, 1, 1)        # [64,in_ch,7,7]
                    resnet.conv1.weight.copy_(w)
                    if resnet.conv1.bias is not None:
                        resnet.conv1.bias.zero_()
                else:
                    nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")
                    if resnet.conv1.bias is not None:
                        nn.init.zeros_(resnet.conv1.bias)

        # Keep spatial stride small via dilation
        resnet.layer3[0].conv2.dilation = (2, 2); resnet.layer3[0].conv2.padding = (2, 2)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer4[0].conv2.dilation = (4, 4); resnet.layer4[0].conv2.padding = (4, 4)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv1.stride = (1, 1)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.l1 = resnet.layer1
        self.l2 = resnet.layer2
        self.l3 = resnet.layer3 if stages > 3 else nn.Identity()
        self.l4 = resnet.layer4 if stages == 5 else nn.Identity()

        in_feat = {5: 512*expand, 4: 256*expand, 3: 128*expand}[stages]
        self.proj32 = nn.Conv2d(in_feat, out_ch, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)            # ~1/8
        if self.stages > 3:
            x = self.l3(x)        # keep ~1/8 via dilation
        if self.stages == 5:
            x = self.l4(x)        # still ~1/8 via dilation
        x = self.proj32(x)        # [B, 32, H/8, W/8] approx
        if self.if_upsample_2x:
            x = self.up2(x)       # optional 2x (parity with some BIT configs)
        return x


# -----------------------------
# BIT Model
# -----------------------------
class BIT(nn.Module):
    """
    BIT: ResNet backbone -> semantic tokenization -> Transformer encoder (tokens) ->
    Transformer decoder (inject tokens into features) -> |x1-x2| -> upsample -> head.

    Instantiate with a preset variant or explicit kwargs.

    Args (common):
      input_nc, output_nc: 3, 2 by default
      variant: one of {"base_transformer_pos_s4", "base_transformer_pos_s4_dd8",
                       "base_transformer_pos_s4_dd8_dedim8"} (optional)
      with_pos: "learned" or None
      resnet_stages_num: 3|4|5 (depth of backbone features considered)
      token_len: number of semantic tokens per image
      enc_depth, dec_depth: encoder/decoder layer counts
      tokenizer: use semantic tokenizer (True) or pooled tokens (False)
      with_decoder: use transformer decoder (True) or simple additive decode (False)
      with_decoder_pos: None | "fix" | "learned" (pos emb on feature map for decoder)
      backbone: "resnet18" | "resnet34" | "resnet50"
      pretrained_backbone: load ImageNet weights for backbone
    """
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        variant: Optional[str] = "base_transformer_pos_s4_dd8",
        *,
        # fine-grained knobs (overridden by variant if provided)
        with_pos: Optional[str] = "learned",
        resnet_stages_num: int = 4,
        token_len: int = 4,
        enc_depth: int = 1,
        dec_depth: int = 8,
        tokenizer: bool = True,
        if_upsample_2x: bool = True,
        pool_mode: str = "max",
        pool_size: int = 2,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        decoder_softmax: bool = True,
        with_decoder_pos: Optional[str] = None,  # None | 'fix' | 'learned'
        with_decoder: bool = True,
    ):
        super().__init__()
        # Apply presets if variant is given
        if variant is not None:
            if variant == "base_transformer_pos_s4":
                with_pos, token_len, resnet_stages_num, enc_depth, dec_depth = "learned", 4, 4, 1, 1
            elif variant == "base_transformer_pos_s4_dd8":
                with_pos, token_len, resnet_stages_num, enc_depth, dec_depth = "learned", 4, 4, 1, 8
            elif variant == "base_transformer_pos_s4_dd8_dedim8":
                # kept for API parity; PyTorch's Transformer doesn't expose head dim directly
                with_pos, token_len, resnet_stages_num, enc_depth, dec_depth = "learned", 4, 4, 1, 8
            else:
                raise ValueError(f"Unknown BIT variant: {variant}")

        self.output_nc = output_nc
        self.token_len = token_len
        self.tokenizer = tokenizer
        self.token_trans = True
        self.with_decoder = with_decoder
        self.with_pos = with_pos
        self.with_decoder_pos = with_decoder_pos
        self.if_upsample_2x = if_upsample_2x
        self.pool_mode = pool_mode
        self.pool_size = pool_size
        self.decoder_softmax = decoder_softmax

        # Backbone to 32-ch features
        self.backbone = ResNetBackbone(
            backbone=backbone,
            stages=resnet_stages_num,
            if_upsample_2x=if_upsample_2x,
            out_ch=32,
            pretrained=pretrained_backbone,
            in_ch=input_nc,
        )

        dim = 32
        mlp_dim = 2 * dim
        nhead = 8

        # Semantic tokenizer (attention over spatial map)
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)

        # Positional embeddings for tokens (learned)
        if self.with_pos == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, dim))
        else:
            self.register_parameter("pos_embedding", None)

        # Decoder positional emb over feature map (optional; created lazily)
        self.register_parameter("pos_embedding_decoder", None)

        # ------- PyTorch Transformers (pre-norm, batch_first) -------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)
        # -------------------------------------------------------------

        # Upsampling and head
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

    # ---------- tokenizers ----------
    def _semantic_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted spatial pooling to token set (B, L, C)."""
        b, c, h, w = x.shape
        attn = self.conv_a(x)                         # [B, L, H, W]
        attn = attn.view(b, self.token_len, -1)
        attn = attn.softmax(dim=-1)                   # spatial softmax
        x_flat = x.view(b, c, -1)                     # [B, C, HW]
        tokens = torch.einsum("bln,bcn->blc", attn, x_flat)  # [B, L, C]
        return tokens

    def _pooled_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: reshape pooled feature map into tokens (B, L, C)."""
        if self.pool_mode == "max":
            y = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == "ave":
            y = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            y = x
        b, c, h, w = y.shape
        return y.permute(0, 2, 3, 1).reshape(b, h * w, c)

    # ---------- transformer helpers ----------
    def _encode_tokens(self, tokens_cat: torch.Tensor) -> torch.Tensor:
        if self.pos_embedding is not None and tokens_cat.shape[1] == self.pos_embedding.shape[1]:
            tokens_cat = tokens_cat + self.pos_embedding
        return self.transformer(tokens_cat)  # [B, 2L, C]

    def _decode_to_feature(self, feat: torch.Tensor, mem_tokens: torch.Tensor) -> torch.Tensor:
        """Inject memory tokens back into spatial feature map via decoder."""
        b, c, h, w = feat.shape
        if self.with_decoder_pos in ("fix", "learned"):
            if (self.pos_embedding_decoder is None) or (self.pos_embedding_decoder.shape[-2:] != (h, w)):
                pe = torch.zeros(1, c, h, w, device=feat.device)
                nn.init.normal_(pe, std=0.02)
                self.pos_embedding_decoder = nn.Parameter(pe) if self.with_decoder_pos == "learned" else pe.detach()
            feat = feat + self.pos_embedding_decoder

        q = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, Nq, C]
        mem = mem_tokens
        if self.decoder_softmax:
            mem = mem.softmax(dim=1)                       # optional quirk kept for parity

        q = self.transformer_decoder(tgt=q, memory=mem)    # [B, Nq, C]
        q = q.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return q

    def _simple_decode(self, feat: torch.Tensor, mem_tokens: torch.Tensor) -> torch.Tensor:
        """Simpler additive broadcast of summed tokens (fallback)."""
        m = mem_tokens.sum(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return feat + m

    # ---------- forward ----------
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 1) shared backbone features
        f1 = self.forward_backbone(x1)   # [B, 32, Hb, Wb]
        f2 = self.forward_backbone(x2)

        # 2) tokens per image
        if self.tokenizer:
            t1 = self._semantic_tokens(f1)   # [B, L, 32]
            t2 = self._semantic_tokens(f2)
        else:
            t1 = self._pooled_tokens(f1)
            t2 = self._pooled_tokens(f2)

        # 3) transformer encoder over concatenated tokens
        tokens_cat = torch.cat([t1, t2], dim=1)       # [B, 2L, 32]
        tokens_cat = self._encode_tokens(tokens_cat)
        t1, t2 = tokens_cat.chunk(2, dim=1)

        # 4) transformer decoder (or simple add) to inject tokens into feats
        if self.with_decoder:
            f1 = self._decode_to_feature(f1, t1)
            f2 = self._decode_to_feature(f2, t2)
        else:
            f1 = self._simple_decode(f1, t1)
            f2 = self._simple_decode(f2, t2)

        # 5) feature differencing + upsample + head
        x = torch.abs(f1 - f2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)        # parity with some configs
        x = self.upsamplex4(x)
        x = self.classifier(x)            # [B, output_nc, H, W]
        return x


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    model = BIT(variant="base_transformer_pos_s4_dd8")  # or "base_transformer_pos_s4"
    model.eval()
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    y = model(x1, x2)
    print("Output:", y.shape)  # expect [1, 2, 256, 256]
