# -*- coding: utf-8 -*-
# Self-contained STANet-like CD model with in-file backbone + attention.
# Returns *logits only* (shape [B,1,H,W]) suitable for BCEWithLogitsLoss.

from typing import Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Small utility blocks
# -----------------------------
def conv_bn_relu(c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None, bias: bool = False):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=bias),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


# -----------------------------
# Tiny CNN backbone (replacement for F_mynet3)
# Produces a feature map with stride = output_stride and channels = f_c
# -----------------------------
class F_mynet3(nn.Module):
    """
    A compact CNN backbone:
      stem -> down blocks until reaching output_stride -> 1x1 conv to f_c
    """
    def __init__(self, backbone: str = "mini", in_c: int = 3, f_c: int = 64, output_stride: int = 32):
        super().__init__()
        assert output_stride in (8, 16, 32), "output_stride must be one of {8,16,32}"
        chans = [64, 128, 256, 256, 256]  # small but effective
        downs_needed = {8: 3, 16: 4, 32: 5}[output_stride]

        layers: List[nn.Module] = []
        c_prev = in_c
        for i in range(downs_needed):
            c_out = chans[i]
            # each stage: stride-2 downsample; then a refinement conv
            layers.append(conv_bn_relu(c_prev, c_out, k=3, s=2))  # downsample
            layers.append(conv_bn_relu(c_out, c_out, k=3, s=1))
            c_prev = c_out

        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_prev, f_c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        x = self.proj(x)
        return x  # [B, f_c, H/OS, W/OS]


# Convenience factory matching your earlier API
def define_F(in_c: int, f_c: int, type: str = 'mynet3', output_stride: int = 32):
    if type == 'mynet3':
        return F_mynet3(backbone='mini', in_c=in_c, f_c=f_c, output_stride=output_stride)
    raise NotImplementedError(f'no such F type: {type!r}')


# -----------------------------
# BAM: Basic self-attention (bitemporal via width concat)
# -----------------------------
class BAM(nn.Module):
    """ Basic self-attention module (operates on concatenated [B,C,H,2W]). """
    def __init__(self, in_dim: int, ds: int = 8, activation=nn.ReLU):
        super().__init__()
        self.ch_in = in_dim
        self.key_ch = max(1, in_dim // 8)
        self.ds = max(1, ds)
        self.pool = nn.AvgPool2d(self.ds)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.key_ch, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=self.key_ch, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,  kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B,C,H,2W] (two times concatenated along width)
        x = self.pool(input)
        B, C, H, W2 = x.shape
        # W = W2 // 2  # (kept for reference)

        q = self.query_conv(x).reshape(B, self.key_ch, H * W2).permute(0, 2, 1)  # [B,HW2,key]
        k = self.key_conv(x).reshape(B, self.key_ch, H * W2)                     # [B,key,HW2]
        v = self.value_conv(x).reshape(B, C,          H * W2)                    # [B,C,HW2]

        sim = torch.bmm(q, k) * (self.key_ch ** -0.5)  # [B,HW2,HW2]
        att = self.softmax(sim)
        out = torch.bmm(v, att.permute(0, 2, 1))       # [B,C,HW2]
        out = out.view(B, C, H, W2)

        # upsample back (no-op when ds==1)
        if self.ds != 1:
            out = F.interpolate(out, size=(H * self.ds, W2 * self.ds), mode='bilinear', align_corners=False)

        out = out + input
        return out


# -----------------------------
# PAM (pyramid) attention working on [B,C,H,2W]
# -----------------------------
class _PAMBlock(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, value_channels: int, scale: int = 1, ds: int = 1):
        super().__init__()
        self.scale = max(1, scale)
        self.ds = max(1, ds)
        self.pool = nn.AvgPool2d(self.ds)

        self.f_key   = nn.Sequential(nn.Conv2d(in_channels, key_channels,   1, 1, 0, bias=False),
                                     nn.BatchNorm2d(key_channels))
        self.f_query = nn.Sequential(nn.Conv2d(in_channels, key_channels,   1, 1, 0, bias=False),
                                     nn.BatchNorm2d(key_channels))
        self.f_value = nn.Conv2d(in_channels, value_channels, 1, 1, 0, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.pool(input) if self.ds != 1 else input
        B, C, H, W2 = x.shape
        W = W2 // 2

        # grid of local windows
        step_h, step_w = max(1, H // self.scale), max(1, W // self.scale)
        local_x, local_y = [], []
        for i in range(self.scale):
            for j in range(self.scale):
                sx, sy = i * step_h, j * step_w
                ex, ey = (H if i == self.scale - 1 else min(sx + step_h, H)), (W if j == self.scale - 1 else min(sy + step_w, W))
                local_x += [sx, ex]; local_y += [sy, ey]

        v = self.f_value(x)
        q = self.f_query(x)
        k = self.f_key(x)

        # split along time (width)
        v = torch.stack([v[..., :W], v[..., W:]], dim=4)  # [B,C,H,W,2]
        q = torch.stack([q[..., :W], q[..., W:]], dim=4)
        k = torch.stack([k[..., :W], k[..., W:]], dim=4)

        def attend(v_local, q_local, k_local):
            Bn, Cv, Hl, Wl, T = v_local.shape  # T=2
            v_flat = v_local.view(Bn, Cv, Hl * Wl * T)                  # [Bn,Cv,N]
            q_flat = q_local.view(Bn, -1, Hl * Wl * T).permute(0, 2, 1) # [Bn,N,Ck]
            k_flat = k_local.view(Bn, -1, Hl * Wl * T)                  # [Bn,Ck,N]
            sim = torch.bmm(q_flat, k_flat) * (k_flat.size(1) ** -0.5)  # [Bn,N,N]
            att = F.softmax(sim, dim=-1)
            ctx = torch.bmm(v_flat, att.permute(0, 2, 1))               # [Bn,Cv,N]
            return ctx.view(Bn, Cv, Hl, Wl, T)

        # parallel over local blocks
        blocks = 2 * self.scale * self.scale
        v_locals = torch.cat([v[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, blocks, 2)], dim=0)
        q_locals = torch.cat([q[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, blocks, 2)], dim=0)
        k_locals = torch.cat([k[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]] for i in range(0, blocks, 2)], dim=0)
        ctx_locals = attend(v_locals, q_locals, k_locals)

        # stitch back
        rows = []
        for i in range(self.scale):
            cols = []
            for j in range(self.scale):
                left = B * (j + i * self.scale)
                right = left + B
                cols.append(ctx_locals[left:right])
            rows.append(torch.cat(cols, dim=3))
        ctx = torch.cat(rows, dim=2)                              # [B,C,H,W,2]
        ctx = torch.cat([ctx[..., 0], ctx[..., 1]], dim=3)        # [B,C,H,2W]

        if self.ds != 1:
            ctx = F.interpolate(ctx, size=(H * self.ds, 2 * W * self.ds), mode='bilinear', align_corners=False)
        return ctx


class PAM(nn.Module):
    """ Pyramid attention over concatenated width (two times). """
    def __init__(self, in_channels: int, out_channels: int, sizes=(1, 2, 4, 8), ds: int = 1):
        super().__init__()
        self.stages = nn.ModuleList([_PAMBlock(in_channels, max(1, out_channels // 8), out_channels, size, ds)
                                     for size in sizes])
        self.conv_bn = nn.Conv2d(in_channels * len(self.stages), out_channels, kernel_size=1, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        priors = [stage(feats) for stage in self.stages]
        out = self.conv_bn(torch.cat(priors, dim=1))
        return out


# -----------------------------
# CDSA wrapper: apply BAM or PAM to (f1,f2)
# -----------------------------
class CDSA(nn.Module):
    """Change-Detection Self-Attention wrapper to process two feature maps."""
    def __init__(self, in_c: int, ds: int = 1, mode: str = 'BAM'):
        super().__init__()
        mode = mode.upper()
        if mode == 'BAM':
            self.att = BAM(in_dim=in_c, ds=ds)
        elif mode == 'PAM':
            self.att = PAM(in_channels=in_c, out_channels=in_c, sizes=(1, 2, 4, 8), ds=ds)
        else:
            raise ValueError(f"Unknown attention mode: {mode!r}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # concat along width, attend, split by *width*, not height
        W = x1.shape[-1]
        x = torch.cat([x1, x2], dim=3)    # [B,C,H, 2W]
        x = self.att(x)                   # [B,C,H, 2W]
        return x[..., :W], x[..., W:]


# -----------------------------
# Top model: STANet-style CD model
# -----------------------------
class STANet(nn.Module):
    """
    Emits *binary logits* [B,1,H,W] (no sigmoid/softmax).
    Set head_type='feat' to predict from |fA-fB| (recommended),
    or 'dist' to predict from per-pixel L2 distance.
    """
    def __init__(
        self,
        in_c: int = 3,
        f_c: int = 64,
        arch: str = 'mynet3',
        output_stride: int = 16,
        sa_mode: str = 'PAM',
        num_classes: int = 1,        # 1 channel for BCEWithLogitsLoss
        head_type: str = 'feat',     # 'feat' | 'dist'
        prior_p: float = 0.05,       # prior positive rate for bias init
        ds_att: int = 1,
    ):
        super().__init__()
        assert num_classes == 1, "This implementation emits a single logit channel."
        assert head_type in ('feat', 'dist')
        self.head_type = head_type

        self.netF = define_F(in_c=in_c, f_c=f_c, type=arch, output_stride=output_stride)
        self.netA = CDSA(in_c=f_c, ds=ds_att, mode=sa_mode)

        if head_type == 'feat':
            # Predict from feature differences (faster learning, wider logit range)
            self.head = nn.Sequential(
                nn.Conv2d(f_c, f_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(f_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_c, 1, 1, bias=True),
            )
            # Bias to match class prior (logit(p))
            with torch.no_grad():
                self.head[-1].bias.fill_(math.log(prior_p / (1 - prior_p)))
        else:
            # Predict from distance map (non-negative). Add mild norm + learnable scale.
            self.pre_head = nn.BatchNorm2d(1, affine=True)
            self.logit_scale = nn.Parameter(torch.tensor(2.0))
            self.classifier = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            with torch.no_grad():
                self.classifier.weight.zero_()
                self.classifier.bias.fill_(math.log(prior_p / (1 - prior_p)))

    def forward_feats(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fA = self.netF(A)
        fB = self.netF(B)
        fA, fB = self.netA(fA, fB)
        return fA, fB

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        fA, fB = self.forward_feats(A, B)

        if self.head_type == 'feat':
            feat = torch.abs(fA - fB)                         # [B,C,h,w]
            logits_low = self.head(feat)                      # [B,1,h,w]
        else:
            dist = torch.norm(fA - fB, p=2, dim=1, keepdim=True)  # [B,1,h,w] >= 0
            x = self.pre_head(dist) * self.logit_scale
            logits_low = self.classifier(x)                   # [B,1,h,w]

        logits = F.interpolate(logits_low, size=A.shape[-2:], mode='bilinear', align_corners=False)
        return logits  # raw logits suitable for BCEWithLogitsLoss


# -----------------------------
# Smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 32, 2, 128, 128
    A = torch.randn(B, C, H, W)
    Bx = torch.randn(B, C, H, W)

    # Try the recommended 'feat' head
    model = STANet(in_c=2, f_c=64, arch='mynet3', output_stride=16, sa_mode='PAM',
                   num_classes=1, head_type='feat', prior_p=0.05)

    logits = model(A, Bx)
    print("logits:", tuple(logits.shape))
    print(f"logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # Uncomment to test the 'dist' head instead
    # model_dist = STANet(in_c=2, f_c=64, arch='mynet3', output_stride=16, sa_mode='PAM',
    #                     num_classes=1, head_type='dist', prior_p=0.05)
    # logits_d = model_dist(A, Bx)
    # print("logits(dist-head):", tuple(logits_d.shape))
    # print(f"logits(dist-head) range: [{logits_d.min().item():.4f}, {logits_d.max().item():.4f}]")