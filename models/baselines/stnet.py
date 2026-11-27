import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from typing import Sequence, List, Optional
import numpy as np

# Utilities from timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ======================================================================================
#                             Helpers for input channel adaptation
# ======================================================================================

def _adapt_in_channels_weight(w: torch.Tensor, in_channels: int) -> torch.Tensor:
    """
    Adapt a conv weight from (out_c, C_old, k, k) to (out_c, in_channels, k, k).
    For grayscale: average RGB. For >3: repeat and trim. For 2: take first two channels.
    """
    if w.shape[1] == in_channels:
        return w
    if in_channels == 1:
        w = w.mean(dim=1, keepdim=True)
    elif in_channels == 2:
        if w.shape[1] >= 2:
            w = w[:, :2, :, :]
        else:
            # If original had 1 channel, repeat once
            w = w.repeat(1, 2, 1, 1)
    elif in_channels > 3:
        reps = int(np.ceil(in_channels / w.shape[1]))
        w = w.repeat(1, reps, 1, 1)[:, :in_channels, :, :]
    else:  # in_channels == 3 but source not 3, or other odd case -> safest is average then expand
        base = w.mean(dim=1, keepdim=True)
        w = base.repeat(1, in_channels, 1, 1)
    return w

# ======================================================================================
#                                      ResNet
# ======================================================================================

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3x3(in_planes, outplanes, stride=1):
    return nn.Conv2d(in_planes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_inplace(out)
        return out

class Resnet(nn.Module):
    def __init__(self, block, layers, out_stride=32, use_stem=False, stem_channels=64, in_channels=3):
        super().__init__()
        self.inplanes = 64
        outstride_to_strides_and_dilations = {
            8:  ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        stride_list, dilation_list = outstride_to_strides_and_dilations[out_stride]
        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels//2, stride=2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),
                conv3x3(stem_channels//2, stem_channels//2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),
                conv3x3(stem_channels//2, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  blocks=layers[0], stride=stride_list[0], dilation=dilation_list[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=stride_list[1], dilation=dilation_list[1])
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=stride_list[2], dilation=dilation_list[2])
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=stride_list[3], dilation=dilation_list[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, contract_dilation=True):
        downsample = None
        dilations = [dilation] * blocks
        if contract_dilation and dilation > 1:
            dilations[0] = dilation // 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, dilation=dilations[0], downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)   # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        return (x1, x2, x3, x4)

def get_resnet18(pretrained=True, in_channels=3):
    model = Resnet(BasicBlock, [2, 2, 2, 2], out_stride=32, use_stem=False, in_channels=in_channels)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        state_dict = checkpoint.get('state_dict', checkpoint)
        # adapt first conv if needed
        if 'conv1.weight' in state_dict and in_channels != state_dict['conv1.weight'].shape[1]:
            state_dict['conv1.weight'] = _adapt_in_channels_weight(state_dict['conv1.weight'], in_channels)
        model.load_state_dict(state_dict, strict=False)
    return model

# ======================================================================================
#                                   Swin Transformer
# ======================================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None; self.W = None
    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        dpr = drop_path if isinstance(drop_path, list) else [drop_path]*depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt; cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0,1,2,3),
                 frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.apply(self._init_weights)
        for i_layer in out_indices:
            layer = norm_layer(self.num_features[i_layer])
            self.add_module(f'norm{i_layer}', layer)
        self._freeze_stages()
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters(): p.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        # no absolute pos embed by default
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

def swin_tiny(pretrained: bool = False, weight_path: Optional[str] = None, in_chans: int = 3):
    model = SwinTransformer(in_chans=in_chans, embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path, map_location='cpu')
        state_dict = old_dict.get('state_dict', old_dict)
        # adapt patch embed if needed
        k = 'patch_embed.proj.weight'
        if k in state_dict and state_dict[k].shape[1] != in_chans:
            state_dict[k] = _adapt_in_channels_weight(state_dict[k], in_chans)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)
    return model

# ======================================================================================
#                                   STNetHead (Decoder)
# ======================================================================================

def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super().__init__()
        self.key_project = self._build_project(key_in_channels, transform_channels, key_query_num_convs)
        self.query_project = self._build_project(query_in_channels, transform_channels, key_query_num_convs)
        self.value_project = self._build_project(key_in_channels, transform_channels, value_out_num_convs)
        self.out_project = self._build_project(transform_channels, out_channels, value_out_num_convs)
        self.transform_channels = transform_channels
    def _build_project(self, in_channels, out_channels, num_convs):
        layers: List[nn.Module] = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for _ in range(num_convs - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    def forward(self, query_feats, key_feats, value_feats):
        B = query_feats.size(0)
        query = self.query_project(query_feats).reshape(B, -1, query_feats.shape[2]*query_feats.shape[3]).permute(0,2,1)
        key   = self.key_project(key_feats).reshape(B, -1, key_feats.shape[2]*key_feats.shape[3])
        value = self.value_project(value_feats).reshape(B, -1, value_feats.shape[2]*value_feats.shape[3]).permute(0,2,1)
        sim_map = torch.matmul(query, key) * (self.transform_channels ** -0.5)
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value).permute(0,2,1).contiguous()
        context = context.view(B, -1, *query_feats.shape[2:])
        context = self.out_project(context)
        return context

class SFF(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big   = conv_1x1(in_channel, in_channel)
        self.catconv    = conv_3x3(in_channel*2, in_channel)
        self.attention  = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels=in_channel,
            transform_channels=in_channel // 2,
            out_channels=in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
    def forward(self, x_small, x_big):
        img_size = (x_big.size(2), x_big.size(3))
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)
        new_x = self.attention(x, x, x_big)
        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out

class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv  = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, xA, xB):
        x_diff = xA - xB
        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))
        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))
        xA = A_weight * xA
        xB = B_weight * xB
        x = self.catconv(torch.cat([xA, xB], dim=1))
        return x

class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.channel_attention = ChannelAttention(in_channel*layer_num)
        self.catconv = conv_3x3(in_channel*layer_num, in_channel)
        self.decoder = nn.Conv2d(in_channel, num_class, 1)
    def forward(self, x1, x2, x3, x4=None):
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
        if self.layer_num == 4 and x4 is not None:
            x4 = F.interpolate(x4, scale_factor=8, mode="bilinear", align_corners=False)
            x = torch.cat([x1, x2, x3, x4], dim=1)
        else:
            x = torch.cat([x1, x2, x3], dim=1)
        out = self.channel_attention(x) * x
        out = self.decoder(self.catconv(out))
        return out

class STNetHead(nn.Module):
    """
    The decoder head.
    """
    def __init__(self, num_class, channel_list, transform_feat, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.tff1 = TFF(channel_list[0], transform_feat)
        self.tff2 = TFF(channel_list[1], transform_feat)
        self.tff3 = TFF(channel_list[2], transform_feat)
        self.tff4 = TFF(channel_list[3], transform_feat) if layer_num == 4 else None
        self.sff1 = SFF(transform_feat)
        self.sff2 = SFF(transform_feat)
        self.sff3 = SFF(transform_feat)
        self.lightdecoder = LightDecoder(transform_feat, num_class, layer_num)
    def forward(self, x):
        featuresA, featuresB = x
        xA1, xA2, xA3, xA4 = featuresA
        xB1, xB2, xB3, xB4 = featuresB
        x1 = self.tff1(xA1, xB1)
        x2 = self.tff2(xA2, xB2)
        x3 = self.tff3(xA3, xB3)
        x4 = self.tff4(xA4, xB4) if (self.layer_num == 4 and self.tff4 is not None) else None
        xlast = x4 if x4 is not None else x3
        x1_new = self.sff1(xlast, x1)
        x2_new = self.sff2(xlast, x2)
        x3_new = self.sff3(x4 if x4 is not None else x3, x3)
        out = self.lightdecoder(x1_new, x2_new, x3_new, x4)
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)
        return out

# ======================================================================================
#                          Two-image Backbone Wrappers
# ======================================================================================

class ResnetBackboneWrapper(nn.Module):
    def __init__(self, pretrained=True, in_channels=2):
        super().__init__()
        self.backbone = get_resnet18(pretrained=pretrained, in_channels=in_channels)
    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> List[torch.Tensor]:
        xA1, xA2, xA3, xA4 = self.backbone(xA)
        xB1, xB2, xB3, xB4 = self.backbone(xB)
        return [xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4]

class SwinBackboneWrapper(nn.Module):
    def __init__(self, pretrained=False, weight_path: Optional[str] = None, in_channels=2):
        super().__init__()
        self.backbone = swin_tiny(pretrained=pretrained, weight_path=weight_path, in_chans=in_channels)
    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> List[torch.Tensor]:
        xA1, xA2, xA3, xA4 = self.backbone(xA)
        xB1, xB2, xB3, xB4 = self.backbone(xB)
        return [xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4]

# ======================================================================================
#                               Top-level Model: STNet
# ======================================================================================

class STNet(nn.Module):
    """
    End-to-end model (backbone + STNetHead).

    Args:
      backbone_name: {"Resnet18", "SwinTiny"}
      num_class: output classes
      channel_list: feature channels per level from backbone
                    ResNet18: [64,128,256,512]
                    SwinTiny: [96,192,384,768]
      transform_feat: internal feature size in STNetHead
      layer_num: 3 or 4
      swin_pretrained / swin_weight_path: load Swin checkpoint if desired
      in_channels: number of input channels per image (default 2)
    """
    def __init__(self,
                 backbone_name: str = "Resnet18",
                 num_class: int = 2,
                 channel_list: Optional[Sequence[int]] = None,
                 transform_feat: int = 128,
                 layer_num: int = 4,
                 swin_pretrained: bool = False,
                 swin_weight_path: Optional[str] = None,
                 in_channels: int = 2):
        super().__init__()
        bb = backbone_name.lower()
        if bb in ("resnet18", "resnet"):
            self.backbone = ResnetBackboneWrapper(pretrained=True, in_channels=in_channels)
            if channel_list is None:
                channel_list = [64, 128, 256, 512]
        elif bb in ("swintiny", "swin_tiny", "swin"):
            self.backbone = SwinBackboneWrapper(pretrained=swin_pretrained,
                                                weight_path=swin_weight_path,
                                                in_channels=in_channels)
            if channel_list is None:
                channel_list = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.head = STNetHead(
            num_class=num_class,
            channel_list=list(channel_list),
            transform_feat=transform_feat,
            layer_num=layer_num,
        )

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(xA, xB)  # [A1..A4, B1..B4]
        featuresA = feats[:4]
        featuresB = feats[4:]
        logits = self.head((featuresA, featuresB))  # (B, num_class, H, W)
        return logits

# ======================================================================================
#                               Config-friendly builder
# ======================================================================================

def build_model_from_config(model_config: dict, in_channels: int = 2) -> nn.Module:
    backbone_cfg = model_config.get("backbone", {})
    decoder_cfg  = model_config.get("decoderhead", {})
    backbone_name = backbone_cfg.get("name", "Resnet18")
    num_class     = int(decoder_cfg.get("num_class", 2))
    channel_list  = decoder_cfg.get("channel_list", None)  # if None, auto by backbone
    transform_feat= int(decoder_cfg.get("transform_feat", 128))
    layer_num     = int(decoder_cfg.get("layer_num", 4))
    return STNet(
        backbone_name=backbone_name,
        num_class=num_class,
        channel_list=channel_list,
        transform_feat=transform_feat,
        layer_num=layer_num,
        in_channels=in_channels,
    )

# ======================================================================================
#                                      Smoke test
# ======================================================================================

if __name__ == "__main__":
    B,C,H,W = 2,2,256,256   # C=2 by default
    net = STNet(
        backbone_name="Resnet18",
        num_class=2,
        channel_list=[64,128,256,512],
        transform_feat=128,
        layer_num=4,
        in_channels=2,   # default
    )
    y = net(torch.randn(B,C,H,W), torch.randn(B,C,H,W))
    print("Output shape:", y.shape)  # (B, 2, H, W)