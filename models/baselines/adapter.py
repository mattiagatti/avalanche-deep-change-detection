import torch.nn as nn
import torch.nn.functional as F


class CDModelAdapter(nn.Module):
    """
    Unifies forward() across different CD backbones.

    Always returns a single-channel logits tensor [B,1,H,W]
    where (H,W) == pre.shape[-2:].
    """

    def __init__(self, core_model: nn.Module, model_name: str):
        super().__init__()
        self.core = core_model
        self.model_name = model_name.lower()

    def _select_output(self, out):
        # If model returns multi-scale [list/tuple], keep the last
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out

    def _to_single_change_logit(self, logits):
        # If the model outputs 2 channels (bg, change), keep the "change" logit
        if logits.ndim == 4 and logits.size(1) == 2:
            logits = logits[:, 1:2, ...]
        return logits

    def _resize_to(self, x, target_spatial):
        if x.shape[-2:] != target_spatial:
            x = F.interpolate(x, size=target_spatial, mode="bilinear", align_corners=False)
        return x

    def forward(self, pre, post):
        name = self.model_name

        if name in {
            "bit",
            "changeformer",
            "siamunet_conc",
            "siamunet_diff",
            "snunet",
            "stanet",
            "stnet",
            "swinunet",
            "tinycd",
        }:
            out = self.core(pre, post)
        else:
            # Fallback: try (pre, post)
            out = self.core(pre, post)

        out = self._select_output(out)
        out = self._to_single_change_logit(out)

        # Always match input spatial size
        out = self._resize_to(out, pre.shape[-2:])
        return out