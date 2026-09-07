"""Landmark regression losses, selectable by config (train.loss: l2 | wing).

Both losses see errors in PIXELS of the model input (coords are stored in
[0, 1]; multiplying by input_size gives the Wing parameters their standard
pixel meaning from the paper and makes L2 magnitudes interpretable).

Wing loss (Feng et al., CVPR 2018), element-wise on coordinate errors:
    wing(x) = w * ln(1 + |x|/eps)   if |x| < w
              |x| - C               otherwise,  C = w - w*ln(1 + w/eps)
It behaves like a scaled log near zero - amplifying the small errors that
dominate the end of landmark training - and like L1 for large errors.
"""

from __future__ import annotations

import torch

from dms_layer1.config import require


def l2_loss(pred_px: torch.Tensor, target_px: torch.Tensor) -> torch.Tensor:
    return ((pred_px - target_px) ** 2).mean()


def wing_loss(pred_px: torch.Tensor, target_px: torch.Tensor,
              w: float, epsilon: float) -> torch.Tensor:
    x = (pred_px - target_px).abs()
    c = w - w * torch.log1p(torch.tensor(w / epsilon, device=x.device))
    return torch.where(x < w, w * torch.log1p(x / epsilon), x - c).mean()


def make_loss(cfg: dict):
    """Returns loss_fn(pred01, target01) -> scalar, reading kind and
    parameters from the config."""
    kind = require(cfg, "train.loss")
    input_size = float(require(cfg, "model.input_size"))
    if kind == "l2":
        return lambda p, t: l2_loss(p * input_size, t * input_size)
    if kind == "wing":
        w = float(require(cfg, "train.wing.w"))
        eps = float(require(cfg, "train.wing.epsilon"))
        return lambda p, t: wing_loss(p * input_size, t * input_size, w, eps)
    raise ValueError(f"Unknown train.loss '{kind}' - expected 'l2' or 'wing'")
