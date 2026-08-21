"""The landmark regression network (milestone 4).

A deliberately small convolutional stack trained from random initialisation
— no pretrained weights, by project design. Input is a grayscale face crop;
output is 24 (x, y) pairs in [0, 1] crop coordinates, regressed directly by
a single linear layer.

Architecture (width = w, default 32):
    stem  : 3x3 conv 1 -> w/2, stride 2          (112 -> 56)
    stage1: 3x3 conv w/2 -> w, 3x3 stride-2 w    (56 -> 28)
    stage2: 3x3 conv w -> 2w, 3x3 stride-2 2w    (28 -> 14)
    stage3: 3x3 conv 2w -> 4w, 3x3 stride-2 4w   (14 -> 7)
    head  : flatten -> linear -> 48

Every conv is conv-BN-ReLU. At width 32 and input 112 this is ~0.59M
parameters, ~2.4 MB in fp32 — comfortably under the 5 MB budget. The final
layer's bias is initialised to 0.5 so the untrained model predicts the crop
centre instead of random scatter.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class LandmarkNet(nn.Module):
    def __init__(self, num_points: int = 24, width: int = 32, input_size: int = 112):
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError(f"input_size must be divisible by 16, got {input_size}")
        w = width
        self.num_points = num_points
        self.features = nn.Sequential(
            _conv_bn_relu(1, w // 2, stride=2),
            _conv_bn_relu(w // 2, w),
            _conv_bn_relu(w, w, stride=2),
            _conv_bn_relu(w, 2 * w),
            _conv_bn_relu(2 * w, 2 * w, stride=2),
            _conv_bn_relu(2 * w, 4 * w),
            _conv_bn_relu(4 * w, 4 * w, stride=2),
        )
        feat_side = input_size // 16
        self.head = nn.Linear(4 * w * feat_side * feat_side, num_points * 2)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.5)   # start at the crop centre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, S, S) normalised grayscale -> (B, num_points, 2) in
        [0, 1] crop coordinates (linear output; slight out-of-range allowed)."""
        f = self.features(x)
        out = self.head(f.flatten(1))
        return out.view(-1, self.num_points, 2)


def model_size_mb(model: nn.Module) -> float:
    """fp32 parameter size in MB (the number reported next to NME)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / 1e6
