"""Training data pipeline: on-the-fly augmentation over the RAM-resident
crop cache (milestone 4).

Augmentations (all ranges from config train.augment): horizontal flip WITH
landmark index remapping (the schema's flip_permutation - the classic silent
bug, unit-tested in tests/test_augment.py), rotation, scale, translation,
brightness/contrast, and blur. Geometry is applied as ONE affine warp from
the cached crop (cache_size px) straight to the model input (input_size px),
and landmark coordinates go through exactly the same matrix.

Determinism: the per-sample RNG is seeded from (base_seed, epoch, index), so
an epoch's augmentations are a pure function of the config seed and the
epoch number - a resumed run regenerates identical batches without having to
persist dataloader RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dms_layer1.config import require
from dms_layer1.data.crops import PAD_VALUE


@dataclass(frozen=True)
class AugmentParams:
    rotation_deg: float
    scale_lo: float
    scale_hi: float
    translate_frac: float
    brightness: float        # +- gray levels
    contrast: float          # alpha in [1-contrast, 1+contrast]
    blur_prob: float
    flip_prob: float

    @staticmethod
    def from_cfg(cfg: dict) -> "AugmentParams":
        a = require(cfg, "train.augment")
        return AugmentParams(
            rotation_deg=float(a["rotation_deg"]),
            scale_lo=float(a["scale"][0]), scale_hi=float(a["scale"][1]),
            translate_frac=float(a["translate_frac"]),
            brightness=float(a["brightness"]), contrast=float(a["contrast"]),
            blur_prob=float(a["blur_prob"]), flip_prob=float(a["flip_prob"]),
        )


def augment_face(crop: np.ndarray, lm01: np.ndarray, rng: np.random.Generator,
                 aug: AugmentParams, out_size: int,
                 flip_perm: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """One augmented sample: (out_size, out_size) uint8 + (24, 2) float in
    [0, 1] of the output. Image and landmarks share every transform."""
    size = crop.shape[0]
    lm01 = lm01.astype(np.float64)

    if rng.random() < aug.flip_prob:
        crop = crop[:, ::-1]
        lm01 = lm01.copy()
        lm01[:, 0] = 1.0 - lm01[:, 0]
        lm01 = lm01[list(flip_perm)]

    angle = rng.uniform(-aug.rotation_deg, aug.rotation_deg)
    scale = rng.uniform(aug.scale_lo, aug.scale_hi)
    tx = rng.uniform(-aug.translate_frac, aug.translate_frac) * out_size
    ty = rng.uniform(-aug.translate_frac, aug.translate_frac) * out_size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale * out_size / size)
    m[0, 2] += out_size / 2 - size / 2 + tx
    m[1, 2] += out_size / 2 - size / 2 + ty
    out = cv2.warpAffine(np.ascontiguousarray(crop), m, (out_size, out_size),
                         flags=cv2.INTER_LINEAR, borderValue=PAD_VALUE)
    pts = lm01 * size
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    lm_out = pts / out_size

    alpha = rng.uniform(1 - aug.contrast, 1 + aug.contrast)
    beta = rng.uniform(-aug.brightness, aug.brightness)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < aug.blur_prob:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out, lm_out.astype(np.float32)


def eval_transform(crop: np.ndarray, lm01: np.ndarray,
                   out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic path: resize the whole crop; [0, 1] coords are
    resolution-independent, so the landmarks are unchanged."""
    out = cv2.resize(crop, (out_size, out_size),
                     interpolation=cv2.INTER_AREA if crop.shape[0] >= out_size
                     else cv2.INTER_LINEAR)
    return out, lm01.astype(np.float32)


class CachedFaceDataset(Dataset):
    """Wraps the in-RAM cache arrays. `augment=None` gives the deterministic
    eval path. Call set_epoch(e) before each training epoch so the
    per-sample RNG seeds advance reproducibly."""

    def __init__(self, crops: np.ndarray, landmarks: np.ndarray,
                 indices: np.ndarray, out_size: int,
                 pixel_mean: float, pixel_std: float,
                 augment: AugmentParams | None = None,
                 flip_perm: tuple[int, ...] | None = None,
                 base_seed: int = 0):
        if augment is not None and flip_perm is None:
            raise ValueError("augmenting dataset needs the flip permutation")
        self.crops = crops
        self.landmarks = landmarks
        self.indices = np.asarray(indices)
        self.out_size = out_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.augment = augment
        self.flip_perm = flip_perm
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        crop = self.crops[idx]
        lm01 = self.landmarks[idx]
        if self.augment is not None:
            rng = np.random.default_rng((self.base_seed, self.epoch, idx))
            img, lm = augment_face(crop, lm01, rng, self.augment,
                                   self.out_size, self.flip_perm)
        else:
            img, lm = eval_transform(crop, lm01, self.out_size)
        x = (torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
             - self.pixel_mean) / self.pixel_std
        return x.unsqueeze(0), torch.from_numpy(lm)
