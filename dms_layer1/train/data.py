"""Training data pipeline: on-the-fly augmentation over the RAM-resident
crop cache (milestone 4, reworked after the milestone-6 framing finding).

Framing is the axis that turned out to matter most. The cache stores a
generous context box around each face (preprocess.crop_expand). A training
sample is cut from that box at a FRAMING FACTOR k, expressed relative to the
canonical box (preprocess.reference_expand): k = 1.0 is the canonical
framing the milestone-5 protocol evaluates, k = 1.5 is a box half again as
wide around the same face. Sampling k over a wide range is what makes the
model tolerate the wider boxes the Haar front end produces at deployment;
milestone 6 measured NME growing roughly linearly with k outside the trained
range, which is why the range is now explicit config.

    base_frac = reference_expand / crop_expand

converts a framing factor into the fraction of the cached crop to cut. The
cache must actually hold that much context: a framing factor whose region
runs past the cached crop would train the model on zero padding where
deployment shows real background, so the trainer refuses it (see
train/loop.py) rather than silently producing a bad run.

Augmentations (ranges from config train.augment): horizontal flip WITH
landmark index remapping (the schema's flip_permutation, the classic silent
bug, unit-tested in tests/test_augment.py), framing, rotation, translation,
brightness/contrast, and blur. Geometry is applied as ONE affine warp from
the cached crop straight to the model input, and landmark coordinates go
through exactly the same matrix.

Determinism: the per-sample RNG is seeded from (base_seed, epoch, index), so
an epoch's augmentations are a pure function of the config seed and the
epoch number, and a resumed run regenerates identical batches without having
to persist dataloader RNG state.
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
    framing_lo: float        # k range, relative to reference_expand
    framing_hi: float
    base_frac: float         # reference_expand / cache crop_expand
    rotation_deg: float
    translate_frac: float
    brightness: float        # +- gray levels
    contrast: float          # alpha in [1-contrast, 1+contrast]
    blur_prob: float
    flip_prob: float

    @property
    def max_region_frac(self) -> float:
        """Fraction of the cached crop the widest framing needs. Above 1.0
        the sample would include padding instead of image."""
        return self.framing_hi * self.base_frac

    @staticmethod
    def from_cfg(cfg: dict, base_frac: float) -> "AugmentParams":
        a = require(cfg, "train.augment")
        if "framing" in a:
            lo, hi = float(a["framing"][0]), float(a["framing"][1])
        elif "scale" in a:
            # Legacy key: a zoom factor applied to the cached crop, so the
            # framing factor is its reciprocal. Kept working so old configs
            # and their snapshots still reproduce.
            lo, hi = 1.0 / float(a["scale"][1]), 1.0 / float(a["scale"][0])
            print("note: train.augment.scale is legacy; read as framing "
                  f"[{lo:.2f}, {hi:.2f}]. Prefer train.augment.framing.")
        else:
            raise KeyError("train.augment needs 'framing' (or legacy 'scale')")
        if lo <= 0 or hi < lo:
            raise ValueError(f"train.augment.framing must be 0 < lo <= hi, got [{lo}, {hi}]")
        return AugmentParams(
            framing_lo=lo, framing_hi=hi, base_frac=float(base_frac),
            rotation_deg=float(a["rotation_deg"]),
            translate_frac=float(a["translate_frac"]),
            brightness=float(a["brightness"]), contrast=float(a["contrast"]),
            blur_prob=float(a["blur_prob"]), flip_prob=float(a["flip_prob"]),
        )


def warp_framed(crop: np.ndarray, lm01: np.ndarray, out_size: int,
                region_frac: float, angle_deg: float = 0.0,
                tx: float = 0.0, ty: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cut the centred region covering `region_frac` of the cached crop and
    map it to out_size, with optional rotation and pixel translation. Image
    and landmarks go through the same matrix."""
    size = crop.shape[0]
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg,
                                out_size / (region_frac * size))
    m[0, 2] += out_size / 2 - size / 2 + tx
    m[1, 2] += out_size / 2 - size / 2 + ty
    out = cv2.warpAffine(np.ascontiguousarray(crop), m, (out_size, out_size),
                         flags=cv2.INTER_LINEAR, borderValue=PAD_VALUE)
    pts = np.asarray(lm01, dtype=np.float64) * size
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return out, (pts / out_size).astype(np.float32)


def augment_face(crop: np.ndarray, lm01: np.ndarray, rng: np.random.Generator,
                 aug: AugmentParams, out_size: int,
                 flip_perm: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """One augmented sample: (out_size, out_size) uint8 + (24, 2) float in
    [0, 1] of the output. Image and landmarks share every transform."""
    lm01 = np.asarray(lm01, dtype=np.float64)

    if rng.random() < aug.flip_prob:
        crop = crop[:, ::-1]
        lm01 = lm01.copy()
        lm01[:, 0] = 1.0 - lm01[:, 0]
        lm01 = lm01[list(flip_perm)]

    k = rng.uniform(aug.framing_lo, aug.framing_hi)
    angle = rng.uniform(-aug.rotation_deg, aug.rotation_deg)
    tx = rng.uniform(-aug.translate_frac, aug.translate_frac) * out_size
    ty = rng.uniform(-aug.translate_frac, aug.translate_frac) * out_size
    out, lm_out = warp_framed(crop, lm01, out_size, k * aug.base_frac,
                              angle, tx, ty)

    alpha = rng.uniform(1 - aug.contrast, 1 + aug.contrast)
    beta = rng.uniform(-aug.brightness, aug.brightness)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < aug.blur_prob:
        k_blur = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k_blur, k_blur), 0)
    return out, lm_out


def eval_transform(crop: np.ndarray, lm01: np.ndarray, out_size: int,
                   base_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic path at framing k = 1.0: cut the canonical region out
    of the cached crop and resize. With base_frac 1.0 (cache stored at the
    reference framing) this is the plain resize it always was."""
    size = crop.shape[0]
    side = base_frac * size
    i0 = max(0, int(round((size - side) / 2.0)))
    i1 = min(size, int(round((size - side) / 2.0 + side)))
    if i1 - i0 < 2:
        raise ValueError(f"base_frac {base_frac} leaves no crop at size {size}")
    sub = crop[i0:i1, i0:i1]
    interp = cv2.INTER_AREA if sub.shape[0] >= out_size else cv2.INTER_LINEAR
    out = cv2.resize(sub, (out_size, out_size), interpolation=interp)
    lm = (np.asarray(lm01, dtype=np.float64) * size - i0) / (i1 - i0)
    return out, lm.astype(np.float32)


class CachedFaceDataset(Dataset):
    """Wraps the in-RAM cache arrays. `augment=None` gives the deterministic
    eval path at framing 1.0. Call set_epoch(e) before each training epoch so
    the per-sample RNG seeds advance reproducibly."""

    def __init__(self, crops: np.ndarray, landmarks: np.ndarray,
                 indices: np.ndarray, out_size: int,
                 pixel_mean: float, pixel_std: float,
                 augment: AugmentParams | None = None,
                 flip_perm: tuple[int, ...] | None = None,
                 base_seed: int = 0, base_frac: float = 1.0):
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
        self.base_frac = base_frac
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
            img, lm = eval_transform(crop, lm01, self.out_size, self.base_frac)
        x = (torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
             - self.pixel_mean) / self.pixel_std
        return x.unsqueeze(0), torch.from_numpy(lm)
