"""Augmentation tests. The flip test is the single most important one in
this file: a horizontal flip without the landmark index remap is the classic
silent landmark-training bug - labels stay plausible, training converges,
and left/right semantics are quietly destroyed."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data.synthetic import canonical_landmarks98, render_face, CANVAS
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.train.data import AugmentParams, augment_face, eval_transform

SCHEMA = load_schema(REPO / "configs" / "landmarks_24.yaml")


def _no_op(flip_prob=0.0, base_frac=1.0) -> AugmentParams:
    return AugmentParams(framing_lo=1.0, framing_hi=1.0, base_frac=base_frac,
                         rotation_deg=0, translate_frac=0, brightness=0,
                         contrast=0, blur_prob=0, flip_prob=flip_prob)


def _face_crop() -> tuple[np.ndarray, np.ndarray]:
    """A rendered schematic face as a 128px cache-style crop with [0,1]
    labels for our 24 points (asymmetric enough to expose flip bugs after
    a small rotation is baked in)."""
    lm98 = canonical_landmarks98()
    img = cv2.cvtColor(render_face(lm98), cv2.COLOR_BGR2GRAY)
    m = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), 8, 1.0)
    img = cv2.warpAffine(img, m, (CANVAS, CANVAS))
    lm98 = np.hstack([lm98, np.ones((98, 1))]) @ m.T
    crop = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    lm01 = lm98[SCHEMA.wflw_indices] / CANVAS
    return crop, lm01.astype(np.float32)


def test_eval_transform_preserves_labels():
    crop, lm01 = _face_crop()
    out, lm = eval_transform(crop, lm01, 112)
    assert out.shape == (112, 112)
    assert np.abs(lm - lm01).max() < 1e-7   # [0,1] coords are size-free


def test_flip_remaps_indices():
    """Forced flip, no other augmentation: every point must land at the
    mirror of its LEFT/RIGHT PARTNER, not of itself."""
    crop, lm01 = _face_crop()
    rng = np.random.default_rng(0)
    out, lm = augment_face(crop, lm01, rng, _no_op(flip_prob=1.0), 128,
                           SCHEMA.flip_permutation)
    mirrored = lm01.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    for p in SCHEMA.points:
        partner = SCHEMA.flip_permutation[p.index]
        assert np.abs(lm[p.index] - mirrored[partner]).max() < 1e-5, \
            f"{p.name} did not take its partner's mirrored position"
    # the named semantic case: the new left pupil is the mirrored right pupil
    lp, rp = SCHEMA.index_of("left_pupil"), SCHEMA.index_of("right_pupil")
    assert np.abs(lm[lp] - mirrored[rp]).max() < 1e-5
    # and the image itself flipped
    assert np.abs(out.astype(int) - crop[:, ::-1].astype(int)).max() <= 1


def test_flip_twice_is_identity():
    crop, lm01 = _face_crop()
    rng = np.random.default_rng(0)
    once_img, once_lm = augment_face(crop, lm01, rng, _no_op(1.0), 128,
                                     SCHEMA.flip_permutation)
    twice_img, twice_lm = augment_face(once_img, once_lm, rng, _no_op(1.0), 128,
                                       SCHEMA.flip_permutation)
    assert np.abs(twice_lm - lm01).max() < 1e-5
    assert np.abs(twice_img.astype(int) - crop.astype(int)).max() <= 2


def test_affine_moves_image_and_labels_together():
    """Draw a bright dot exactly at one landmark; after a random affine, the
    brightest pixel must sit at the transformed landmark position."""
    crop = np.zeros((128, 128), dtype=np.uint8)
    lm01 = np.full((24, 2), 0.5, dtype=np.float32)
    lm01[18] = (0.62, 0.41)                       # nose_tip, off-centre
    x, y = (lm01[18] * 128).round().astype(int)
    crop[y, x] = 255
    aug = AugmentParams(framing_lo=0.9, framing_hi=1.1, base_frac=1.0,
                        rotation_deg=20, translate_frac=0.05, brightness=0,
                        contrast=0, blur_prob=0, flip_prob=0)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        out, lm = augment_face(crop, lm01, rng, aug, 112, SCHEMA.flip_permutation)
        by, bx = np.unravel_index(np.argmax(out), out.shape)
        px, py = lm[18] * 112
        assert abs(bx - px) <= 2 and abs(by - py) <= 2, \
            f"seed {seed}: dot at ({bx},{by}), label at ({px:.1f},{py:.1f})"


def test_same_seed_same_output():
    crop, lm01 = _face_crop()
    aug = AugmentParams(framing_lo=0.85, framing_hi=1.15, base_frac=1.0,
                        rotation_deg=15, translate_frac=0.05, brightness=30,
                        contrast=0.2, blur_prob=0.5, flip_prob=0.5)
    a_img, a_lm = augment_face(crop, lm01, np.random.default_rng((7, 3, 11)),
                               aug, 112, SCHEMA.flip_permutation)
    b_img, b_lm = augment_face(crop, lm01, np.random.default_rng((7, 3, 11)),
                               aug, 112, SCHEMA.flip_permutation)
    assert np.array_equal(a_img, b_img) and np.array_equal(a_lm, b_lm)
    c_img, _ = augment_face(crop, lm01, np.random.default_rng((7, 4, 11)),
                            aug, 112, SCHEMA.flip_permutation)
    assert not np.array_equal(a_img, c_img)       # epoch changes the draw


def test_framing_factor_changes_apparent_face_size():
    """k is the framing factor: k = 1.0 is the canonical box, larger k is a
    wider box around the same face, so the face covers LESS of the output
    and the landmark spread shrinks proportionally."""
    crop, lm01 = _face_crop()
    spreads = {}
    for k in (1.0, 1.3, 1.6):
        aug = AugmentParams(framing_lo=k, framing_hi=k, base_frac=1.0,
                            rotation_deg=0, translate_frac=0, brightness=0,
                            contrast=0, blur_prob=0, flip_prob=0)
        _, lm = augment_face(crop, lm01, np.random.default_rng(0), aug, 112,
                             SCHEMA.flip_permutation)
        spreads[k] = float(lm[:, 0].max() - lm[:, 0].min())
    assert spreads[1.0] > spreads[1.3] > spreads[1.6]
    # the shrink is the inverse of k, within interpolation tolerance
    assert abs(spreads[1.3] * 1.3 - spreads[1.0]) < 0.02
    assert abs(spreads[1.6] * 1.6 - spreads[1.0]) < 0.02
    # and points stay centred on the same face
    for k in (1.3, 1.6):
        aug = AugmentParams(framing_lo=k, framing_hi=k, base_frac=1.0,
                            rotation_deg=0, translate_frac=0, brightness=0,
                            contrast=0, blur_prob=0, flip_prob=0)
        _, lm = augment_face(crop, lm01, np.random.default_rng(0), aug, 112,
                             SCHEMA.flip_permutation)
        assert abs(lm.mean(axis=0)[0] - 0.5) < 0.05


def test_base_frac_cuts_the_canonical_region():
    """With a context-rich cache (base_frac < 1), framing k = 1.0 must
    reproduce the canonical framing: the same face size as a cache stored at
    the reference framing."""
    crop, lm01 = _face_crop()                       # 128px at reference framing
    wide = cv2.copyMakeBorder(crop, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=0)
    lm_wide = (lm01.astype(np.float64) * 128 + 32) / 192   # same face, 1.5x context
    aug_ref = _no_op()
    aug_wide = _no_op(base_frac=128 / 192)
    _, lm_a = augment_face(crop, lm01, np.random.default_rng(0), aug_ref, 112,
                           SCHEMA.flip_permutation)
    _, lm_b = augment_face(wide, lm_wide, np.random.default_rng(0), aug_wide, 112,
                           SCHEMA.flip_permutation)
    assert np.abs(lm_a - lm_b).max() < 0.01
    # the deterministic eval path agrees with augmentation at k = 1.0
    _, lm_c = eval_transform(wide, lm_wide, 112, 128 / 192)
    assert np.abs(lm_a - lm_c).max() < 0.01


def test_from_cfg_reads_framing_and_legacy_scale():
    base = {"rotation_deg": 15, "translate_frac": 0.05, "brightness": 30,
            "contrast": 0.2, "blur_prob": 0.15, "flip_prob": 0.5}
    a = AugmentParams.from_cfg({"train": {"augment": {**base, "framing": [0.85, 1.6]}}}, 0.59)
    assert (a.framing_lo, a.framing_hi) == (0.85, 1.6)
    assert abs(a.max_region_frac - 1.6 * 0.59) < 1e-9
    # legacy zoom factors are the reciprocal of the framing factors
    b = AugmentParams.from_cfg({"train": {"augment": {**base, "scale": [0.85, 1.15]}}}, 1.0)
    assert abs(b.framing_lo - 1 / 1.15) < 1e-9 and abs(b.framing_hi - 1 / 0.85) < 1e-9
