"""Tests for the 24-point schema: validation, and the horizontal-flip
permutation (the classic silent landmark-training bug)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data.synthetic import CANVAS, canonical_landmarks98
from dms_layer1.landmarks.schema import SchemaError, load_schema

SCHEMA_PATH = REPO / "configs" / "landmarks_24.yaml"


def test_schema_loads_and_is_complete():
    s = load_schema(SCHEMA_PATH)
    assert len(s.points) == 24
    assert s.index_of("left_pupil") == 12
    assert s.points[12].wflw == 96
    assert s.index_of("right_pupil") == 13
    assert s.points[13].wflw == 97
    assert s.indices_of_group("eyelids") == list(range(12))
    assert s.indices_of_group("pupils") == [12, 13]
    assert s.indices_of_group("mouth") == [14, 15, 16, 17]
    assert s.indices_of_group("axis") == [18, 19]
    assert s.indices_of_group("contour") == [20, 21, 22, 23]
    # NME reference points are the outer eye corners
    assert s.nme_left_index == s.index_of("left_eye_outer")
    assert s.nme_right_index == s.index_of("right_eye_outer")


def test_flip_permutation_is_involution():
    s = load_schema(SCHEMA_PATH)
    perm = list(s.flip_permutation)
    assert sorted(perm) == list(range(24))
    for i in range(24):
        assert perm[perm[i]] == i, f"flip not an involution at {i}"


def test_flip_permutation_geometric():
    """On a face that is left-right symmetric, mirroring the coordinates and
    applying the flip permutation must reproduce the original 24 points.
    This is the property the augmentation relies on."""
    s = load_schema(SCHEMA_PATH)
    lm98 = canonical_landmarks98()          # symmetric about x = CANVAS/2
    ours = lm98[s.wflw_indices]             # (24, 2)

    mirrored = ours.copy()
    mirrored[:, 0] = CANVAS - mirrored[:, 0]
    flipped = mirrored[list(s.flip_permutation)]

    err = np.abs(flipped - ours).max()
    assert err < 1e-6, f"flip permutation breaks symmetry, max error {err:.4f}px"


def test_flip_permutation_moves_sides():
    """Every left_* point must map to the matching right_* point and vice
    versa; midline points map to themselves."""
    s = load_schema(SCHEMA_PATH)
    for p in s.points:
        partner = s.points[s.flip_permutation[p.index]]
        if "left" in p.name:
            assert partner.name == p.name.replace("left", "right")
        elif "right" in p.name:
            assert partner.name == p.name.replace("right", "left")
        else:
            assert partner.index == p.index


def _load_raw() -> dict:
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)


def _expect_schema_error(raw: dict, tmp_name: str):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        path = f.name
    try:
        load_schema(path)
    except SchemaError:
        return
    raise AssertionError(f"malformed schema '{tmp_name}' loaded without error")


def test_mediapipe_block_loads_and_validates():
    s = load_schema(SCHEMA_PATH)
    assert s.mediapipe_indices is not None and len(s.mediapipe_indices) == 24
    assert len(set(s.mediapipe_indices)) == 24
    assert all(0 <= i < 478 for i in s.mediapipe_indices)
    # the iris centres feed the pupils
    assert s.mediapipe_indices[s.index_of("left_pupil")] == 468
    assert s.mediapipe_indices[s.index_of("right_pupil")] == 473

    raw = _load_raw()
    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["mediapipe"]["indices"][1] = bad["mediapipe"]["indices"][0]  # duplicate
    _expect_schema_error(bad, "duplicate mediapipe index")
    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["mediapipe"]["indices"][0] = 478                             # out of range
    _expect_schema_error(bad, "mediapipe index out of range")
    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["mediapipe"]["indices"] = bad["mediapipe"]["indices"][:23]   # wrong count
    _expect_schema_error(bad, "short mediapipe index list")


def test_malformed_schemas_are_rejected():
    raw = _load_raw()
    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["points"][5]["wflw"] = bad["points"][4]["wflw"]      # duplicate source index
    _expect_schema_error(bad, "duplicate wflw index")

    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["flip_permutation"][0], bad["flip_permutation"][1] = (
        bad["flip_permutation"][1], bad["flip_permutation"][0])   # breaks involution/names
    _expect_schema_error(bad, "corrupted flip permutation")

    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["points"] = bad["points"][:23]                        # wrong count
    _expect_schema_error(bad, "missing point")

    bad = yaml.safe_load(yaml.safe_dump(raw))
    bad["points"][0], bad["points"][1] = bad["points"][1], bad["points"][0]
    _expect_schema_error(bad, "entries out of output order")


def test_optional_alternative_mediapipe_mappings():
    """indices_alt is optional, takes one list or several named lists, is
    validated like indices, and is absent by default: the project ships one
    mapping and the others exist only for a sensitivity check."""
    import tempfile

    import yaml as _yaml

    raw = _yaml.safe_load(SCHEMA_PATH.read_text())
    assert load_schema(SCHEMA_PATH).mediapipe_alt_maps == ()

    good = list(raw["mediapipe"]["indices"])
    good[2] = 157
    mixed = list(raw["mediapipe"]["indices"])
    mixed[5], mixed[10] = 163, 390
    # compared as dicts: yaml round-tripping does not preserve key order
    cases = [(good, True, {"alt": tuple(good)}),
             ({"proximity": good, "mixed": mixed}, True,
              {"proximity": tuple(good), "mixed": tuple(mixed)}),
             (good[:10], False, None),
             ([9999] * 24, False, None),
             ({"bad": good[:3]}, False, None)]
    with tempfile.TemporaryDirectory() as tmp:
        for alt, ok, expected in cases:
            path = Path(tmp) / "schema.yaml"
            raw["mediapipe"]["indices_alt"] = alt
            path.write_text(_yaml.safe_dump(raw))
            try:
                loaded = load_schema(path)
            except SchemaError:
                assert not ok, f"a valid indices_alt was rejected: {alt}"
                continue
            assert ok, f"an invalid indices_alt was accepted: {alt}"
            assert dict(loaded.mediapipe_alt_maps) == expected
