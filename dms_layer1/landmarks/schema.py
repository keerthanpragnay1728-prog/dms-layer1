"""The 24-point landmark schema: identities, WFLW source indices, flip map.

Loads configs/landmarks_24.yaml and validates it hard on load, because every
other component (loader, augmentation, both detectors, Layer 2) trusts this
structure blindly. A mistake here is invisible in downstream metrics, so it
must be impossible to load a malformed schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

NUM_POINTS = 24
NUM_WFLW_POINTS = 98

# Expected group sizes; anything else means the schema file was edited badly.
EXPECTED_GROUP_SIZES = {"eyelids": 12, "pupils": 2, "mouth": 4, "axis": 2, "contour": 4}


class SchemaError(Exception):
    """Raised when the landmark schema file is inconsistent."""


@dataclass(frozen=True)
class LandmarkPoint:
    index: int
    name: str
    group: str
    wflw: int


@dataclass(frozen=True)
class LandmarkSchema:
    points: tuple[LandmarkPoint, ...]
    flip_permutation: tuple[int, ...]
    nme_left_index: int   # our index of the left NME reference point
    nme_right_index: int  # our index of the right NME reference point
    source_file: str
    # MediaPipe Face Mesh source index per output point (478-point mesh),
    # or None when the schema file has no mediapipe block yet.
    mediapipe_indices: tuple[int, ...] | None = None
    mediapipe_refine: bool = True
    # Optional second MediaPipe mapping, used only to answer "would a
    # different index choice change the ablation's conclusion". Populated by
    # hand from a scripts/verify_mediapipe_mapping.py run on the TRAIN split;
    # never the mapping the project claims to use.
    mediapipe_indices_alt: tuple[int, ...] | None = None

    @property
    def wflw_indices(self) -> list[int]:
        """WFLW source index for each of our 24 outputs, in output order."""
        return [p.wflw for p in self.points]

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.points]

    def index_of(self, name: str) -> int:
        for p in self.points:
            if p.name == name:
                return p.index
        raise KeyError(f"No landmark named '{name}' in schema")

    def indices_of_group(self, group: str) -> list[int]:
        return [p.index for p in self.points if p.group == group]

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for p in self.points:
            if p.group not in seen:
                seen.append(p.group)
        return seen


def _derive_flip_from_names(points: list[LandmarkPoint]) -> list[int]:
    """Independently derive the flip permutation from the left_/right_ naming:
    a point maps to the point whose name has left<->right swapped; names
    without a side are their own mirror. Used to cross-check the explicit
    flip_permutation list in the YAML."""
    name_to_index = {p.name: p.index for p in points}
    derived = []
    for p in points:
        if "left" in p.name:
            partner = p.name.replace("left", "right")
        elif "right" in p.name:
            partner = p.name.replace("right", "left")
        else:
            partner = p.name
        if partner not in name_to_index:
            raise SchemaError(
                f"Landmark '{p.name}' has no mirror partner '{partner}'; "
                "left/right names must come in pairs."
            )
        derived.append(name_to_index[partner])
    return derived


def load_schema(path: str | Path) -> LandmarkSchema:
    path = Path(path)
    if not path.is_file():
        raise SchemaError(f"Landmark schema file not found: {path.resolve()}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    entries = raw.get("points")
    if not isinstance(entries, list) or len(entries) != NUM_POINTS:
        raise SchemaError(
            f"Schema must define exactly {NUM_POINTS} points, "
            f"got {len(entries) if isinstance(entries, list) else 'none'} in {path}"
        )

    points = []
    for pos, e in enumerate(entries):
        pt = LandmarkPoint(index=int(e["index"]), name=str(e["name"]),
                           group=str(e["group"]), wflw=int(e["wflw"]))
        if pt.index != pos:
            raise SchemaError(
                f"Schema entry at position {pos} declares index {pt.index}; "
                "entries must be listed in output order."
            )
        if not (0 <= pt.wflw < NUM_WFLW_POINTS):
            raise SchemaError(f"Point '{pt.name}' has WFLW index {pt.wflw} outside 0..97")
        points.append(pt)

    names = [p.name for p in points]
    if len(set(names)) != NUM_POINTS:
        raise SchemaError("Duplicate landmark names in schema")
    wflw = [p.wflw for p in points]
    if len(set(wflw)) != NUM_POINTS:
        dupes = sorted({i for i in wflw if wflw.count(i) > 1})
        raise SchemaError(f"WFLW indices used more than once: {dupes}")

    group_sizes = {g: sum(1 for p in points if p.group == g)
                   for g in {p.group for p in points}}
    if group_sizes != EXPECTED_GROUP_SIZES:
        raise SchemaError(
            f"Group sizes {group_sizes} do not match expected {EXPECTED_GROUP_SIZES}"
        )

    flip = raw.get("flip_permutation")
    if not isinstance(flip, list) or sorted(flip) != list(range(NUM_POINTS)):
        raise SchemaError("flip_permutation must be a permutation of 0..23")
    for i, j in enumerate(flip):
        if flip[j] != i:
            raise SchemaError(
                f"flip_permutation is not an involution: {i} -> {j} -> {flip[j]}"
            )
    derived = _derive_flip_from_names(points)
    if derived != flip:
        raise SchemaError(
            "flip_permutation disagrees with the left/right naming:\n"
            f"  explicit: {flip}\n  derived : {derived}"
        )

    nme = raw.get("nme_normalisation")
    if not isinstance(nme, dict) or "left" not in nme or "right" not in nme:
        raise SchemaError("nme_normalisation must define 'left' and 'right' point names")
    name_to_index = {p.name: p.index for p in points}
    for side in ("left", "right"):
        if nme[side] not in name_to_index:
            raise SchemaError(f"nme_normalisation.{side}='{nme[side]}' is not a landmark name")

    mp_indices: tuple[int, ...] | None = None
    mp_refine = True
    mp_indices_alt = None
    mp_block = raw.get("mediapipe")
    if mp_block is not None:
        if not isinstance(mp_block, dict) or "indices" not in mp_block:
            raise SchemaError("mediapipe block must be a mapping with 'indices'")
        idx = mp_block["indices"]
        if not isinstance(idx, list) or len(idx) != NUM_POINTS:
            raise SchemaError(f"mediapipe.indices must list exactly {NUM_POINTS} entries")
        if len(set(idx)) != NUM_POINTS:
            raise SchemaError("mediapipe.indices contains duplicates")
        if not all(isinstance(i, int) and 0 <= i < 478 for i in idx):
            raise SchemaError("mediapipe.indices must be ints in 0..477")
        mp_indices = tuple(idx)
        mp_refine = bool(mp_block.get("refine_landmarks", True))
        alt = mp_block.get("indices_alt")
        if alt is not None:
            if (not isinstance(alt, list) or len(alt) != NUM_POINTS
                    or not all(isinstance(i, int) and 0 <= i < 478 for i in alt)):
                raise SchemaError(
                    f"mediapipe.indices_alt must list exactly {NUM_POINTS} "
                    "ints in 0..477 when present")
            mp_indices_alt = tuple(alt)

    return LandmarkSchema(
        points=tuple(points),
        flip_permutation=tuple(flip),
        nme_left_index=name_to_index[nme["left"]],
        nme_right_index=name_to_index[nme["right"]],
        source_file=str(path.resolve()),
        mediapipe_indices=mp_indices,
        mediapipe_refine=mp_refine,
        mediapipe_indices_alt=mp_indices_alt,
    )


def format_mapping_table(schema: LandmarkSchema) -> str:
    """The mapping as a fixed-width text table for terminal printing and
    result files."""
    lines = [
        f"24-point schema  (source: {schema.source_file})",
        "left/right are IMAGE space: 'left' = viewer's left = subject's right",
        "",
        f"{'ours':>4}  {'name':<22} {'group':<8} {'wflw':>4}   flips with",
        "-" * 62,
    ]
    for p in schema.points:
        partner = schema.points[schema.flip_permutation[p.index]]
        partner_desc = "itself" if partner.index == p.index else f"{partner.index:>2} ({partner.name})"
        lines.append(f"{p.index:>4}  {p.name:<22} {p.group:<8} {p.wflw:>4}   {partner_desc}")
    lines += [
        "-" * 62,
        "EAR (either eye, local offsets within its 6-point block):",
        "    EAR = (|p1-p5| + |p2-p4|) / (2*|p0-p3|)",
        "    left eye block = ours 0..5, right eye block = ours 6..11",
        "MAR = |ours16 - ours17| / |ours14 - ours15|",
        f"NME normalisation: inter-ocular distance |ours{schema.nme_left_index} - ours{schema.nme_right_index}| (outer eye corners)",
    ]
    return "\n".join(lines)
