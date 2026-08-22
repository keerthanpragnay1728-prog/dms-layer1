#!/usr/bin/env python3
"""Verify the MediaPipe -> 24-point index mapping against ground truth.

Why this exists twice. The mapping in configs/landmarks_24.yaml was first
measured on the legacy `mediapipe.solutions` Face Mesh. That API was removed
from the package (docs/dependency_notes.md), so the wrapper now runs the
Tasks API. Tasks returns the same 478-point topology, so the indices should
carry over unchanged, but "should" is not a measurement: this script re-runs
the check on whatever mesh the installed bundle actually produces.

What it measures, on real WFLW faces with 98-point annotations:

  1. Bundle identity and mesh size. 478 points means the iris head is
     present. That head is what the legacy API called refine_landmarks, and
     the pupil indices (468, 473) do not exist without it.
  2. Per-point offset between MediaPipe's mapped point and the WFLW ground
     truth point it is supposed to mean, as a percentage of inter-ocular
     distance, median and p90 over the sample. A mapping error shows up as
     one point sitting far from its ground truth while its neighbours are
     fine, which is exactly what an accuracy number hides.
  3. The control that actually verifies the CHOICE of index: for each of our
     24 points, search all 478 mesh points for the one closest to ground
     truth across the sample. If the configured index is that argmin, or
     close to it, the index is right. If some other index is materially
     better, this prints it, and the mapping is a decision to revisit rather
     than a fact to assume.

Section 2 is a tolerance check, section 3 is the interesting one. Both are
reported per point, never as a single aggregate.

Usage (Kaggle):
    python scripts/verify_mediapipe_mapping.py --config configs/layer1_base.yaml \
        --split test --limit 300

Locally, without the dataset, --synthetic runs the same checks on schematic
faces. That is a code smoke test, not verification: the cartoon face has no
real skin texture and MediaPipe's mesh on it means little.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import square_box_around
from dms_layer1.evaluation.matching import match_target
from dms_layer1.landmarks.schema import load_schema

# Tolerance on the MEDIAN offset from ground truth, per group, in % of IOD.
# These bound "is this index the right anatomy", not "is MediaPipe accurate":
# they are loose enough that ordinary landmark error passes and tight enough
# that a point on the wrong feature cannot. Rationale per group:
#   pupils   the two meshes define the iris centre identically, so anything
#            beyond a couple of % means the iris indices are wrong
#   eyelids  corners and lid midpoints are the same anatomy in both, but lid
#            arcs are sampled at slightly different parameters
#   mouth    same, on the outer lip
#   axis     nose tip and chin are both annotation-convention sensitive (how
#            far under the chin the point sits), so a looser bound
#   contour  a KNOWN approximation, documented in the schema file: mesh 234 /
#            58 / 454 / 288 are near but not on WFLW 4 / 8 / 28 / 24. The
#            bound only asks that they stay on the right side of the right
#            face; section 3 is what says whether a better index exists.
TOLERANCE_PCT_IOD = {"eyelids": 8.0, "pupils": 3.0, "mouth": 8.0,
                     "axis": 10.0, "contour": 30.0}

# In section 3, flag the configured index only when another mesh index beats
# it by more than this many IOD points. Below that the two indices are
# describing the same place and the choice does not matter.
BETTER_INDEX_MARGIN_PCT_IOD = 1.5

MIN_SAMPLE = 30      # below this the medians are not worth reading
SYNTHETIC_FACES = 40  # smoke-test sample, kept above MIN_SAMPLE

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def synthetic_frames(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Schematic faces at varied pose/scale, as (bgr frame, 98 points)."""
    from dms_layer1.data.synthetic import generate_face

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(generate_face(angle_deg=rng.uniform(-12, 12),
                                 scale=rng.uniform(0.8, 1.1),
                                 shift=(rng.uniform(-20, 20), rng.uniform(-20, 20))))
    return out


def wflw_frames(cfg: dict, split: str, limit: int, seed: int
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Seeded sample of WFLW images, each with its target face's 98 points.
    The target is the largest annotated face, the same rule the ablation
    uses, since MediaPipe is asked for a single face."""
    records, paths = wflw.load_split(cfg, split)
    by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    rels = sorted(by_image)
    if limit and limit < len(rels):
        rng = random.Random(seed)
        rng.shuffle(rels)
        rels = sorted(rels[:limit])
    out = []
    for rel in rels:
        recs = by_image[rel]
        target = max(recs, key=lambda r: square_box_around(r.landmarks98, 1.0).side)
        frame = cv2.imread(str(wflw.image_path(paths, recs[0])), cv2.IMREAD_COLOR)
        if frame is None:
            raise wflw.WFLWError(f"cv2 could not decode {rel}")
        out.append((frame, target.landmarks98.astype(np.float64)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300,
                    help="images to process (seeded sample); 0 = all")
    ap.add_argument("--out-dir", default="/kaggle/working/m6_mapping")
    ap.add_argument("--synthetic", action="store_true",
                    help="schematic faces instead of WFLW (code smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if schema.mediapipe_indices is None:
        say("FAIL: the schema has no mediapipe block; nothing to verify.")
        return 1

    from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                      MediaPipeUnavailable)
    try:
        det = MediaPipeLandmarkDetector(cfg, schema)
    except MediaPipeUnavailable as e:
        say(f"FAIL: mediapipe unavailable, mapping not verified:\n{e}")
        return 1
    try:
        return verify(cfg, schema, det, args, out_dir)
    finally:
        det.close()


def verify(cfg: dict, schema, det, args, out_dir: Path) -> int:
    """The measurement itself. Split from main() so the landmarker is closed
    on every exit path, including the failing ones."""
    indices = list(schema.mediapipe_indices)
    seed = int(require(cfg, "seed"))

    import mediapipe as mp_pkg
    say("=== 1. What produced these numbers ===")
    say(f"  mediapipe {mp_pkg.__version__}, Tasks API "
        f"(solutions present: {hasattr(mp_pkg, 'solutions')})")
    say(f"  bundle: {det.model_path} "
        f"({det.model_path.stat().st_size / 1e6:.2f} MB)")
    say(f"  opencv {cv2.__version__}")
    say(f"  schema: {schema.source_file}")
    say(f"  mapped indices: {indices}")

    samples = (synthetic_frames(SYNTHETIC_FACES, seed) if args.synthetic
               else wflw_frames(cfg, args.split, args.limit, seed))
    say(f"  source: {'schematic faces' if args.synthetic else args.split + ' split'}"
        f", {len(samples)} images")
    if args.synthetic:
        say("  NOTE: synthetic mode is a code smoke test. Cartoon faces are not "
            "evidence about the mapping; run it on WFLW for that.")

    # ---- collect: full mesh + ground truth, on faces MediaPipe actually found
    meshes, gts, iods = [], [], []
    n_none = n_unmatched = 0
    mesh_size = None
    for n_img, (frame, pts98) in enumerate(samples, 1):
        mesh = det.mesh(frame)
        if mesh is None:
            n_none += 1
            continue
        mesh_size = mesh.shape[0] if mesh_size is None else mesh_size
        gt24 = pts98[schema.wflw_indices]
        if not match_target(mesh[indices].astype(np.float64), gt24):
            n_unmatched += 1
            continue
        meshes.append(mesh.astype(np.float64))
        gts.append(gt24)
        iods.append(np.linalg.norm(gt24[schema.nme_right_index]
                                   - gt24[schema.nme_left_index]))
        if n_img % 100 == 0:
            say(f"  {n_img}/{len(samples)} images")

    say(f"\n  usable faces: {len(meshes)} "
        f"(no face: {n_none}, found a different face: {n_unmatched})")
    if mesh_size is None:
        say("FAIL: MediaPipe found no face at all; nothing to measure.")
        return 1
    iris_ok = mesh_size >= 478
    say(f"  mesh points returned: {mesh_size} -> iris head "
        f"{'PRESENT' if iris_ok else 'ABSENT'} "
        "(the Tasks equivalent of the old refine_landmarks=True; the pupil "
        "indices 468/473 live in this head)")
    if not iris_ok:
        say("FAIL: this bundle cannot supply the pupils.")
        return 1
    if len(meshes) < MIN_SAMPLE:
        say(f"FAIL: only {len(meshes)} usable faces, below the {MIN_SAMPLE} "
            "needed for the medians to mean anything.")
        return 1

    mesh_arr = np.stack(meshes)                    # (N, 478, 2)
    gt_arr = np.stack(gts)                         # (N, 24, 2)
    iod_arr = np.maximum(np.array(iods), 1e-9)     # (N,)

    # ---- 2: configured index vs its ground truth point --------------------
    say("\n=== 2. Offset from ground truth, per point (% of IOD) ===")
    say("  A mapping error is one point far out while its neighbours are "
        "fine, not a uniformly raised table.")
    picked = mesh_arr[:, indices, :]                          # (N, 24, 2)
    off = (np.linalg.norm(picked - gt_arr, axis=2)
           / iod_arr[:, None]) * 100.0                        # (N, 24)
    med = np.median(off, axis=0)
    p90 = np.percentile(off, 90, axis=0)
    say(f"  {'ours':>4}  {'name':<22} {'group':<8} {'mesh':>4}  "
        f"{'median':>7} {'p90':>7}  {'tol':>5}  verdict")
    per_point = []
    failed = []
    for p in schema.points:
        tol = TOLERANCE_PCT_IOD[p.group]
        ok = med[p.index] <= tol
        if not ok:
            failed.append(p)
        say(f"  {p.index:>4}  {p.name:<22} {p.group:<8} {indices[p.index]:>4}  "
            f"{med[p.index]:6.2f}% {p90[p.index]:6.2f}%  {tol:4.0f}%  "
            f"{'ok' if ok else 'OVER TOLERANCE'}")
        per_point.append({"index": p.index, "name": p.name, "group": p.group,
                          "mesh_index": indices[p.index],
                          "median_pct_iod": float(med[p.index]),
                          "p90_pct_iod": float(p90[p.index]),
                          "tolerance_pct_iod": tol, "within_tolerance": bool(ok)})

    say("\n  by group (median of the group's per-point medians):")
    for g in schema.groups:
        gi = schema.indices_of_group(g)
        say(f"    {g:<8} {np.median(med[gi]):6.2f}%   "
            f"worst point {schema.points[gi[int(np.argmax(med[gi]))]].name} "
            f"at {med[gi].max():.2f}%")

    # ---- 3: would another mesh index be better? ---------------------------
    say("\n=== 3. Control: the closest mesh index to each ground truth point ===")
    say("  Over the same faces, every one of the 478 mesh points is scored "
        "against each ground truth point. This is what verifies the CHOICE "
        "of index rather than just its accuracy.")
    # (N, 24, 478) distances, normalised per face, then the median over faces
    d = np.linalg.norm(gt_arr[:, :, None, :] - mesh_arr[:, None, :, :], axis=3)
    d = d / iod_arr[:, None, None] * 100.0
    med_all = np.median(d, axis=0)                            # (24, 478)
    best_idx = med_all.argmin(axis=1)
    best_val = med_all.min(axis=1)
    flagged = []
    say(f"  {'ours':>4}  {'name':<22} {'cfg':>4} {'median':>7}   "
        f"{'best':>4} {'median':>7}   gap")
    for p in schema.points:
        gap = med[p.index] - best_val[p.index]
        mark = "" if gap <= BETTER_INDEX_MARGIN_PCT_IOD else "   <-- REVIEW"
        if mark:
            flagged.append((p, int(best_idx[p.index]), float(best_val[p.index]),
                            float(gap)))
        say(f"  {p.index:>4}  {p.name:<22} {indices[p.index]:>4} "
            f"{med[p.index]:6.2f}%   {best_idx[p.index]:>4} "
            f"{best_val[p.index]:6.2f}%   {gap:5.2f}{mark}")

    # ---- verdict -----------------------------------------------------------
    say("\n=== Verdict ===")
    if failed:
        say(f"  MAPPING FAIL: {len(failed)} point(s) beyond tolerance: "
            + ", ".join(f"{p.name} ({med[p.index]:.2f}%)" for p in failed))
    else:
        say("  MAPPING OK: every point sits within its group's tolerance of "
            "the ground truth point it claims to be.")
    say(f"  pupils: left {med[12]:.2f}% / right {med[13]:.2f}% of IOD "
        "(the iris-head check; these are the indices that would silently "
        "vanish on a 468-point bundle)")
    if flagged:
        say(f"  {len(flagged)} index/indices where another mesh point is more "
            f"than {BETTER_INDEX_MARGIN_PCT_IOD}% of IOD closer:")
        for p, bi, bv, gap in flagged:
            say(f"    {p.name}: configured {indices[p.index]} at "
                f"{med[p.index]:.2f}%, mesh {bi} at {bv:.2f}% (gap {gap:.2f})")
        say("  These are a decision to make, not automatically a bug: an index "
            "chosen for semantics can lose to one chosen by proximity.")
    else:
        say(f"  No mesh index beats a configured one by more than "
            f"{BETTER_INDEX_MARGIN_PCT_IOD}% of IOD.")

    results = {
        "mediapipe_version": mp_pkg.__version__,
        "api": "tasks",
        "bundle": str(det.model_path),
        "bundle_mb": round(det.model_path.stat().st_size / 1e6, 3),
        "source": "synthetic" if args.synthetic else args.split,
        "images": len(samples),
        "usable_faces": len(meshes),
        "no_face": n_none,
        "wrong_face": n_unmatched,
        "mesh_points": int(mesh_size),
        "iris_head_present": bool(iris_ok),
        "per_point": per_point,
        "better_index_available": [
            {"name": p.name, "configured": indices[p.index],
             "configured_median_pct_iod": float(med[p.index]),
             "best_mesh_index": bi, "best_median_pct_iod": bv, "gap_pct_iod": gap}
            for p, bi, bv, gap in flagged],
        "verdict": "fail" if failed else "pass",
    }
    with open(out_dir / "mediapipe_mapping.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nwrote {out_dir / 'mediapipe_mapping.yaml'} and report.txt")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
