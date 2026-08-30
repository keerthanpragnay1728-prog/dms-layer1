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
BOOTSTRAP_DRAWS = 2000  # paired resamples for correlation differences
SYNTHETIC_FACES = 40  # smoke-test sample, kept above MIN_SAMPLE

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def mesh_features() -> dict[str, tuple[set[int], list[int]]]:
    """The mesh features MediaPipe publishes, as (vertex set, ordered cycle).

    This is what lets a "some other index is closer" finding be read properly:
    a neighbouring vertex on the SAME eyelid ring is the same anatomy sampled
    at a different point along the lid, while an index on another feature
    would be a real mapping error. Names are in IMAGE space, the opposite of
    MediaPipe's own naming (its RIGHT_EYE is the viewer's left eye).
    """
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    def parts(conns):
        adj: dict[int, set[int]] = {}
        for c in conns:
            adj.setdefault(c.start, set()).add(c.end)
            adj.setdefault(c.end, set()).add(c.start)
        start = min(adj)
        order, prev, cur = [start], None, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev and v not in order), None)
            if nxt is None:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        # a feature with several concentric rings (the lips) walks only one of
        # them; the vertex set still covers the whole feature
        return set(adj), order

    named = [("eye ring, image-left", C.FACE_LANDMARKS_RIGHT_EYE),
             ("eye ring, image-right", C.FACE_LANDMARKS_LEFT_EYE),
             ("brow, image-left", C.FACE_LANDMARKS_RIGHT_EYEBROW),
             ("brow, image-right", C.FACE_LANDMARKS_LEFT_EYEBROW),
             ("iris, image-left", C.FACE_LANDMARKS_RIGHT_IRIS),
             ("iris, image-right", C.FACE_LANDMARKS_LEFT_IRIS),
             ("lips", C.FACE_LANDMARKS_LIPS),
             ("nose", C.FACE_LANDMARKS_NOSE),
             ("face oval", C.FACE_LANDMARKS_FACE_OVAL)]
    return {name: parts(conns) for name, conns in named}


def describe_alternative(cfg_idx: int, alt_idx: int,
                         features: dict[str, tuple[set[int], list[int]]]) -> str:
    """How a proposed index relates to the configured one, in mesh terms."""
    def feature_of(i):
        return next((n for n, (verts, _) in features.items() if i in verts), None)

    fc, fa = feature_of(cfg_idx), feature_of(alt_idx)
    where_cfg = f"configured is on {fc}" if fc else "configured is not on a named feature"
    if fa is None:
        return ("alternative is an INTERIOR mesh vertex, on no named feature "
                f"({where_cfg})")
    if fc != fa:
        return f"alternative is on a DIFFERENT feature, {fa} ({where_cfg})"
    ring = features[fa][1]
    if cfg_idx in ring and alt_idx in ring:
        i, j, n = ring.index(cfg_idx), ring.index(alt_idx), len(ring)
        d = min((i - j) % n, (j - i) % n)
        return f"same {fa}, {d} vertex/vertices along the ring"
    return f"same {fa}"


def eye_ear(p6: np.ndarray) -> float:
    """EAR on one eye, from the six points in the schema's per-eye order
    [corner, upper, upper, corner, lower, lower]. This is the quantity Layer 2
    consumes, so it is the mapping's real acceptance test."""
    return float((np.linalg.norm(p6[1] - p6[5]) + np.linalg.norm(p6[2] - p6[4]))
                 / (2 * max(np.linalg.norm(p6[0] - p6[3]), 1e-9)))


def chord_skew(p6: np.ndarray) -> float:
    """How far the two EAR chords are from perpendicular to the corner axis,
    as a fraction of eye width. EAR assumes vertical chords: the numerator is
    read as lid separation, so a chord that leans along the eye picks up eye
    WIDTH as well as opening. Zero is a pair of vertical chords."""
    axis = p6[3] - p6[0]
    w = max(float(np.linalg.norm(axis)), 1e-9)
    u = axis / w
    return float((abs(u @ (p6[1] - p6[5])) + abs(u @ (p6[2] - p6[4]))) / (2 * w))


def chord_skew_one(p6: np.ndarray, which: int) -> float:
    """Skew of a single EAR chord: 0 = the outer chord p1-p5, 1 = the inner
    chord p2-p4. Reported separately because the two chords do not respond
    alike to an index change, which an average conceals."""
    axis = p6[3] - p6[0]
    w = max(float(np.linalg.norm(axis)), 1e-9)
    u = axis / w
    a, b = ((1, 5) if which == 0 else (2, 4))
    return float(abs(u @ (p6[a] - p6[b])) / w)


def eye_stats(pts24: np.ndarray) -> tuple[float, float]:
    """Mean EAR and mean chord skew over the two eyes of one face."""
    eyes = [pts24[0:6], pts24[6:12]]
    return (float(np.mean([eye_ear(e) for e in eyes])),
            float(np.mean([chord_skew(e) for e in eyes])))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of two per-face measurements of the same thing. The
    pairing is what makes a small sample usable: face-to-face variation is
    common to both and cancels."""
    d = a - b
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean_diff": float(d.mean()), "se": se, "n": n,
            "ci95": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)),
            "b_closer_pct": float(100 * np.mean(b < a))}


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
    # face size as a fraction of the shorter image side, kept per outcome: a
    # low usable count is a property of the detector, and which faces it
    # drops decides what population the medians below describe
    size_found, size_missed = [], []
    for n_img, (frame, pts98) in enumerate(samples, 1):
        h, w = frame.shape[:2]
        face_frac = float(square_box_around(pts98, 1.0).side / max(1, min(h, w)))
        mesh = det.mesh(frame)
        if mesh is None:
            n_none += 1
            size_missed.append(face_frac)
            continue
        mesh_size = mesh.shape[0] if mesh_size is None else mesh_size
        gt24 = pts98[schema.wflw_indices]
        if not match_target(mesh[indices].astype(np.float64), gt24):
            n_unmatched += 1
            continue
        meshes.append(mesh.astype(np.float64))
        gts.append(gt24)
        size_found.append(face_frac)
        iods.append(np.linalg.norm(gt24[schema.nme_right_index]
                                   - gt24[schema.nme_left_index]))
        if n_img % 100 == 0:
            say(f"  {n_img}/{len(samples)} images")

    say(f"\n  usable faces: {len(meshes)} "
        f"(no face: {n_none}, found a different face: {n_unmatched})")
    if size_found and size_missed:
        say(f"  target face size, fraction of the shorter image side: "
            f"median {np.median(size_found):.3f} where MediaPipe found it, "
            f"{np.median(size_missed):.3f} where it found nothing. The Tasks "
            "bundle uses a short-range face detector, so small faces in wide "
            "web photos drop out; the sample below is the faces it keeps.")
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
        "of index rather than just its accuracy. Read the relation column "
        "before acting: a neighbouring vertex on the same ring is the same "
        "anatomy sampled slightly differently, and moving to it fits WFLW's "
        "parameterisation rather than fixing a mapping error.")
    # (N, 24, 478) distances, normalised per face, then the median over faces
    d = np.linalg.norm(gt_arr[:, :, None, :] - mesh_arr[:, None, :, :], axis=3)
    d = d / iod_arr[:, None, None] * 100.0
    med_all = np.median(d, axis=0)                            # (24, 478)
    best_idx = med_all.argmin(axis=1)
    best_val = med_all.min(axis=1)
    features = mesh_features()
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

    if flagged:
        say("\n  Each flagged point, with what the alternative index actually "
            "is and whether the gap survives the sample size:")
        for pt, bi, bv, gap in flagged:
            rel = describe_alternative(indices[pt.index], bi, features)
            st = paired(off[:, pt.index], d[:, pt.index, bi])
            sig = ("significant" if st["ci95"][0] > 0 else "NOT significant")
            say(f"    {pt.name} ({pt.group}): {indices[pt.index]} -> {bi}")
            say(f"      relation: {rel}")
            say(f"      paired gain {st['mean_diff']:.2f} +- {st['se']:.2f} "
                f"points of IOD (95% CI {st['ci95'][0]:.2f} to "
                f"{st['ci95'][1]:.2f}, n={st['n']}), {sig}; the alternative is "
                f"closer on {st['b_closer_pct']:.0f}% of faces")
            if pt.group == "eyelids":
                # what the single swap does to the EAR chords on that eye:
                # the eyelid indices exist to make those chords vertical, so
                # a gain in NME that costs chord geometry is not a gain
                lo = 0 if pt.index < 6 else 6
                swapped = list(indices)
                swapped[pt.index] = bi
                before = float(np.mean([chord_skew(m[indices[lo:lo + 6]])
                                        for m in mesh_arr]))
                after = float(np.mean([chord_skew(m[swapped[lo:lo + 6]])
                                       for m in mesh_arr]))
                say(f"      EAR chord skew on that eye: {before:.4f} "
                    f"configured -> {after:.4f} with the swap "
                    f"({'worse' if after > before else 'better'})")

    # ---- 4: the candidate mappings on what they are used for --------------
    say("\n=== 4. Candidate mappings compared on what they are used for ===")
    alt_indices = [int(b) for b in best_idx]
    candidates = [("configured", list(indices))]
    candidates += [(name, list(idx)) for name, idx in schema.mediapipe_alt_maps]
    candidates.append(("proximity-optimal", alt_indices))

    eye_idx = schema.indices_of_group("eyelids")
    per_face_nme, per_face_ear = {}, {}
    for name, idx in candidates:
        o = (np.linalg.norm(mesh_arr[:, idx, :] - gt_arr, axis=2)
             / iod_arr[:, None]) * 100.0
        per_face_nme[name] = (o.mean(axis=1), o[:, eye_idx].mean())
    st_nme = paired(per_face_nme["configured"][0],
                    per_face_nme["proximity-optimal"][0])

    say("  NME over all 24 points, per face:")
    say(f"    {'mapping':<20} {'mean':>8} {'median':>8} {'eyelids only':>14}")
    for name, _ in candidates:
        v, eyelid = per_face_nme[name]
        say(f"    {name:<20} {v.mean():7.3f}% {np.median(v):7.3f}% "
            f"{eyelid:13.3f}%")
    say(f"    Switching every index to the proximity-optimal one would move "
        f"MediaPipe's NME by {-st_nme['mean_diff']:.3f} +- {st_nme['se']:.3f} "
        f"points (n={st_nme['n']}). That is the bound on how far the mapping "
        "choice can flatter or hurt either detector.")

    say("\n  EAR, which is what Layer 2 consumes and the reason the eyelid "
        "indices are chosen at all. Correlation is the criterion: Layer 2 "
        "calibrates per driver, so a constant offset is absorbed and a weaker "
        "relationship with the truth is not. It is a PROXY, measured across "
        "different faces because WFLW has no eye-state labels and no subject "
        "ids; the quantity Layer 2 really needs is within-driver separation "
        "of open from closed.")
    ear_gt = np.array([eye_stats(g)[0] for g in gt_arr])
    skew_gt = float(np.mean([eye_stats(g)[1] for g in gt_arr]))
    per_face_ear["ground truth"] = ear_gt
    rows = []
    for name, idx in candidates:
        e = np.array([eye_stats(m[idx])[0] for m in mesh_arr])
        per_face_ear[name] = e
        # per chord, not averaged: the two chords do not behave alike, and
        # the average hid that when this check was first written
        skew_a = float(np.mean([chord_skew_one(m[idx][lo:lo + 6], 0)
                                for m in mesh_arr for lo in (0, 6)]))
        skew_b = float(np.mean([chord_skew_one(m[idx][lo:lo + 6], 1)
                                for m in mesh_arr for lo in (0, 6)]))
        rows.append((name, e, float(np.mean(np.abs(e - ear_gt))),
                     float(np.corrcoef(e, ear_gt)[0, 1]), skew_a, skew_b))
    say(f"    {'mapping':<20} {'mean EAR':>9} {'|EAR-gt|':>9} {'corr':>7} "
        f"{'skew outer':>11} {'skew inner':>11}")
    say(f"    {'ground truth':<20} {ear_gt.mean():9.4f} {0.0:9.4f} "
        f"{1.0:7.3f} {skew_gt:11.4f} {'':>11}")
    for name, e, mae, r, sa, sb in rows:
        say(f"    {name:<20} {e.mean():9.4f} {mae:9.4f} {r:7.3f} "
            f"{sa:11.4f} {sb:11.4f}")
    say("    skew is the chord's tilt away from perpendicular to the corner "
        "axis, in eye widths; outer is the p1-p5 chord, inner is p2-p4.")

    # correlation differences carry a confidence interval, because two point
    # estimates a few hundredths apart decide nothing on their own
    say("\n  correlation difference against the configured mapping "
        f"(paired bootstrap over faces, {BOOTSTRAP_DRAWS} draws):")
    rng = np.random.default_rng(seed)
    base = per_face_ear["configured"]
    corr_stats = {}
    for name, e, _, r, _, _ in rows:
        if name == "configured":
            continue
        draws = np.empty(BOOTSTRAP_DRAWS)
        n = len(ear_gt)
        for k in range(BOOTSTRAP_DRAWS):
            j = rng.integers(0, n, n)
            draws[k] = (np.corrcoef(e[j], ear_gt[j])[0, 1]
                        - np.corrcoef(base[j], ear_gt[j])[0, 1])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        verdict = ("better than configured" if lo > 0 else
                   "worse than configured" if hi < 0 else
                   "indistinguishable from configured")
        say(f"    {name:<20} {draws.mean():+.4f}  95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")
        corr_stats[name] = {"delta": float(draws.mean()), "ci95": [float(lo), float(hi)],
                            "verdict": verdict}

    np.save(out_dir / "ear_ground_truth.npy", ear_gt)
    for name, e in per_face_ear.items():
        if name != "ground truth":
            np.save(out_dir / f"ear_{name.replace(' ', '_')}.npy", e)

    # ---- verdict -----------------------------------------------------------
    say("\n=== Verdict ===")
    if failed:
        say(f"  MAPPING FAIL: {len(failed)} point(s) beyond tolerance: "
            + ", ".join(f"{pt.name} ({med[pt.index]:.2f}%)" for pt in failed))
    else:
        say("  MAPPING OK: every point sits within its group's tolerance of "
            "the ground truth point it claims to be.")
    say(f"  pupils: left {med[12]:.2f}% / right {med[13]:.2f}% of IOD "
        "(the iris-head check; these are the indices that would silently "
        "vanish on a 468-point bundle)")
    same_ring = [f for f in flagged
                 if describe_alternative(indices[f[0].index], f[1],
                                         features).startswith("same ")]
    if flagged:
        say(f"  {len(flagged)} index/indices where another mesh point is more "
            f"than {BETTER_INDEX_MARGIN_PCT_IOD}% of IOD closer, "
            f"{len(same_ring)} of them a neighbouring vertex on the same "
            "feature. Same-feature neighbours are a difference in where along "
            "the feature each convention samples, not a mapping error, and "
            "chasing them optimises against WFLW's annotation style.")
    if not args.synthetic and args.split == "test":
        say("  NOTE: this ran on the test split. Its per-point table is a "
            "verification result, but the proximity-optimal indices in "
            "section 3 are FITTED to this data: adopting them from a test-split "
            "run is fitting the mapping to the evaluation set. Re-run with "
            "--split train if you intend to change any index.")

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
        "face_frac_median_found": float(np.median(size_found)) if size_found else None,
        "face_frac_median_missed": float(np.median(size_missed)) if size_missed else None,
        "per_point": per_point,
        "better_index_available": [
            {"name": pt.name, "configured": indices[pt.index],
             "configured_median_pct_iod": float(med[pt.index]),
             "best_mesh_index": bi, "best_median_pct_iod": bv, "gap_pct_iod": gap,
             "relation": describe_alternative(indices[pt.index], bi, features),
             "paired": paired(off[:, pt.index], d[:, pt.index, bi])}
            for pt, bi, bv, gap in flagged],
        # the fitted mapping, recorded so a decision to adopt it can be
        # traced to the run that produced it. Fitted on THIS split.
        "proximity_optimal_indices": alt_indices,
        "nme_by_mapping_pct": {name: float(per_face_nme[name][0].mean())
                               for name, _ in candidates},
        "nme_paired_configured_vs_proximity": st_nme,
        "ear": {"gt_mean": float(ear_gt.mean()), "gt_chord_skew": skew_gt,
                "by_mapping": {name: {"mean": float(e.mean()), "mae_vs_gt": mae,
                                      "corr_vs_gt": r, "skew_outer_chord": sa,
                                      "skew_inner_chord": sb}
                               for name, e, mae, r, sa, sb in rows},
                "corr_difference_vs_configured": corr_stats},
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
