#!/usr/bin/env python3
"""Milestone 7: frame-to-frame stability, measured in the order the milestone
6 findings imply.

Accuracy on stills said the pipeline is dominated by acquisition, not by the
landmark model, and that 86% of the remaining centre error is the Haar box
landing differently on each face. If that carries over to time, box motion
is the dominant jitter source and a landmark jitter number measured without
it would mostly be measuring the box. So the harness measures the box first
and reports the model's own contribution separately:

  1. Box jitter: how much the stage-1 crop moves and resizes, in box-side
     units, against a box that tracks the face perfectly. Same units as
     face_detector.box_shift_x/y, so it is comparable to the per-face
     scatter measured in milestone 6.
  2. Model jitter: landmarks predicted in a per-frame ground-truth box, so
     acquisition is perfect and only the image changes. This is the model's
     own noise floor.
  3. End-to-end jitter: the deployed path. The difference from (2) is what
     acquisition contributes.
  4. What refinement does to jitter: stage 1 alone against two stages. Stage
     2 rebuilds its box from stage 1's points, which is a feedback path: it
     can damp box jitter or amplify it, and milestone 6 gives no way to
     predict which.
  5. The Layer 2 signals: EAR, MAR and the gaze proxy, with their jitter and
     with a false-crossing rate against a per-sequence blink threshold. A
     landmark comparison only matters insofar as it changes these.

MediaPipe runs beside ours for (3) and (5), so the ablation extends to
stability rather than stopping at accuracy.

Sequences are synthesised from WFLW stills (see dms_layer1/stability/
sequences.py): a smooth random walk plus sensor noise, with the annotation
warped alongside, so ground truth is known per frame and jitter is measured
on the RESIDUAL rather than on the raw trajectory. A 2D warp cannot produce
out-of-plane rotation, blinking or expression, so these numbers are a lower
bound. --video runs the same measurements on a real clip, where there is no
ground truth and jitter is estimated from the trajectory's high-frequency
part instead; that estimator is not comparable to the synthesised one and is
reported in its own section.

Usage (Kaggle):
    python scripts/stability_harness.py --config configs/layer1_base.yaml
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
from dms_layer1.detect.interface import as_gray
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.evaluation import metrics, signals
from dms_layer1.evaluation.matching import match_target
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.stability.metrics import (box_jitter, crossing_rate, gap_runs,
                                          highfreq_jitter, residual_jitter,
                                          rule_of_three, split_jumps,
                                          sustained_crossing_rate,
                                          threshold_margin)
from dms_layer1.stability.sequences import SequenceSpec, make_sequence

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def load_m6_scatter(path: str | None) -> dict | None:
    """The across-faces centre scatter from a milestone 6 run. Read from the
    file rather than written into this script: a hardcoded reference number
    is a claim the code cannot check, and one written here from memory was
    wrong by a factor of two before anyone noticed."""
    if not path:
        return None
    data = yaml.safe_load(Path(path).read_text())
    ce = (data or {}).get("centre_error")
    if not ce or "shift_x_scatter" not in ce:
        return None
    return ce


def spec_from_cfg(cfg: dict) -> SequenceSpec:
    return SequenceSpec(
        frames=int(require(cfg, "stability.frames")),
        translate_frac=float(require(cfg, "stability.translate_frac")),
        rotate_deg=float(require(cfg, "stability.rotate_deg")),
        scale_frac=float(require(cfg, "stability.scale_frac")),
        smoothness=float(require(cfg, "stability.smoothness")),
        noise_sigma=float(require(cfg, "stability.noise_sigma")),
        brightness_amp=float(require(cfg, "stability.brightness_amp")))


def pick_faces(cfg: dict, split: str, n: int, seed: int) -> list:
    """Seeded sample of large single-face images: the cabin analogue, and the
    population where every detector under test can be expected to work, so a
    jitter number is not dominated by frames nobody found."""
    records, paths = wflw.load_split(cfg, split)
    by_image: dict[str, list] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    singles = [rel for rel, recs in by_image.items() if len(recs) == 1]
    rng = random.Random(seed)
    rng.shuffle(singles)
    chosen = []
    for rel in singles:
        rec = by_image[rel][0]
        img = cv2.imread(str(wflw.image_path(paths, rec)), cv2.IMREAD_COLOR)
        if img is None:
            continue
        side = square_box_around(rec.landmarks98, 1.0).side
        if side / max(1, min(img.shape[:2])) < 0.25:
            continue
        chosen.append((rel, img, rec.landmarks98.astype(np.float64)))
        if len(chosen) >= n:
            break
    return chosen


MIN_VALID_FRAMES = 8   # below this a standard deviation over time is noise
# A box centre step larger than this fraction of the box side is a discrete
# event (a different candidate winning, or a different pyramid level) rather
# than wobble. Cascade detectors do both and the two need different fixes.
JUMP_THRESHOLD = 0.05


def summarise(per_seq: list, key: str | None = None) -> float:
    """Median over sequences, skipping those with too few usable frames. A
    sequence nobody could track is a dropout result, not a jitter result."""
    vals = [d if key is None else d[key] for d in per_seq if d is not None]
    return float(np.median(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sequences", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--m6-yaml", default=None,
                    help="milestone 6 m6_diagnosis.yaml, so the across-faces "
                         "centre scatter is read rather than quoted")
    ap.add_argument("--static", action="store_true",
                    help="with --video: the subject is stationary, so jitter "
                         "is the raw trajectory spread rather than its "
                         "high-frequency part")
    ap.add_argument("--ablate-perturbation", action="store_true",
                    help="repeat section 1 with motion only and with noise "
                         "only, to attribute box jitter to each")
    ap.add_argument("--video", default=None,
                    help="a real clip; scored on self-consistency, no truth")
    ap.add_argument("--synthetic", action="store_true",
                    help="schematic faces and a briefly trained model "
                         "(code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from dms_layer1.data.synthetic import synthetic_trained_setup
        cfg, ckpt = synthetic_trained_setup(cfg)
        cfg["detector"]["weights"] = str(ckpt)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    seed = int(require(cfg, "seed"))
    out_dir = Path(args.out_dir or require(cfg, "stability.out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    expand = float(require(cfg, "preprocess.reference_expand"))
    n_seq = args.sequences or int(require(cfg, "stability.sequences"))
    spec = spec_from_cfg(cfg)
    if args.frames:
        spec = SequenceSpec(**{**spec.__dict__, "frames": args.frames})
    blink_frac = float(require(cfg, "stability.blink_threshold_frac"))

    ours = OurLandmarkDetector(cfg, weights=args.weights)
    mp_det = None
    try:
        from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                          MediaPipeUnavailable)
        try:
            mp_det = MediaPipeLandmarkDetector(cfg, schema)
        except MediaPipeUnavailable as e:
            say(f"mediapipe unavailable, ours-only: {e}")
    except ImportError as e:
        say(f"mediapipe import failed, ours-only: {e}")

    say("\n=== Sequences ===")
    if args.synthetic:
        from dms_layer1.data.synthetic import generate_face
        rngp = random.Random(seed)
        faces = []
        for _ in range(min(n_seq, 4)):
            img, p98 = generate_face(angle_deg=rngp.uniform(-8, 8))
            faces.append(("synthetic", img, p98.astype(np.float64)))
        say("  NOTE: synthetic mode is a code smoke test, not a measurement.")
    else:
        faces = pick_faces(cfg, args.split, n_seq, seed)
    say(f"  {len(faces)} sequences of {spec.frames} frames, from {args.split} "
        "single-face images at least 0.25 of the shorter side")
    say(f"  per-frame motion: translate {spec.translate_frac:.3f} of the face "
        f"side, rotate {spec.rotate_deg:.2f} deg, scale {spec.scale_frac:.3f}, "
        f"smoothness {spec.smoothness:.2f}, noise sigma {spec.noise_sigma:.1f}")
    say("  synthesised motion is a 2D warp: no out-of-plane rotation, no "
        "blinking, no expression. These jitter numbers are a LOWER BOUND.")

    rows = ["ours_gt_box", "ours_stage1", "ours_deploy"]
    if mp_det is not None:
        rows.append("mediapipe")
    jit: dict[str, list] = {r: [] for r in rows}
    nme_seq: dict[str, list] = {r: [] for r in rows}
    sig_jit: dict[str, list] = {r: [] for r in rows}
    boxes_seq, dropouts = [], {r: [] for r in rows}
    jumps_seq: list[dict] = []
    runs_seq: dict[str, list] = {r: [] for r in rows}
    margins: dict[str, list] = {r: [] for r in rows}
    cross_events = {r: 0 for r in rows}
    cross_frames = {r: 0 for r in rows}
    group_jit: dict[str, list] = {r: [] for r in rows}

    for n_face, (rel, image, pts98) in enumerate(faces, 1):
        rng = np.random.default_rng(seed + n_face)
        face_side = square_box_around(pts98, 1.0).side
        frames, seq98 = make_sequence(image, pts98, spec, rng, face_side)
        gt24 = seq98[:, schema.wflw_indices, :]
        iod = np.array([signals.interocular_of(g, schema) for g in gt24])

        pred = {r: [] for r in rows}
        drop = {r: 0 for r in rows}
        stage1_boxes, gt_boxes = [], []
        for t, frame in enumerate(frames):
            gray = as_gray(frame)
            g_box = square_box_around(seq98[t], expand)
            gt_boxes.append(g_box)
            pred["ours_gt_box"].append(ours.predict_in_box(gray, g_box))

            out = ours.detect(frame)
            box = getattr(ours, "last_stage1_box", None)
            stage1_boxes.append(box)
            # A dropped frame contributes nothing rather than holding the
            # previous prediction: holding turns the face's real motion into
            # measured jitter, so a detector that fails often would look
            # unsteady for the wrong reason. Dropout is reported separately.
            if out is None or not match_target(out.points.astype(np.float64), gt24[t]):
                drop["ours_deploy"] += 1
                pred["ours_deploy"].append(None)
            else:
                pred["ours_deploy"].append(out.points.astype(np.float64))
            # stage 1 alone, from the same box: no second Haar pass needed
            if box is None:
                drop["ours_stage1"] += 1
                pred["ours_stage1"].append(None)
            else:
                pred["ours_stage1"].append(ours.predict_in_box(gray, box))
            if mp_det is not None:
                mp_out = mp_det.detect(frame)
                if mp_out is None or not match_target(
                        mp_out.points.astype(np.float64), gt24[t]):
                    drop["mediapipe"] += 1
                    pred["mediapipe"].append(None)
                else:
                    pred["mediapipe"].append(mp_out.points.astype(np.float64))

        boxes_seq.append(box_jitter(stage1_boxes, gt_boxes))
        centres = np.array([[(b.x0 + b.side / 2.0) / b.side,
                             (b.y0 + b.side / 2.0) / b.side]
                            if b is not None else [np.nan, np.nan]
                            for b in stage1_boxes])
        gcent = np.array([[(g.x0 + g.side / 2.0) / max(b.side, 1e-9),
                           (g.y0 + g.side / 2.0) / max(b.side, 1e-9)]
                          if b is not None else [np.nan, np.nan]
                          for b, g in zip(stage1_boxes, gt_boxes)])
        resid = centres - gcent
        ok = ~np.isnan(resid).any(axis=1)
        if ok.sum() > 2:
            jumps_seq.append(split_jumps(resid[ok], JUMP_THRESHOLD))
        for r in rows:
            runs_seq[r].append(gap_runs([v is not None for v in pred[r]]))
        for r in rows:
            keep = [t for t, v in enumerate(pred[r]) if v is not None]
            dropouts[r].append(drop[r] / len(frames))
            if len(keep) < MIN_VALID_FRAMES:
                jit[r].append(None)
                nme_seq[r].append(None)
                group_jit[r].append(None)
                sig_jit[r].append(None)
                continue
            arr = np.stack([pred[r][t] for t in keep])
            gt_keep, iod_keep = gt24[keep], iod[keep]
            jit[r].append(residual_jitter(arr, gt_keep, iod_keep))
            nme_seq[r].append(float(metrics.nme_per_face(arr, gt_keep, schema).mean()))
            group_jit[r].append({
                g: float(jit[r][-1]["per_point_pct"][schema.indices_of_group(g)].mean())
                for g in schema.groups})
            s_pred = signals.signal_series(arr)
            s_gt = signals.signal_series(gt_keep)
            thr = float(blink_frac * np.mean(s_gt["ear_mean"]))
            sig_jit[r].append({
                "ear": float(np.std(s_pred["ear_mean"] - s_gt["ear_mean"])),
                "mar": float(np.std(s_pred["mar"] - s_gt["mar"])),
                "gaze": float(np.mean(np.std(s_pred["gaze"] - s_gt["gaze"], axis=0))),
                "ear_false_crossings": crossing_rate(s_pred["ear_mean"], thr),
                "ear_sustained": sustained_crossing_rate(s_pred["ear_mean"], thr),
                "ear_true_crossings": crossing_rate(s_gt["ear_mean"], thr)})
            margins[r].append(threshold_margin(s_pred["ear_mean"], thr))
            cross_events[r] += int(round(
                crossing_rate(s_pred["ear_mean"], thr) * (len(arr) - 1)))
            cross_frames[r] += len(arr) - 1
        if n_face % 5 == 0:
            say(f"  {n_face}/{len(faces)} sequences")

    # ---- 1. box jitter -----------------------------------------------------
    say("\n=== 1. Stage-1 box, in box-side units ===")
    say("  Measured first because milestone 6 put 86% of the centre residual "
        "in per-face scatter; if that carries over to time, everything below "
        "is downstream of it.")
    good = [b for b in boxes_seq if b.get("n")]
    if good:
        for k, label in (("centre_bias_x", "centre bias x"),
                         ("centre_bias_y", "centre bias y"),
                         ("centre_jitter_x", "centre jitter x"),
                         ("centre_jitter_y", "centre jitter y"),
                         ("centre_jitter", "centre jitter, combined"),
                         ("side_jitter", "side jitter")):
            v = np.median([b[k] for b in good])
            say(f"  {label:<26} {v:+.4f}")
        say("  Bias is where the box sits, jitter is how much it moves. Only "
            "the second reaches the landmarks as noise.")
        ref = load_m6_scatter(args.m6_yaml)
        if ref is None:
            say("  The comparable across-faces number is centre_error."
                "shift_x_scatter and shift_y_scatter in the milestone 6 "
                "diagnosis YAML. Pass --m6-yaml to have it printed here "
                "rather than quoted from memory.")
        else:
            comb = float(np.hypot(ref["shift_x_scatter"], ref["shift_y_scatter"]))
            say(f"  across FACES, from {args.m6_yaml}: "
                f"x {ref['shift_x_scatter']:.4f}, y {ref['shift_y_scatter']:.4f}, "
                f"combined {comb:.4f}")
            v = np.median([b["centre_jitter"] for b in good])
            say(f"  across FRAMES, here: {v:.4f}, a ratio of {v / max(comb, 1e-9):.2f}. "
                "Same units and same definition, so the two are comparable.")
    if jumps_seq:
        say(f"\n  discrete against continuous, split at a {JUMP_THRESHOLD:.2f} "
            "box-side step:")
        say(f"    jump rate          {np.median([j['jump_rate'] for j in jumps_seq]):.4f} "
            "of frame pairs")
        say(f"    largest jump       {np.median([j['largest_jump'] for j in jumps_seq]):.4f} "
            "box sides")
        say(f"    continuous wobble  {np.median([j['continuous_sd'] for j in jumps_seq]):.4f} "
            "box sides per frame")
        say("    A cascade searches a scale pyramid on a grid and merges "
            "neighbours, so a small image change can flip which candidate "
            "wins or at which level, and the box steps rather than slides. "
            "One aggregate standard deviation cannot tell a few large steps "
            "from constant small wobble, and the two call for different "
            "fixes: temporal smoothing helps the second and not the first.")

    # ---- 1b. is section 1 measuring the detector or the perturbation? ------
    if args.ablate_perturbation and not args.synthetic:
        say("\n=== 1b. Attributing box jitter to motion against sensor noise ===")
        say("  Synthesised sequences perturb two ways at once, and the box "
            "jitter above is the sum. If it is mostly the noise term then the "
            "number is a property of the chosen sigma rather than of the "
            "detector, and only a real clip can set that sigma. Same faces, "
            "same seeds, one perturbation at a time.")
        variants = {
            "motion only": SequenceSpec(**{**spec.__dict__, "noise_sigma": 0.0,
                                           "brightness_amp": 0.0}),
            "noise only": SequenceSpec(**{**spec.__dict__, "translate_frac": 0.0,
                                          "rotate_deg": 0.0, "scale_frac": 0.0}),
            "both (as above)": spec,
        }
        say(f"  {'perturbation':<18} {'centre jitter':>14} {'jump rate':>11} "
            f"{'continuous':>11}")
        ablation = {}
        for label, vspec in variants.items():
            per_seq, per_jump = [], []
            for n_face, (rel, image, pts98) in enumerate(faces, 1):
                rng = np.random.default_rng(seed + n_face)
                fs = square_box_around(pts98, 1.0).side
                vframes, vseq = make_sequence(image, pts98, vspec, rng, fs)
                bx, gb = [], []
                for t, frame in enumerate(vframes):
                    ours.detect(frame)
                    bx.append(getattr(ours, "last_stage1_box", None))
                    gb.append(square_box_around(vseq[t], expand))
                stats = box_jitter(bx, gb)
                if stats.get("n"):
                    per_seq.append(stats["centre_jitter"])
                    res = np.array([[(b.x0 + b.side / 2.0 - g.x0 - g.side / 2.0) / b.side,
                                     (b.y0 + b.side / 2.0 - g.y0 - g.side / 2.0) / b.side]
                                    for b, g in zip(bx, gb) if b is not None])
                    if len(res) > 2:
                        per_jump.append(split_jumps(res, JUMP_THRESHOLD))
            if per_seq:
                cj = float(np.median(per_seq))
                jr = float(np.median([j["jump_rate"] for j in per_jump])) if per_jump else float("nan")
                cs = float(np.median([j["continuous_sd"] for j in per_jump])) if per_jump else float("nan")
                ablation[label] = {"centre_jitter": cj, "jump_rate": jr,
                                   "continuous_sd": cs}
                say(f"  {label:<18} {cj:14.4f} {jr:11.4f} {cs:11.4f}")
        say("  Read it this way: if 'noise only' is close to 'both', section 1 "
            "is measuring the noise model and the real clip must set the "
            "sigma before the number means anything. If 'motion only' "
            "dominates, the box is stepping in response to sub-pixel movement "
            "of the face, which is detector behaviour that any real sequence "
            "would also produce.")
    else:
        ablation = None

    # ---- 2 to 4. landmark jitter ------------------------------------------
    say("\n=== 2-4. Landmark jitter (residual standard deviation, % of IOD) ===")
    say(f"  {'row':<14} {'jitter':>8} {'worst pt':>9} {'bias':>8} {'NME':>8} "
        f"{'dropout':>8}")
    for r in rows:
        usable = sum(1 for d in jit[r] if d is not None)
        say(f"  {r:<14} {summarise(jit[r], 'mean_pct'):8.3f} "
            f"{summarise(jit[r], 'worst_pct'):9.3f} "
            f"{summarise(jit[r], 'bias_mean_pct'):8.3f} "
            f"{100 * summarise(nme_seq[r]):8.3f} "
            f"{100 * float(np.median(dropouts[r])):7.1f}% "
            f"  ({usable}/{len(faces)} sequences usable)")
    say("  ours_gt_box is the model's own noise floor (perfect acquisition). "
        "ours_deploy minus that is what the front end contributes. "
        "ours_stage1 against ours_deploy is what refinement does to jitter: "
        "lower means the second stage damps box motion, higher means its "
        "feedback amplifies it.")

    say("\n  availability, which is its own result and not a footnote:")
    say("  a dropped frame is a frame with no drowsiness estimate, and Layer 2 "
        "aggregates over a window, so the length of a gap matters as much as "
        "the rate: the same 3% lost one frame at a time and lost in one "
        "blackout are different failures.")
    say(f"  {'row':<14} {'dropout':>9} {'longest gap':>12} {'gaps/seq':>9} "
        f"{'worst gap':>12}")
    fps = 30.0
    for r in rows:
        drop = 100 * float(np.median([g["dropout"] for g in runs_seq[r]]))
        longest = float(np.median([g["longest_gap"] for g in runs_seq[r]]))
        worst = max(g["longest_gap"] for g in runs_seq[r]) if runs_seq[r] else 0
        gaps = float(np.median([g["gaps"] for g in runs_seq[r]]))
        say(f"  {r:<14} {drop:8.2f}% {longest:11.1f}f {gaps:9.1f} "
            f"{worst / fps:11.2f}s (worst, at {fps:.0f} fps)")

    say("\n  per group:")
    say(f"  {'row':<14}" + "".join(f"{g:>11}" for g in schema.groups))
    for r in rows:
        vals = {g: summarise([d[g] if d else None for d in group_jit[r]])
                for g in schema.groups}
        say(f"  {r:<14}" + "".join(f"{vals[g]:10.3f} " for g in schema.groups))

    # ---- 5. the Layer 2 signals -------------------------------------------
    say("\n=== 5. What reaches Layer 2 ===")
    say("  EAR and MAR jitter are standard deviations of the error against "
        "ground truth over the sequence; gaze is the pupil's position inside "
        "the eye, normalised by eye width, which is what a zone classifier "
        "consumes. False crossings are threshold crossings on a sequence "
        f"where the truth crosses none (threshold {blink_frac:.2f} of the "
        "sequence's own mean EAR, the per-driver calibration Layer 2 does).")
    say(f"  {'row':<14} {'EAR jitter':>11} {'MAR jitter':>11} "
        f"{'gaze jitter':>12} {'false cross':>12}")
    for r in rows:
        say(f"  {r:<14} "
            f"{summarise([d['ear'] if d else None for d in sig_jit[r]]):11.4f} "
            f"{summarise([d['mar'] if d else None for d in sig_jit[r]]):11.4f} "
            f"{summarise([d['gaze'] if d else None for d in sig_jit[r]]):12.4f} "
            f"{100 * summarise([d['ear_false_crossings'] if d else None for d in sig_jit[r]]):11.2f}%")
    say(f"\n  {'row':<14} {'margin (sd)':>12} {'sustained':>11} "
        f"{'95% bound':>11}")
    for r in rows:
        m = summarise(margins[r])
        sus = summarise([d["ear_sustained"] if d else None for d in sig_jit[r]])
        bound = rule_of_three(cross_events[r], cross_frames[r])
        say(f"  {r:<14} {m:12.2f} {100 * sus:10.2f}% {100 * bound:10.3f}%")
    say("  margin is how far the EAR sits from the threshold in units of its "
        "own jitter, and it is the number that generalises: zero crossings on "
        "these sequences is a fact about these sequences, a margin of five "
        "standard deviations is a property of the system. Sustained counts "
        "only runs of three frames or more, which is what a blink detector "
        "would act on; single-frame dips are not blinks.")
    n_pairs = cross_frames[rows[0]]
    worst_gap_s = 1.0 / max(rule_of_three(0, n_pairs) * 30.0, 1e-9)
    say(f"  the 95% bound is the rule of three: no events in {n_pairs} frame "
        "pairs bounds the rate at 3/n, not at zero. At 30 fps that is one "
        f"event every {worst_gap_s:.1f} seconds in the worst case consistent "
        "with seeing none. Sequences buy this bound linearly and a real "
        "recording buys it in the regime that matters, which is why a longer "
        "clip is worth more here than more synthetic frames.")
    true_cross = summarise([d["ear_true_crossings"] if d else None
                            for d in sig_jit[rows[0]]])
    say(f"  control, ground-truth EAR crossing the same threshold: "
        f"{100 * true_cross:.2f}%. Any excess above this is invented by the "
        "landmarks, and at the decision layer it is a false blink.")

    # ---- optional: a real clip --------------------------------------------
    video_out = None
    if args.video:
        say("\n=== Real clip (different estimator, not comparable above) ===")
        say(f"  {args.video}, treated as "
            + ("STATIC: the subject is stationary, so the raw spread of the "
               "trajectory is the jitter and no smoothing assumption is "
               "needed. This is the strongest ground-truth-free estimator "
               "available and the reason a still segment is worth recording."
               if args.static else
               "MOVING: jitter is the deviation from the trajectory's own "
               "moving average, which assumes real motion is slower than the "
               "window. Weaker than the static estimator."))
        cap = cv2.VideoCapture(args.video)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pts_ours, boxes_v, valid_ours, valid_mp, pts_mp = [], [], [], [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = ours.detect(frame)
            box = getattr(ours, "last_stage1_box", None)
            valid_ours.append(out is not None)
            if out is not None:
                pts_ours.append(out.points.astype(np.float64))
            if box is not None:
                boxes_v.append(box)
            if mp_det is not None:
                m = mp_det.detect(frame)
                valid_mp.append(m is not None)
                if m is not None:
                    pts_mp.append(m.points.astype(np.float64))
        cap.release()
        say(f"  {len(valid_ours)} frames at {fps_v:.1f} fps")

        video_out = {}
        if len(boxes_v) > 3:
            c = np.array([[(b.x0 + b.side / 2.0) / b.side,
                           (b.y0 + b.side / 2.0) / b.side] for b in boxes_v])
            jit_box = (float(np.linalg.norm(c.std(axis=0))) if args.static
                       else highfreq_jitter(c))
            jm = split_jumps(c, JUMP_THRESHOLD)
            sides = np.array([b.side for b in boxes_v], dtype=np.float64)
            say(f"  box centre jitter {jit_box:.4f} box sides, jump rate "
                f"{jm['jump_rate']:.4f}, largest jump {jm['largest_jump']:.4f}, "
                f"continuous {jm['continuous_sd']:.4f}")
            say(f"  box side, relative spread {sides.std() / max(sides.mean(), 1e-9):.4f}")
            video_out["box"] = {"centre_jitter": jit_box, **jm}
            say("  THIS is the number section 1 needs: the same quantity on "
                "real sensor noise and real motion, so the synthesised one "
                "can be calibrated against it rather than trusted.")

        for label, pts, valid in (("ours", pts_ours, valid_ours),
                                  ("mediapipe", pts_mp, valid_mp)):
            if len(pts) < 10:
                say(f"  {label}: too few frames with a face ({len(pts)})")
                continue
            arr = np.stack(pts)
            iod = np.array([signals.interocular_of(f, schema) for f in arr])
            flat = arr.reshape(len(arr), -1) / iod[:, None]
            sig = signals.signal_series(arr)
            lm = (float(np.linalg.norm(flat.std(axis=0)) / np.sqrt(arr.shape[1]))
                  if args.static else highfreq_jitter(flat) / np.sqrt(arr.shape[1]))
            ear_j = (float(sig["ear_mean"].std()) if args.static
                     else highfreq_jitter(sig["ear_mean"]))
            runs = gap_runs(valid)
            thr = blink_frac * float(np.median(sig["ear_mean"]))
            say(f"  {label:<10} landmark jitter {100 * lm:6.3f}% of IOD, "
                f"EAR jitter {ear_j:.4f}, dropout "
                f"{100 * runs['dropout']:.2f}%, longest gap "
                f"{runs['longest_gap'] / max(fps_v, 1):.2f}s")
            say(f"             sustained EAR crossings "
                f"{100 * sustained_crossing_rate(sig['ear_mean'], thr):.2f}% of "
                "frames; on a real clip these include GENUINE blinks, so read "
                "them against the still segment where there should be none")
            video_out[label] = {"landmark_jitter_pct": 100 * lm,
                                "ear_jitter": ear_j, **runs}
        say("  This section uses a different estimator from the ones above "
            "and the two must not be put in one table.")

    results = {
        "sequences": len(faces), "frames": spec.frames,
        "spec": spec.__dict__,
        "box": {k: float(np.median([b[k] for b in good])) for k in
                ("centre_bias_x", "centre_bias_y", "centre_jitter_x",
                 "centre_jitter_y", "centre_jitter", "side_jitter")} if good else None,
        "landmark_jitter_pct": {r: summarise(jit[r], "mean_pct") for r in rows},
        "nme_pct": {r: 100 * summarise(nme_seq[r]) for r in rows},
        "dropout_pct": {r: 100 * float(np.median(dropouts[r])) for r in rows},
        "usable_sequences": {r: sum(1 for d in jit[r] if d is not None) for r in rows},
        "group_jitter_pct": {r: {g: summarise([d[g] if d else None
                                               for d in group_jit[r]])
                                 for g in schema.groups} for r in rows},
        "signals": {r: {k: summarise([d[k] if d else None for d in sig_jit[r]])
                        for k in ("ear", "mar", "gaze", "ear_false_crossings")}
                    for r in rows},
        "ear_true_crossing_rate": true_cross,
        "video": video_out,
        "ear_margin_sd": {r: summarise(margins[r]) for r in rows},
        "ear_crossing_95pct_bound": {r: rule_of_three(cross_events[r],
                                                      cross_frames[r])
                                     for r in rows},
        "availability": {r: {
            "dropout_pct": 100 * float(np.median([g["dropout"] for g in runs_seq[r]])),
            "longest_gap_frames": int(max((g["longest_gap"] for g in runs_seq[r]),
                                          default=0)),
            "gaps_per_sequence": float(np.median([g["gaps"] for g in runs_seq[r]]))}
            for r in rows},
        "perturbation_ablation": ablation,
        "box_jumps": {k: float(np.median([j[k] for j in jumps_seq]))
                      for k in ("jump_rate", "continuous_sd", "largest_jump")}
        if jumps_seq else None,
    }
    with open(out_dir / "m7_stability.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    if mp_det is not None:
        mp_det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
