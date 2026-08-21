# Milestone 1 verdict: the 98→24 mapping is correct

**Date:** 2026-08-21
**Dataset:** WFLW test split (2,500 faces), Kaggle mount
**Tooling:** `scripts/verify_layout.py` + `scripts/diagnose_layout_failures.py`
at commit `6aa6d27` (the diagnostic run analysed exactly the verifier's failure
set — its reproduction section matched the verifier's pass rates digit for
digit: 89.47 / 91.26 / 85.46 / 89.10).

## Context

`verify_layout.py` runs statistical checks of the assumed WFLW 98-point index
layout against the real annotation file. Every ordering and containment check
passed (pupil-inside-eye at 99.9%), while four symmetry/midline checks came
back FAIL on the frontal (`pose==0`) population: chin-is-lowest, contour-pair
height match for (4,28) and (8,24), and nose-on-midline. The magnitude
diagnostic below attributes all four to head-pose sensitivity of the checks
themselves, not to wrong mapping indices.

## Table 1 — pass rates by population

Populations: frontal = WFLW `pose` flag 0; no-flag = all six attribute flags 0;
low |yaw| = the no-flag faces below that subset's median contour-asymmetry yaw
proxy.

| check          | frontal (pose==0) | no-flag | no-flag & low \|yaw\| |
|----------------|------------------:|--------:|----------------------:|
| chin lowest    | 89.47%            | 92.88%  | 98.13%                |
| pair (4,28)    | 91.26%            | 92.32%  | 99.63%                |
| pair (8,24)    | 85.46%            | 87.83%  | 99.63%                |
| nose midline   | 89.10%            | 91.01%  | 100.00%               |

## Table 2 — failure rate vs estimated head yaw

Yaw proxy = contour half-width asymmetry about the eye axis; cross-checked
against an independent eye-width-asymmetry proxy (r = −0.852, i.e. both track
the same head-yaw signal).

| check          | fail % at \|yaw\| 0.00–0.05 | fail % at \|yaw\| ≥ 0.30 | median \|yaw\|, failing | median \|yaw\|, passing |
|----------------|---------------------------:|-------------------------:|------------------------:|------------------------:|
| chin lowest    | 1.0%                       | 27.7%                    | 0.344                   | 0.143                   |
| pair (4,28)    | 0.2%                       | 22.7%                    | 0.339                   | 0.144                   |
| pair (8,24)    | 0.2%                       | 40.8%                    | 0.348                   | 0.130                   |
| nose midline   | 0.2%                       | 38.3%                    | 0.401                   | 0.137                   |

Additional pose evidence:

* Signed nose offset vs signed yaw correlates at −0.886, and **99.58%** of
  nose-midline failures have the offset sign opposite to the yaw sign — the
  locked-direction signature of a 3D nose swinging off the eye axis under
  yaw, not of a wrong index.
* Chin failures are **not** near-ties: only 1.75% lie within 1% of IOD of
  point 16, and the lowest index spreads across 14–18 (44.98% at 15 or 17).
  With chin margin correlating at +0.374 with |yaw|, the jaw contour is
  sliding along the visible face boundary under rotation — annotation
  behaviour, not misindexing.

## Table 3 — chin tolerance: absolute vs IOD-relative pass rates (frontal)

The chin check originally used an absolute 2.0px tolerance — confirmed to be
the wrong shape (scale-dependent: stricter for large faces).

| tolerance            | pass rate |
|----------------------|----------:|
| 0.5% of IOD          | 64.63%    |
| 1% of IOD            | 74.33%    |
| 2% of IOD            | 83.99%    |
| **3% of IOD**        | **89.56%**|
| 5% of IOD            | 94.16%    |
| 2.0px absolute (old) | 89.47%    |

## Verdict

The 98→24 index mapping defined in `configs/landmarks_24.yaml` is correct.
All four failing checks are pose-sensitive by construction — they measure
symmetry and midline alignment, which real 3D heads violate under yaw — and
WFLW's `pose` flag only marks large poses, so moderately turned faces remain
in the "frontal" population. On the cleanest available subset (no attribute
flags, below-median yaw) all four checks pass at 98.13–100%, failure rates
climb monotonically with the yaw proxy across all four checks, and the nose
offset direction is locked to the yaw direction on 99.58% of its failures.
The residual failures are therefore pose-driven measurement effects, two of
them amplified by defects in the checks themselves (the absolute chin
tolerance, and a nose tolerance that ignores nose protrusion). Following this
verdict the chin check was switched to a 3%-of-IOD relative tolerance
(`dms_layer1/data/frame.py`, `CHIN_TOL_IOD`), which reproduces the old
operating point (89.56% vs 89.47%) while being scale-fair. No mapping index
was changed.
