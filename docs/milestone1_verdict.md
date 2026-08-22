# Milestone 1 verdict: the 98 to 24 mapping is correct

Date: 2026-08-21. Dataset: WFLW test split (2,500 faces) on Kaggle.
Tooling: `scripts/verify_layout.py` and `scripts/diagnose_layout_failures.py`
at commit `6aa6d27`. The diagnostic reproduced the verifier's pass rates
digit for digit (89.47 / 91.26 / 85.46 / 89.10), so both analysed the same
set of failing faces.

## Background

`verify_layout.py` checks the assumed WFLW 98 point index layout against
the real annotation file. Every ordering and containment check passed, with
pupil-inside-eye at 99.9%. Four symmetry and midline checks failed on the
frontal population (faces with the pose flag unset): chin is lowest, the
height match of contour pairs (4,28) and (8,24), and nose on midline. The
diagnostic below shows all four failures come from head pose sensitivity in
the checks themselves, not from wrong mapping indices.

## Table 1: pass rates by population

Populations: frontal means the WFLW pose flag is 0. No-flag means all six
attribute flags are 0. Low yaw means the no-flag faces below that subset's
median yaw proxy (contour asymmetry).

| check          | frontal (pose==0) | no-flag | no-flag and low \|yaw\| |
|----------------|------------------:|--------:|------------------------:|
| chin lowest    | 89.47%            | 92.88%  | 98.13%                  |
| pair (4,28)    | 91.26%            | 92.32%  | 99.63%                  |
| pair (8,24)    | 85.46%            | 87.83%  | 99.63%                  |
| nose midline   | 89.10%            | 91.01%  | 100.00%                 |

## Table 2: failure rate against estimated head yaw

The yaw proxy is contour half width asymmetry about the eye axis. It was
cross checked against an independent proxy from eye width asymmetry, r =
-0.852, so it tracks real head yaw.

| check          | fail % at \|yaw\| 0.00-0.05 | fail % at \|yaw\| >= 0.30 | median \|yaw\|, failing | median \|yaw\|, passing |
|----------------|---------------------------:|--------------------------:|------------------------:|------------------------:|
| chin lowest    | 1.0%                       | 27.7%                     | 0.344                   | 0.143                   |
| pair (4,28)    | 0.2%                       | 22.7%                     | 0.339                   | 0.144                   |
| pair (8,24)    | 0.2%                       | 40.8%                     | 0.348                   | 0.130                   |
| nose midline   | 0.2%                       | 38.3%                     | 0.401                   | 0.137                   |

Two further pieces of evidence:

* The signed nose offset correlates with signed yaw at -0.886, and 99.58%
  of the nose midline failures have the offset sign opposite to the yaw
  sign. A wrong index would not produce a direction locked to head yaw; a
  3D nose swinging off the eye axis under yaw does.
* The chin failures are not near ties. Only 1.75% of them sit within 1% of
  the inter-ocular distance of point 16, and the lowest index spreads
  across 14 to 18 (44.98% at 15 or 17). The chin margin correlates with
  yaw at +0.374, which is the jaw contour sliding along the visible face
  boundary as the head turns. That is annotation behaviour, not a wrong
  index.

## Table 3: chin tolerance, absolute versus relative (frontal faces)

The chin check originally used an absolute 2.0 px tolerance. That is the
wrong shape, since a fixed pixel tolerance is stricter for large faces than
for small ones.

| tolerance            | pass rate |
|----------------------|----------:|
| 0.5% of IOD          | 64.63%    |
| 1% of IOD            | 74.33%    |
| 2% of IOD            | 83.99%    |
| 3% of IOD            | 89.56%    |
| 5% of IOD            | 94.16%    |
| 2.0 px absolute (old)| 89.47%    |

## Verdict

The 98 to 24 index mapping in `configs/landmarks_24.yaml` is correct. The
four failing checks measure symmetry and midline alignment, which real 3D
heads violate under yaw, and WFLW's pose flag only marks large poses, so
moderately turned faces stay inside the "frontal" population. On the
cleanest available subset (no flags, below median yaw) all four checks pass
at 98.13% to 100%, the failure rate climbs with yaw for every check, and
the nose offset direction is locked to the yaw direction. The residual
failures are pose driven measurement effects, and two of them were made
worse by defects in the checks themselves: the absolute chin tolerance and
a nose tolerance that ignores nose protrusion. After this verdict the chin
check moved to a tolerance of 3% of the inter-ocular distance
(`CHIN_TOL_IOD` in `dms_layer1/data/frame.py`), which reproduces the old
operating point (89.56% against 89.47%) while treating all face sizes the
same. No mapping index was changed.
