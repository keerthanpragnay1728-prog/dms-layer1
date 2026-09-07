# Milestone 5 results: test set evaluation

Date: 2026-08-22. WFLW test split, 2,500 faces, ground truth box crops (the
standard protocol, so these numbers measure the landmark model alone, not
the face detector). All three trained checkpoints evaluated with
`scripts/evaluate.py`; per run outputs (m5_results.yaml, per face NME
array, worst-12 renders, config snapshots) are saved with the committed
notebook version.

## Main table

|                  | l2 w32  | wing w32 | wing w48 |
|------------------|--------:|---------:|---------:|
| overall NME      | 9.312%  | 7.346%   | 7.731%   |
| median NME       | 6.876%  | 5.026%   | 5.126%   |
| failure @ 10%    | 28.16%  | 18.88%   | 20.48%   |
| eyelids          | 8.668%  | 6.360%   | 6.859%   |
| pupils           | 8.848%  | 6.607%   | 7.057%   |
| mouth            | 9.415%  | 7.679%   | 7.910%   |
| axis             | 10.923% | 9.054%   | 9.321%   |
| contour          | 10.567% | 9.488%   | 9.709%   |
| no_flags NME     | 5.804%  | 4.089%   | 4.271%   |
| size             | 2.37 MB | 2.37 MB  | 4.43 MB  |
| CPU ms/frame     | 2.84    | 2.90     | 4.06     |

Wing width 32 wins every row. Baseline locked: Wing loss, width 32,
2.37 MB, 7.346% overall test NME, 4.089% on no-flag faces. The width 48
probe is closed, confirmed worse on the test set as well.

## Wing width 32 per WFLW subset

| subset       | NME     | failure @ 10% |
|--------------|--------:|--------------:|
| largepose    | 17.123% | 73.93%        |
| occlusion    | 9.182%  | 26.09%        |
| blur         | 8.146%  | 21.35%        |
| makeup       | 7.331%  | 17.96%        |
| expression   | 7.668%  | 18.15%        |
| illumination | 6.776%  | 14.76%        |
| no_flags     | 4.089%  | 2.81%         |

## Reading of the results

1. Eyelids and pupils are the two best groups on every run, and contour is
   the worst. That ordering suits this project, since gaze depends on pupil
   precision and contour matters least. To state it carefully: the ordering
   is also the intrinsic difficulty ordering of the task (eye points are
   small, well defined structures; jawline points slide along the visible
   boundary, as the milestone 1 diagnostics showed), so the claim is that
   the error budget lands where the pipeline can afford it, not that the
   model specially prioritised eyes. One forward pointer: per point NME is
   a proxy for what Layer 2 consumes. The eye aspect ratio depends on small
   vertical lid distances, and the relative pupil in eye position drives
   gaze; correlated errors partly cancel in those relative quantities,
   which is exactly what the milestone 7 stability harness measures
   directly.
2. Mean 7.346% against median 5.026% means a hard tail, not a broadly weak
   model. The subset table shows where the tail lives: largepose fails on
   73.93% of faces while no_flags fails on 2.81%. The worst-12 renders are
   near total profiles, deep shadow, heavy motion blur, and one face that
   is mostly the back of a head; several are inputs where a person would
   struggle, and some WFLW test annotations on such faces are themselves
   estimates.
3. The 4.089% no-flag figure is the closer estimate for the in-cabin
   condition, since a driver is roughly frontal, lit and in focus, and the
   report gives both numbers with that reasoning. Two qualifiers belong
   next to it: no-flag WFLW is a proxy (web photographs, not cabin footage;
   night driving with IR illumination and eyeglass reflections are not
   represented), and occlusion is the second most relevant subset for the
   cabin (sunglasses, hands on face) at 9.182% NME and 26.09% failure.
