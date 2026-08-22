# Milestone 4 notes: baseline training run — report material

**Date:** 2026-08-22 · **Hardware:** Kaggle T4, crop cache attached ·
**Config:** `loss: l2`, width 32, input 112, AdamW 1e-3, cosine over 120
epochs, seed 42 (snapshot saved next to the checkpoints).

## Baseline (L2) run

* best **val NME 7.285% at epoch 99**; final epoch 119 at 7.415% — twenty
  epochs past best with no improvement, i.e. converged under the schedule,
  not cut off.
* validation curve: 22.6% at epoch 0, steady descent, flattening ~epoch 95.
* train loss 10.51 vs val loss 9.85 at the end — val below train is expected
  (train batches carry augmentation, validation does not); no overfitting.
* median epoch time **3.3 s** (~7 minutes for the full run) — an order of
  magnitude under the 30 s dataloader budget, so the RAM-cache design did
  its job.
* model 2.37 MB, 592,480 params; train/val split 6,750 / 750 (seeded).

Caveat: val NME is measured on a random split of the TRAIN cache and exists
to drive early stopping. Citable numbers come from the milestone-5 test-set
evaluation (overall / per group / per subset / failure rate).

## Ceiling probes before locking the baseline

Each is one ~7-minute T4 run; all are judged on the milestone-5 test-set
report (especially the pupil and eyelid groups — the pipeline turns on eye
region precision, and overall NME hides it):

1. **Wing loss** (`train.loss: wing`) — the planned comparison axis, and
   the most promising lever: Wing amplifies exactly the small-error regime
   that dominates late training.
2. **Width 48** (`model.width: 48`) — the largest width inside the 5 MB
   budget (~1.11 M params, ~4.4 MB fp32; width 64 would be ~7 MB, over).
3. **Longer schedule** (`train.epochs: 200`, fresh run — the cosine's
   T_max changes, so this is not a resume) — likely small gains; the 120
   epoch run converged with the LR floor, not against the patience limit.

## Operational fix from this run

Kaggle mounts notebook outputs at `/kaggle/input/notebooks/<user>/<slug>/`,
not `/kaggle/input/<slug>/` — the configured cache path missed. `cache.dir`
now defaults to `auto`: the loader searches `/kaggle/input` for the cache
files, uses a single unambiguous hit with a printed note, and refuses to
guess between several. An explicit path still pins it.
