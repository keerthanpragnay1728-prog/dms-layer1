# Milestone 4 notes: training runs

Hardware: Kaggle T4 with the crop cache attached. Width 32, input 112,
AdamW at 1e-3, cosine schedule over 120 epochs, seed 42. Each run saves its
config snapshot next to its checkpoints. Model: 2.37 MB, 592,480
parameters. Train/val split 6,750 / 750 (seeded; WFLW has no subject IDs,
so a random face split is the only option). Median epoch time 3.3 s for
both runs, roughly 7 minutes per full run, an order of magnitude under the
30 s dataloader budget, so the RAM cache did its job.

## L2 run (first baseline)

Best val NME 7.285% at epoch 99. The final epoch (119) sat at 7.415%,
twenty epochs past the best with no improvement, so the run converged under
its schedule rather than being cut off. The validation curve starts at
22.6% and descends steadily, flattening around epoch 95. Train loss 10.51
against val loss 9.85 at the end. Val below train is expected here, since
train batches carry augmentation and validation does not; there is no
overfitting.

## Wing run (the baseline)

Best val NME 5.386% at epoch 113, a 26% relative reduction over L2 from a
one line config change (`train.loss: wing`, w=10, epsilon=2 in input
pixels). Converged cleanly, plateaued from about epoch 100, train 4.95
against val 4.93, no overfit. Same epoch time as L2. Wing is the baseline
going forward. Both checkpoints are kept, and both go through the milestone
5 test set evaluation so the comparison is made on the per group numbers,
not on val alone.

Note on comparing the loss values themselves: L2 and Wing are different
functions, so their raw loss magnitudes are not comparable. The comparison
lives in the NME column, which is loss independent.

## Ceiling probes, results

All three runs, same schedule, same seed, same seeded validation split of
750 faces (identical across runs, so the comparison is on the same faces):

| run           | params    | size    | best val NME | epoch | s/epoch |
|---------------|----------:|--------:|-------------:|------:|--------:|
| l2, width 32  | 592,480   | 2.37 MB | 7.285%       | 99    | 3.3     |
| wing, width 32| 592,480   | 2.37 MB | 5.386%       | 113   | 3.3     |
| wing, width 48| 1,106,424 | 4.43 MB | 5.631%       | 111   | 3.8     |

1. Wing loss delivered: a 26% relative NME reduction over L2 at identical
   size and speed.
2. Width 48 did not: nearly double the parameters for a slightly worse
   validation NME, converged (plateau from about epoch 105). On the size of
   the effect: the standard error of a mean NME over 750 validation faces
   is roughly 0.1 points, and run to run seed variance for models this size
   is of the same order, so the 0.245 point deficit is at the edge of
   noise. The robust claim is not "width 48 is worse" but "the extra
   capacity buys nothing". The overfitting check was done on the metrics
   CSVs: width 32 finished with train 4.95 against val 4.93, width 48 with
   train 5.143 against val 5.080. Val sits slightly below train in both,
   same direction and similar magnitude, so there is no overfitting
   signature and the deficit is not attributed to any mechanism. The
   conclusion for the report: at 6,750 training faces and this
   augmentation, the 24 point task at 112 px input is not capacity limited
   at width 32, and the smaller model wins on every axis.
3. The 200 epoch schedule probe was skipped. Both width runs converged as
   the cosine reached its floor rather than against the patience limit, and
   width 48 gave no sign the schedule was binding.

Baseline locked: Wing loss, width 32, pending confirmation by the milestone
5 test set numbers.

## Val numbers versus citable numbers

Val NME is measured on a random split of the train cache and exists to
drive early stopping. The numbers that go in the report come from the
milestone 5 test set evaluation: overall, per group, per subset, and the
failure rate.

## Operational fixes from these runs

Kaggle mounts notebook outputs at `/kaggle/input/notebooks/<user>/<slug>/`,
not `/kaggle/input/<slug>/`, so the configured cache path missed on the
first run. `cache.dir` now defaults to `auto`: the loader searches
`/kaggle/input` for the cache files, uses a single unambiguous hit with a
printed note, and refuses to guess between several. An explicit path still
pins it. The Wing run confirmed the auto discovery works.

Training checkpoints carry optimiser and RNG state and weigh 6.9 MB.
`scripts/export_weights.py` strips one down to deployable weights (2.37 MB
for width 32), records the architecture and the val NME inside the file,
and verifies the export by reloading it and matching the forward pass
exactly. The exported file is what milestone 6 ships and what the report
quotes as the model size.
