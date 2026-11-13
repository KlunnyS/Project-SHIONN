# Recovery target and timing audit — 2026-11-12

Reproduce: `analyze_recovery_target_timing.py --output NEW_FILE.json`.
Actual artifact: `docs/reviews/recovery_target_timing_20261112.json`.
The report compares the same five TRAIN episodes/806targets across source,
uniform control and turn-rehearsal predictions from the completed report.
No new inference, gradients, gameplay or label edits. Synthetic arithmetic checks
passed for missed turn/sign/class reconstruction counts.

All five raw episodes have ordered sample times, nondecreasing receive times
and valid snapshot brackets. Sample spans are consistent with nominal24Hz after
accounting for the first-to-last versus full-frame-count duration. Median frame
ages range19–29ms; worst120ms. This does not establish hardware event/exposure
alignment or justify shifting labels. Existing physical focus-check logs and
virtual preflight yaw responses show no clear gross multi-fold input gain error;
physical aggregate estimates vary and are asynchronous, so they cannot calibrate
or justify changing mouse scale.

|Stage|Frames|Expert mean abs dx|Turn model mean abs dx|Turn dx MAE|Assigned target-bin decode MAE|
|---|---:|---:|---:|---:|---:|
|First8 active per episode|40|172.725|122.1|91.325|72.175|
|Next16|80|88.675|92.713|46.463|23.863|
|Remainder|686|23.867|18.353|22.166|7.614|

The largest horizontal output representative is134. First8TRAIN class agreement
improves from55% to75% under turn rehearsal; the remaining raw error is partly
representation error even when assigned classes are correct. Assigned-bin
reconstruction MAE is not the minimum possible MAE over all representatives and
is not a calibrated game-angle error. Saturation does not by itself explain
frozen wall states or prove that changing camera heads will improve navigation.

Later trajectory steering also remains weak:127/remaining nonzero dx targets
are predicted zero by the turn model (versus124source). Fitting the first turn
therefore has not learned the entire recovery-to-exit controller. No more
camera-cap sweeps or training extensions follow from this audit.

The existing Gaussian residual-v5 policy and binned policy share a512-feature
encoder and binary head interfaces (`models/imitation/network.py`). Continuous
camera outputs already exist; a new architecture must not be assumed necessary
without comparison. Replacing heads would change initialization/targets and
requires a separate controlled experiment rather than silently migrating source
checkpoints.

Before another head/training experiment, a model-independent expert-action
replay is the useful discriminating check: compare a successful demonstration's
raw actions with its source-bin encoded/decoded actions under the actual input
pipeline. Existing `wrapper.play_csv` interpolates10mouse steps per tick while
policy `apply_action` uses one, so it is not an equivalent oracle replay path.
Any new replay must use the policy submission path, preserve all safeguards,
and distinguish initial-pose/input/timing faults from quantization effects.

Original test1 episode20260915_185555_520895 was inspected: same staircase-room
layout is visible, raw221frames exist, first active target35excludes initial
loading/fade. Its initial view faces the closed entry door, unlike current
scored starts facing the room. Metadata stores only the map, without exact pose
or map hash. Do not replay it from today's spawn and claim that differences
measure model or bin quality. A reliable oracle test needs a fresh successful
spawn-to-exit demonstration with recorded initial pose and verified matching
reset. Preparing that workflow is the next engineering step; gameplay remains
user-operated and no new human correction batch is requested by this audit.
