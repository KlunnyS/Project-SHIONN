# Recovery turn fitting diagnosis — 2026-11-12

Read-only CPU inference on the five new TRAIN corrections, using the original
four-frame histories and human action windows. Reproduce with:

```
.venv/bin/python analyze_reviewed_recovery_turns.py --output NEW_REPORT.json
```

Actual report: `docs/reviews/recovery_turn_fit_20261112.json`. This is TRAIN fit,
not validation or gameplay improvement. CPU float32 inference may differ slightly
from previous CUDA/TF32 aggregates; do not interpret one-frame differences as a
training effect without controlling precision.

|Horizontal measure|Deployed|Old-data control|Expanded candidate|
|---|---:|---:|---:|
|Zero prediction on first8 active frames (40nonzero targets)|14|13|12|
|Correct turn sign on those40frames|65%|67.5%|70%|
|Zero prediction on first24 active frames (112nonzero targets)|20|18|17|
|Zero prediction on complete clips (503nonzero targets)|144|145|147|

Keyboard agreement on the first40 frames is92.5% for the source despite its14
missed camera turns. A keyboard-only score therefore conceals this failure.
Full-clip steering zero predictions do not improve with the additional data.
The policy has not reliably fitted the demonstrated corrective camera behavior,
so failure cannot be attributed solely to transfer to unseen rollout views.

Deterministic draw reconstruction matches the experiment schedule: the first8
active frames of these5episodes contribute50/10000training targets (0.5%). The
new806frames collectively contribute1005/10000targets. Exposure report:
`docs/reviews/recovery_turn_exposure_20261112.json`. This supports testing whether
stronger corrective-turn exposure can fit the labels; it does not establish that
more exposure will improve gameplay or preserve existing behavior. Prior broader
onset experiments did not establish gameplay gains, so do not simply repeat them
or extend this completed training run.

The next discriminating offline experiment should be a small isolated fitting
probe on these labelled turn windows, without deployable checkpoint publication,
to determine whether the current representation/action heads can learn the
corrections under deliberately concentrated supervision. It must be explicitly
bounded and report only training fit, preserve source/bins/checkpoints and all
heldout roles, and must not be treated as a gameplay candidate. Only after that
result should a matched correction-weighting experiment be considered. No new
human recording is currently needed.
