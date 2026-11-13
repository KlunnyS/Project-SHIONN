# Reviewed recovery data comparison — 2026-11-12

Purpose: determine whether the additional reviewed wall-to-route-to-exit human
corrections improve a small feedforward correction experiment while preserving
startup and navigation. This follows the completed24/12/8Hz integration screens,
which did not show a chamber-completion advantage. Decision rate stays24Hz.

`run_reviewed_recovery_pair.py` defaults to plan-only. The reviewed configuration
is `docs/reviews/recovery_correction_plan_20261112.json`; new raw-data review is
`docs/reviews/recovery_collection_20261112.json`, plus the earlier calibration
review. The supplemental exit-only clip is excluded. Both arms start from the
unchanged deployed `v5_recovery_proactive_v3/best.pt`, SHA256
`2ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483`.

| Setting | Control | Expanded corrections |
|---|---|---|
| Human TRAIN pool |7episodes,801frames|12episodes,1607frames|
| Complete development holdouts |2episodes,272frames|same|
| Successful updates |1000|1000|
| Batch |7original TRAIN +1opening +2human|same composition|
| Opening pool |first8 active-window frames of original TRAIN episodes|same draws|
| Optimizer |fresh AdamW, LR1e-5|same|
| Architecture/loss/action semantics |source-compatible feedforward|same|

Human draws are balanced across frames within each pool, not equal episodes.
Source preprocessing, mouse bins, augmentation, loss weights, smoothing and
clipping are retained. Source-model ancestry, frozen TRAIN/validation membership,
raw human files, source-attempt ownership, review/config/code and actual sample
schedules are checked. Autonomous/context observations never become expert
labels. Similar new viewpoints stay together in TRAIN; the old holdouts are
already used development diagnostics, not untouched generalization evidence.

The independent code gate passed;24 focused tests and370 full regression tests
passed. Real plan-only inspection confirmed801/1607/272 frame counts. The approved
execution uses service `shionn-reviewed-recovery-pair-v1`, external30min cap,
LimitNOFILE8192, MemoryHigh16G, MemoryMax20G and MemorySwapMax2G. Output:
`models/imitation/checkpoints/runs/reviewed_recovery_pair_20261112`.

Both fixed checkpoints remain pending until the complete pair passes final input
verification. Failures do not trigger retries, extension, deployment or promotion.
The report records source and both candidate evaluations on all39769 navigation
frames, the same TRAIN diagnostics and complete development corrections. Existing
navigation guards remain: binary exact drop at most0.005, mouse MAE ratio at most
1.05, first8-frame idle allowance0.01. Offline eligibility permits a user-operated
autonomous comparison; it does not demonstrate chamber completion improvement.

## Completed result

The job completed successfully in749.613s (12min30s in the experiment report;
service wall12min35s), with1000successful float32 updates per arm. Service host
memory peak14.3GiB and swap peak850.6MiB; no runtime or memory limit failure.
Both endpoints load and match report hashes, all recorded input/code hashes were
rechecked, source unchanged, and checkpoint training lists exclude held episodes.
Machine-readable review: `docs/reviews/recovery_pair_result_20261112.json`.

| Offline measure |Deployed source|Previous-data control|Expanded data|
|---|---:|---:|---:|
|Navigation binary exact|84.604%|84.913%|84.943%|
|Development episode-mean binary exact|69.473%|68.386%|68.013%|
|Development horizontal mouse MAE|20.368|20.451|20.640|
|New TRAIN episode-mean binary exact|80.348%|81.649%|83.251%|
|Navigation/startup screen|reference|pass|pass|

The additional data improves binary agreement on its own TRAIN examples but does
not improve held-out recovery agreement or held-out horizontal steering. No
learned gameplay improvement is claimed. Navigation eligibility permits one
limited nine-attempt user-operated screen: deployed/control/expanded checkpoints,
map-first test1/test11/test12, one repeat,60s,24Hz, persistent mouse, preflight and
3s post-preflight settle. Use fresh root
`model_attempts/sequences/reviewed_recovery_pair_20261112_screen`.
This is a screen, not a promotion decision or generalization evaluation. Any
promising result requires balanced-order confirmation rather than pooling old
source runs. No further training or budget extension is authorized by this result.

## Autonomous screen completed

Source: `model_attempts/sequences/reviewed_recovery_pair_20261112_screen/sequence_20261112_142443_155533`.
All9attempts valid with no capture-review flags. All3checkpoints timed out on
test1/test11 and reached test12: deployed6.1496s, control6.1878s, expanded6.1499s.
No completion gain; no promotion or repeat confirmation is justified by this
screen. Single attempts do not establish statistical equivalence.

Sampled visual review of deployed/candidate test1 and candidate test11 shows
wall approaches, candidate stairs/exit visibility followed by leaving the route,
and later stationary wall-facing views. On test1 the candidate diagnostic freeze
starts27.73s;841/1440ticks have all-zero actions. Source has937/1430all-zero ticks;
control105/1433 but still no exit. On test11 candidate has147/1440all-zero ticks,
so inactivity alone does not explain the whole navigation failure.

Offline replay of three late candidate-test1 views (four decoded observations per
sample) makes all3checkpoints emit all-zero actions. This supports a learned
policy response at those views rather than a runtime input failure; it does not
supply a correct expert action or prove why the policy learned that response.
Actual candidate logs at a late frozen view put horizontal zero-bin probability
around0.88, so a simple threshold adjustment is not justified as the general fix.

The806new active labels include63horizontal deltas beyond the largest source
representative magnitude134 and44vertical deltas beyond23. Reconstructing their
assigned bin representatives gives mean absolute error12.43/1.46(dx/dy). This
quantization mismatch limits exact imitation of fast turns; it does not explain
freezing by itself or authorize a checkpoint/action-space migration. Preserve
source bins for the completed experiment. Further work should first distinguish
insufficient fitting of corrective turns from viewpoint/distribution coverage,
using offline labelled correction comparisons; do not repeat collection/training
or change movement thresholds solely on these screen outcomes.

Detailed artifacts: `docs/reviews/recovery_pair_gameplay_20261112.json` and
`docs/reviews/recovery_mouse_quantization_20261112.json`. Existing deployed
checkpoint remains selected; trained endpoints retained as experimental artifacts.
