# Model benchmarking and improvement roadmap

Updated 2026-10-25. This document records the evidence for the current
candidate, the limits of that evidence, and the order in which the next model
changes should be made. Recheck local artifacts before treating paths or
checkpoint names as current.

## Executive recommendation

Keep `v5_recovery_proactive_v3` as the current candidate. Its frozen-validation
action metrics are slightly better than `v5_balanced_recovery_v1`, especially
for horizontal mouse error and left/right movement recall. That improvement is
small and does not establish better live navigation: both models solve only
the easy `dataset_test12` live chamber reliably, and the latest live runs are
confounded by possible cursor escape and Gamescope pointer-capture failures.

Do not start a larger architecture or collect a large unstructured batch yet.
First make the input/capture measurement trustworthy, then make controlled
data and sampler changes on the same V5 architecture. Move to a small temporal
V6 only if that clean experiment still shows failures caused by missing temporal
context rather than bad input capture or missing demonstrations.

## Benchmarks and evidence

### 2026-10-25 persistent mouse: local capture confirmation passed

The five-repeat balanced test11 confirmation at
`/tmp/shionn-persistent-mouse-confirmation/sequence_20261025_175754_528314/`
passed every automatic capture check: zero recorded cursor exits, focus losses,
unknown camera observations, or input submission failures. Bidirectional mouse
preflight, effective confinement-rule probes, and guard restoration passed on
all five attempts. Steering was exercised over 2,720 mouse-active policy ticks.
Together with the two-run pilot, seven consecutive persistent-mode attempts
had no recorded cursor/focus faults. This supports the current local X11 Steam
configuration with `--hyprland-confine-pointer --persistent-mouse`; it does not
isolate the cause or establish reliability on other configurations.

Keyboard assistance was permitted, so these are capture diagnostics rather
than autonomous model scores. No further capture-only reruns are needed unless
the configuration changes or another fault appears. Continue controlled data
and sampling work; clean, unassisted candidate comparisons remain outstanding.
See [benchmarking](BENCHMARKING.md) for configuration and evidence details.

### 2026-10-25 hover guard: focus retained, test11 pointer escapes remain

The six-attempt balanced test1/test11 diagnostic in
`/tmp/shionn-focus-guard/sequence_20261025_162021_137292/` completed with
zero observed focus losses and successful hover-setting restoration after every
attempt. Every test11 run recorded a cursor exit, starting at about 23.2, 46.5,
and 34.5 seconds; test1 inference recorded none, although one test1 run had no
mouse commands. All six timed out. This supports separating focus switching
from pointer confinement, without proving the root cause of either. This
historical test preceded the verified rule matching and persistent-mouse
confirmation above.

### 2026-10-25 direct runs: policy failures with no observed capture faults

Two subsequent direct test1 runs passed the automatic capture checks, with no
observed cursor exits, focus losses, unknown camera replies, or submission
failures. The new candidate's run in `/tmp/shionn-direct-stress/` progressively
commanded downward mouse motion, reached 89-degree pitch at 13.29 seconds, then
became idle at about 13.36 seconds; Escape ended the run at 186.4 seconds.
The balanced run in `/tmp/shionn-balanced-direct/` moved/turned until its final
nonidle prediction at 37.53 seconds, then requested no controls while facing a
wall. Its pitch ended near -0.53 degrees, with a range of -5.64 to 33.02 degrees;
Escape ended the run at 59.1 seconds. Final-frame inspection confirms the wall.

These are specific examples of downward-view drift and wall-state idling under
observed stable capture, useful for the Phase 1 data/sampling audit. Most of the
new model's long attempt was idle and did not exercise continuous mouse capture.
One direct attempt per checkpoint does not establish repeatability or ranking,
and does not resolve the earlier sequence's grab/focus faults.

### 2026-10-25 instrumented sequence: stopped for capture failure

The sequence at
`/tmp/shionn-v5-comparison/sequence_20261025_150806_092817/` planned 30
attempts but completed only eight gameplay attempts. Job 9 stopped the entire
sequence when preflight measured zero left yaw despite submitted mouse pulses
and observed game focus. Both checkpoints completed test12 once and timed out
on test1 twice and test11 once. These are incomplete, capture-confounded results.

Five of eight gameplay attempts have capture-review flags. There were three
focus-loss episodes to Firefox and 98 cursor-exit transitions across three
attempts (1, 4, and 93). The final successful preflight produced ±1.351593
degrees rather than the usual ±4.054779; the next failed at zero. Camera
telemetry parsed successfully throughout gameplay, and policy submission
failures were zero. Thus the new observability path exposed live input-path
instability, while the cause of confinement/grab failures remains unresolved.
Cursor exits alone do not establish loss of relative game input; short zero-yaw
intervals must also account for asynchronous engine and observation timing.

The generated benchmark summary and capture diagnostics are beside the manifest
in temporary storage. Preserve those artifacts before `/tmp` is cleared. Fix
and validate confinement/reacquisition with short runs before restarting the
full comparison. Do not promote a checkpoint from these partial results.

### Live sequence: useful failure evidence, not a clean score

Sequence: `model_attempts/sequences/sequence_20261020_185151_464090/`.
It ran three checkpoints on `dataset_test1`, `dataset_test11`, and
`dataset_test12`, three repeats each (27 attempts total):

| Checkpoint | Test 1 | Test 11 | Test 12 | Total |
|---|---:|---:|---:|---:|
| `v5_recovery_proactive_v3` | 0/3 | 0/3 | 3/3 | 3/9 |
| `v5_balanced_recovery_v1` | 0/3 | 0/3 | 3/3 | 3/9 |
| `v5_holdout_11_12` | 0/3 | 0/3 | 2/3 | 2/9 |

Treat these as a warning that the difficult chambers remain unsolved. Do not
rank the models from these rates: the runs showed cursor-edge observations and
possible capture loss. The runner sampled desktop state at about one-second
intervals. It can miss short cursor escapes, and an edge coordinate is not
proof that a pointer event reached the game. New-model mouse predictions were
nonzero on the difficult chambers, so a policy can be blamed for apparent
freeze or mouse failure even when the capture layer is at fault. Test 12 also
had no predicted mouse movement, so its success says little about steering.

### Frozen expert-action validation: comparable for the two V5 checkpoints

Reports use the same frozen validation split: 200 episodes on `dataset_test11`
and `dataset_test12`, totaling 39,769 usable frames after leading-idle
trimming. They compare predicted actions to demonstrations; they do not measure
whether the predicted action reaches the intended game state.

| Metric | `v5_recovery_proactive_v3` | `v5_balanced_recovery_v1` |
|---|---:|---:|
| Exact binary-action match | 0.8460 | 0.8440 |
| Mouse dx MAE | 6.2919 | 7.0865 |
| Mouse dy MAE | 2.3614 | 2.3478 |
| Validation loss | 2.6179 | 2.6496 |

By chamber:

| Checkpoint / chamber | Frames | Exact match | dx MAE | dy MAE | W F1 | A F1 | D F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| New / test11 | 23,952 | .8241 | 7.5134 | 2.4975 | .9784 | .3281 | .3932 |
| Balanced / test11 | 23,952 | .8226 | 8.5764 | 2.4501 | .9771 | .2169 | .4128 |
| New / test12 | 15,817 | .8793 | 4.4421 | 2.1553 | .9812 | .4105 | .5055 |
| Balanced / test12 | 15,817 | .8764 | 4.8304 | 2.1930 | .9805 | .3808 | .4864 |

The candidate improved dx and left movement recall on these samples. Its dy
error is marginally worse overall. The improvement is modest, and exact-action
match remains high partly because forward movement dominates. These scores are
not a proxy for completion rate.

Demonstration mouse activity is still under-predicted:

| Chamber | Expert dx/dy nonzero | New candidate dx/dy nonzero | Balanced dx/dy nonzero |
|---|---:|---:|---:|
| Test11 | 43.2% / 35.8% | 30.8% / 19.0% | 30.5% / 18.4% |
| Test12 | 28.3% / 27.3% | 17.2% / 12.8% | 17.2% / 12.3% |

This supports continued investigation of mouse action calibration and sampling.
It does not by itself prove that zero-bin bias is the root cause; missing
context, conflicting labels, action scaling, and capture can all contribute.

The new model's best checkpoint is epoch 3. Validation loss rose at epochs 4
and 5 (2.7958 and 2.9039) while training loss kept falling. Use `best.pt`, not
`last.pt`, for this run. The older `v5_holdout_11_12` report used a smaller,
different validation frame set (`validation_cuda_v5_holdout_11_12/summary.json`)
and should not be compared numerically to the full newer reports.

Report locations:

- `model_attempts/benchmarks/validation_cuda_v5_recovery_proactive_v3/summary.json`
- `model_attempts/benchmarks/validation_cuda_v5_recovery_proactive_v3/action_comparison.csv`
- `model_attempts/benchmarks/validation_cuda_v5_balanced_recovery_v1/summary.json`
- `model_attempts/benchmarks/validation_cuda_v5_holdout_11_12/summary.json` (older, non-comparable subset)
- `model_attempts/sequences/sequence_20261020_185151_464090/benchmark_summary.json`

These are local generated artifacts and may live on the external `local_storage`
drive. Preserve their split manifests and configuration sidecars with reports.

## What the model and data look like now

- Current candidate: `models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt`.
- Frozen split: `frozen-2026-10-18-v3`; 1,100 base training episodes, 20
  proactive-turn training clips, 31 recovery training clips, and 200 validation
  episodes on test11/test12. Evaluation maps are excluded from the cache/split.
- Training data has 286,087 base raw frames, 3,290 proactive-turn frames, and
  5,212 recovery frames (8,502 correction frames total).
- The 20 proactive clips are 10 left and 10 right across test1–5. They are all
  training data; they cannot be repurposed as a held-out test set after the
  model has already trained on them.
- The correction caches currently share `--train-extra-cache-dir` handling.
  The recorded config has `correction_sampling_fraction=0.0`; with the
  chamber-balanced sampler, corrections are expected to be only about 2.99%
  of training draws. Separate correction-group sampling should be implemented
  and evaluated before collecting many more clips.
- Base training action prevalence: W 74.2%, A 12.6%, S 0.7%, D 11.6%;
  mouse dx/dy are zero on 58.0%/65.4% of frames. Validation mouse dx/dy are
  zero on 69.5%/73.5%. Recovery clips: 41.7%/51.1% zero. Proactive turns:
  37.1%/46.7% zero.
- V5 is a 13.2M-parameter residual CNN with a four-frame input stack and
  binned mouse output. It has no persistent recurrent state. A temporal model
  is a hypothesis for later, not yet the demonstrated fix.

## Roadmap

Time windows are planning estimates from the current project state, not firm
deadlines. Start each phase only when its entry condition is met.

### Phase 0 — repair benchmark observability (next work session, 1–3 days)

The 2026-10-25 implementation provides 15 Hz desktop sampling, timestamped
submission results, asynchronous camera observations, and per-attempt mouse
preflight with a map reload. See [benchmarking](BENCHMARKING.md) for usage and
uncertainty handling. Five user-run confirmation attempts passed all automatic
capture checks under real inference load, meeting the observability exit
condition for the documented local configuration. Clean autonomous model
comparisons remain separate work in Phase 2.

1. Keep recent cursor-escape runs classified as capture-confounded. Do not use
   their completion rates to promote or reject a checkpoint.
2. Record compositor focus, workspace, cursor-in-window, policy action, action
   injection result, and camera response at higher frequency (target 10–20 Hz
   or event-driven) with monotonic timestamps. Include actual window identity
   and distinguish unknown telemetry from `false`.
3. Add a short, low-risk mouse capture check before a long benchmark. Confirm
   left/right game yaw response and that the game window retains focus; abort
   if a direction is ineffective. Do not inject camera motion during saved
   expert demonstrations.
4. Separate model idle/freeze diagnostics from compositor focus/capture faults.
   A `policy_freeze` label alone is not evidence of the cause.

**Exit condition:** logs can tell whether the game had focus, whether cursor
containment was observed, whether actions were sent, and whether camera motion
followed mouse input. Unknown values are explicitly recorded as unknown.

### Phase 1 — controlled V5 data and sampling ablation (1–2 weeks)

1. Preserve frozen v3. Create a new split revision for all newly collected
   clips; assign independent validation clips before any training sees them.
2. Record targeted corrections from model failures before contact: forward
   approaches, left/right turns, lateral movement, and recovery from wall
   contact. Vary approach angle and distance. Avoid bulk duplicates of the
   current 20-turn pilot.
3. Keep base, proactive-turn, and recovery groups separately identifiable in
   the sampler. Compare no correction oversampling with controlled fractions
   (for example 5% and 10%), holding architecture, random seed policy, split,
   and total update budget fixed.
4. Compare action metrics by chamber and behavior group, not only global exact
   match. Review mouse calibration and predicted-vs-expert action rates.

**Exit condition:** the best sampling choice improves held-out difficult-scene
metrics without degrading the easy chamber or overfitting known correction
clips. If it does not, inspect labels and camera-state coverage before adding
more volume.

### Phase 2 — clean live promotion benchmark (after Phase 0, about 1 week)

1. Compare `v5_recovery_proactive_v3` against `v5_balanced_recovery_v1` with
   identical chamber order, start state, input capture, and settings.
2. Run at least five attempts per model per chamber (test1, test11, test12) to
   reduce noise from the previous three-repeat sample. Record completion,
   time-to-goal, freeze/wall/turn-loop events, focus losses, cursor exits, and
   camera response. Randomize or interleave model order to limit session drift.
3. Keep the old v5 baseline only if there is a specific question; its validation
   report uses a different subset and live performance is already poor.

**Exit condition:** a repeatable improvement on the hard maps with no new
capture faults or material regression on test12. If all candidates fail in the
same way, treat it as a data/control diagnosis rather than a checkpoint race.

### Phase 3 — decide whether V6 is justified (2–6 weeks)

If clean-input V5 plus targeted data still fails because decisions require
motion history, create one small temporal variant: a compact GRU or temporal
convolution over existing CNN features. Keep action heads, frame resolution,
and training recipe fixed where possible so the temporal component is the main
change. Compare it to the V5 candidate on the same frozen split and live
protocol. Do not jump directly to a large transformer while correction data is
small and evaluation coverage is weak.

**Exit condition:** temporal V6 wins on held-out navigation-relevant measures,
not just training loss. If it does not, retain V5 and correct the demonstrator,
action labels, or game-input layer.

### Phase 4 — broaden generalization (1–3+ months, as recording time allows)

1. Build a truly unseen map/layout test set. Test11/12 have been validation
   chambers and evaluation1/2 have been inspected in earlier experiments; do
   not describe them as untouched. Keep new final-test episodes isolated.
2. Use DAgger-style collection: let the current policy reach unfamiliar states,
   then record human corrections from those exact states with capture verified.
3. Expand movement cases beyond static wall recovery: stairs approached at an
   angle, open-space exits, narrow turns, lateral alignment, and delayed
   recovery after a policy stall. Maintain metadata for state/context and
   recording quality.
4. Consider reinforcement learning only after resets, terminal signals, and
   reward are dependable and imitation reaches a stable baseline.

## Rules for future dataset growth

- Freeze a new manifest before training and preserve every earlier split.
- Split by full recording episode, and reserve independent maps/layouts when
  possible. Do not use a clip for validation after training on it.
- Keep evaluation data out of preprocessing/training caches where practical.
- Track group size, source map, left/right direction, start distance/angle,
  outcome, mouse activity, and quality approval.
- Add examples for measured gaps. Do not chase a larger episode count alone.
- Keep raw video/action pairs so preprocessing can be regenerated and audited.
