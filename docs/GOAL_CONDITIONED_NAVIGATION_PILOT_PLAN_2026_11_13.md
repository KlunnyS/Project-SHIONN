# Goal-Conditioned Navigation Pilot and Reliability Follow-Up

> Dated snapshot. Current status: [navigation guide](NAVIGATION.md). Current collection commands: [paired-goal protocol](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md). Later implementation may supersede statements below.

**Date:** 2026-11-13

**Status:** Updated 2026-11-14. Recorder guards, a goal interface, and an annotation validator are implemented and CPU-tested. No gameplay, recording, or training was run.

## Current decision

The next learning capability should make the same visual state produce useful
actions for a supplied destination. Existing records are promising for
annotating the chamber exit, but they do not contain frame-aligned goal or
waypoint labels. The existing policy also has no goal input. Do not start
goal-conditioned training until annotations and counterfactual examples exist.

One small recorder correctness fix was made: overlapping physical keys that
map to the same logical action are now tracked independently under a lock.
This prevents releasing one crouch key from clearing the action while another
crouch key is still held.

## Recording and annotation inventory

### What is available

- The behavior-cloning dataset returns four-frame RGB stacks and ten action
  values, keyed by episode and map. It has no goal, destination, waypoint, or
  per-frame action-owner field (`models/imitation/dataset.py`).
- The reviewed Nov. 5 and Nov. 9 correction set has nine human-action episodes:
  seven train episodes (999 raw rows; 801 after leading-idle trimming) and two
  development holdouts (340 raw rows; 272 after trimming). Membership and review are recorded in
  `data/annotations/route_expansion_split_20261109.json` and
  `data/annotations/route_expansion_validated_plan_v1.json`. The review labels
  one Nov. 9 clip as substantive `route_recovery`; two are short `exit_finalization`
  clips. This is useful recovery/action evidence, not goal-conditioning data.
- Five Nov. 12 recovery episodes are stored under
  `model_attempts/targeted_navigation_20261112/recovery_episodes_pending/goal_reached/`.
  They contain 801 action rows in total, are on test1/test9, and have a
  `source_attempt` link. Their metadata has map/source/timing, not a destination
  or takeover frame. The separate observation-context recordings explicitly
  say `expert_targets: false`, `labels: none`, `training_approved: false`.
- I sampled the start, middle, and final video frame of each of those five
  episodes. The final frame shows the exit; this suggests an exit waypoint can
  be annotated. Sampling does not establish when the exit first became visible
  or which earlier frames provide usable goal supervision.

### Data boundary

Keep all existing train/development roles unchanged. Keep test11 validation-only
and test12 regression-only; neither can enter training. The Nov. 12 session is a
single source session and remains quarantined from training until each clip is
reviewed and assigned a role. Its human recovery clips start after takeover;
the autonomous attempt and unlabelled context video are separate artifacts.
Do not concatenate them or fabricate action labels across their handoff gap.

Before using any candidate, review the full clip and verify frame/action
alignment, target visibility, source attempt linkage, outcome, and action owner.
Keep a frame-indexed annotation ledger outside the raw recordings, with hashes
for the video, action CSV, and metadata. Never infer missing target locations
by interpolating across occlusion, dropped frames, or the takeover gap.

## Concrete first goal representation

For the first visible-target pilot, use a manually specified egocentric waypoint
projected into the current image. The annotation ledger stores these fields per
frame:

```text
goal_t = [x_norm, y_norm, visible, target_kind]
```

`x_norm` and `y_norm` are float32 target coordinates normalized to `[-1,1]`
with top-left origin and positive y downward (`x=2u/(W-1)-1`,
`y=2v/(H-1)-1`). An invisible target has null ledger coordinates and is encoded
as zeros. `target_kind` is an integer ID (`0=exit`, `1=waypoint`) mapped by a
learned 8-value embedding. Thus the model receives 11 encoded values
(`x`, `y`, visibility, 8 embedding values); the caller supplies the stable
`goal_id` separately with every descriptor. As the camera moves, the pixel
projection is updated while the requested goal ID remains unchanged. This is
an egocentric visible-point interface, not a world coordinate or a location
representation for a fully occluded target.

For the initial recurrent candidate, pass this vector alongside the existing
512-value GRU state to a small residual goal adapter that adds action logits to
the frozen baseline heads. Zero-initialize the adapter's final layer so the
candidate initially reproduces the source GRU exactly. Freeze the visual
encoder, GRU, and original action heads during the first pilot; train only the
adapter. Use a separate architecture ID/configuration and preserve the source
checkpoint. The goal ID and descriptor are supplied every step by the caller,
so the objective persists as an interface value; the pilot only claims
visible-target reaching. A future target-image embedding can let the GRU reason
about an occluded target, and a learned goal selector can choose the target,
after the point-conditioned pilot is understood.

This representation supplies a destination; it does not recognize exits or
choose goals. Pixel coordinates in training are human annotations. At inference,
`PolicyInference.set_goal()` requires its caller to provide the current visual
projection; the current visual encoder does not locate that target. Thus this
pilot isolates whether the low-level policy uses a supplied visible waypoint,
not autonomous goal recognition or full-chamber planning. Goal selection and
projection must later be learned or supplied by a perception/high-level module.

## Pilot data and controls

1. First annotate reviewed frames from the seven existing training episodes
   only, for feasibility. Keep the two existing development episodes whole and
   out of gradients. Their current labels point to one exit route; they cannot
   alone train or validate counterfactual goal choice.
2. Before training, collect an explicitly approved, separate goal-labelled
   pilot set: four matched start states with two requested goals each for train
   (8 full human episodes), and four other matched start states with the same
   two goals for held-out counterfactual analysis (8 episodes). Keep each pair
   and source session together. Choose reachable destinations whose directions
   differ from the same observation. Record stable goal identity, per-frame
   point/visibility, requested goal, and action owner. No autonomous actions
   are expert targets.
3. Freeze the annotation manifest, hashes, start-state/source-session
   membership, source checkpoint/hash, decoder, and update budget before
   training. Proposed bounded pilot config, matched across both candidate and
   goal-blind control: 500 Adam updates, batch size 2, sequence length 16,
   burn-in 16, learning rate `1e-3`, label smoothing `0.01`, jump-positive
   weight `3.0`, and seed `20261114`. Freeze the recurrent base and train only
   the goal adapter and target-kind embedding. The goal-blind control uses the
   same adapter with a constant invisible neutral goal; use identical sampled
   trajectories/order and action losses in both arms. This is a proposed
   experiment contract, not an executable training command.
4. Reserve six additional start states for live evaluation; do not expose their
   views or trajectories to training or development. Use the same start states
   and two requested goals for candidate, goal-blind control, and deployed
   feedforward regression reference. A deterministic reset/start adapter is
   not available yet, so these scores remain blocked until repeatable starts
   can be verified.
5. Use the frozen 512-unit source
   `models/imitation/checkpoints/runs/temporal_readout_comparison_v1/gru/endpoint.pt`
   (SHA-256
   `338906faa5328ccbe6ace33b946b6adce1775c2cb5487ed2cecb237fd2a1f41b`)
   for both the goal-blind recurrent baseline and the warm-start for the
   goal-conditioned candidate. It uses no observation feature gate and was
   trained with sequence length 16 and burn-in 16. Keep the deployed feedforward
   checkpoint as a separate regression reference, not the paired training
   baseline. This controls the architecture change in the candidate comparison.
6. Retain test11 as validation-only and test12 as regression-only; neither may
   enter gradients. Use whole start-state groups for all splits. Require a
   reproducible reset/start registry before interpreting a live comparison;
   the current repo does not provide a verified deterministic scenario-reset
   adapter.
7. Do not promote from imitation loss or action accuracy. Live destination
   completion remains a separate, user-run evaluation requiring approval.

### Counterfactual and closed-loop measures

- **Same-view goal-use check:** hold the observation/history and hidden state
  fixed, supply left-versus-right target vectors, and measure whether signed
  camera steering and the first navigation choice change toward the requested
  target. On at least 20 held-out paired view/history examples across at least
  four start groups, require at least 16/20 to steer toward each requested
  target and at least 16/20 pairs to reverse steering correctly when the goal
  is swapped. Report raw actions and probabilities. This is a necessary
  diagnostic, not proof of reaching.
- **Paired destination reaching:** from matched starts, run each requested
  target with the same 60-second cap. For an initial bounded screen, use six
  held-out start states, two goals per start, and compare the goal candidate,
  the matched goal-blind adapter, and the deployed policy (36 attempts; 36
  minutes maximum policy time plus resets/preflight). Require the candidate to
  reach at least 9/12 targets, at least 4/6 per goal, and at least 3 more
  targets than the goal-blind adapter; neither per-goal rate may be over 1/6
  below the deployed policy. Record wrong-target rate, time-to-target,
  timeouts, wall stalls, focus/capture failures, and route progress. These
  thresholds define an exploratory pilot gate only; this small sample cannot
  establish broad generalization or justify checkpoint promotion. Invalid
  safety captures stop the comparison and are not scored as policy failures.
- **Reject/no-go:** reject the candidate for this pilot if it misses any
  counterfactual or paired-reaching threshold above, if either goal regresses
  more than one success out of six against the deployed reference, if captures
  are invalid, or if frozen membership/provenance checks fail. Passing means
  only that a larger held-out study is justified; it does not authorize
  promotion.
- **Persistence/recovery:** after temporary occlusion or a blocked route,
  measure whether the original goal ID remains active, the agent resumes
  progress toward it, and it avoids silently switching targets.
- Split by entire start/route group and source session, never individual frames.
  Save every candidate output separately and keep failed captures invalid.

### Progress-aware recovery design (deferred)

Current recordings contain ordered images, action snapshots, frame IDs and
receive times, but no trusted player pose, target distance, obstruction state,
or frame-aligned recovery labels. They support time-based action review and
visual inspection; they cannot establish physical progress toward a goal.
After the goal pilot is label-valid, a small auxiliary head can predict
short-window progress toward the supplied goal, commanded movement without
observable displacement, recovery success, and renewed progress toward the
same goal after a recovery maneuver. Prefer target-relative progress from
trusted offline annotations or independently verified game telemetry, while
keeping that privileged signal out of inference. Train/evaluate by whole
episode and start group, report calibration and false-alarm rate, and first
compare this predictor offline before connecting it to action selection. Do not
train a progress head from action magnitude or image change alone: turning the
camera can change pixels without moving toward the destination.

### Annotation-ledger contract and validator

The validator in `models/imitation/goals.py` requires schema
`shionn_goal_annotations_v1`, coordinate system
`normalized_frame_xy_origin_top_left_y_down`, one row for every frame index,
stable goal IDs unless an explicit transition has a reason, action ownership per
row, source hashes for each annotated episode, and disjoint source-session/start
groups across train and held-out roles. It rejects test11/test12 in train,
quarantined train data, non-human expert action rows, incomplete frame labels,
noncanonical hidden-target coordinates, and unsupported coordinate systems.
`annotation_only` is available for a feasibility ledger; it cannot satisfy the
counterfactual pair gate or grant training eligibility. The validator only
validates the ledger; it does not edit recordings or membership.

After creating a ledger, validate it with:

```bash
.venv/bin/python -m models.imitation.goals path/to/goal_annotations.json
# Add this flag for the paired two-goal training/evaluation manifest:
.venv/bin/python -m models.imitation.goals path/to/goal_annotations.json --require-counterfactual-pairs
```

Reviewed recordings do not currently satisfy this schema because they contain
no per-frame target projection or action-owner alignment. The nine reviewed
episodes remain eligible only for their documented route-correction role; the
five Nov. 12 episodes remain quarantined. Neither group may be silently
reclassified or treated as a clean goal holdout.

## User-run mouse decoder comparison

This is a diagnostic of the same frozen deployed checkpoint, not a training run
or promotion test. It uses test11 and test12 as validation/regression screens,
three 60-second attempts per map and decoder (12 minutes maximum policy time,
plus map loads and capture checks). Expected decoding remains experimental.

Before running, start Portal 2 yourself on the desktop and use a terminal in the
active Wayland session. Confirm the environment and capture outputs:

```bash
printf 'XDG_RUNTIME_DIR=%s\nWAYLAND_DISPLAY=%s\nDISPLAY=%s\n' "$XDG_RUNTIME_DIR" "$WAYLAND_DISPLAY" "$DISPLAY"
wf-recorder -L
```

Use the active session's real environment values; do not copy values from an
SSH shell if it is not attached to the graphical session. If the capture output
is unavailable or mouse preflight fails, stop there and fix the session setup;
do not disable preflight or the focus/confinement safeguards.

Run these two commands from the repository root. They differ only by decoder
and output directory:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint deployed=models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test11 \
  --map dataset_test12 \
  --repeats 3 \
  --max-seconds 60 \
  --fps 24 \
  --model-fps 24 \
  --countdown 3 \
  --post-preflight-settle 3 \
  --output DP-1 \
  --persistent-mouse \
  --keep-focused \
  --hyprland-confine-pointer \
  --no-launch \
  --verbose \
  --mouse-decode argmax \
  --recording-root model_attempts/sequences/mouse_decoder_live_20261113/argmax
```

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint deployed=models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test11 \
  --map dataset_test12 \
  --repeats 3 \
  --max-seconds 60 \
  --fps 24 \
  --model-fps 24 \
  --countdown 3 \
  --post-preflight-settle 3 \
  --output DP-1 \
  --persistent-mouse \
  --keep-focused \
  --hyprland-confine-pointer \
  --no-launch \
  --verbose \
  --mouse-decode expected \
  --recording-root model_attempts/sequences/mouse_decoder_live_20261113/expected
```

Checkpoint SHA-256 at preparation time:
`2ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483`.
Before the first arm, save code and checkpoint identity:

```bash
mkdir -p model_attempts/sequences/mouse_decoder_live_20261113
{
  git rev-parse HEAD
  sha256sum \
    models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
    models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.json
} > model_attempts/sequences/mouse_decoder_live_20261113/provenance.txt
```

Keep the sequence JSONL, attempt logs, videos, and provenance file. Compare chamber outcomes,
timeouts, startup freezes, wall stalls, turn direction, recovery, and visible
overshoot. Action MAE alone cannot decide the result. Because the two arms run
sequentially, treat this first screen as diagnostic; repeat with reversed arm
order before making a decoder decision.

## Remaining reliability work

- **Focus:** current focus guard fails closed on lost/unknown focus and cursor
  escape. It cannot prove Gamescope relative-mouse capture; the bidirectional
  camera preflight is still required for every attempt.
- **Mouse delivery:** dataset targets are physical evdev deltas, while live
  control emits relative pulses. No matched yaw-per-input calibration exists.
  Measure under the actual Proton/Wayland setup before changing scale or pulse
  behavior.
- **Recorder keyboard state:** overlapping-key release bug is fixed and covered
  by CPU tests. Recorder key state and key-tap latches now share a lock.
- **Recorder device/camera faults:** read-loop failures, unexpected camera EOF,
  partial frames, missing/stale samples, frame-ID gaps, and writer/sync failures
  now invalidate the capture. Partial artifacts are retained under `invalid/`
  with a reason and are excluded by preprocessing. Hardware disconnect and
  Wayland timing behavior still need a supervised live check before a new batch.
- **Frame/action timing:** recorder timestamps bracket snapshots and frames are
  timestamped on full receipt, not display exposure. Keep this limitation in
  mind for fine-grained waypoint/action alignment.

## Smallest next action

The smallest non-game action for the user is to create one `annotation_only`
ledger for a reviewed training episode and run the validator, to estimate the
frame-labeling effort without changing its existing dataset membership. To
demonstrate goal use after that feasibility check, the remaining requirements
are 8 training human trajectories across 4 matched starts/two goals, 8 held-out
counterfactual trajectories across 4 other matched starts, and 6 distinct live
evaluation starts. No goal-training command is ready because these labels and
a reproducible start registry do not yet exist. The separate argmax-versus-
expected user-run diagnostic remains ready under the commands above.

### Implementation status (2026-11-14)

- Recorder faults now fail closed, preserve partials in a non-training `invalid`
  directory, add `capture_valid` and `capture_failure_reason` metadata, and are
  filtered from future preprocessing. Historical metadata without these fields
  is left readable and untouched.
- `GoalCondition` validates stable identity, target kind, visibility and
  normalized current-frame coordinates. A separate goal-GRU architecture uses
  an 8-value target-kind embedding and a zero-initialized residual logit
  adapter; a caller-owned goal persists across observations and clears on
  `PolicyInference.reset()`.
- The baseline `RecurrentBinnedPolicy` default outputs and checkpoint keys are
  unchanged. The goal policy is not connected to the general trainer yet; the
  annotation and paired data gate must pass before adding a bounded goal-only
  training command.
- The goal adapter for the selected 512-unit GRU has 73,022 trainable
  parameters (about 0.28 MiB of FP32 weights, or about 0.84 MiB including two
  Adam state tensors and gradients). This excludes activations and the frozen
  baseline; no GPU runtime or peak-VRAM measurement was made. Training runtime
  remains unknown until a controlled profile is authorized.
- CPU fault-injection and goal-interface tests pass (see final report for the
  full regression results). No action-matching or gameplay benefit is claimed.
