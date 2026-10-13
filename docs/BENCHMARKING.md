# Benchmarking a navigation checkpoint

Run these commands from the repository root after training a candidate. Replace the checkpoint path with the actual `best.pt`. The updated trainer saves its validation episode names inside the checkpoint, so the offline scorer uses the same held-out recordings that were excluded from training.

The [frozen split](DATA_SPLIT.md) defines test1–10 as training, test11/12 as validation, and evaluation1/2 as evaluation-only. The latter two have had prior live runs, which should be disclosed with results; neither is part of the training cache. Check the checkpoint's split ID and manifest hash when comparing candidates trained after the split was introduced.

## 1. Compare actions on held-out expert frames

```bash
.venv/bin/python benchmark_expert.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --cache-dir data/datasets/cached_frames
```

This uses the checkpoint's validation episodes by default. It creates `summary.json` with per-chamber binary-action precision, recall, F1, exact binary-action match, and mouse mean absolute error. `action_comparison.csv` contains each human action and the model's prediction and probability on the **same recorded frame**. The report also breaks out the final 24 frames of each episode; those approximate the goal approach only when the source episodes were successful. `--map NAME` can limit the report to one or more maps.

For checkpoints made before validation episode names were saved, use a new checkpoint from the updated trainer. `--subset all` is for an independent evaluation cache; using it on training data produces an optimistic score.

## 2. Measure autonomous success

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint candidate=models/imitation/checkpoints/runs/candidate/best.pt \
  --map dataset_test1 --map dataset_test5 --map dataset_test10 \
  --map evaluation1 --map evaluation2 \
  --repeats 10 --keep-focused
```

Add `--output NAME` if the default Wayland output is wrong. Each attempt writes a video and an action/diagnostic JSONL. The sequence writes `sequence.jsonl` and, when it finishes, `benchmark_summary.json` under a timestamped `model_attempts/sequences/sequence_*` directory. For an interrupted or older sequence, generate the summary afterward:

```bash
.venv/bin/python benchmark_sequence.py model_attempts/sequences/sequence_TIMESTAMP
```

The report gives successes, valid attempts, success rate, median successful run duration, outcome counts, invalid attempts, and focus losses by model and map. A runner error or manual cancellation is marked invalid instead of counting as a navigation failure. The summary also retains each attempt's video and log path. Compare multiple checkpoints by repeating `--checkpoint LABEL=PATH` in the sequence command.

Review the beginning and at least one later frame before accepting a sequence. With `--keep-focused`, Hyprland should report a nonzero fullscreen state and the runner requests the same visible portal-gun viewmodel used in the demonstrations. Reject runs containing a zoomed opening, desktop bars, window borders, an open Steam overlay, or other large visual mismatches. Input activation requires `xte`: the runner briefly unbinds `MOUSE1`, delivers an XWayland left click to capture input, and restores the prior binding so no portal is fired. An enabled but closed Steam overlay is harmless; hide any visible performance HUD before the authoritative repeated benchmark.

## 3. Record separate references for evaluation chambers

Freeze the candidate checkpoint first. Human reference runs on `evaluation1` and `evaluation2` belong outside the training recording and cache directories:

```bash
.venv/bin/python record_dataset.py \
  --map evaluation1 --map evaluation2 --episodes 10 \
  --episodes-root episodes_eval

.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes_eval/goal_reached \
  --cache-dir data/datasets/evaluation_cached_frames

.venv/bin/python benchmark_expert.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --cache-dir data/datasets/evaluation_cached_frames \
  --subset all
```

Do not pass `episodes_eval` to the training preprocessor or merge its cache into `data/datasets/cached_frames` while it is being used as a held-out evaluation set. The offline comparison asks which actions the model would take on expert images. Once the model controls the game, its images can diverge from the human recording; use autonomous success and attempt videos to judge those runs.

## 4. Diagnose wall-facing and mouse failures

For a binned-mouse checkpoint, annotate visually confirmed wall-facing ranges in
the expert comparison. `action_comparison.csv` provides episode names and frame
indices; the annotation uses inclusive ranges:

```csv
episode,start_frame,end_frame
episode_YYYYMMDD_HHMMSS_000000,42,58
```

Then run:

```bash
.venv/bin/python analyze_mouse_validation.py \
  model_attempts/benchmarks/candidate/action_comparison.csv \
  --wall-ranges model_attempts/benchmarks/candidate/wall_ranges.csv
```

The report compares mouse-bin entropy, peak separation, opposing-direction
peaks, and jump false positives inside and outside the annotated ranges. The
ranges must come from video inspection; the dataset has no automatic wall
label.

For a live failure, save video and diagnostics, note every wall-facing time
window, and analyze the corresponding ticks:

```bash
.venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/candidate/best.pt \
  --map dataset_test9 --max-seconds 30 --record-video --verbose

.venv/bin/python analyze_mouse_rollout.py \
  model_attempts/attempt_TIMESTAMP.jsonl \
  --wall-window 12:18 --wall-window 23:27
```

Opposing peaks may indicate conflicting demonstrations, while a flat
distribution may indicate low confidence. Neither result proves a cause by
itself; interpret it alongside the video, expert-frame metrics, visual motion,
and focus diagnostics.

The runner also writes explicit `diagnostic` records when one of these applied-
action patterns lasts at least one second:

- `policy_freeze`: no movement, interaction, firing, jumping, or mouse action;
- `wall_stuck`: movement is commanded while sparse-frame visual delta remains
  at or below `0.05`; or
- `turn_loop`: movement is combined with horizontal mouse output of at least
  `45` in the same direction.

Each record has a `start` or `end` phase, and the final summary counts detected
starts. These are triage signals, not proof of a cause; confirm them against
the matching video.

To isolate whether extreme mouse bins cause circular failures, compare the
unchanged baseline against the same sequence with a cap:

```bash
.venv/bin/python run_model_sequence.py \
  --checkpoint v5_best=models/imitation/checkpoints/runs/v5_holdout_11_12/best.pt \
  --map dataset_test1 --map dataset_test10 --map dataset_test11 \
  --map dataset_test12 --map evaluation2 \
  --repeats 3 --keep-focused --mouse-max-abs 28
```

The cap is opt-in and does not modify the checkpoint. Every tick logs the raw
model output as `raw_action`, the capped action sent to the controller as
`action`, and whether they differ as `mouse_limited`. Keep all other settings
and chamber order fixed when comparing it with the uncapped baseline.

On 2026-10-06, an experimental `74 × tanh(raw/74)` response was compared with
the uncapped balanced candidate over three repeats of `dataset_test1`,
`dataset_test11`, and `dataset_test12`. It reduced steering too aggressively in
closed-loop use: zero-mouse raw predictions rose from 84.0% to 98.3%, freeze
events rose from 7 to 25, `dataset_test11` fell from 1/3 to 0/3 successes, and
`dataset_test12` fell from 3/3 to 2/3. The experiment was rejected and its CLI
option was not retained. Use uncapped behavior for future recovery-data work.

## 5. Calibrate jump and forward decisions

Sweep jump thresholds only on held-out expert predictions:

```bash
.venv/bin/python benchmark_jump.py \
  model_attempts/benchmarks/candidate/action_comparison.csv
```

Choose a precision/recall trade-off before live evaluation, then pass that
threshold consistently to `run_imitation.py` or `run_model_sequence.py` with
`--jump-threshold`. Thresholds are checkpoint-specific; do not promote one from
a different model merely because it performed well there.

If a policy remains idle despite expert-frame evidence that forward movement is
correct, use `--move-w-threshold FLOAT` for a short calibration run. Change one
decision rule at a time and leave mouse decoding and other thresholds fixed.
Threshold overrides test calibration; they do not retrain the model or prove
that it can complete a chamber. Confirm any candidate with repeated autonomous
runs.

## 6. Record targeted recovery demonstrations

Use recovery recording only after calibration shows a genuinely unfamiliar
policy-induced state. Stop a model attempt while Portal 2 is still at the
failure state; the runner releases held controls during cleanup. Without
reloading the chamber, record a human recovery:

```bash
.venv/bin/python record_recovery.py \
  --map dataset_test9 \
  --source-attempt model_attempts/attempt_TIMESTAMP.jsonl \
  --max-seconds 30
```

For a supervised collection session, the handoff and chamber cycling can be
automated instead:

```bash
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_balanced_recovery_v1/best.pt \
  --map dataset_test1 --map dataset_test5 --map dataset_test9 \
  --target-corrections 30
```

The session loads a training chamber and runs the model. Press **P** at a useful
failure: the model releases its controls and the recovery recorder starts at
the *same position*, after recentering the confined pointer, reacquiring Portal 2
input, and a five-second focus delay. Sweep the physical mouse toward the other monitor and back during that
wait, then return the view near its starting angle. The runner checks the
Hyprland cursor position and Portal focus throughout the sweep, as well as
camera response; a detected escape or large view shift retries the same
chamber without writing a correction. If confinement still looks wrong once
recording starts, press **R** to reject that correction; its partial clip is
kept under `episodes/recovery/discarded/` for review, not training. Reach the goal to save a
successful correction and automatically start the model on the next chamber.
The five-second handoff also logs physical mouse counts, game angles, and
Hyprland cursor/focus to `handoff_diagnostics.jsonl` in the attempt folder.
If substantial mouse movement produces almost no camera turn, the runner
reacquires input once and then retries the chamber without a clip.
If automated focus verification blocks collection, run the session with
`--manual-focus`: after the first P handoff, check confinement and re-align the
view at your own pace, then press P again to begin the recovery MP4 and actions
CSV. No pre-start mouse movement is included in that recovery clip. This mode
does not automatically reject an escaped pointer, though it rejects clearly
ineffective physical mouse movement; use R if the take is invalid.
For a feature check without filling the dataset with unusable clips, add
`--test-mode` (optionally alongside `--manual-focus`). The policy still sees
live frames, but no model or recovery MP4/CSV is written. Test diagnostics
go under `model_attempts/recovery_tests/` instead of the normal session folder.
For a real correction run without the preceding model-attempt MP4, use
`--no-model-video`; this helps isolate whether model-video encoding affects
the handoff while keeping recovery MP4/CSV collection enabled.
**R** restarts the current chamber from a new model attempt, whether pressed
during model control or human correction. **K** stops the whole session;
during correction it saves the partial clip under `interrupted`. Esc remains
the model runner's emergency stop, not the handoff key. During recovery recording,
Esc pauses/resumes; capture waits three seconds after the resume press so the
Portal 2 pause menu is not written to the clip. P is read from the
physical keyboard and may also reach Portal 2, so ensure it has no in-game
binding. The runner temporarily unbinds Portal 2's R key and restores its
original binding at the end of each phase.
The `wall_stuck` diagnostic remains a hint to the human, not an automatic
handoff trigger. If the model reaches the goal before takeover, the same
chamber is reloaded for another attempt; no recovery is recorded or counted.
Model time limits also retry the same chamber. Session logs and model videos
are saved under `model_attempts/recovery_sessions/`; corrections remain under
`episodes/recovery/`. The command accepts only `dataset_test1`–`dataset_test10`
because validation/evaluation chambers must stay out of recovery training.
The model keeps its configured `--model-seconds` limit (60 seconds by default),
regardless of the chamber's built-in timer. The session requires the updated
`shionn_events.nut` installed in Portal 2: it disables the chamber timeout
with an acknowledged command before running the model and rechecks the guard
before human recording. The correction has no local code time limit by default
(`--recovery-seconds 0`); K and genuine goal/failure events still end it.
An unexpected map reload invalidates the clip. If timeout suppression is not acknowledged, the session
stops rather than collecting a potentially invalid correction.

Keep successful corrections separate from the base dataset and its validation
split:

```bash
.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes/recovery/goal_reached \
  --cache-dir data/datasets/recovery_cached_frames

.venv/bin/python -m models.imitation.train_bc \
  --cache-dir data/datasets/cached_frames \
  --train-extra-cache-dir data/datasets/recovery_cached_frames \
  --correction-sampling-fraction 0.10 \
  --start-window-frames 8 --start-sampling-boost 5 \
  --checkpoint-dir models/imitation/checkpoints/runs/candidate_recovery
```

Extra caches are added to training only after the original episode split, so
they do not contaminate the preserved base validation set. Use a new checkpoint
directory, review every correction video, and include several distinct
recoveries before assigning a large sampling fraction. After preprocessing new
corrections, generate and review a *new* split manifest as described in
[DATA_SPLIT.md](DATA_SPLIT.md), then pass it with `--split-manifest` when
training. The original frozen v1 manifest must not be silently overwritten.

## Acceptance criteria

A candidate is ready for promotion only when:

- offline reports use data excluded from training;
- live trials use clean, training-compatible frames;
- runner errors and manual cancellations are marked invalid rather than failed;
- each evaluation chamber has repeated autonomous attempts;
- videos and diagnostics support the reported outcomes; and
- evaluation recordings remain isolated from training data.

The [CLI reference](CLI_REFERENCE.md) documents every benchmark, calibration,
and recovery option.
