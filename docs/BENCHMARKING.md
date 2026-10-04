# Benchmarking a navigation checkpoint

Run these commands from the repository root after training a candidate. Replace the checkpoint path with the actual `best.pt`. The updated trainer saves its validation episode names inside the checkpoint, so the offline scorer uses the same held-out recordings that were excluded from training.

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

Review the beginning and at least one later frame before accepting a sequence. With `--keep-focused`, Hyprland should report a nonzero fullscreen state and the runner requests the same visible portal-gun viewmodel used in the demonstrations. Reject runs containing a zoomed opening, desktop bars, window borders, an open Steam overlay, or other large visual mismatches. Keep `MOUSE4` unbound so the runner can acquire XWayland input without invoking Portal 2's default `MOUSE3` zoom action. An enabled but closed Steam overlay is harmless; hide any visible performance HUD before the authoritative repeated benchmark.

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
recoveries before assigning a large sampling fraction.

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
