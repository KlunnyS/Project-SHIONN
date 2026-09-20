# Mouse-bin and recovery experiment

The 850-episode checkpoint uses a Gaussian mouse head. Train a fresh binned checkpoint with the same episode split, chamber sampling, binary class weighting, seed, and default training settings:

```bash
.venv/bin/python -m models.imitation.train_bc \
  --cache-dir data/datasets/cached_frames \
  --mouse-head binned --mouse-bins 15 \
  --sampling chamber-balanced --binary-class-weighting balanced \
  --device cuda \
  --checkpoint-dir models/imitation/checkpoints/runs/850_binned_v1 \
  --checkpoint-every 0
```

The 15-class bins are fitted from nonzero mouse magnitudes in the usable training frames. Positive and negative movements get separate bins, and zero gets its own bin. The trainer prints the fitted edges and saves them in the checkpoint. The old 850 checkpoint still loads with its Gaussian head. Use a new checkpoint directory for every run; never resume a Gaussian checkpoint into a binned run.

## Compare expert actions and live behavior

```bash
.venv/bin/python benchmark_expert.py \
  --checkpoint models/imitation/checkpoints/runs/850_binned_v1/best.pt \
  --output-dir model_attempts/benchmarks/850_binned_v1

.venv/bin/python benchmark_jump.py \
  model_attempts/benchmarks/850_binned_v1/action_comparison.csv
```

The expert comparison uses the checkpoint's validation episodes. `dataset_test9` has 40 training and 10 validation episodes in the original 850 split, so its failure is not a fully held-out chamber failure. The 850 checkpoint made 54 false-positive jump predictions on its `dataset_test9` validation frames at the default threshold; none of those frames has a human jump label. Across the original 147,287 usable training frames, `use`, `fire_left`, and `fire_right` are all zero, so those heads cannot learn successful use or portal-fire behavior from that dataset.

To measure wall-facing frames in the **validation set**, review the expert episodes and create a CSV of inclusive frame ranges. Episode names and frame indices appear in `action_comparison.csv`:

```csv
episode,start_frame,end_frame
episode_YYYYMMDD_HHMMSS_000000,42,58
```

Then run:

```bash
.venv/bin/python analyze_mouse_validation.py \
  model_attempts/benchmarks/850_binned_v1/action_comparison.csv \
  --wall-ranges model_attempts/benchmarks/850_binned_v1/wall_ranges.csv
```

This scores every annotated validation frame, including bin entropy and peak gaps, and compares jump false-positive rates on wall-facing frames against the rest. The ranges need to come from visual inspection; the dataset has no existing wall-facing labels. Use `--jump-threshold` to inspect a calibration candidate.

For a short live run, omit `--keep-focused` because that option caused view snaps in the previous tests:

```bash
.venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/850_binned_v1/best.pt \
  --map dataset_test9 --max-seconds 30 --record-video --verbose
```

Review the attempt video and note the seconds when the model faces a wall or freezes. Then analyze **every tick** in those windows:

```bash
.venv/bin/python analyze_mouse_rollout.py \
  model_attempts/attempt_TIMESTAMP.jsonl \
  --wall-window 12:18 --wall-window 23:27
```

The JSON summary and CSV show per-axis normalized entropy, the gap between the two most probable bins, whether those bins point in opposite directions, and probability mass on negative, zero, and positive turns. Opposing peaks can support the conflicting-demonstrations hypothesis; a flat distribution can suggest low confidence. Neither measure alone proves a cause, so compare the video and expert action report as well. The rollout report compares jump action rates in wall windows with the rest of the attempt; the validation report can count actual jump false positives because expert labels are available there.

## Diagnose forward suppression before recording more corrections

In the `850_binned_v1` checkpoint, the forward head was trained with class weights `[7.4176, 0.5361]` for no-forward and forward. Under weighted cross-entropy, a 0.5 output threshold can require roughly 93% underlying confidence to press forward. This is a plausible cause of the idle starts: the original training set includes nearly identical opening views, yet the live policy did not press forward. On `dataset_test9` validation frames, a forward threshold of 0.2 improves forward recall from 91.2% to 95.1%, while precision changes from 97.2% to 95.8%.

Try a short run with only the forward threshold changed. Leave jump and mouse decoding as they were:

```bash
.venv/bin/python run_imitation.py \
  --checkpoint models/imitation/checkpoints/runs/850_binned_v1/best.pt \
  --map dataset_test9 --max-seconds 15 \
  --move-w-threshold 0.2 --record-video --verbose
```

If the model leaves the start but still freezes at walls, forward suppression and wall turning are separate issues. A threshold test does not retrain the policy or establish that it can reach the goal. A fresh unweighted binary-loss run is the next training experiment if forward suppression is confirmed; defer additional recovery recording until that result is clear.

## Sweep jump weight separately

Keep every setting above fixed except jump's positive-class weight and checkpoint directory. The 850 Gaussian run had jump weights `[0.5044, 10.0]`, an effective positive/no-jump ratio of about 19.8. Try a ratio of 3, then 5:

```bash
.venv/bin/python -m models.imitation.train_bc \
  --mouse-head binned --mouse-bins 15 --jump-positive-weight 3 \
  --sampling chamber-balanced --binary-class-weighting balanced --device cuda \
  --checkpoint-dir models/imitation/checkpoints/runs/850_binned_jump3 \
  --checkpoint-every 0

.venv/bin/python -m models.imitation.train_bc \
  --mouse-head binned --mouse-bins 15 --jump-positive-weight 5 \
  --sampling chamber-balanced --binary-class-weighting balanced --device cuda \
  --checkpoint-dir models/imitation/checkpoints/runs/850_binned_jump5 \
  --checkpoint-every 0
```

Score each checkpoint with `benchmark_expert.py`, then run `benchmark_jump.py` on its `action_comparison.csv`. Compare precision and recall across thresholds and chambers before choosing a live candidate. The runner uses binary argmax by default; pass `--jump-threshold 0.7` (or a threshold supported by the validation report) to test calibrated jump behavior live. Compare live jump behavior only after selecting a checkpoint and threshold.

## Record corrections from stuck states

Use this only when the policy still reaches a genuinely unseen failure state after testing forward calibration. Stop a live run at a wall or another failure state with its local time limit or Ctrl+C. Leave Portal 2 on that state. The model runner releases held actions when it exits. Then record your recovery without loading the map again:

```bash
.venv/bin/python record_recovery.py \
  --map dataset_test9 \
  --source-attempt model_attempts/attempt_TIMESTAMP.jsonl \
  --max-seconds 30
```

Only successful corrections in `episodes/recovery/goal_reached` should be cached for training. Keep this cache separate from the original 850 frames so the original validation split stays untouched:

```bash
.venv/bin/python -m models.imitation.preprocess \
  --recordings-dir episodes/recovery/goal_reached \
  --cache-dir data/datasets/recovery_cached_frames

.venv/bin/python -m models.imitation.train_bc \
  --cache-dir data/datasets/cached_frames \
  --train-extra-cache-dir data/datasets/recovery_cached_frames \
  --correction-sampling-fraction 0.10 \
  --start-window-frames 8 --start-sampling-boost 5 \
  --mouse-head binned --mouse-bins 15 \
  --sampling chamber-balanced --binary-class-weighting balanced \
  --device cuda \
  --checkpoint-dir models/imitation/checkpoints/runs/850_binned_recovery_v1 \
  --checkpoint-every 0
```

`--train-extra-cache-dir` adds these snippets to training after splitting the original episodes. A few clips would otherwise receive almost no optimizer draws in the 850-episode dataset, so set `--correction-sampling-fraction 0.10` when you have several distinct corrections. Use `--start-window-frames 8 --start-sampling-boost 5` to emphasize the first actions of every episode. On the existing 850 split, this changes the expected share of first-eight-frame openings from 4.2% to 17.7% before correction weighting. The original validation episodes remain untouched. Run a fresh training job because adding correction frames changes the fitted mouse bins and training data. Never preprocess the entire `episodes` tree into the base cache when keeping these corrections isolated.
