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
