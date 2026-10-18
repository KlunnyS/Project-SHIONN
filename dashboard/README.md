# SHIONN Control Room

A local Plotly Dash interface for the existing Portal 2 imitation-learning
workflow. The starter app's layout and navigation have been reused, but its
Heart Disease/MNIST/CIFAR classifier workflow has been replaced with controls
for this repository's recorders, frozen-v3 trainer, checkpoints, and
benchmarks.

## Run

From the repository root, install the interface dependencies into the same
Python environment used by SHIONN, then start the app:

```bash
.venv/bin/python -m pip install -r dashboard/requirements.txt
cd dashboard
../.venv/bin/python app.py
```

Open `http://127.0.0.1:8050`. The app binds to localhost only. Keep the SHIONN
virtual environment active/available because dashboard jobs run its Python
interpreter and call scripts from the repository root.

For the separate CUDA environment, first install the project runtime
dependencies with the provided setup script, then use `./run_gpu.sh` (or its
Fish/Windows counterpart). See the script comments and project
[training runbook](../../docs/TRAINING_RUNBOOK.md) for hardware-specific
environment details.

## Pages

- **Overview:** latest frozen-v3 train/validation counts, cache availability,
  and checkpoints with valid JSON contracts.
- **Record:** standard demonstrations, pre-wall left/right corrections, and
  model-assisted recovery. Map selection is restricted to training chambers.
- **Benchmark:** offline expert comparison on validation chambers, or a live
  model sequence on validation/evaluation chambers. Dry-run is available for
  live prediction without applying controls.
- **Train:** launches `models.imitation.train_bc` with frozen v3, selected
  correction caches, and a new checkpoint directory.
- **Results:** reads training metrics from checkpoint `metrics.jsonl` files and
  lists recent live benchmark sequence summaries.

## Safety and behavior

The dashboard constructs argument arrays for a small allow-list of project
commands; it does not invoke a shell or accept arbitrary commands/paths. It
allows one active job at a time, writes its command and output under
`model_attempts/interface_jobs/`, and requests a graceful interrupt when
**Stop active task** is pressed. Process status and recent output are polled in
the task bar.

Recording and live benchmark buttons launch real game processes. A recording
requires operator interaction; proactive capture starts on physical **P**.
Recovery mode uses the existing P/R/K controls. Live benchmark may control
Portal 2, so prepare the game/compositor before starting. The interface cannot
reliably recover in-memory job status if the Dash server itself is restarted.

Training defaults to `data/splits/frozen_v3.json` and both correction caches.
It will refuse a run name if the checkpoint destination already exists.
Validation remains on dataset_test11/12; evaluation1/2 are only offered for
live benchmark and are never added to training.

The old generic teaching backend was replaced; its external benchmark datasets
and app-local `runs/` archive are no longer used. SHIONN's real training logs,
checkpoints, recordings, and benchmark outputs stay in their existing project
locations.
