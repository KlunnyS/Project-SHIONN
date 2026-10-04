# Training runbook

This is the operational source of truth for training SHIONN's current behavior
cloning policy on the local Arch Linux workstation. For model design and data
contracts, see the [program guide](PROGRAM_GUIDE.md). For every trainer flag,
see the [CLI reference](CLI_REFERENCE.md).

## Current training status

Status recorded on 2026-10-04 at 18:10 CEST: a new from-scratch balanced-data
candidate is being started. It uses the 1,300 successful-episode cache, with
`dataset_test11` and `dataset_test12` held out in full. The 450 new successful
episodes include planned wall-correction, turn, stair, and chamber-specific
variation; they are part of the base dataset rather than the separate
policy-induced recovery cache.

| Item | Value |
|---|---|
| Run directory | `models/imitation/checkpoints/runs/v5_balanced_recovery_v1` |
| Architecture | Residual v5 with binned mouse heads |
| Training holdouts | `dataset_test11`, `dataset_test12` |
| Cached data | 1,300 successful episodes; 1,100 training and 200 held out |
| Training / validation batch | `8` / `32` |
| Initial learning rate | `2e-4` |
| Plateau schedule | First miss reduces LR by `0.25`; second consecutive miss stops |
| Workers / checkpoint interval | `0` / `10,000` optimizer steps |
| Service | `shionn-train-v5-balanced-recovery-v1` |
| Baseline to beat | validation loss `2.6779654485835236` from the completed run's epoch 2 |

The prior `v5_holdout_11_12_tuned_v1` trial was intentionally stopped without a
checkpoint after about five minutes. Keep the completed baseline's `best.pt` as
the deployed policy until this candidate has completed expert and live
comparison.

## Completed v5 baseline run

Status recorded on 2026-09-28: completed successfully at 11:19 CEST after
early stopping. The collected `shionn-train-v5-resume` user service exited
with status 0.

| Item | Value |
|---|---|
| Run directory | `models/imitation/checkpoints/runs/v5_holdout_11_12` |
| Architecture | Residual v5 with binned mouse heads |
| Training holdouts | `dataset_test11`, `dataset_test12` |
| Batch size / workers | `2` / `0` |
| Service outcome | `shionn-train-v5-resume`: success (collected after exit) |
| Latest checkpoint | `last.pt`: epoch 6 complete |
| Global optimizer step | `494,292` |
| Best checkpoint | `best.pt`: epoch 2 |
| Best validation loss | `2.6779654485835236` |
| Completed epoch losses | validation `2.879048`, `2.677965`, `2.713138`, `2.801036`, `2.988689`, `3.248011` |
| Early stopping | Triggered after unimproved epochs 4, 5, and 6 |

Training loss continued to improve through epoch 6, but validation loss did
not beat epoch 2 and worsened in the final epochs. Use `best.pt`, not
`last.pt`, for evaluation; the divergence indicates overfitting after epoch 2.

Checkpoint cleanup completed on 2026-10-04. The run retains `best.pt/json`,
`last.pt/json`, `metrics.jsonl`, and the newest periodic recovery pair
`step_000490000.pt/json`; all three retained checkpoints passed a load check.
Removing 321 older periodic pairs recovered about 47.37 GiB and reduced the
run directory to about 453 MiB. These artifacts are not tracked by Git.
`last.pt` contains the final optimizer state, while `best.pt` contains the best
validation candidate.

## Live benchmark and visual-state correction

Review on 2026-10-04 found another visual-state mismatch in the 2026-09-30
sequence: input activation clicked `MOUSE3`, which this Portal 2 installation
binds to `+zoom`. The model has no zoom action, and zoom changes both its visual
input and the apparent effect of mouse motion. Treat
`model_attempts/sequences/sequence_20260930_112316_263060/` as diagnostic only,
not an authoritative model score. The runner now activates XWayland input with
unbound `MOUSE4` instead. Verify an unzoomed opening in new videos before using
their results, and rerun the repeated benchmark before making promotion or
retraining decisions.

A live smoke attempt on 2026-10-04 verified the fix on `dataset_test12`. The
opening remained unzoomed, XWayland input activated, all 148 logged ticks
applied their actions, compositor fullscreen remained active with zero focus
losses, and the run reached the goal. The recording and diagnostics are
`model_attempts/attempt_20261004_135245_185703.mp4` and `.jsonl`. This validates
the input-activation fix only; a repeated clean benchmark is still required to
measure policy reliability.

A clean five-chamber sequence later on 2026-10-04 confirmed the remaining
policy failures without the zoom mismatch. `dataset_test12` reached the goal in
7.03 seconds; `dataset_test1`, `dataset_test10`, `dataset_test11`, and
`evaluation2` reached the 60-second limit. All five attempts had zero focus
losses. Video and action-log review found sustained idle output, large
same-direction mouse/movement loops, and commanded wall approaches with little
visual progress. The sequence is
`model_attempts/sequences/sequence_20261004_135851_169801/`. The runner now
records explicit `policy_freeze`, `wall_stuck`, and `turn_loop` events and
supports an opt-in `--mouse-max-abs` cap. The next controlled live comparison
should repeat the same chamber order with `--mouse-max-abs 28`; the uncapped
default remains the baseline.

The first 2026-09-28 sequence was invalidated after video inspection. Portal 2
was not fullscreen: whole-output capture included the desktop bar and window
border, while the portal-gun viewmodel visible throughout the demonstrations
was absent. Focus, input activation, and tick delivery were healthy, but this
was a material train/live visual mismatch. Preserve the diagnostic sequence at
`model_attempts/sequences/sequence_20260928_123405_985146/`, but do not use its
0/3 result as a model score.

The live runner now uses `--keep-focused` to enter Hyprland compositor
fullscreen after each map load, request `r_drawviewmodel 1`, equip
`weapon_portalgun`, and record the prepared window state in the attempt
metadata. A corrected one-attempt sequence then produced:

| Map | Result | Diagnostics |
|---|---|---|
| `dataset_test1` | 0/1, time limit | 1,434 ticks; 1,407 idle and 27 forward; became stuck against a wall |
| `dataset_test11` | 0/1, time limit | 1,433 ticks; remained active but oscillated, chiefly 682 forward and 465 backward ticks |
| `dataset_test12` | 1/1, goal reached | 566 ticks; goal signal at about 23.8 seconds |

All corrected attempts recorded compositor fullscreen state `2`,
`input_ready=true`, and zero focus losses. Videos confirm clean game framing and
the portal gun on `dataset_test11` and `dataset_test12`; on `dataset_test1` the
agent immediately drove against a wall, where the viewmodel moved out of frame.
The corrected sequence is under
`model_attempts/sequences/sequence_20260928_125751_837912/`.

The previously suspected performance overlay was the Steam HUD and is not a
policy-run blocker. The 1/1 held-out success on `dataset_test12` shows
that the checkpoint can complete an unseen chamber, but one attempt per map is
not enough to estimate reliability. Keep the existing deployed policy as the
baseline until repeated clean trials are available.

## Preflight checks

Run commands from the repository root. Confirm the extra drive and CUDA before
starting a service:

```bash
findmnt /mnt/extra
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The cached dataset and model directories are symlinks into `/mnt/extra`. The
current checkpoint path resolves to:

```text
/mnt/extra/Project-SHIONN/models/imitation/checkpoints/runs/v5_holdout_11_12
```

On 2026-09-27 this was `/dev/sdb4`, mounted at `/mnt/extra`. Do not start
recording, preprocessing, training, or inference when that mount is absent.

## Resume command used

The completed run was resumed in its own systemd service with the following
command. Do not run it again unless deliberately extending the completed run:
at the time this command ran, early-stopping patience still started fresh on
every resume. The separate cgroup prevents a training failure from taking down
the editor, and `python -u` makes progress messages immediately visible in the
journal.

```bash
systemd-run --user --unit=shionn-train-v5-resume --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 \
  -p MemoryHigh=16G \
  -p MemoryMax=20G \
  -p MemorySwapMax=2G \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  -m models.imitation.train_bc \
  --holdout-map dataset_test11 \
  --holdout-map dataset_test12 \
  --device cuda \
  --workers 0 \
  --batch-size 2 \
  --early-stop-patience 3 \
  --lr-scheduler none \
  --checkpoint-every 10000 \
  --checkpoint-dir models/imitation/checkpoints/runs/v5_holdout_11_12 \
  --resume models/imitation/checkpoints/runs/v5_holdout_11_12/last.pt
```

Use a new systemd unit name if that name is still loaded. Do not change the
dataset, holdouts, batch size, optimizer settings, sampling settings, mouse
head, or augmentation flags when resuming: the trainer validates these values
against the checkpoint. Checkpoint frequency, logging frequency, workers, and
early-stop patience may be changed without invalidating the saved model state.

A mid-epoch resume reconstructs the deterministic epoch loader and reads past
the previously completed batches before optimization continues. That initial
skip can look idle; the printed resume position and advancing checkpoint files
are the reliable indicators.

## Monitoring and control

Follow unbuffered trainer output:

```bash
journalctl --user -fu shionn-train-v5-resume
```

Inspect service and memory state:

```bash
systemctl --user status shionn-train-v5-resume
systemctl --user show shionn-train-v5-resume -p MemoryCurrent -p MemoryPeak
```

Inspect saved progress independently of journal output:

```bash
tail -n 5 models/imitation/checkpoints/runs/v5_holdout_11_12/metrics.jsonl
ls -lhtr models/imitation/checkpoints/runs/v5_holdout_11_12/step_*.pt | tail -n 3
```

Stop cleanly with:

```bash
systemctl --user stop shionn-train-v5-resume
```

Periodic saves are atomic, so an interrupted write leaves the preceding
`last.pt` available. Check its epoch and optimizer step before the next resume.

## Resource findings

Two unconstrained training attempts launched inside VS Code were killed by the
kernel OOM killer after the Python process reached roughly 24-25 GiB resident
memory. Because Python shared VS Code's application scope, the editor was also
lost. At the time the scope peaked near 27 GiB of RAM and 10.6 GiB of swap.

The contained batch-2 run remained below the 20 GiB hard limit and recorded no
OOM or hard-limit events. Its cgroup hovered around the 16 GiB soft limit with
about 2 GiB swap. Most resident memory observed in the contained run was
file-backed cache from the memory-mapped frame dataset, not Python heap. Soft-
limit reclaim events are therefore expected; an increasing `oom_kill` or
`max` count is not.

The 850 episodes open both a frame and action memory map. A transient service
with the default descriptor limit failed during dataset construction with
`OSError: [Errno 24] Too many open files`. Keep `LimitNOFILE=8192` for this
dataset. `--workers 0` avoids additional loader processes and is the validated
setting on this workstation.

An isolated synthetic v5 benchmark on the RTX 4060 measured approximately
108 samples/s at batch 2 and 133 samples/s at batch 8, while peak allocated
CUDA memory rose only from 0.36 GiB to 0.73 GiB. The next controlled run should
therefore use training batch 8 and validation batch 32. Host memory is still
governed mainly by the memory-mapped dataset, so retain the existing cgroup
limits and `--workers 0`.

## Checkpoint and disk policy

Each v5 checkpoint is approximately 151 MiB. Saving every 1,000 optimizer
steps produced roughly 13 GiB of checkpoints per epoch and nearly filled the
extra drive. Use `--checkpoint-every 10000` for resumable long runs, or `0` to
keep only epoch-end `last.pt` and improved `best.pt`.

Before a run, check both the resolved path and free space:

```bash
readlink -f models/imitation/checkpoints/runs/v5_holdout_11_12
df -h models/imitation/checkpoints/runs/v5_holdout_11_12
du -sh models/imitation/checkpoints/runs/v5_holdout_11_12
```

Old periodic checkpoints are redundant once a newer `last.pt` has been
validated. Keep `best.pt`, `last.pt`, their JSON contracts, `metrics.jsonl`,
and optionally one recent periodic checkpoint. Review exact deletion targets
before removing generated files.

## Resume and early-stopping semantics

`last.pt` stores model weights, AdamW state, epoch, step within epoch, global
step, best validation loss, and the configuration contract. `best.pt` is for
evaluation; `last.pt` is normally the correct resume source.

The completed run used the old behavior in which early-stopping patience was
not persisted. Its resume reset the counter, so it ran epochs 4, 5, and 6
before stopping. New checkpoints persist both the consecutive-unimproved count
and learning-rate scheduler state. Legacy checkpoints recover their count from
`metrics.jsonl`, and a resume exits without optimizing when patience was
already satisfied. Legacy checkpoints created before scheduler configuration
was recorded must be resumed with `--lr-scheduler none` to preserve their
fixed-learning-rate contract.

## Tuned experiment command (not active)

The stopped no-checkpoint trial used the following command, which remains the
candidate configuration for a deliberate future run. The completed epoch-2
`best.pt` remains the baseline. The first validation miss reduces the learning
rate from `2e-4` to `5e-5`; a second consecutive miss stops the run. This gives
the model one lower-rate recovery epoch instead of three full-rate misses.

```bash
systemd-run --user --unit=shionn-train-v5-tuned-v1 --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 \
  -p MemoryHigh=16G \
  -p MemoryMax=20G \
  -p MemorySwapMax=2G \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  -m models.imitation.train_bc \
  --holdout-map dataset_test11 \
  --holdout-map dataset_test12 \
  --device cuda \
  --workers 0 \
  --batch-size 8 \
  --validation-batch-size 32 \
  --early-stop-patience 2 \
  --lr-scheduler plateau \
  --lr-reduction-patience 0 \
  --lr-reduction-factor 0.25 \
  --checkpoint-every 10000 \
  --checkpoint-dir models/imitation/checkpoints/runs/v5_holdout_11_12_tuned_v1
```

If run later, this remains an experiment rather than a guaranteed improvement.
Compare its `best.pt` against the preserved baseline with the expert benchmark
and repeated live chamber runs before promotion.

## Documentation maintenance

When training state or behavior changes:

1. Update the dated **Current training status** section in this file whenever a
   run is started, stopped, resumed, completed, or superseded.
2. Update the README's short milestone summary only after a meaningful result;
   keep exact operational state here.
3. Update `CLI_REFERENCE.md` when commands, flags, or defaults change.
4. Update `PROGRAM_GUIDE.md` when architecture, data flow, or contracts change.
5. Update `CHAMBER_AUTHORING.md` or `BENCHMARKING.md` when those workflows
   change; do not copy the procedure into another document.
6. Add a concise entry to `CHANGELOG.md`.
7. Never commit datasets, recordings, checkpoints, or diagnostic media.
