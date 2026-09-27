# Training runbook

This is the operational source of truth for training SHIONN's current behavior
cloning policy on the local Arch Linux workstation. For model design and data
contracts, see the [program guide](PROGRAM_GUIDE.md). For every trainer flag,
see the [CLI reference](CLI_REFERENCE.md).

## Current v5 run

Status recorded on 2026-09-27: intentionally stopped and safe to resume.

| Item | Value |
|---|---|
| Run directory | `models/imitation/checkpoints/runs/v5_holdout_11_12` |
| Architecture | Residual v5 with binned mouse heads |
| Training holdouts | `dataset_test11`, `dataset_test12` |
| Batch size / workers | `2` / `0` |
| Latest checkpoint | `last.pt`: epoch 4, step 55,854 within epoch |
| Global optimizer step | `303,000` |
| Best checkpoint | `best.pt`: epoch 2 |
| Best validation loss | `2.6779654485835236` |
| Completed epoch losses | validation `2.879048`, `2.677965`, `2.713138` |
| Early stopping | Three newly completed unimproved epochs after resume |

The epoch-3 validation loss was slightly worse than epoch 2 even though the
training loss continued to improve. Keep `best.pt` for evaluation unless a
later epoch establishes a new validation minimum.

The run directory currently also contains 303 legacy `step_*.pt` files from
the original 1,000-step checkpoint interval. They occupy about 44.7 GiB of the
46 GiB run directory. They are not tracked by Git. `last.pt` contains the exact
resume position, while `best.pt` contains the best validation candidate.

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

## Safe resume command

Run training in its own systemd service instead of a VS Code terminal. The
separate cgroup prevents a training failure from taking down the editor, and
`python -u` makes progress messages immediately visible in the journal.

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

Early-stopping patience is not persisted. It starts fresh whenever the trainer
is resumed. With patience 3, the current run will stop after three newly
completed epochs without beating `2.6779654485835236`. The total `--epochs`
value remains the absolute run target rather than a count of additional epochs.

## Documentation maintenance

When training state or behavior changes:

1. Update the dated **Current v5 run** section in this file.
2. Update the README's current-progress summary after a meaningful milestone.
3. Update `CLI_REFERENCE.md` when flags or defaults change.
4. Update `PROGRAM_GUIDE.md` when architecture, data flow, or contracts change.
5. Add a concise entry to `CHANGELOG.md`.
6. Never commit datasets, recordings, checkpoints, or diagnostic media.
