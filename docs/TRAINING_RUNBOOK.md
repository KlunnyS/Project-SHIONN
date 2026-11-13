# Training runbook

This is the operational source of truth for training SHIONN's current behavior
cloning policy on the local Arch Linux workstation. For model design and data
contracts, see the [program guide](PROGRAM_GUIDE.md). For every trainer flag,
see the [CLI reference](CLI_REFERENCE.md).

## Current training status

Status recorded on 2026-10-04 at 21:16 CEST: the from-scratch balanced-data
candidate completed successfully after epoch 4, when early stopping followed
two consecutive validation misses. It uses the 1,300 successful-episode cache,
with `dataset_test11` and `dataset_test12` held out in full. The 450 new
successful episodes include planned wall-correction, turn, stair, and
chamber-specific variation; they are part of the base dataset rather than the
separate policy-induced recovery cache.

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
| Service outcome | `shionn-train-v5-balanced-recovery-v1`: success |
| Latest checkpoint | `last.pt`: epoch 4 / global step 117,608 |
| Best checkpoint | `best.pt`: epoch 2 / validation loss `2.649578924484107` |
| Training duration | 3h 06m 35.8s wall time; 46m 37.4s average epoch |

The numerical best validation loss is lower than the completed baseline's
`2.6779654485835236`, but the held-out chambers now include 100 episodes each
rather than 50. Do not compare the losses as a promotion decision: compare
both checkpoints on the same frozen expert-frame evaluation and repeated live
chamber sequence. Keep the completed baseline's `best.pt` as the deployed
policy until those comparisons are complete.

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
not an authoritative model score. The runner was changed at the time to activate
XWayland input with unbound `MOUSE4`; it now uses a temporarily unbound `MOUSE1`
left click. Verify an unzoomed opening in new videos before using
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

For new training runs, the [frozen split](DATA_SPLIT.md) is enforced by default.
The historical commands below document runs started before that manifest was
introduced; the same test11/test12 holdouts are now selected automatically.

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

## Controlled group-sampling comparison — prepared 2026-10-31

Control status, verified 2026-10-31: `v5_binned_sampling_control_v1` completed
four epochs and 121,276 optimizer steps. Validation losses were 2.71493,
2.63910, **2.58747** and 2.76430; retain epoch-3 `best.pt` for comparison and
epoch-4 `last.pt` for reproducibility. Summed epoch time was about 3h 44m.
Checkpoint metadata confirms seed 0, frozen-v3 membership (1,151 training /
200 validation episodes), fixed learning rate 0.0002, opening boost 1 and no
explicit correction oversampling. The final epoch worsened validation despite
lower training loss. This is an offline control result, not navigation evidence.
The matched 90/5/5 candidate has also completed four epochs and 121,276 steps.
Its best epoch is 3, with validation loss 2.74413; epoch 4 worsened to 2.85681.
Saved settings match the control apart from the intended correction grouping.
Do not claim navigation improvement from these losses; neither new checkpoint
has a live score yet. See [the comparison](GROUP_SAMPLING_COMPARISON_2026_10_31.md).
Paired expert-frame scoring also did not support the grouped recipe: exact
binary match fell from 84.43% to 83.52% and horizontal MAE rose from 6.503 to
6.744. Do not automatically start a higher correction fraction or promote the
new run. Both training commands below now document a completed experiment.

The new `--correction-group-fraction` option can express 90% base / 5%
proactive-turn / 5% recovery draws. Both extra caches remain separate and
frozen v3 keeps validation membership unchanged. These are expected sampling
shares; individual replacement-sampled epochs fluctuate. Rescaling groups
preserves relative weights within each group but can change global map shares.
Omitting the new options retains the existing sampler. All loaded correction
groups must receive a positive explicit fraction when group mode is enabled.

Run a matched control first, then the group-sampled candidate, sequentially.
Both use the same four complete epochs, seed 0, fixed learning rate, architecture,
split, batch size and augmentation. The older candidate's different stopping
schedule does not replace this control. Do not resume either experiment from
an existing trained policy: initialize both from scratch. Checkpoint selection
uses the best validation epoch; preserve `last.pt` as well. The candidate run
directory below satisfies #26's naming convention; the separate turn-onset
strategy criterion remains outstanding until annotated and tested.

From the repository root, with the external storage mounted and the destination
directory absent, start the control:

```bash
systemd-run --user --unit=shionn-sampling-control-v1 --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  -m models.imitation.train_bc \
  --split-manifest data/splits/frozen_v3.json \
  --train-extra-cache-dir data/datasets/proactive_turn_cached_frames \
  --train-extra-cache-dir data/datasets/recovery_cached_frames \
  --device cuda --workers 0 --batch-size 8 --validation-batch-size 32 \
  --epochs 4 --seed 0 --sampling chamber-balanced \
  --learning-rate 0.0002 --lr-scheduler none --early-stop-patience 0 \
  --binary-class-weighting none --jump-positive-weight 3 \
  --brightness-augmentation 0.05 --contrast-augmentation 0.1 \
  --start-window-frames 8 --start-sampling-boost 1 \
  --checkpoint-every 0 --log-every 500 \
  --checkpoint-dir models/imitation/checkpoints/runs/v5_binned_sampling_control_v1
```

Monitor with:

```bash
journalctl --user -u shionn-sampling-control-v1 -f
```

After this finishes successfully, repeat the training command with unit
`shionn-turn-recovery-v1`, checkpoint directory
`models/imitation/checkpoints/runs/v5_binned_turn_recovery_v1`, and these two
additional options:

```text
--correction-group-fraction proactive_turn=0.05
--correction-group-fraction recovery=0.05
```

Startup output must show expected proactive and recovery shares of 5% each.
The recipe and exact correction episode/group mapping are saved in checkpoint
configuration. Resume rejects changed fractions or membership; legacy
checkpoints without these fields remain compatible when group mode is off.
Only resume an interrupted experiment with its original options and directory.

Compare held-out action and turn-onset results, then run the same autonomous
live protocol. Do not combine opening boosts or new losses into this first
comparison. No improvement in navigation has been demonstrated yet.

## Bounded recovery-initiation probe — prepared 2026-11-01

Status verified 2026-11-02: **both arms completed in 11m27s**, each at 1,000
updates with full evaluation. `paired_complete` is true and the parent model
is unchanged. The boost did not improve the initial all-zero decisions: both
endpoints still predict idle at the first recovery frame in 6/6 assessment clips.
See [the result review](GROUP_SAMPLING_COMPARISON_2026_10_31.md).
The commands below document the completed experiment; its default output now
exists, so **do not rerun it** or extend it automatically.

This is a short warm-start experiment, not another from-scratch training
run. Default invocation prints a plan without
constructing a GPU model, changing weights or creating an output directory:

```bash
.venv/bin/python run_recovery_probe.py
```

Both arms start independently from the exact weights in
`v5_recovery_proactive_v3/best.pt`, with fresh AdamW optimizers and fixed source
mouse bins, class weights, preprocessing and action heads. Both use90% base /
5% proactive /5% recovery draws, seed0, batch 8 and learning rate0.00001.
Control leaves recovery openings unboosted; the second arm multiplies the
first 24 usable frames (one second at 24FPS) of each recovery episode by8, then
renormalizes within recovery so its total share stays5%. Other groups are
unchanged. Expected opening draws are about73 versus256 in8,000 sampled
frames per arm; actual consumed group/opening counts are reported separately.
These frames may be sampled repeatedly; they are not additional recordings.

Each arm has at most **1,000 optimizer updates**, in float32 to avoid unequal
AMP-skipped updates in this small budget. Both are evaluated at that fixed
endpoint; no repeated selection on assessment is performed. The six fresh
assessment clips are read only for evaluation and never enter gradient data.
Full frozen validation and the unchanged source baseline are evaluated too.
The broad data mix and new optimizer/LR are common to both arms; only recovery-
opening emphasis differs between them.

Budget: **roughly12–18 minutes total for both arms and evaluation**, based on
recent workstation timings, not a measured full probe. The Python soft deadline
is18 minutes across preparation, baseline, both arms and evaluation. The command
below adds a20-minute service runtime limit with at most one second of shutdown
grace for blocked GPU/I/O. If time runs out, the result is incomplete; do not
automatically rerun, extend it, compare unequal update counts, or promote an
endpoint. The unchanged source remains available regardless.

After reviewing the plan, run once from the repository root, with storage
mounted and output directory `recovery_onset_probe_v1` still absent:

```bash
systemd-run --user --unit=shionn-recovery-probe-v1 --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G \
  -p RuntimeMaxSec=20min -p TimeoutStopSec=1s \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  run_recovery_probe.py --execute
```

Monitor with:

```bash
journalctl --user -u shionn-recovery-probe-v1 -f
```

Output: `models/imitation/checkpoints/runs/recovery_onset_probe_v1/report.json`,
with `control/endpoint.pt` and `boosted/endpoint.pt` plus sidecars. Existing
output is refused. Partial progress may have `partial.pt`; it is not a
completed comparison. Only a report with `paired_complete: true` has both
equal step budgets and complete evaluation. No model is automatically selected,
deployed or promoted, even when the pair completes.

The endpoints record zero complete dataset epochs, actual update count, source
hash, source progress, script hash, split identity, exact train/validation
membership, group recipe and opening boost. Source target processing is copied
unchanged. These are new fine-tuning endpoints, not resumptions of the source
optimizer. Loading them for inference is supported; generic full-trainer resume
is explicitly rejected. An interrupted probe requires review before another
experiment is prepared.

Preparation verification: all219 project tests pass, including synthetic CPU
tests that exercise exact update budgets,
fresh initial weights, deadline interruption, incomplete evaluation, source
immutability and checkpoint compatibility. Real frozen-v3 plan verifies1,151
training /200 validation /6 assessment episodes and expected draw shares.
A single real float32 GPU forward/backward batch produced finite gradients
without optimizer updates or checkpoint writes; allocated tensor memory plus
estimated fresh Adam moments was about0.82GiB, excluding device/library overhead.
The real paired probe completed as recorded above; no navigation improvement
has been established and no endpoint is automatically selected.

## Recovery history probe — 2026-11-02

Status: user-run experiment completed in 227.339 seconds (3m47s), with
`comparison_complete: true`, 1,000 updates, 8,000 draws, and full 39,769-frame
validation / 879-frame assessment evaluation. All recorded parent, reference,
helper and new-script hashes still match. The endpoint records 1,000 completed
updates and `complete_fixed_endpoint`; the control was not retrained.

No useful initiation improvement was established. First-action idle output
changed from 6/6 control clips to 5/6; the one changed prediction was forward
movement for an expert turn (`dx=-13, dy=7`), so it is not a demonstrated correct
recovery. First-eight-frame idle outputs fell from 13/48 to 11/48, but horizontal
MAE remained 45.729 and binary exact match fell from 83.33% to 79.17%. Full
assessment horizontal MAE worsened slightly from 20.854 to 20.940. Normal
validation binary match changed 84.946% to 84.978%, with horizontal MAE
6.225 to 6.268. Do not promote or automatically extend this recipe. No live
game test is justified by these results alone; inspect training/inference
history construction and image/action timing before another training decision.

The previous opening-boost
probe completed without fixing first-frame initiation (all six independent
assessment starts remained idle). Repeating the newest image across its history
at evaluation increased idle predictions from 19/144 to 84/144 frames. This
shows sensitivity to history, but removes legitimate motion cues and introduces
distribution shift; it does not prove a cause or that augmentation will help.

`run_recovery_history_probe.py` tests one hypothesis: repeat the newest RGB image
across all four history slots on 50% of sampled recovery draws, keeping the
expert target. Base and proactive-turn samples stay unchanged. The verified
sample order produces 221 augmented draws out of 424 recovery draws; total
draws remain 8,000 (7,173 base / 424 recovery / 403 proactive).

Only one new arm trains: 1,000 updates, batch 8, LR `1e-5`, fresh AdamW from
`v5_recovery_proactive_v3/best.pt`. It reuses the completed unboosted control
from `recovery_onset_probe_v1`, checking helper/source/config/split/cohorts,
sample-order hashes, endpoint progress and cache timestamps. The earlier run
did not record library versions or full pixel hashes, so historical bitwise
identity cannot be established. The plan records this limitation explicitly.
The six assessment clips remain evaluation-only. No automatic promotion occurs.

Estimated total runtime: **4–8 minutes**, with a **10-minute internal budget**
including preparation/evaluation and a **12-minute outer limit**. No extension
or automatic retry follows a timeout. Default invocation only prints the plan:

```bash
.venv/bin/python run_recovery_history_probe.py
```

Preparation checks: all 228 project tests pass, including nine new CPU synthetic
tests for unchanged labels, per-draw recovery masks, global RNG preservation,
fresh parent initialization, reference immutability and incomplete evaluation.
A read-only protocol review and the real plan passed before execution; the
original helper hash remains unchanged.

Historical execution command (completed; do not rerun the existing output):

```bash
systemd-run --user --unit=shionn-recovery-history-probe-v1 --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G \
  -p RuntimeMaxSec=12min -p TimeoutStopSec=1s \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  run_recovery_history_probe.py --execute
```

Follow with `journalctl --user -u shionn-recovery-history-probe-v1 -f`.
Output is `models/imitation/checkpoints/runs/recovery_history_probe_v1/report.json`.
Only `comparison_complete: true` with 1,000 completed updates and complete full
validation/assessment evaluation counts as a finished comparison. A partial
endpoint or a report left `running` by an external kill does not. Existing output
is refused. Review the resulting action metrics against `reused_control` before
deciding whether any user-run game test is justified.

## Offline history/timing audit — 2026-11-02

The follow-up audit found exact cached label/video and training/live input
agreement on 53 saved episodes (2,691 decoded frames). Existing live logs mostly
ran near 24 Hz. Saved-video motion analysis did not identify a consistent label
shift. Capture receive timing was absent, so frame age/reuse remained unknown.
Diagnostics have now been added, without changing labels, model or controls.
No training or game action was launched. See the
[audit and one short manual timing check](HISTORY_TIMING_AUDIT_2026_11_02.md).
Keep this diagnostic recording outside all frozen gradient and assessment data.

The user subsequently completed the 20-second manual calibration (481 aligned
frames). Receive age median was 11.76 ms, with five reused selections; visible
mouse response followed input by about one to two rows. No prolonged capture
freeze or justified target shift was established. Read-only model comparison
shows history augmentation reduces repeated-history assessment idle from
81/144 to 53/144, while binary exact match worsens 47.92% to 42.36%. Even on
training clips, 19/31 initial repeated-history actions remain idle. A bounded
recovery-fit sanity check on existing training clips is the proposed next
experiment, not started or scheduled. See the audit for assumptions and full
results; no new recording, full training run or promotion is requested now.

## Recovery-fit sanity check — 2026-11-02

Status: user-run diagnostic completed successfully in 34.651 seconds. The report
records `diagnostic_complete: true`, exactly 400 updates / 3,200 draws and complete
before/after scoring on all 31 starts. Source checkpoint, sidecar and all recorded
code hashes still match. The output directory contains only `report.json`.

| Training-start metric | Before | After |
|---|---:|---:|
| All eight binary actions and both mouse classes correct | 3/31 | 31/31 |
| Predicted idle | 19/31 | 0/31 |
| Backward actions recovered | 2/8 | 8/8 |
| Horizontal raw mouse MAE | 29.290 | 0.903 |

After-fit mouse MAE equals the source bin-quantization floor (0.903); all
encoded classes are correct. This establishes fit on the 31 known stationary
inputs under concentrated exposure with image augmentation disabled. It does
not establish generalization, preservation of navigation or whether sampling
alone caused the earlier failures. No assessment data was loaded and no policy
checkpoint was saved or promoted. Do not repeat this completed diagnostic.

The next proposed step is a controlled candidate fine-tune combining explicit
stationary recovery-start replay with base-navigation rehearsal, followed by
complete held-out scoring before a user-run game test. That recipe has not been
prepared or started. Keep assessment and calibration data outside gradients.

Historical rationale and protocol: the timing/learning audit found
19/31 initial stationary-history actions still idle on the history model's own
training recordings. This diagnostic tests whether that warm start can fit
these known examples under a concentrated fixed budget before another full run.

`run_recovery_fit_probe.py` reads only the 31 frozen-v3 recovery TRAIN clips,
selecting each first active action and repeating its newest RGB across all four
history slots. It keeps their targets and source mouse bins unchanged. Exact
duplicate inputs with conflicting encoded labels are rejected; none exist in
the current 31 examples. Base, validation, six assessment clips and the timing
calibration are not loaded. This is intentional memorization, not a navigation
or generalization score.

It starts from `recovery_history_probe_v1/endpoint.pt` with fresh AdamW, LR
`5e-5`, batch 8 and exactly 400 updates. Private seeded permutation cycles give
each start 103 or 104 draws (3,200 total). Brightness/contrast/flips are disabled;
network dropout, binary weights, mouse bins, label smoothing and gradient
clipping remain as in the source. The goal is all 31 encoded action vectors
correct: eight binary decisions and both mouse classes. Raw mouse MAE is
reported alongside its bin-quantization floor; nonzero output alone is not a
correct action. Missing the goal under this budget does not prove the
architecture cannot fit the examples.

Estimated total runtime: **1–3 minutes**, a **four-minute internal budget** and
**five-minute outer limit**, without automatic retry/extension. Only a JSON
report is saved; no model checkpoint or optimizer state is written, and the
source/current model stays untouched. Default invocation only prints the plan:

```bash
.venv/bin/python run_recovery_fit_probe.py
```

Historical execution command (completed; existing output is refused):

```bash
systemd-run --user --unit=shionn-recovery-fit-probe-v1 --collect \
  -p WorkingDirectory=/home/user/Documents/GitHub/Project-SHIONN \
  -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G \
  -p RuntimeMaxSec=5min -p TimeoutStopSec=1s \
  /home/user/Documents/GitHub/Project-SHIONN/.venv/bin/python -u \
  run_recovery_fit_probe.py --execute
```

Follow with `journalctl --user -u shionn-recovery-fit-probe-v1 -f`.
Output is `model_attempts/diagnostics/recovery_fit_probe_v1/report.json`.
Existing output is refused. Require `diagnostic_complete: true`, all 400 updates,
and complete 31-start before/after scoring. `after.fit_goal_met` answers the
fitting question separately from whether execution completed. A partial or
externally killed report is not a completed diagnostic. A successful fit only
establishes learning on the known starts; it does not justify deployment. No
game test or training run automatically follows either result.

Verification: all 245 project tests pass, including 11 new CPU synthetic tests
for training membership, finite/active targets, conflicting labels, exact class
scoring versus quantization, source preservation, private balanced draws,
report-only execution and deadline failure. The real default plan confirms 31
starts, 103–104 exposures each, and no output creation. No real optimizer update
was performed during preparation.

## Prepared stationary-replay candidate — 2026-11-02

Status updated 2026-11-03: user-run candidate completed in 651.212 seconds
(10m51s), with all 2,000 updates and complete before/after evaluation. Source
hash is unchanged; candidate hash and endpoint metadata match the report.

The navigation imitation screen passed: binary exact match changed 84.978% to
84.802%, horizontal MAE 6.268 to 6.363, and first-eight-frame idle count 184 to
176/1,600. All 31 trained stationary starts became fully correct (3/31 before).
However, stationary assessment starts remain 0/6 fully correct; idle falls
5/6 to 3/6 while backward recall remains 0/1. Normal-history assessment first8
idle falls 11/48 to 6/48, but binary exact match worsens 79.17% to 75.00%.
This demonstrates training fit and limited reduction in idleness, not reliable
recovery transfer. Do not promote or automatically extend this candidate.
Further review should focus on why learned start actions fail to transfer,
rather than repeating the same sampling recipe. No game test was run by the agent.

Historical preparation: no real candidate training had started at this point. The preceding
report-only fit diagnostic reached 31/31 correct actions on known stationary
recovery starts. That supports testing a mixed candidate; it does not establish
recovery transfer or navigation preservation.

`run_stationary_replay.py` starts from the saved
`recovery_history_probe_v1/endpoint.pt`, using a fresh AdamW optimizer, LR
`5e-5`, batch 8 and exactly 2,000 updates. Each batch contains two repeated-current
RGB recovery starts and six regular frames. The 31 starts come only from frozen
TRAIN recovery episodes. Stationary inputs keep their expert targets and receive
no image augmentation; regular frames retain the source augmentation, including
matching target changes if flipped. Bins, class weights, smoothing, dropout and
gradient clipping are preserved.

Regular frames follow the conditional 90/5/5 base/recovery/proactive recipe.
The total expected mix is 67.5% base, 28.75% recovery (including 25% stationary
starts), and 3.75% proactive. The seeded real plan has 10,749 base regular,
649 recovery regular, 602 proactive and 4,000 stationary draws: 16,000 total.
Each recovery start gets 129 or 130 practices. Private samplers do not consume
the Torch RNG used by augmentation/dropout. Actual recipe and new-phase progress
are recorded in checkpoint metadata; generic optimizer resume is blocked.

Only one candidate trains; no control model is retrained. The run scores the
unchanged warm-start model and fixed final endpoint afresh on all 39,769
validation frames and 879 frames in the six assessment clips. It also scores
the 31 stationary training starts and six stationary assessment starts using
all eight binary decisions and both mouse classes. Assessment data is evaluation
only and does not select a checkpoint or enter gradients. No epoch/step sweep,
automatic promotion, retry or budget extension occurs.

The predeclared validation screen compares the candidate with its warm-start
source: binary exact match may drop by at most 0.5 percentage points; horizontal
and vertical mouse MAE may each rise by at most 5%; first-eight-frame idle rate
may rise by at most 1 percentage point. This is an imitation regression screen,
not a live success criterion. A passing screen alone does not justify deployment;
review correct recovery actions and compare with the deployed candidate before
requesting a user-run game test.

Estimated total runtime, including fresh full before/after scoring:
**6–10 minutes**, with a **12-minute internal budget** and **14-minute outer
limit**. Runtime is not verified on the real GPU until the user executes it.
Default invocation only validates data and prints the plan:

```bash
.venv/bin/python run_stationary_replay.py
```

Run from the project directory (one line):

```bash
systemd-run --user --unit=shionn-stationary-replay-v1 --collect -p WorkingDirectory="$PWD" -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G -p RuntimeMaxSec=14min -p TimeoutStopSec=1s "$PWD/.venv/bin/python" -u run_stationary_replay.py --execute
```

Follow with `journalctl --user -u shionn-stationary-replay-v1 -f`.
Output is `models/imitation/checkpoints/runs/v5_stationary_replay_v1/`.
Require `report.json` with `candidate_complete: true`, all 2,000 updates and
complete before/after cohorts before considering `candidate.pt`. Its final name
is absent until all evaluation and deadline checks finish. A staged `pending.pt`
or an incomplete/running report is not usable evidence for a game test. Existing
output is refused; no existing model is overwritten. If a timeout occurs, stop
and review the partial report rather than automatically restarting.

Source, sidecar/progress, code, stationary pixel/label snapshots and sampled draw
hashes are recorded and checked. Cache fingerprints hash metadata/actions and
frame-file size/mtime, not all pixel contents; in-place changes with restored
file stats cannot be excluded. Completed source and data checks are required.
Preparation verification: all 256 project tests pass, including 11 new synthetic
CPU checks for batch composition, augmentation isolation, target immutability,
complete evaluation, source/progress guards, checkpoint metadata/resume and
deadline interruption even after checkpoint staging. Real default plan and
read-only protocol review pass; no candidate output was created or real model
optimizer update performed during preparation.

## Sustained recovery review — 2026-11-03

Implemented and verified the additional 22/9/6 development split enforcement
and sustained offline scorer; all 264 tests pass. Full read-only scoring of the
balanced source, deployed candidate and replay candidate completed with normal
and repeated-current histories. For the six diagnostic clips, normal-history
first 48 binary exact match improves 57.99% deployed to 63.19% replay, idle fraction
falls 7.64% to 4.51%, and backward recall improves 5/11 to 7/11 (one episode supplies
all 11 backward frames). Repeated-history backward recall remains 0/11. The nine
new recovery-validation clips were already trained on by both later candidates;
their scores are explicitly exposed-data diagnostics, not clean transfer.

The earlier 0/6 exact first-action result was too narrow to rule out useful
sustained behavior. Together with the completed full navigation regression
screen, these mixed sustained results justify a small six-attempt user-run
autonomous comparison of the existing deployed/replay checkpoints before more
training. This is a smoke comparison, not promotion evidence by itself. The
proposed clean-source paired training experiment is deferred, not scheduled.
See the [protocol and prepared live command](RECOVERY_TRANSFER_PROTOCOL_2026_11_03.md).
Only `--plan-only` was executed by the agent; the user runs all game work.

## Prepared turn-onset comparison — 2026-11-03

Status: prepared only; no real optimizer update or game action has been run.
The test1 review found late steering and active correction loops rather than
long all-zero freezes. Matching right-turn demonstrations already exist, but
the latest replay fine-tune sampled only 40 first-eight-onset frames across 20
proactive clips versus 4,000 recovery starts. This is recent-run exposure, not
lifetime exposure; existing models still reproduce much of the trained turning
signal. Sampling is a hypothesis to test, not an established cause.

`run_turn_onset_comparison.py` tests one factor with two matched arms:

- `whole_episode`: proactive frames sampled from whole usable recordings.
- `onset_window`: proactive frames sampled from
  `[max(context_start_frame, first_active_frame), turn_start_frame+8)`.

Both arms use the exact same proactive episode order and shared random fractions,
base draws and stationary-start draws. The clipped onset pools contain 8–20
frames. Only reviewed onsets are used; provisional turn-end annotations are not.
The annotation file's original integration note describes its creation-time
state; this helper now consumes the reviewed onset/context fields only.

Each batch has five base-navigation frames, two stationary recovery starts and
one normal-history proactive frame:62.5% /25% /12.5% of draws. Augmentation
applies to the six normal-history examples only. Both arms preserve source bins,
loss weights, smoothing, dropout and gradient clipping, with fresh AdamW at
LR `5e-5` and exactly 1,500 updates each. Fresh model/RNG resets isolate the arms
from source-evaluation RNG consumption. Per arm there are 7,500 base draws,
3,000 recovery-start draws and 1,500 proactive draws (125 per proactive TRAIN
episode). This deliberately replaces the earlier proposed first-start versus
48-frame recovery comparison: the observed active test1 failure prioritizes
pre-wall turn initiation while keeping recovery rehearsal fixed.

Training membership is 1,100 base + 22 recovery + 12 proactive episodes. Nine
recovery and eight proactive recordings on reserved maps 2/5/8 remain excluded
from gradients (map 8 has no proactive clips). Six assessment clips remain
diagnostic-only. The pinned initial source is the earlier balanced checkpoint,
which has no recorded overlap with either correction set. The later replay
candidate cannot substitute as this clean initialization. Source legacy split
fields are absent, so exact base/validation membership and hashes are checked;
the new experiment has its own identity and explicit held-role metadata. This
is a development recording holdout on already-known navigation maps, not unseen
layouts or an untouched final test.

Protocol: `data/annotations/turn_onset_comparison_v1.json`, referencing the
unchanged recovery development partition and pinned onset annotations. Raw CSV
hashes, cache/CSV label equality, reviewed statuses, map roles, FPS and frame
bounds are checked. Training helpers that require all 31 recovery episodes are
not used to construct this experiment's training set.

The source is evaluated once; each fixed endpoint is evaluated on all 39,769
navigation validation frames, the nine reserved recovery windows and eight
reserved proactive windows. Per-episode/per-map, equal-episode and pooled
metrics include idle runs, direction agreement, nonzero-to-zero errors,
backward recall/false positives and mouse MAE. Target/prediction traces are saved.
The six diagnostic clips are scored only after both fixed arms finish. No
evaluation result selects a step, extends training or automatically promotes
an endpoint. Review held proactive improvement, recovery tradeoffs and the same
navigation regression screen; a single improved clip or increased motion alone
does not establish a winner. This remains offline imitation, not wall escape.

Estimated total runtime: **12–18 minutes**, including source and both full
evaluations. Shared internal limit: **18 minutes**; outer limit: **20 minutes**.
Default/max updates were reduced from the initial 2,000-per-arm proposal to 1,500
to leave evaluation headroom based on the measured previous run. GPU runtime is
still an estimate. No automatic timeout extension or retry occurs. A matched
control costs an extra arm, but old controls either saw these reserved recordings
or used different sources/recipes and cannot establish this effect.

Default invocation only prints the validated plan:

```bash
.venv/bin/python run_turn_onset_comparison.py
```

Run from the project directory, as one line:

```bash
systemd-run --user --unit=shionn-turn-onset-comparison-v1 --collect -p WorkingDirectory="$PWD" -p LimitNOFILE=8192 -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=2G -p RuntimeMaxSec=20min -p TimeoutStopSec=1s "$PWD/.venv/bin/python" -u run_turn_onset_comparison.py --execute
```

Follow with `journalctl --user -u shionn-turn-onset-comparison-v1 -f`.
Output: `models/imitation/checkpoints/runs/turn_onset_comparison_v1/report.json`
and, only for a complete pair, `whole_episode/endpoint.pt` and
`onset_window/endpoint.pt` with sidecars. Existing output is refused. A staged
`pending.pt`, one arm's completed steps or an individual filename never makes
the comparison valid: require `paired_complete: true`, both 1,500-step budgets
and all declared evaluations. Publication across the two directories is not
atomic; the paired report is authoritative. Generic full-trainer optimizer
resume is blocked. No existing model is overwritten.

Preparation verification: 273 full project tests pass, including nine new CPU
synthetic checks of matching/private schedules, 5/2/1 batches, reviewed annotation
hashes and clipped windows, source/heldout boundaries, fresh optimizer states,
checkpoint metadata and incomplete-pair reporting even when expiry occurs
during final hashing. Independent protocol review passes. The real default plan
validates all declared cohorts and creates no output directory. Source/data/code
fingerprints are checked; cache metadata/actions are hashed and frame pixels
are represented by file stats rather than exhaustive content hashes.

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
