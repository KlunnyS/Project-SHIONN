# Recovery transfer development protocol — 2026-11-03

Supports Kanban #33 steps1–3, #25 diagnosis, #26 training and #27 evaluation.

## Decision and evidence

Stop using exact first-action matches as the sole recovery score. Five sampled
frames across each of 31 recovery and six assessment clips were visually reviewed
alongside recorded actions. The sequences show initial correction followed by
route traversal and an exit view; several first labels are small adjustments
before a much larger turn. This was sampled review, not continuous playback or
frame-precise certification of wall clearance. Contact sheets are archived with
the diagnostics. No new recording, model training or game action was performed.

Use the first 48 usable frames (two seconds at 24 FPS) of each clip as the
predeclared offline correction window, followed by the full remaining episode
as a secondary navigation-imitation window. Index ranges in the manifest have
exclusive ends. All 37 clips have at least 48 usable frames. The automatic
horizontal-onset candidate is exploratory only (abs(dx)>=3 in at least 3 of 5
frames); it is not a visually certified escape boundary or a training label.

## Scoring contract

Offline scores measure imitation on the same expert-observed states, not the
outcome of the model's proposed alternative actions. Report each episode and
map, then equal-episode aggregates alongside frame-weighted totals:

- All-zero prediction fraction and longest consecutive idle run in seconds.
- Mouse sign agreement separately per axis on expert movements with magnitude
  at least 3 counts; report the supported frame counts and abstain if none.
- Expert-nonzero/predicted-zero frequency, backward recall and false backward
  predictions, binary exact match and mouse MAE. Keep exact mouse-bin/full-action
  matching as supporting metrics, not the definition of escape.
- Normal four-frame history is primary. Repeated-current history is a separate
  sensitivity intervention; never merge the two into one success rate.
- Preserve complete frozen test11/test12 expert-frame navigation scoring.

Actual recovery success requires a user-run autonomous rollout: the agent
visibly leaves the obstructed state and resumes progress for at least one
second, or reaches the goal first. Record time to escape, subsequent relapse,
goal completion and capture validity. Motion alone, an isolated turn, exact
expert action matching, or forward input without progress is insufficient.
Video-based judgments and unavailable escape boundaries remain explicit review
items. Assistance invalidates that attempt as an autonomous score. The one-second
rule is a proposed operational criterion to apply prospectively, not a claim
that past attempts have already passed it.

## Development split and source provenance

`data/annotations/recovery_transfer_development_v1.json` records 22 recovery
training episodes, nine recovery-validation episodes and the six diagnostic
assessment episodes. All recovery clips on test2, test5 and test8 are reserved
together (three per map), reducing repeated same-chamber recovery positions
across these two recovery partitions. Original frozen-v3 roles are untouched;
the new experiment adds exclusions, rather than rewriting history.

All recent recovery candidates already trained on all 31 clips. They cannot
provide a clean initialization for this new holdout. The earlier
`v5_balanced_recovery_v1/best.pt` has 1,100 recorded base training episodes and
zero episode-ID overlap with the 31 recovery clips. Its checkpoint/sidecar hashes
and parent split digest are recorded. Initialization must pass this membership
check; renaming a holdout does not remove exposure from a warm start.

The balanced source already knows these maps through base navigation. Base-data
visual overlap has not been exhaustively excluded. This is transfer to held-out
recovery recordings, not unseen layouts or an untouched final test. Six assessment
clips have already guided diagnosis and remain diagnostic-only, outside gradients.
The timing calibration is excluded entirely.

## Concrete next experiment proposal

Before execution, implement a loader that enforces this additional manifest;
existing scripts requiring all 31 recovery training clips are incompatible.
Do not launch them with this new source or pretend their current split checks
enforce these exclusions.

Compare two fixed-budget arms from the same eligible balanced checkpoint:

1. **First-start replay:** two stationary first-active samples from the 22 TRAIN
   recoveries plus six base-navigation frames per batch.
2. **Window replay:** identical recipe, except each recovery draw is selected
   from the first 48 usable frames of those same 22 episodes, still using repeated
   newest RGB. Targets remain the corresponding recorded actions.

This isolates replay-window coverage; source, stationary representation, base
rehearsal, loss, targets and update count are shared. Use no proactive clips in
either arm for this first controlled comparison. That avoids another changing
correction mix and limits analogous correction leakage into the development
holdout. Do not describe the existing replay candidate as this matched control:
its source and recovery membership differ.

Proposed bounds: fresh AdamW, LR5e-5, batch 8, 2,000 updates per arm, private seeded
draws; no augmentation on stationary samples and identical source augmentation
on base frames. Score the unchanged source once, then both fixed endpoints on
the nine reserved recovery episodes and full navigation validation. Score the
six diagnostic clips only after both fixed endpoints, without selecting a step
or extending training based on them. Keep both endpoints for review; no automatic
promotion. A matched control is necessary here because prior control weights
already saw the new recovery holdout or used a different source/recipe.

Budget planning estimate: 12–18 minutes total including full scoring, with a
proposed 18-minute internal/20-minute outer limit. These are proposal values, not
a prepared command or measured runtime. Explain the two-arm cost before the
user launches anything. No run is scheduled, and implementation must preserve
incomplete reporting, source immutability and explicit checkpoint provenance.

Prefer window replay only if episode-balanced held-out initiation and correction
metrics improve without material navigation regression. Report ambiguous tradeoffs
instead of selecting a winner. Neither offline result proves physical escape;
only a justified candidate proceeds to the user-run navigation gate.

## Status

- [x] Review sampled sequences and explain first-frame metric limits.
- [x] Define offline windows and separate live recovery success from imitation.
- [x] Record and validate the 22/9/6 development partition and clean source membership.
- [x] Specify the changed factor, matched comparison and proposed runtime budget.
- [x] Implement and test experiment-manifest enforcement and sustained scoring.
- [x] Re-score existing candidates with sustained metrics as exposed-data diagnostics.
- [ ] Prepare the bounded comparison command after those checks pass.
- [ ] User executes the experiment; review before any game test.

## Implemented scoring and decision — 2026-11-03

`models/imitation/recovery_transfer.py` enforces the protocol and returns only
base training plus the 22 permitted recovery manifests for future training.
It rejects reserved-episode exposure and requires the pinned initial checkpoint
and sidecar hashes with matching base train/validation membership. Scoring other
checkpoints is permitted, with recorded training exposure labeled explicitly;
missing membership is unknown, not clean. Full ancestral/pixel novelty is not
established by those labels.

`benchmark_recovery_transfer.py` performs read-only inference, using decoded
runtime actions. It reports initial48, full post-idle reference and remainder
windows separately, per episode/map and with equal-episode/denominator-weighted
aggregates. Normal and repeated histories are separate. NaN/Inf logits are
rejected before argmax. Completion requires all declared cohorts and unchanged
checkpoint/config, protocol and cache fingerprints. Cache pixels themselves are
represented by file size/mtime, not exhaustive content hashes.

Validation-only example (choose a new output path):

```bash
.venv/bin/python benchmark_recovery_transfer.py --validate-only --output /tmp/recovery-protocol-validation.json
```

Scoring example:

```bash
.venv/bin/python benchmark_recovery_transfer.py --checkpoint models/imitation/checkpoints/runs/v5_stationary_replay_v1/candidate.pt --history normal --history repeated --output /tmp/recovery-window-scores.json
```

All 264 project tests passed, including eight focused protocol/metric/numerical
checks. Independent read-only review passed after the NaN decoding guard was
added. Real protocol validation and GPU inference completed; no optimizer,
capture, game input or model checkpoint write occurred. Raw final results:
`model_attempts/diagnostics/recovery_transfer_review_20261103/sustained_scores.json`.

For the six diagnostic episodes, first 48 frames each:

| Normal-history metric | Balanced source | Deployed candidate | Replay candidate |
|---|---:|---:|---:|
| Idle fraction, equal-episode mean | 6.60% | 7.64% | 4.51% |
| Binary action exact match | 59.03% | 57.99% | 63.19% |
| Backward recall | 9/11 | 5/11 | 7/11 |

The 11 backward frames all come from one episode, not 11 independent trials.
Under repeated-current intervention, replay idle falls from 45.14% deployed to
23.61%, but backward recall remains 0/11 and binary exact match falls 43.75% to
42.01%. These mixed results show some sustained imitation improvements while
leaving a stationary-input weakness. The nine newly reserved recovery clips
were already in the deployed/replay training sets, so their apparent improvements
are exposed-data diagnostics and cannot establish clean holdout transfer.

The earlier 0/6 exact first-frame result was too narrow to justify treating every
recovery action as ineffective. The sustained result plus the completed full
navigation regression screen supports a small user-run autonomous screen of
the existing checkpoints before spending on the proposed paired training run.
The clean-source training proposal remains unlaunched and is deferred pending
this screen; the scoring implementation does not itself implement that trainer.

### Next user-run screen

Compare deployed and replay once each on test1, test11 and test12 (six attempts,
60 seconds each, interleaved by map). This is a descriptive smoke comparison,
not enough attempts for a readiness/promotion decision. Preserve the deployed
checkpoint and capture settings. No manual unsticking during scored attempts;
use K/Esc only to stop an attempt if necessary. Invalid capture/runner errors
are not policy failures or success evidence. Review videos for actual escape,
progress and relapse using the prospective live criterion above.

The following command was validated with `--plan-only` (no game was opened or
controlled by the agent). The user executes the live version:

```bash
.venv/bin/python run_model_sequence.py --checkpoint deployed=models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt --checkpoint replay=models/imitation/checkpoints/runs/v5_stationary_replay_v1/candidate.pt --map dataset_test1 --map dataset_test11 --map dataset_test12 --repeats 1 --max-seconds 60 --keep-focused --hyprland-confine-pointer --persistent-mouse --no-launch --verbose --recording-root model_attempts/sequences/stationary_replay_screen
```

Allow roughly 8–12 minutes including map loads/preflights. If the input guard
stops the sequence, retain its logs for review and do not count it as a clean
model comparison. No automatic promotion or additional training follows.

## User-run replay screen — 2026-11-03

Completed sequence: `model_attempts/sequences/stationary_replay_screen/sequence_20261103_181134_301957`. The earlier `sequence_20261103_180958_942772` contains no result rows and is not combined with the completed screen.

| Model | test1 | test11 | test12 | Total |
|---|---|---|---|---|
| Deployed | time limit | time limit | goal6.73s | 1/3 |
| Replay | time limit | goal35.43s | goal6.81s | 2/3 |

All six attempts passed automatic capture checks and mouse preflight; no recorded focus losses, cursor exits or unknown observations; guards restored. This does not prove every submitted event reached the game. Autonomy follows the supplied hands-off protocol; no intervention was reported. One attempt per model/map is insufficient for promotion or navigation readiness.

Tick-derived longest all-zero output streaks: deployed/replay test1 approximately0.17/0.21s, deployed test11 34.42s (25.58s through60s), replay test11 0.14s. Six sampled video frames per attempt1–4 show the deployed test11 wall-facing stall and replay reaching the exit. On test1, both remain or revisit wall-facing states while requesting nonzero actions. Therefore these test1 failures cannot be explained as a prolonged all-zero policy freeze. Sampled frames do not certify exact wall-clearance times or continuous recovery events.

Decision: keep replay as a promising experimental candidate, preserve deployed, and prioritize reviewing active wall-facing/backtracking behavior on test1. Do not automatically promote, run another capture-only check or increase training merely because2/3 exceeds1/3. Clean-source window-coverage experiment remains a proposal; its scorer and partition enforcement are ready, but its trainer is not yet implemented.

## Test1 diagnosis — 2026-11-03

Reviewed two-second video samples through both60-second test1 attempts and
five-second action summaries. Both start by walking toward the first obstacle;
replay requests W on every logged tick and zero horizontal mouse throughout
seconds1–4. Later both revisit wall/floor/stair views while turning/strafing.
Approximately43.0% deployed and41.25% replay ticks request abs(dx)=134, their
largest decoded horizontal magnitude. These are requested commands, not measured
yaw or a validated count of bad turns. Sparse frames do not certify collision
or escape times. This supports investigating late pre-wall steering and unstable
correction rather than attributing this run to prolonged all-zero outputs.

The four existing test1 proactive clips were inspected at context, annotated
onset,+8 and+24 frames. Two right-turn examples show the wall/stairs approach
seen in the failed rollout; corresponding demonstrations exist, so additional
recording is not currently justified. This is visual situation similarity,
not exact pose equivalence or proof the labels are optimal.

Reconstructed the completed replay run's private regular draw sequence using
its unchanged recipe/seed:602 proactive regular draws total; only26 fall within
the annotated12-frame pre-onset contexts and40 within the first8 frames after
onset, across all20 clips. The four test1 clips get3 context and4 onset draws;
one right-turn clip gets neither. Compare4,000 explicitly replayed recovery
starts. These counts describe this fine-tune only, not lifetime exposure: the
source models already trained on proactive data. Provisional turn endpoints
were not used for this count.

Read-only inference on all20 reviewed onset windows (first8 frames) finds
horizontal sign agreement119/156 deployed versus121/156 replay, with40 versus37
expert-nonzero horizontal targets decoded as zero. Both models agree in sign
on6/8 significant frames in each test1 right-turn demonstration, with2/8 zero
outputs. This suggests initiation remains imperfect even on trained examples;
it does not prove lack of data, identify the causal source of rollout failure,
or establish that simple oversampling will solve it.

Next hypothesis: explicit replay of reviewed approach/turn-onset windows,
retaining navigation and recovery rehearsal. Before any launch, define the
comparison and preserve source/holdout provenance; the existing22/9/6 split
cannot make the already-exposed replay model clean. If using the clean balanced
source, additionally reserve proactive clips on the reserved maps rather than
leaking analogous correction examples across that development split. No new
training command has been prepared or run. Do not impose an arbitrary live
mouse cap as a substitute for learned steering.

## Prepared comparison handoff

The next hypothesis is now concretely implemented in
`run_turn_onset_comparison.py`, with a pinned separate experiment manifest
`data/annotations/turn_onset_comparison_v1.json`. It compares uniform proactive
recording draws against reviewed onset-window draws, holding the 5/2/1 batch
mix, source and episode/base/recovery schedules fixed. The clean source allows
22 recovery and 12 proactive TRAIN clips; nine recovery/eight proactive clips
remain reserved, and six diagnostics remain out of gradients. This supersedes
the earlier proposed recovery-window-only comparison; it has not been run.

The budget is 1,500 updates per arm, estimated 12–18 minutes including evaluation,
with 18-minute internal/20-minute outer limits. It saves fixed endpoints for manual
review, never promotes one automatically, and rejects incomplete comparisons.
See the [complete runbook and command](TRAINING_RUNBOOK.md#prepared-turn-onset-comparison--2026-11-03).
