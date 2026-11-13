# Autonomous navigation readiness — 2026-10-31

## Protocol and evidence

The user ran the planned candidate/balanced comparison on test1, test11 and
test12, five attempts per model/map, interleaved in map-first order, with a
60-second policy limit. Instructions required no keyboard/mouse assistance;
the user reported starting freezes, wall stalls and occasional better turning.
No intervention was reported. Results assume those instructions were followed.

Sequence: `model_attempts/sequences/navigation_readiness/sequence_20261031_112225_795708/`.
The folder contains `sequence.jsonl`, `benchmark_summary.json`, and per-attempt
JSONL/video pairs. Candidate is `v5_recovery_proactive_v3/best.pt`; comparison
is `v5_balanced_recovery_v1/best.pt`. Default output thresholds were retained,
with no mouse cap. The sequence used one persistent mouse/process, X11 outer
Gamescope window, verified Hyprland confinement, and mouse preflight/map reload.

## Results

| Model | test1 | test11 | test12 | Total |
|---|---:|---:|---:|---:|
| Candidate | 0/5 | 2/5 | 5/5 | 7/15 |
| Balanced | 0/5 | 0/5 | 5/5 | 5/15 |

All other attempts ended at the time limit; there were no runner errors.
Median successful durations: candidate test11 12.26 seconds, candidate test12
6.81 seconds, balanced test12 7.44 seconds. Five attempts per map are a small
sample: the two additional test11 completions are encouraging, not proof of a
reliable or statistically established improvement. Test1 is a training chamber;
these results do not establish generalization to unseen layouts.

All 30 attempts passed the automatic capture checks: no recorded cursor exits,
focus losses, unknown desktop/camera observations, input submission failures,
or sampling-gap flags. Every preflight passed and every guard cleanup restored
settings. There were zero suppressed/unapplied logged policy ticks. Thus these
failures are actionable model/control evidence under observed stable capture;
sampling and submission logs still cannot prove every event reached the game.

## Starting and later idle predictions

An idle prediction here means every raw binary and mouse output is zero;
it is distinct from a nonzero movement command that fails to move the player.

- Balanced first requested a nonidle action on test11 at 12.81, 9.77, 6.94,
  24.23 and 0.48 seconds across the five repeats. Four starts were delayed
  by more than five seconds. Every test11 run later had an idle streak of
  about 29–47 seconds, and all failed.
- Candidate requested its first nonidle action on every test11 run within
  0.15 seconds. It therefore improves initiation in this sequence, while
  only two runs completed the route.
- Both models requested an action on every attempt; none remained all-zero
  for the entire attempt. Brief initial actions can still be followed by long
  idle periods and look like a freeze near the spawn to an observer.
- On test1, the candidate's longest idle streaks were 50.6, 53.4, 31.1,
  34.7 and 47.2 seconds. The major test1 failure is not solely delayed startup.
  Selected video frames show wall-facing idle states after initial movement.

First-action times are measured from the runner's elapsed timer; they include
initial capture/inference overhead. Idle streaks are reconstructed from raw
actions and consecutive tick timestamps, ending at the last logged tick.

## Visual review and diagnostic limitations

Inspected saved frames at approximately 1, 5, 15 and 55 seconds for jobs
1, 4, 8, 9, 21 and 27. These show initial movement followed by wall-facing idle,
continued forward commands near obstacles, and misalignment/large turns on
failed routes. This is sampled review, not a full collision annotation of all
videos. It agrees with the user's live observations of poor wall avoidance
and recovery, while preserving the candidate's occasional successful turns.

The built-in diagnostic counted 6 candidate and 23 balanced `policy_freeze`
episodes, plus one `turn_loop` episode per model. It counted zero `wall_stuck`
episodes for both. These are heuristic event counts, not task-failure counts:
freeze detection accepts any one-second all-zero streak, while wall detection
requires a very low image delta during movement. The zero wall count conflicts
with the visual/user evidence and must not be interpreted as no wall stalls.
Exact wall-collision and recovery durations need visual annotation or an
independently validated motion measure.

## Decision and next experiment

Navigation is not ready to serve as a dependable button/cube approach policy.
Keep puzzle training deferred. Retain the candidate for the next controlled
experiment, but do not claim it is reliably deployed or generally better from
7/15 versus 5/15. The older baseline was not included in this bounded comparison.

Prioritize #25: review initial idle and first-turn windows, compare them with
expert start/approach coverage, and freeze the turn-onset annotations. Existing
proactive clips do not need to be rerecorded merely for their leading idle
padding; training already excludes leading idle target frames. That does not
explain or cure idle predictions in later wall-facing states.

Then complete #26 with measured, separate proactive/recovery group sampling
and the selected onset strategy. Change one factor at a time: compare a matched
baseline with group sampling first, then onset/start-window boosting separately
if supported by the audit. Retain the same split, update budget and evaluation
thresholds. Do not lower thresholds, add forced forward movement, or install
an automatic unsticking routine and describe it as a learned improvement.

No additional capture-only stress run is needed on the unchanged setup. The
live comparison requirement shared by #31/#27 now has evidence; #27's explicit
turn-onset comparison and remaining acceptance criteria still need review.

## Follow-up sampling audit

Read cached training actions for all 1,151 frozen-v3 training episodes and
evaluated the existing trainer's expected sampling weights, without training
or controlling the game. After the dataset's leading-idle exclusion, there are
242,550 usable training frames: 235,214 base, 4,477 recovery and 2,859 proactive.
These are usable-frame statistics, not earlier raw-frame counts.

The candidate's saved config confirms chamber-balanced sampling, opening boost
1, opening window 8, and correction fraction 0. Under that recipe, expected
training draws are 97.01% base, 1.88% recovery and 1.11% proactive. The correction
clips are present but together receive only 2.99% of draws. Horizontal mouse
targets are zero on 48.88% of usable base frames, 32.08% of recovery and 27.67%
of proactive frames. This supports testing increased correction exposure; it
does not establish that exposure alone causes the failures.

Opening eight-frame windows account for 4.06% of baseline draws. The existing
2x opening boost increases that to 7.79%; 4x to 14.41%, retaining map balance.
The first eight usable base frames request forward movement on 88.47% of
frames on average across episodes (80.50% on test1). Start actions therefore
exist in the dataset, but label coverage alone cannot establish that training
images match live spawn/door states.

The existing combined-correction option is not independent group control:
requesting 10% corrections produces approximately 6.30% recovery and 3.70%
proactive; requesting 20% produces 12.60% and 7.40%. It cannot directly express
the intended 90/5/5 base/proactive/recovery experiment. Implement and validate
explicit group draws before claiming that recipe has been tested.

Recommended next order: verify representative start/first-turn image-label
windows; implement independent group sampling; run a matched unchanged recipe
against 90/5/5 with the same seed, split and update budget; retain a subsequent
80/10/10 trial only if the first comparison warrants it. Evaluate opening/turn
predictions and repeat the autonomous navigation protocol. Compare opening
boost separately rather than changing it in the same first experiment.

A label-only turn-onset detector was used for exploratory exposure estimates;
its windows are not visually verified annotations and are not yet suitable as
the frozen onset set required by #25.
