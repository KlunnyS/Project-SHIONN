# Startup timing review — 2026-11-06

## Evidence and change

The same deployed checkpoint has three goal_reached model segments in assisted-collection sessions on2026-11-05, while earlier scored test1 attempts timed out. These collection segments are not scored autonomous successes and cannot be combined with scored totals; human-input absence is not independently certified by the logs.

Archived scored failure: `model_attempts/sequences/stationary_replay_screen/sequence_20261103_181134_301957/001_deployed_dataset_test1_r1/attempt_20261103_181144_251321.jsonl`.
Collection example: `model_attempts/targeted_navigation_20261105/sessions/session_20261105_161705_075500/001_dataset_test1/attempt_20261105_161715_596402.jsonl`.

Initial reported camera angles are approximately pitch0/yaw90 in both. First five forward probabilities are[.111,.786,.281,.333,.925] in the archived failure versus[.931,.944,.935,.925,.995] in the collection example. Initial video views differ in entry-door phase. These observations motivate a startup investigation but do not prove timing caused success; date, context, input history, capture conditions and collection/scoring safeguards differ.

Code inspection found a real protocol discrepancy: initial countdown occurs before mouse preflight. Preflight then reloads the map, restores focus and waits only0.25seconds before policy control. Recovery-collection mode has no preflight reload and receives its countdown after the final load.

The initial wait is preserved for capture/preflight readiness. After the final preflight reload, run_imitation now waits max(0.25,countdown) by default. `--post-preflight-settle` explicitly overrides that delay in both runner and sequence CLI;0.25reproduces the old final-settle timing. Values must be finite/nonnegative and require mouse preflight. Existing yaw preflight, confinement/focus guards and thresholds remain enabled. Focus is queried again after the wait, and cancellation/focus loss sends no policy actions. No-preflight collection startup is unchanged.

Metadata records startup timing version2, initial countdown and requested final settle. startup_settle events record requested and elapsed wait. This is a timing correction, not verification that every door animation or capture pipeline has finished. Extra settling also consumes chamber-script wall-clock time; keep the same maps/budgets and report timeout provenance.

## Engineering verification

All311tests pass, including final reload→settle→policy ordering, default/override forwarding, invalid inputs, cancellation and focus loss before model control. Independent read-only review passes. The exact three-attempt command below passed plan-only validation. No gameplay has been launched by the agent and no completion improvement is claimed.

## User-run baseline

With Portal2 already running onDP-1, use the same SSH environment setup as before. The current home Wayland socket was observed aswayland-1. Physical keyboard/mouse takeover is not required for this autonomous scored test. Do not interact with the game during attempts. Stop on capture/preflight failure; do not disable guards.

```fish
set -gx XDG_RUNTIME_DIR /run/user/(id -u)
set -gx WAYLAND_DISPLAY wayland-1

.venv/bin/python run_model_sequence.py \
  --checkpoint deployed_settled=models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --repeats 3 \
  --max-seconds 60 \
  --countdown 3 \
  --post-preflight-settle 3 \
  --output DP-1 \
  --persistent-mouse \
  --keep-focused \
  --hyprland-confine-pointer \
  --no-launch \
  --verbose \
  --recording-root model_attempts/sequences/startup_settle_20261106
```

Expected4–6minutes including loading, at most3minutes of policy control. Each attempt may end early on goal/failure. Keep this timing-v2cohort separate from earlier scoring. The first purpose is to verify startup behavior and capture validity, not establish causality or promote a model. A controlled old-vs-new delay comparison may follow only if these results justify it.

Some historical tests were stored in/tmpand are no longer present after restart; the new recording root is durable project storage. Do not discard existing collection recordings or change dataset roles. No new neural training or checkpoints are involved.

## First user-run timing-v2baseline

Sequence: `model_attempts/sequences/startup_settle_20261106/sequence_20261106_114944_247736`. All3capture-valid, no focus losses, all yaw preflights passed. Measured final settle3.00015/3.00024/3.00028s. Outcomes: timeout, goal_reached8.437s, timeout. This cohort is separate from historical timing-v1scoring.

First forward probabilities are approximately0.93in every attempt (first five all>=0.919), and forward movement starts immediately. Dense early trajectory samples show the two failures approach the first wall and stop; the successful attempt turns onto the stairs and crosses the exit. Longest approximate all-zero-policy runs54.67/0.04/22.17s. These are later policy stalls rather than an initial failure to move. A three-run cohort does not prove a causal success-rate gain from settling, nor reliable navigation.

Next predeclared baseline extension: source checkpoint only, same timing-v2settings,3attempts each on test11andtest12. Keep all6capture-valid/invalid outcomes and record all thresholds/settings unchanged. Test11remains validation-only; test12is the regression check. No new training or model promotion. This establishes the updated startup baseline across the existing evaluation chambers before choosing further policy changes.

## Validation extension — 2026-11-09

Sequence `model_attempts/sequences/startup_settle_validation_20261109/sequence_20261109_125621_643215` completed all6attempts after SSH display environment and Gamescope pointer-grab recovery. All6capture-valid, no focus losses, no capture review required. Test11:0/3, all time limits. Test12:3/3, median successful duration6.189s. Earlier launch/capture/preflight failures are invalid setup attempts and are not combined with scored results.

Together with the separate2026-11-06timing-v2test1cohort, updated baseline is test1:1/3, test11:0/3, test12:3/3. Dates differ and the cohorts remain separately recorded. The settle change does not establish reliable navigation or a causal success improvement. No further broad repetition or model promotion is justified from these results; inspect latest test11 failure trajectories and retain targeted human recovery collection for desktop availability.
