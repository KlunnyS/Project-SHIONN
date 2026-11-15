# Navigation development iteration — 2026-11-14

> Dated snapshot. Current status: [navigation guide](NAVIGATION.md). Current collection commands: [paired-goal protocol](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md). Later implementation may supersede statements below.

## Decision and verified state

Not ready for goal-conditioned training. The smallest missing data item is a
verified common-start pair reaching two different visible intermediate waypoints.
Repeatable start verification and an explicit projection supplier are also gates.
No game, recording, training, checkpoint or membership changes occurred.

Recent commits inspected: `83d4c55` recurrent benchmark/event correction,
`4217c1c` opt-in decoders, `a595f37` audit clarification. Optional goal adapter,
recorder safeguards and feasibility ledger are present in the worktree; do not
confuse local implementation with published commits. The independent model
review itself was not located in the repository inventory; its contents were not
used as evidence. The updated roadmap is available under docs.

## Technical evaluation readiness

`PolicyInference` retains argmax as default. Expected decoding rounds the weighted
mean, so opposing directions can cancel and small means round to zero.
Hierarchical decoding thresholds zero probability, chooses a direction by side
mass and then the strongest bin on that side; ties choose left. These are distinct
control hypotheses. None is established as a gameplay improvement by offline MAE.
The pinned-checkpoint argmax/expected test11/test12 commands in
`GOAL_CONDITIONED_NAVIGATION_PILOT_PLAN_2026_11_13.md` remain the first bounded
user screen: 12 attempts, 60 seconds each, same FPS, settle, guarded preflight and
recording settings. No default change or fresh training is warranted beforehand.
Treat it as exploratory because repeated map starts/reset order have not been
calibrated; counterbalance decoder order in a later replication. Do not silently
add hierarchical decoding to the registered two-arm comparison. If the first
screen supports another arm, preregister its threshold (0.5 initially), run order,
map/repeat budget and checkpoint fingerprint separately.

Report completion, timeouts, mouse nonzero rate, signed turn response, sign-flip
rate, overshoot and wall-facing duration, with per-map videos and invalid counts.
Define overshoot by crossing a manually reviewed intended turn/target direction,
not by magnitude alone. Jitter proxies are supporting diagnostics, not proof of
poor navigation. Physical evdev displacement versus injected relative pulses and
actual camera response remain uncalibrated; a supervised user check is needed.

`live_input_telemetry.latest()` defaults to a 0.25-second freshness limit and
invalidates stale focus. The live runner releases held actions and calls
reset_memory on unknown/lost focus; it rechecks focus before action submission.
This clears recurrent history while preserving the external goal. Episode reset
clears both. Frequent unknown focus can therefore repeatedly destroy useful
history even while preserving goal identity; measure these reset events before
attributing wandering to learned memory. Recording fault tests cover missing and
stale frames, partial EOF, writer and required-device failures and state locking.
These CPU tests do not certify hardware unplug behavior or Wayland timing.

Training uses bounded sequence windows/burn-in, while live recurrent inference
carries hidden state step by step through an episode. Offline expert sequential
scoring is covered by regression tests. The hidden-state horizon difference is
intentional but remains a distribution concern; matching offline imitation does
not validate long-horizon live memory or guarantee full-episode training exposure.

## Strategy and next implementation

Movement control: complete the user-run decoder baseline with valid capture.
Goal selection: externally declare a destination first; the present adapter does
not recognize destinations. Pathfinding: start with goal-dependent choices, then
collect alternate obstacle detours to the same destination. Multiple routes teach
robustness but cannot substitute for alternative-goal choices.

After the first pair passes, Codex should add bounded annotation-to-sequence
loading and source-hash verification, requested-goal/start metadata plumbing, and
a pose/state verification interface before a goal-aware training command. Preserve
whole trajectories for recurrent comparisons; keep architecture and data changes
separable. Goal IDs persist externally, but invisible input alone has no world
direction or unique identity embedding; temporary occlusion requires an earlier
observed goal and useful recurrent history.

Existing recovery clips retain their original training/development/diagnostic
roles. They can be retrospectively annotated for visible waypoints and reviewed
escape/resumption phases where evidence permits. They cannot become clean held-out
goal data after model selection exposure. Begin with two full recovery-to-goal
clips and reviewed negatives only after the paired goal protocol is valid. Score
escape, no immediate re-entry and resumed progress to the original goal; wandering
after escape is incomplete recovery. Progress labels need trusted displacement
or reviewed outcome windows; camera change and action magnitude are insufficient.

PPO remains gated by reset/action/observation validity, fresh terminal signals,
progress labels, time limits and measured environment throughput. Proposed reward
review: terminal target success, time cost and potential-based goal-distance
progress where valid, with obstruction penalties only after calibration. Test
oscillation reward farming, wrong-goal success, premature failure, unsafe shortcuts,
teleport/reset leakage and stationary-camera penalties. Euclidean distance can
penalize necessary detours; use validated route-aware progress or terminal rewards
until that is resolved. Any BC rehearsal during PPO must retain frozen partitions
and matched evaluation; no PPO implementation is justified in this iteration.

## Data preparation and user action

See `DATA_REQUIREMENTS_AND_COLLECTION_PLAN.md` for seven capability rows with
scenarios, labels, coverage/gaps, diversity, quantities, effort, dependencies,
priority, dates and validation, and `TWO_WAYPOINT_COLLECTION_PROTOCOL.md` for the
immediate procedure, raw paths, development-only eligibility and validator command.
The user can identify/map two target regions and prepare interaction variants
while Codex works on offline integration. The critical path is reliable controls,
verified starts, paired goals, ingestion and a bounded evaluation. Revised dates
are conditional; the older November button timetable is at risk. Keep the March
2027 demonstration narrow if advanced planning threatens the core milestone.

Next user action is the setup/target-selection step, followed by only one human
trajectory per goal after pose/capture checks pass. `SELECTED_MAP` in the command
is an explicit placeholder; no numeric world pose or chamber suitability was
invented. The command prepares recording and neither loads nor resets the map.

## Validation and changes

Created this report, the proactive data roadmap and the two-waypoint protocol.
No model architecture, runtime defaults, dataset membership or checkpoint changes
were needed. The existing feasibility ledger validates successfully and the pair
requirement rejects it. Focused CPU run:

```bash
.venv/bin/python -m unittest tests.test_mouse_bins tests.test_recurrent tests.test_benchmarking tests.test_policy_decision_rate tests.test_recording_fail_closed -q
```

Result: 37 tests passed in 11.649 seconds; output captured in
`/tmp/shionn-navigation-readiness-tests.log`. Recorder and sequence `--help`
commands verify the documented options, including all three decoders. No live
inference/capture or GPU measurements were performed. Remaining uncertainties are
real input calibration, repeatable reset, label truth/effort, target perception,
limited layout diversity and goal-conditioned action utility. No navigation
improvement is claimed.
