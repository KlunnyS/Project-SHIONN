# Goal-Directed Navigation Audit — 2026-11-13

## Scope and source limits

Audited the worktree beginning at commit `45d07a6` plus the user-provided,
currently untracked roadmap `SHIONN_Goal_Directed_Navigation_Roadmap_v2.md`.
The roadmap refers to a 15-page independent model review, but that PDF was not
present in the repository or available attachment directory. Findings below
verify summarized claims against code; they cannot check the reviewer's full
evidence or page-level context. The review itself was static-only and did not
run tests, training, or gameplay. At the first inspection the cache mount
appeared empty; it is now populated with 1,300 manifests, including all 200
validation episodes saved in the GRU checkpoint. The completed real-cohort
scores and their limits are documented below.

## Review issue status

| Review area | Current evidence | Status |
|---|---|---|
| Mouse zero-bin bias | `PolicyInference` originally selected the maximum individual bin (`models/imitation/inference.py`); zero could beat each movement bin despite lower combined movement probability. Existing `GROUP_SAMPLING_COMPARISON_2026_10_31.md` reports global zero gates worsened horizontal MAE and raised false movement. A new post-hoc comparison on saved validation probabilities found weighted-expected decoding modestly lowers MAE, while increasing movement frequency beyond expert rates on several axes. | Argmax remains the default. `argmax`, `expected`, and `hierarchical` are now opt-in decoders; expected is an offline candidate only. No closed-loop benefit is established; run a user-operated paired screen before considering any default change. |
| Spatial augmentation | Trainer supports brightness/contrast and horizontal flip, but no spatial shift (`models/imitation/train_bc.py`, `models/imitation/recurrent.py`). | Absence confirmed; overfitting explanation untested. |
| Ordinal mouse loss | Binned mouse heads use categorical targets; nearby bins are not treated as closer by the loss (`models/imitation/train_bc.py`). | Limitation confirmed; benefit of ordinal targets unknown. |
| EMA | No EMA state/update or EMA checkpoint selection was found in the trainer. | Appears absent; no evidence it would improve navigation. |
| Data-loader throughput | Worker count is configurable (`--workers`, `models/imitation/train_bc.py`). No current paired profile justifies a throughput change. | Partially present; speed remains unmeasured. |
| Fragmented console events | `record_dataset.py` already buffered partial lines. Live inference and `EpisodeRecorder.run()` previously classified raw chunks independently; this audit centralizes parsing in `episode_events.py` and uses it in both paths. The P handoff now waits up to 100 ms to finish an already-started line. Retained partial lines are capped at 64 Ki characters; an overlong line is discarded through its next delimiter. | Dataset, policy, handoff, and low-level recorder paths preserve split lines and have focused tests, including overlong-line recovery. Generic `Portal2Controller.check_event()` remains a legacy chunk-local buffer, but has no current production call sites. |
| Focus-unknown memory reset | The live guard releases controls and resets policy state when focus telemetry is unknown (`run_imitation.py`). | Behavior confirmed; whether retaining memory safely would improve navigation is unknown. Keep input release safety; measure before changing reset semantics. |
| Mouse delivery parity | Training labels come from physical evdev deltas; live control sends one relative pulse through the controller. | Paths differ; no controlled Proton/Wayland calibration demonstrates a harmful mismatch. |
| Recorder reliability | `.in_progress` publication and timing/discard metadata exist, but some recorder paths can leave partial/error conditions for later review rather than automatically invalidating them. | Safeguards are partial; audit and fault-injection are still needed before new large recording batches. |
| Recurrent expert benchmark | `benchmark_expert.py` previously called the policy on flat frame batches without carrying/resetting episode state. The GRU requires ordered observations and episode-local hidden state (`models/imitation/recurrent.py`). This audit changes scoring to use `PolicyInference.predict_stacked_tensor()` in order and reset between episodes. | Corrected score completed for both GRU and deployed checkpoints on the same saved validation split: 200/200 episodes and 39,769 eligible frames each. Reports and per-frame output are in `model_attempts/benchmarks/{gru,deployed}_validation_corrected_20261113/`. These are expert action scores, not autonomous performance evidence. |
| Other tooling/configuration | A complete independent audit of checkpoint overwrite, split defaults, wrapper waits, atomic writes, and dashboard recipes could not be performed from the missing review PDF. | Unknown; inspect each claim before changing it. |

None of the structural findings establishes the cause of poor closed-loop
navigation. The expected mouse decoder is available only as an opt-in
experiment; no decoder, recurrent horizon, augmentation, model, or checkpoint
was promoted by this audit.

## Existing capabilities and missing interfaces

- The base policy already has a residual visual encoder and action heads; an
  optional observation-conditioned gate and optional GRU are available.
- GRU training and inference exist, but the current trainer uses short
  sequence chunks (default length 8, burn-in 4). This is a design limitation,
  not demonstrated evidence that a longer horizon improves play.
- `PolicyInference` accepts image observations and internal history only. The
  cached dataset contains frames, actions, episode identity, and map name; it
  has no goal, waypoint, distance-to-target, or subgoal-success labels.
- No goal-conditioned policy path or progress/stuck training targets were
  found. No PPO trainer, game-environment adapter, validated navigation reward,
  or deterministic scenario-reset interface is documented as implemented.
- Existing navigation experiments show repeatable failures on difficult
  routes and some familiar-map successes; they do not establish general
  goal-directed behavior or unseen-layout transfer. Keep test11 validation-only
  and preserve the existing frozen cohort roles.

## Work completed in this audit

1. `benchmark_expert.py` now scores recurrent checkpoints in full episode
   order, advances memory on every observation including leading idle frames,
   resets hidden state at episode boundaries, and reports only the established
   non-idle scoring range. Feedforward scoring uses the same action decoder as
   live inference, including optional jump/forward thresholds and mouse
   decoder selection.
2. `PolicyInference.predict_stacked_tensor()` exposes the one-observation
   recurrent step for cached-frame evaluation; `decode_outputs()` shares the
   action decoding behavior between live inference and both benchmark paths.
   Binned mouse decoding now supports opt-in `argmax`, `expected`, and
   `hierarchical` modes; `argmax` remains the default. Live runner, sequence
   runner, benchmark CLI, JSONL metadata, and CLI reference carry the choice.
3. `episode_events.py` is now the shared partial-line parser for the dataset
   recorder, live runner, and low-level recorder. It caps an incomplete line at
   64 Ki characters and discards an overlong line through its delimiter. The
   handoff path performs a bounded drain for partial lines. Tests split
   terminal events across every boundary, cover a handoff whose failure line
   finishes on the next read, and verify overlong-line resynchronization.
4. The focused decoder/inference/sequence/pipeline/event suite passed: 133
   tests. The full regression suite passed: 406 tests in 25.996 seconds. These
   tests do not validate real Proton or Wayland behavior.
5. The actual checkpoint
   `temporal_readout_comparison_v1/gru/endpoint.pt` loaded and completed the
   recurrent scorer on two synthetic episodes, producing two scored rows with
   separate episode resets. The report recorded checkpoint SHA-256
   `338906faa5328ccbe6ace33b946b6adce1775c2cb5487ed2cecb237fd2a1f41a`.
   This validates the real architecture path, not cohort accuracy or navigation
   behavior. A focused decoder-threshold test also confirmed that benchmark
   scoring honors the same optional threshold as live inference.
6. Using the real checkpoint and cached frames from one saved validation
   episode per map, the benchmark's stacked-tensor path and live `predict()`
   path produced identical actions, binary probabilities, mouse-bin
   probabilities, and GRU hidden state for 16 ordered frames on each of
   test11 and test12.
7. A regression test now checks the benchmark report's checkpoint SHA-256
   against a known file payload. Current hashes are GRU
   `338906faa5328ccbe6ace33b946b6adce1775c2cb5487ed2cecb237fd2a1f41b` and
   deployed `2ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483`.
8. Both checkpoint validation scores used the exact same saved 200-episode
   split (test11: 23,952 frames; test12: 15,817). The deployed checkpoint
   scored 82.40%/87.93% exact binary match and mouse MAE (dx,dy) of
   (7.51,2.50)/(4.44,2.16). The GRU scored 82.19%/87.43% and
   (8.21,2.59)/(4.92,2.23). Current checkpoint hashes and full reports are in
   the paired benchmark folders above. CPU scoring took 29m37s deployed and
   31m00s GRU (four PyTorch threads; CUDA unavailable). Action matching is not
   navigation success, and these results do not justify GRU promotion.
9. Post-hoc decoding of saved per-bin probabilities on the unchanged cohort
   lowered expected-decoder mouse MAE for both checkpoints on both maps. For
   deployed, dx/dy MAE changed from 7.51/2.50 to 7.24/2.37 on test11 and
   4.44/2.16 to 4.38/2.02 on test12. Its predicted dx nonzero rate rose from
   30.8% to 65.0% on test11 and 17.2% to 45.6% on test12 (expert rates are
   43.2% and 28.3%). This could increase overshoot; it is an offline hypothesis
   only. The complete argmax/expected/hierarchical sweep is
   `model_attempts/benchmarks/mouse_decoder_posthoc_20261113.json`. No live A/B
   was run.

## Dependency-ordered development plan

| Order | Milestone and surfaces | Acceptance evidence | Cost / constraints |
|---|---|---|---|
| 0A | Correct episode-sequential GRU scoring, share action decoding, and compare one-step outputs against live inference. Re-audit generic console helpers such as `Portal2Controller.check_event()` before relying on them. | Complete: event fragmentation/bounds tests; 200 episodes scored for deployed and GRU on identical saved validation membership; hashes/configs recorded; exact action/probability/hidden-state parity verified on real cached frames. | CPU inference took 29m37s deployed and 31m00s GRU for 39,769 rows each. No training or game control. |
| 0B | Evaluate mouse decoding without training. `argmax` stays default; optional `expected` and `hierarchical` are implemented and swept against saved per-bin probabilities. | Offline candidate metrics are available. Next acceptance gate is a user-run matched autonomous diagnostic screen with frozen starts and attempt count; compare completion/timeouts, start freeze, wall stalls, turn direction, recovery, and overshoot. Do not promote from offline fit. | Post-hoc sweep took seconds; full CPU scoring was already paid for in 0A. Live A/B is user-operated and remains diagnostic on validation maps. |
| 0C | Audit input/recorder safety: focus-unknown release/reset behavior, physical input edge cases, missing-frame/error publication, and live mouse pulse calibration. | Fault-injection tests prove failed/stale input and incomplete captures are visible and safely invalidated. Hardware calibration reports yaw-per-input under the actual game setup; no scale change without evidence. | Software checks are CPU-only; mouse calibration requires a user-operated live session. No new dataset until recorder gate passes. |
| 1 | Produce a trustworthy, read-only navigation baseline on the deployed checkpoint. Reconstruct idle periods, first turns, wall-facing loops, repeated paths, and route progress from valid logs/videos. | Frozen checkpoint/config hashes; capture-valid attempts; per-map completion and failure categories; visually checked diagnostic labels; no contamination of validation/diagnostic episodes. | Offline diagnostics are bounded by data volume. Any scored game sequence needs user operation and a preregistered attempt budget. |
| 2 | Goal-conditioned navigation pilot. First create a goal interface and data protocol (initially a visually specified destination or annotated relative waypoint), then add goal inputs to policy/inference without changing existing default behavior. Include same observation with different goals to test whether the policy uses the goal. | Matched baseline/candidate on held-out starts and layouts: destination completion, goal persistence, blocked-route detours, and regressions. Demonstrate goal conditioning changes actions appropriately for same-view/different-goal cases. | Small code and CPU tests first; training size, VRAM, and wall time require a measured pilot profile. Do not begin before labels and evaluation starts are frozen. |
| 3 | Progress-aware recovery, conditioned on the same persistent goal. Define labels for blocked motion, purposeful escape, resumed progress, and false alarms; test a small predictor before changing the action policy. | Calibrated held-out state predictions plus unassisted recovery that returns to the original target on excluded starts. | Dataset annotation and offline fitting first; live evaluation remains separate. No scripted unsticking. |
| 4 | PPO only after scenario resets, terminal handling, reward, and actual environment steps/sec are validated. Start from a goal-conditioned imitation policy and preserve a BC retention comparison. | Matched imitation-only vs imitation-plus-PPO results on frozen navigation starts with no unacceptable regressions. | Compute cannot be responsibly estimated until environment throughput and reset reliability are measured. RTX 4060 8GB is a hard profiling constraint. |
| 5–6 | Add high-level subgoals and button/cube interactions only after navigation and progress gates pass. Validate subgoal outcomes and changed layouts before any unfamiliar-puzzle claim. | Multi-step goal sequence with observed interaction-state verification on held-out variants. | March 2027 remains a conditional target; reduce puzzle scope if earlier gates consume time. |

## Implementation handoff: files, tests, and bounded cost

| Work item | First files and tests | Bounded work / cost estimate | Acceptance gate |
|---|---|---|---|
| Phase 0A complete | `benchmark_expert.py`, `models/imitation/inference.py`; event fixes in `episode_events.py`, `record_dataset.py`, `run_imitation.py`, `recorder.py`. Tests: `tests/test_benchmarking.py`, `tests/test_imitation_pipeline.py`, `tests/test_record_dataset.py`. | CPU-only inference; 29m37s deployed and 31m00s GRU. No training or game control. | Same 200 episodes/39,769 rows, matching decode, state reset/parity, checkpoint hashes, reports preserved. |
| Phase 0B live validation pending | Optional decoder code in `models/imitation/inference.py`, flags through `run_imitation.py` and `run_model_sequence.py`, report metadata in `benchmark_expert.py`; `tests/test_mouse_bins.py`, `tests/test_benchmarking.py`, `tests/test_run_model_sequence.py`. Post-hoc output: `model_attempts/benchmarks/mouse_decoder_posthoc_20261113.json`. | No training. A 2-map × 2-decoder × 3-repeat pilot at 60 seconds per attempt has a 12-minute policy-time cap, plus capture/startup overhead. User runs it only after confirming game/capture readiness. | Argmax remains default. Keep expected only if autonomous completion/recovery improves without unacceptable stalls, overshoot, or regressions; this small screen cannot justify checkpoint promotion. |
| 0C input/recording audit | `run_imitation.py`, `recorder.py`, `record_dataset.py`, controller/input modules; `tests/test_focus_guard.py`, `tests/test_live_input_telemetry.py`, `tests/test_capture_frame_timing.py`, `tests/test_record_dataset.py`. | CPU fault-injection tests are bounded. Mouse yaw-per-input calibration requires a separate user-operated game session; no remote game run is part of this work. | Focus loss, stale telemetry, disconnect and incomplete captures release controls or visibly invalidate data; hardware calibration is reported separately. |
| Phase 1 read-only baseline | Existing attempt logs, sequence manifests, `benchmark_sequence.py`, and focused diagnostic scripts under `model_attempts/`. | Offline analysis only; runtime scales with selected logs/videos. Freeze model/config hashes and a small set of test1, test11 and test12 attempts before visual review. | Publish per-map completion/failure categories, startup idle, wall-stall/off-route/recovery timing, and reviewed route-progress examples. Exclude training-exposed runs from clean holdout claims. |
| Phase 2 goal-conditioned pilot (later) | Likely extension points: `models/imitation/network.py`, `models/imitation/recurrent.py`, `models/imitation/dataset.py`, `models/imitation/train_bc.py`, and `models/imitation/inference.py`; add isolated `tests/test_goal_conditioning.py` only after label and goal-interface design is frozen. | Do not train now. First audit eligible episodes and annotate a small pilot; measure a 5-minute GPU throughput/VRAM profile before estimating a bounded training run on the RTX 4060 8GB. | Same observation with different goals changes behavior appropriately; matched held-out starts show destination progress and no unacceptable regression. |
| Phase 3 recovery, then PPO | Progress labels and auxiliary heads should extend the selected goal-conditioned data/model path; PPO only after a resettable environment adapter exists. | Dataset annotation and offline fitting are measurable after Phase 2. PPO budget is unknown until resets, reward validation, and environment steps/second are measured. | Recovery resumes the original goal without human input; PPO beats an imitation-only baseline on frozen starts while retaining prior skills. |

### Immediate next implementation milestone

Next, the user may run the paired argmax-versus-expected diagnostic screen on
test11/test12 using the new `run_model_sequence.py --mouse-decode` option.
Review completion and failure categories, not just action metrics. Then proceed
through 0C and Phase 1 before goal-conditioned training. Do not begin
goal-conditioned training until a trustworthy navigation baseline establishes
which goal interface and destination labels can be evaluated. Preserve the
deployed checkpoint and all existing split membership. No training, live game
control, recording, or checkpoint promotion was authorized or performed.
