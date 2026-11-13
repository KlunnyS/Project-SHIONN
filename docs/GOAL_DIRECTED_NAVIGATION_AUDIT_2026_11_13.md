# Goal-Directed Navigation Audit — 2026-11-13

## Scope and source limits

Audited the worktree at commit `45d07a6` plus the user-provided, currently
untracked roadmap `SHIONN_Goal_Directed_Navigation_Roadmap_v2.md`. The roadmap
refers to a 15-page independent model review, but that PDF was not present in
the repository or the available attachment directory. The findings below can
verify summarized claims against code; they cannot independently check the
reviewer's complete evidence or page-level context. The review itself was
static-only and did not run tests, training, or gameplay.

## Review issue status

| Review area | Current evidence | Status |
|---|---|---|
| Mouse zero-bin bias | `PolicyInference` still selects the maximum individual bin (`models/imitation/inference.py`); a zero bin can beat each movement bin despite lower combined movement probability. | Structural possibility confirmed; behavioral cause and cited activity rates unverified. Keep `argmax` as baseline. |
| Spatial augmentation | Trainer supports brightness/contrast and horizontal flip, but no spatial shift (`models/imitation/train_bc.py`, `models/imitation/recurrent.py`). | Absence confirmed; overfitting explanation untested. |
| Ordinal mouse loss | Binned mouse heads use categorical targets; nearby bins are not treated as closer by the loss (`models/imitation/train_bc.py`). | Limitation confirmed; benefit of ordinal targets unknown. |
| EMA | No EMA state/update or EMA checkpoint selection was found in the trainer. | Appears absent; no evidence it would improve navigation. |
| Data-loader throughput | Worker count is configurable (`--workers`, `models/imitation/train_bc.py`). No current paired profile justifies a throughput change. | Partially present; speed remains unmeasured. |
| Fragmented console events | `record_dataset.py` already buffered partial lines. Live inference and `EpisodeRecorder.run()` previously classified raw chunks independently; this audit centralizes parsing in `episode_events.py` and uses it in both paths. The P handoff now waits up to 100 ms to finish an already-started line. | Dataset, policy, handoff, and low-level recorder paths now preserve split lines and have focused tests. Generic `Portal2Controller.check_event()` remains a legacy chunk-local buffer, but has no current production call sites. |
| Focus-unknown memory reset | The live guard releases controls and resets policy state when focus telemetry is unknown (`run_imitation.py`). | Behavior confirmed; whether retaining memory safely would improve navigation is unknown. Keep input release safety; measure before changing reset semantics. |
| Mouse delivery parity | Training labels come from physical evdev deltas; live control sends one relative pulse through the controller. | Paths differ; no controlled Proton/Wayland calibration demonstrates a harmful mismatch. |
| Recorder reliability | `.in_progress` publication and timing/discard metadata exist, but some recorder paths can leave partial/error conditions for later review rather than automatically invalidating them. | Safeguards are partial; audit and fault-injection are still needed before new large recording batches. |
| Recurrent expert benchmark | `benchmark_expert.py` previously called the policy on flat frame batches without carrying/resetting episode state. The GRU requires ordered observations and episode-local hidden state (`models/imitation/recurrent.py`). This audit changes scoring to use `PolicyInference.predict_stacked_tensor()` in order and reset between episodes. | Code defect fixed; synthetic test and an actual GRU checkpoint scored against a two-episode synthetic cache passed on CPU. The mounted cached-frame directory has no episodes, so real-cohort scoring remains pending. |
| Other tooling/configuration | A complete independent audit of checkpoint overwrite, split defaults, wrapper waits, atomic writes, and dashboard recipes could not be performed from the missing review PDF. | Unknown; inspect each claim before changing it. |

None of the structural findings establishes the cause of poor closed-loop
navigation. No mouse decoder, recurrent horizon, augmentation, model, or
checkpoint was promoted by this audit.

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
   non-idle scoring range. Feedforward scoring uses the same decoder as live
   inference, including optional jump and forward-movement thresholds.
2. `PolicyInference.predict_stacked_tensor()` exposes the one-observation
   recurrent step for cached-frame evaluation; `decode_outputs()` shares the
   action decoding behavior between live inference and both benchmark paths.
3. `episode_events.py` is now the shared partial-line parser for the dataset
   recorder, live runner, and low-level recorder. The handoff path performs a
   bounded drain for partial lines. Tests split terminal events across every
   boundary and cover a handoff whose failure line finishes on the next read.
4. Focused tests passed: 116. Full regression suite passed: 403 tests in
   27.125 seconds. These tests do not validate real Proton/Wayland behavior.
5. The actual checkpoint
   `temporal_readout_comparison_v1/gru/endpoint.pt` loaded and completed the
   recurrent scorer on two synthetic episodes, producing two scored rows with
   separate episode resets. The report recorded checkpoint SHA-256
   `338906faa5328ccbe6ace33b946b6adce1775c2cb5487ed2cecb237fd2a1f41a`.
   This validates the real architecture path, not cohort accuracy or navigation
   behavior. A focused decoder-threshold test also confirmed that benchmark
   scoring honors the same optional threshold as live inference.

## Dependency-ordered development plan

| Order | Milestone and surfaces | Acceptance evidence | Cost / constraints |
|---|---|---|---|
| 0A | Run an actual GRU checkpoint through the corrected scorer on the actual held-out cache and compare its one-step outputs to live inference on identical ordered frames. Re-audit generic console helpers such as `Portal2Controller.check_event()` before relying on them. | Real cohort scoring completes with per-episode resets, matching decode, checkpoint/config identity, and no cross-episode state. Preserve old reports with qualification notes if a scoring path changed. | Code and CPU tests are bounded. The real checkpoint passed a tiny synthetic CPU score; the cached-frame directory is empty, so real-cohort score runtime cannot yet be measured. No gameplay or training. |
| 0B | Evaluate mouse decoding without training. Add opt-in hierarchical/expected decoders only after recording unchanged-cohort argmax rates and per-axis errors. | Synthetic probability tests, unchanged frozen cohort, per-map nonzero rate/direction/MAE/jitter results, then a user-run paired autonomous comparison before selecting a decoder. | Offline inference only; exact wall time must be measured on the selected cache/device. Live A/B remains user-operated. |
| 0C | Audit input/recorder safety: focus-unknown release/reset behavior, physical input edge cases, missing-frame/error publication, and live mouse pulse calibration. | Fault-injection tests prove failed/stale input and incomplete captures are visible and safely invalidated. Hardware calibration reports yaw-per-input under the actual game setup; no scale change without evidence. | Software checks are CPU-only; mouse calibration requires a user-operated live session. No new dataset until recorder gate passes. |
| 1 | Produce a trustworthy, read-only navigation baseline on the deployed checkpoint. Reconstruct idle periods, first turns, wall-facing loops, repeated paths, and route progress from valid logs/videos. | Frozen checkpoint/config hashes; capture-valid attempts; per-map completion and failure categories; visually checked diagnostic labels; no contamination of validation/diagnostic episodes. | Offline diagnostics are bounded by data volume. Any scored game sequence needs user operation and a preregistered attempt budget. |
| 2 | Goal-conditioned navigation pilot. First create a goal interface and data protocol (initially a visually specified destination or annotated relative waypoint), then add goal inputs to policy/inference without changing existing default behavior. Include same observation with different goals to test whether the policy uses the goal. | Matched baseline/candidate on held-out starts and layouts: destination completion, goal persistence, blocked-route detours, and regressions. Demonstrate goal conditioning changes actions appropriately for same-view/different-goal cases. | Small code and CPU tests first; training size, VRAM, and wall time require a measured pilot profile. Do not begin before labels and evaluation starts are frozen. |
| 3 | Progress-aware recovery, conditioned on the same persistent goal. Define labels for blocked motion, purposeful escape, resumed progress, and false alarms; test a small predictor before changing the action policy. | Calibrated held-out state predictions plus unassisted recovery that returns to the original target on excluded starts. | Dataset annotation and offline fitting first; live evaluation remains separate. No scripted unsticking. |
| 4 | PPO only after scenario resets, terminal handling, reward, and actual environment steps/sec are validated. Start from a goal-conditioned imitation policy and preserve a BC retention comparison. | Matched imitation-only vs imitation-plus-PPO results on frozen navigation starts with no unacceptable regressions. | Compute cannot be responsibly estimated until environment throughput and reset reliability are measured. RTX 4060 8GB is a hard profiling constraint. |
| 5–6 | Add high-level subgoals and button/cube interactions only after navigation and progress gates pass. Validate subgoal outcomes and changed layouts before any unfamiliar-puzzle claim. | Multi-step goal sequence with observed interaction-state verification on held-out variants. | March 2027 remains a conditional target; reduce puzzle scope if earlier gates consume time. |

### Immediate next implementation milestone

Verify one actual GRU checkpoint/cache pairing. Then proceed through 0B/0C and Phase 1. Do not begin
goal-conditioned training until the trustworthy baseline establishes which goal
interface and destination labels can be evaluated. Preserve the deployed
checkpoint and all existing split membership. No training, live game control,
recording, or checkpoint promotion was authorized or performed in this audit.
