> Research roadmap/proposal; not an execution log. See [current navigation status](NAVIGATION.md) and [current data plan](DATA_REQUIREMENTS_AND_COLLECTION_PLAN.md).

# SHIONN — Goal-Directed Navigation & Reasoning Roadmap (Review-Integrated v2)

**Proposed timeline:** November 2026 – March 2027

**Long-term vision:** An autonomous Portal 2 agent that can understand unfamiliar chambers, formulate and verify plans, and learn from experience.

**Near-term priority:** First verify that inference, live control, recording, and evaluation are trustworthy; then reduce aimless wandering, improve goal recognition, detect lack of progress, and learn reliable stuck recovery.
**Revision:** 2026-11-13 — incorporates the independent 15-page *SHIONN Model Review: Issues and Improvements* (code review of commit `45d07a6`).

## Guiding approach

Build on SHIONN's existing CNN, GRU, feature gating, imitation-learning pipeline, and planned PPO reinforcement learning. **First repair and verify the measurement/control pipeline.** Then begin with a **goal-conditioned hybrid navigation controller** and progressively add a **hierarchical planner**. Do not replace working components or implement a full world model before measuring their need.

The independent review was **static code inspection only**: no tests, training, or live experiments were run by the reviewer. Treat its code-path findings as **issues to reproduce and verify against the current branch**, and its performance explanations as **hypotheses**. Commit `45d07a6` may differ from current local changes.

Primary evidence of progress should be **closed-loop autonomous behavior**, not merely imitation accuracy.

### Development phases

0. **Stabilize measurement, inference, and live/recording correctness** (new review-driven gate).
1. Diagnose the existing policy using trustworthy measurements.
2. Learn goal-conditioned navigation.
3. Learn progress monitoring and recovery.
4. Reinforce navigation skills with PPO.
5. Introduce high-level subgoals and persistent goal selection.
6. Demonstrate button-and-cube puzzle solving.

## Current evidence and constraints

As of the November 3, 2026 evaluations:

| Chamber | Deployed | Replay |
|---|---|---|
| Test 1 | Time limit | Time limit |
| Test 11 | Time limit | Goal in 35.43 s |
| Test 12 | Goal in 6.73 s | Goal in 6.81 s |

Replay completed **2/3** chambers and deployed completed **1/3**, but these were only one attempt per model per chamber. Neither result establishes navigation readiness or warrants automatic promotion. Test 1 failures showed active wall-facing/backtracking behavior, not simply prolonged all-zero output.

**Independent review (Nov 13, 2026):** Reports a possible zero-bin bias in 15-way mouse decoding, stream-fragment parsing errors, mismatched/offline GRU evaluation, focus-unknown memory resets, recorder-label risks, train/live mouse delivery differences, overfitting with limited augmentation, and short recurrent training horizons. These are **not yet measured fixes**; the review ran no experiments.

Other relevant evidence:

- Stationary recovery fitting achieved **31/31** training starts, but the earlier six-start assessment showed **0/6** exact first-action matches. Exact action matching is not sufficient to assess actual recovery.
- Later sustained offline scoring showed some improvements but used checkpoints exposed to reserved data; it must be labeled diagnostic rather than clean holdout performance.
- A development recovery partition exists: **22 TRAIN / 9 recovery-validation / 6 diagnostic-only**. Preserve its membership restrictions and provenance checks.
- The deployed checkpoint remains unchanged. Local, uncommitted changes should not be treated as published implementation.
- Game-involved runs, new recordings, and training remain user-run/approval-gated under the existing workflow.

### Project constraints

- Main hardware target: Ryzen 7 5800XT, RTX 4060 8GB VRAM, 32GB RAM.
- Keep candidate experiments bounded and specify estimated cost before execution.
- Maintain evaluation separation; mark data-contaminated results clearly.
- No automatic checkpoint promotion, extra retries, or runtime extensions.
- Preserve capture checks, hashes, source membership, and regression screens.
- Keep no more than five project items **In Progress** simultaneously.
- March 2027 is a target, not a guarantee.

---

## Phase 0 — Technical stabilization and validated baselines (NEW)

**Target:** November 2026; must precede interpreting new model comparisons or approving retraining.

**Training:** None for the initial fixes; all changes opt-in or behavior-preserving where feasible.

### 0A. Verify and repair scoring integrity — first

- [ ] **Reproduce issue #6** on the current branch: feed console events split at every character boundary through live runner, recorder, wrapper/camera telemetry as applicable. Centralize the existing complete-line `EpisodeEventStream` pattern; retain partial TCP lines across reads. Ensure exactly-once event accounting, bounded buffers, and safe disconnect handling.
- [ ] **Reproduce issue #10:** test `benchmark_expert.py` against a real GRU checkpoint. Run episodes sequentially via recurrent `step()`, reset hidden state only at episode boundaries, and share all action-decode logic/flags with live inference. Until fixed, do not offer GRU offline scoring as if it were valid.
- [ ] Write parity tests for keyboard thresholds, mouse decoding, frame/history handling, success/failure events, and checkpoint identity. Re-score existing checkpoints if any historical results depended on broken paths; keep old reports with a superseded/qualification note.
- [ ] Verify whether past `time_limit` results are affected by missed events using console logs/video; **do not retroactively change scores without evidence**.

**Gate 0A:** Event fragmentation tests pass; recurrent offline scoring works; exact policy/config and decoder match live behavior; benchmark results have a traceable validity flag.

### 0B. Test mouse decoding without training — highest-leverage model-side experiment

The review identifies a potential structural bias: an argmax over **15 mouse bins** (7 left, 1 still, 7 right) compares the whole still probability with individual movement bins. This can select still even when summed movement probability is greater. The report cites expert-vs-model nonzero dx on Test11 of **43.2% vs 30.8%**, and on Test12 of **28.3% vs 17.2%**. This is an explanatory hypothesis, **not proof of causation**.

- [ ] Preserve `argmax` as the default/reproducible baseline. Implement **opt-in** `--mouse-decode {argmax,hierarchical,expected}` and configurable move threshold; record decoder and threshold in results.
- [ ] Hierarchical decode: first still/move by `P(zero)` threshold, then left/right by summed mass, then select magnitude within chosen direction. Document tie/threshold handling and direction-bin mapping; unit-test representative probability vectors and each axis.
- [ ] Expected decode: probability-weighted mouse bin representatives; test cancellation and clipping. Do not presume it is better.
- [ ] Compare offline **nonzero dx/dy rates, signed direction correctness, per-axis mouse MAE, jitter and action distribution** on an unchanged expert cohort. Because offline score agreement may not predict closed-loop performance, designate the most informative decoder(s) for a **user-run paired autonomous** test with frozen starts, capture, and limits.
- [ ] Monitor overshoot, oscillation, and whether different decoding causes more wall encounters. Do not promote based solely on closer activity rates.

**Gate 0B:** Valid offline parity and at least a bounded, repeatable live comparison support a decoder choice; otherwise retain baseline and document the outcome.

### 0C. Stabilize control, input safety, and recordings

- [ ] **Issue #7, focus telemetry:** count existing `focus_unknown` and confirmed-loss events before editing. On unknown focus, preserve GRU/history if safe, but **do not indefinitely retain active key holds**; use bounded unknown-duration release, known-focus requirements and explicit safety tests. Confirmed focus loss must release inputs/reset appropriately. Test stale telemetry, focus recovery, disconnects, and emergency stop.
- [ ] **Issue #8, mouse delivery:** compare per-tick burst vs opt-in substeps using camera yaw-per-unit telemetry in the real Proton/Wayland setup. Do not assume pointer acceleration; keep thread/timing/input-safety controls and avoid blocking the model loop.
- [ ] **Issue #9, recorder labels:** test `BTN_TOUCH` vs `BTN_LEFT`, prefer relative-motion pointing devices, reject/flag absolute touchpad movements, make short-tap latches atomic, track physical crouch keys separately, wait for valid first capture, and discover keyboards from capabilities. Add mocked evdev tests and audit existing fire/jump/use outliers before requesting new recordings. Do not silently relabel old data.
- [ ] **Issue #11, run safety:** refuse nonempty checkpoint destinations without explicit resume/overwrite; verify split defaults (`frozen_v3.json` vs `frozen_v1.json`) against current rules; verify Python/PyTorch requirements; align dashboard recipe with runbook; bound wrapper self-test waits; make checkpoint/metadata writes atomic or integrity-verifiable; avoid a training chamber as the default live run; guard GRU `--model-fps < --fps` mismatch (or advance recurrent state each frame).
- [ ] Record training/live video compression mismatch as a possible domain gap; do not change capture format mid-comparison. If exploring lossless/raw frames, make it a separate versioned data experiment.

**Gate 0C:** No known silent recorder corruption in covered paths; input release is fail-safe; benchmarks/capture remain comparable; old recipes and checkpoints stay reproducible.

### 0D. Training-quality improvements — only after measurement and decode gates

These require **separate, bounded experiments**, not one combined retraining run. Keep the main goal-conditioned work in view; stop if closed-loop behavior does not improve.

| Review issue | Proposed controlled test | Interpret cautiously |
|---|---|---|
| #5 Data throughput | Profile `--workers 0` vs 4–6; then AMP batch 8/16/32; consider nonblocking recurrent transfer and sequence augmentation vectorization | Speedup not measured by review; larger batch changes optimization and needs controlled settings |
| #2 Spatial augmentation | Opt-in random shift `0` vs `6` pixels, with identical offset across all 4 stacked frames **and the full GRU sequence** | Compare validation, mouse MAE, held navigation and live outcome; later best epoch alone does not prove generalization |
| #3 Ordered mouse targets | After shift experiment, compare hard CE with opt-in local ordinal soft targets; preserve sharp still/move boundary | Compare near-bin errors, directional accuracy, MAE and closed-loop behavior |
| #4 EMA | Last, compare raw vs EMA weights (e.g. decay 0.999), with resume-safe state handling | Benefits not guaranteed; use fair checkpoint-selection criteria |
| GRU history horizon | Audit 4 burn-in + 8 sequence steps versus episode-length inference; consider longer training chunks or state carried across chunks with truncated BPTT | Longer history costs VRAM and may help or hurt; benchmark actual benefit |

Every trial must pin its training data, source checkpoint, config, random seeds, update/runtime budget, stopping criteria, baseline and eligible holdouts. Avoid combining shifts, ordinal loss, EMA and recurrent-length changes in one unexplained candidate.

**Gate 0D:** At most one justified training adjustment is advanced at a time based on clean and autonomous navigation evidence; defer others if goal-conditioning is the more important remaining bottleneck.

### Technical review issue map

| Issue | Status in this plan | Review's assessment |
|---|---|---|
| #1 Mouse argmax zero-bin bias | 0B | High potential impact; no retraining |
| #2 Missing spatial shift | 0D | Overfitting hypothesis; retraining |
| #3 Unordered mouse-bin objective | 0D | Potential precision issue; retraining |
| #4 No EMA | 0D | Optional training stability test |
| #5 Data-loader throughput | 0D | Profile first, no accuracy claim |
| #6 TCP event fragmentation | 0A | Benchmark correctness |
| #7 Focus-unknown memory reset | 0C | Verify logs; preserve safety |
| #8 Burst mouse delivery | 0C | Unverified train/live difference |
| #9 Recorder label corruption risks | 0C | Fix before new recording |
| #10 Offline GRU / decode mismatch | 0A | Benchmark correctness |
| #11 Tooling/configuration | 0C | Execution reliability |

**Source:** *SHIONN Model Review: Issues and Improvements*, 2026-11-13, pp. 1–15; inspected repo commit `45d07a6`. The document did not run tests, model experiments or game execution.

---

## Phase 1 — Diagnose existing behavior

**Target:** After Phase 0 baseline verification (tentatively November–December 2026)

**Training:** None initially

### Objective

Determine whether SHIONN's aimless wandering and failed recovery stem primarily from missing goal information, weak progress awareness, ineffective action selection, limited demonstration coverage, or some combination.

| Hypothesis | Diagnostic |
|---|---|
| Poor goal recognition | Check whether the visual representation discriminates exits, doors, buttons, and other task-relevant objects. |
| Poor progress awareness | Test whether recent observations/actions predict useful movement versus wall-facing loops. |
| Poor recovery decisions | Check whether failure is detectable but followed by ineffective actions. |

### Tasks

- [ ] Build a reproducible, read-only navigation diagnostic suite from existing recordings.
- [ ] Inspect visual embeddings/features and their association with task-relevant objects, avoiding unsupported claims of semantic recognition.
- [ ] Measure turning, forward/backward action, idle duration, wall-facing loops, and repeated-path behavior.
- [ ] Compare deployed, replay, and earlier eligible checkpoints with transparent training-exposure labels.
- [ ] Review test 1 failures and annotate meaningful progress/recovery windows.
- [ ] Publish an evidence report and rank competing failure hypotheses.

**Exit gate:** Document which capability is the best justified next target. Phase 1 does not require solving a chamber.

---

## Phase 2 — Goal-conditioned navigation

**Target:** December 2026 (conditional on Phase 0 gates)

**Priority:** Highest near-term architectural learning improvement

### Objective

Make navigation depend on a persistent desired destination, not only current visuals and action history.

Existing policy (conceptually):

\[
a_t \sim \pi_\theta(\cdot \mid o_t, h_t)
\]

Proposed goal-conditioned policy:

\[
a_t \sim \pi_\theta(\cdot \mid o_t, h_t, g_t)
\]

Here, \(g_t\) is the selected goal representation. Supplying a goal vector by itself does not teach the model what an exit is; the target representation must be grounded in data and tested.

### Goal interfaces to consider

| Goal format | Example | Role |
|---|---|---|
| Target image | Image of an exit | Visual destination grounding |
| Target embedding | Learned encoding of a target | Compact neural conditioning |
| Relative target location | Waypoint direction/distance | Easier supervised pilot or training-only signal |
| Learned high-level subgoal | Approach cube; reach button | Later planner interface |

Game coordinates and object states may be useful as **privileged training or evaluation signals**, but the final visual-only agent must not depend on them at inference.

### Tasks

- [ ] Audit existing episodes for reliable, identifiable goal labels and target visibility.
- [ ] Specify goal representation and a persistent goal interface with the existing CNN/GRU.
- [ ] Implement a minimal goal-conditioned training and inference path, without automatic training.
- [ ] Define an explicit baseline using the current policy.
- [ ] Use eligible imitation data to train a bounded candidate only after review/approval.
- [ ] Test visible-target reaching, varied orientations, blocked routes, and same-view/different-goal cases.
- [ ] Confirm SHIONN actually uses goal conditioning rather than ignoring the extra input.

**Exit gate:** Reach specified destinations more reliably than the baseline on excluded starts, and maintain the target despite temporary obstacles. Quantify improvement per map and start condition.

---

## Phase 3 — Progress awareness and learned recovery

**Target:** December 2026–January 2027 (conditional on reliable goal metrics)

### Objective

Enable SHIONN to infer whether its recent actions are achieving meaningful progress and to alter its behavior when obstructed **without forgetting the active goal**.

A potential shared representation:

\[
h_t = \operatorname{GRU}(h_{t-1},\operatorname{CNN}(o_t),a_{t-1})
\]

Possible output heads:

\[
\begin{aligned}
a_t &\sim \pi_\theta(\cdot \mid h_t,g_t)\\
\hat p_{\mathrm{stuck}} &= f_{\mathrm{stuck}}(h_t,g_t)\\
\hat p_{\mathrm{progress}} &= f_{\mathrm{progress}}(h_t,g_t)
\end{aligned}
\]

An optional auxiliary learning objective:

\[
L = L_{\mathrm{BC}} + \lambda_p L_{\mathrm{progress}} + \lambda_s L_{\mathrm{stuck}}
\]

These predictions alone do not guarantee better recovery; the policy must learn or be designed to use the resulting information effectively.

### Tasks

- [ ] Define observable labels for blocked motion, purposeful escape, resumed progress, and false positives.
- [ ] Use history and held-out episodes to test whether stuck/progress states are predictable.
- [ ] Compare normal-history and repeated-history evaluations where appropriate.
- [ ] Prototype small auxiliary predictors and measure their accuracy/calibration separately from action prediction.
- [ ] Define a bounded learning experiment that changes one major factor.
- [ ] Evaluate unassisted recovery from excluded starts, including whether the original navigation goal is resumed.
- [ ] Check full navigation regression and reject unsupported improvements.

**Exit gate:** Repeatedly escape representative obstructions, return to the same objective, and outperform a declared baseline on valid autonomous attempts—without scripted unsticking masquerading as learned recovery.

---

## Phase 4 — Reinforcement-learning navigation curriculum

**Target:** January–February 2027 (dependent on stabilization and environment throughput)

**Dependency:** Reliable resets, meaningful rewards, and a measurable goal interface

### Objective

Initialize PPO from the goal-conditioned imitation policy and reinforce effective navigation through closed-loop experience.

### Suggested curriculum

1. **Reach a visible target:** Short unobstructed navigation.
2. **Navigate around a wall:** Reward successful detours rather than direct but blocked motion.
3. **Recover from bad starts:** Randomize positions and orientations, including obstructed states.
4. **Navigate corridors and route choices:** Corners and multiple alternatives.
5. **Complete short chambers:** Combine the skills in new starts and layouts.

A simple illustrative reward is:

\[
r_t = \alpha(d_{t-1}-d_t)+\beta\,\mathbb{1}_{\mathrm{goal}}-\gamma\,\Delta t
\]

where \(d_t\) is distance to the target, \(\mathbb{1}_{\mathrm{goal}}\) indicates goal completion, and \(\Delta t\) represents time cost. This is **not** a validated SHIONN reward. Test for reward hacking, unreachable goals, loops, and incentives that conflict with the task.

### Tasks

- [ ] Implement or verify deterministic-enough scenario resets and reliable success measurement.
- [ ] Benchmark **actual game environment steps per second**, including loading, reset, capture, and input overhead.
- [ ] Define train/evaluation start distributions and validity criteria before training.
- [ ] Specify reward terms, horizon, exploration settings, and runtime/update budget.
- [ ] Consider a behavior-cloning retention loss to reduce forgetting:

\[
L_{\mathrm{total}} = L_{\mathrm{PPO}} + \lambda L_{\mathrm{BC}}
\]

- [ ] Compare imitation-only vs. imitation-plus-RL on frozen navigation evaluations.
- [ ] Review regressions and preserve the deployed checkpoint until repeatable evidence supports promotion.

**Exit gate:** PPO delivers repeatable, measurable navigation improvement on valid evaluation starts without unacceptable loss of previously learned skills.

---

## Phase 5 — Hierarchical subgoals and persistent objectives

**Target:** February 2027 onward, if low-level navigation is ready

### Objective

Separate **which objective to pursue** from **how to carry it out**.

Proposed architecture:

```text
Visual observation + history
           |
       CNN + GRU
           |
  Goal / subgoal selector
  (explore, approach object,
   interact, reach exit)
           |
  Persistent goal embedding
           |
 Goal-conditioned movement
    + learned recovery
           |
   Mouse and keyboard
           |
  Observe outcome / verify
           |
  Continue or choose new goal
```

Goals should persist across multiple low-level actions and terminate based on observed success, failure, or justified replanning—not arbitrary per-frame switching.

### Tasks

- [ ] Specify goal-setting, termination, and result-reporting interfaces.
- [ ] Start with supplied high-level goals to validate the controller, then train a goal selector.
- [ ] Collect/derive trustworthy subgoal outcome labels, including failed interactions.
- [ ] Evaluate multi-subgoal execution and recovery while retaining the current objective.
- [ ] Plan a later learned world-model prototype for predicting the consequences of candidate subgoals.

**Exit gate:** Execute short sequences of subgoals and detect when a subgoal succeeds or fails, with evidence that goal switching is meaningful.

---

## Phase 6 — Simple puzzle-solving demonstration

**Target:** February–March 2027 (stretch target if earlier gates pass)

### Objective

Combine navigation, objective selection, and interaction in basic Portal 2 puzzles.

### Milestones

- [ ] Instrument a **button-opens-door** chamber.
- [ ] Instrument a **carry-cube-to-button** chamber.
- [ ] Hand-validate interaction outcomes and positive pickup/use/carry/placement labels.
- [ ] Define train and evaluation variants with changes in positions and layouts.
- [ ] Train an appropriately bounded interaction pilot after navigation readiness.
- [ ] Demonstrate a sequence: identify target → navigate → interact → verify changed state → reach exit.
- [ ] Report full success, intermediate subgoal success, failures, and capture validity.

**Exit gate:** Demonstrate combined navigation and interaction on simple held-out variations. Do not interpret familiar puzzle completion alone as proof of broad reasoning or unknown-chamber generalization.

---

## Proposed initial readiness metrics

These thresholds are **engineering proposals**, not measured current results. Confirm feasibility in Phase 1 and preregister exact scenario sets before scored experiments.

| Metric | Initial proposed gate |
|---|---|
| Reach a specified visible destination | ≥80% of 20 valid attempts |
| Escape designated stuck starts | ≥70% of 20 valid attempts |
| Resume original target after successful escape | ≥70% of successful recoveries |
| Repeat solved short navigation chambers with varied starts | ≥80% of valid attempts |
| Autonomous operation | No manual unsticking during scored runs |
| Regression | No material degradation on frozen baseline cohort |

Twenty attempts are an initial development screen, not evidence of broad statistical generalization. Report per-map/per-start results, denominators, uncertainty, invalid captures, and failure cases.

## Immediate next sprint — review-first sequence

**Sprint objective:** Produce a trustworthy, comparable baseline before investing in more demonstration collection or retraining.

1. [ ] Inspect current branch against review commit `45d07a6`; make a status matrix **still present / already fixed / not reproducible** with code locations and tests.
2. [ ] Reproduce/fix fragmented events (#6) and GRU/offline decoder divergence (#10); preserve old reports and identify potentially affected benchmarks.
3. [ ] Implement opt-in mouse decoding modes (#1); run frozen-cohort offline comparisons and prepare **one bounded user-run** live A/B after correctness gates pass.
4. [ ] Audit focus gates (#7), burst-delivery calibration (#8), recorder labels (#9), checkpoint overwrite/split/config issues (#11); fix demonstrated safety/data faults before new recordings.
5. [ ] Only then run the original **test1 wall-facing / goal-awareness / progress** diagnostics, using the newly validated instrumentation.
6. [ ] Choose between a small controlled generalization experiment (#2, #3, #4, or GRU horizon) **or** a goal-conditioning pilot, based on evidence; do not schedule both automatically.

**Go/no-go checkpoint:** Record the date, completed integrity fixes, valid navigation comparison, remaining model defects, estimated experiment cost, and the effect on the March interaction milestone. If Phase 0 consumes more time, reduce the March demonstration scope explicitly rather than silently skipping safety or evaluation gates.

### Execution rules for Codex

1. Preserve the deployed checkpoint, prior experiments, versioned partitions, source hashes, and reproducibility safeguards. **First check current code before implementing a review finding.**
2. Keep the **22/9/6** recovery development split enforced; never claim exposed models provide clean holdout evidence.
3. Do not start training, game execution, new recording sessions, or checkpoint promotion without explicit user approval.
4. Change only one major learning factor in a controlled experiment; decode and instrumentation fixes are not evidence of learning until closed-loop tests confirm improvement.
5. Provide estimated compute/game runtime and an explicit stopping policy before each user-run experiment.
6. Validate ordinary navigation and recovery jointly, not through isolated action-matching scores alone.
7. Keep code interfaces compatible with a future high-level planner and world model.
8. Do not introduce hard-coded navigation solutions that conceal underlying policy weaknesses.
9. Commit/push only according to the user's repository workflow; do not report uncommitted local findings as published.
10. Preserve the project's March milestone as a **target**, and record navigation go/no-go decisions before expanding interaction work.

## Beyond March 2027 — Reasoning in unfamiliar chambers

Longer-term SHIONN should learn to **observe → understand → predict → plan → act → verify → revise**.

Potential research directions:

- A learned world model of navigation and interaction outcomes.
- Latent-space planning over candidate actions or subgoals.
- Object-state and causal-relation prediction (e.g. button activates door).
- Active experimentation when mechanics or relationships are uncertain.
- Evaluation on unseen layouts, recombinations of familiar mechanics, and altered causal relationships.

Test reasoning through **transfer and adaptation**, not simply successful replay of familiar solutions. Compare planning-enabled SHIONN against a matched reactive baseline and track whether it learns from predictions that fail.

## Core design principle

**Imitation teaches basic skills; reinforcement learning improves them; goal conditioning makes behavior purposeful; progress feedback enables recovery; hierarchical planning eventually lets SHIONN compose skills into solutions for unfamiliar puzzles.**

Avoid building the entire hierarchy prematurely. Establish measurable, goal-directed navigation first.

## Existing local project references (as provided)

- `docs/TRAINING_RUNBOOK.md`
- `docs/HISTORY_TIMING_AUDIT_2026_11_02.md`
- `docs/MODEL_ROADMAP.md`
- `docs/MARCH_2027_PLAN.md`
- `docs/RECOVERY_TRANSFER_PROTOCOL_2026_11_03.md`
- `data/annotations/recovery_transfer_development_v1.json`
- `data/annotations/turn_onset_comparison_v1.json`
- `model_attempts/diagnostics/recovery_transfer_review_20261103/sustained_scores.json`
- `docs/EXPERT_ACTION_REPLAY_2026_11_12.md` (identified by external review; verify file exists)
- *SHIONN Model Review: Issues and Improvements* (independent review, Nov 13, 2026; commit `45d07a6`)

*This document is a proposed plan based on user-provided results and the independent PDF review. It does not claim to have inspected the current repository, reproduced bugs, run tests, trained models, or executed the game. Reported review findings may already differ from the current branch.*
