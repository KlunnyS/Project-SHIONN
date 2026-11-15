# Data requirements and collection plan

Updated 2026-11-15. Maintained planning source; [earlier revisions are archived](archive/navigation/DATA_REQUIREMENTS_AND_COLLECTION_PLAN_HISTORY_2026_11_15.md).
Read alongside [current navigation status](NAVIGATION.md). Dates/quantities below
are provisional planning budgets, not evidence that a dataset will be sufficient.

## Evidence and integrity

- Seven reviewed route-expansion TRAIN clips and two development clips retain
  their original roles. The five November 12 recordings remain quarantined.
- Two goal_choice_v1 A/B pairs (four trajectories) are exposed development data.
  They have reviewed annotations/crops and valid initial capture checks; neither
  is approved for gradients or a clean holdout.
- The visual-goal encoder, recurrent readout, loader, provenance gates and CPU
  evaluation/preparation tools are implemented. There is no trained goal policy.
- No historical observations/actions, frozen splits or checkpoints are changed.
  Later training approval requires an explicit versioned review record pinning
  source video/actions/metadata, plus approved ledger and crop descriptors.

## Immediate collection matrix

**Four training-candidate + two held-out demonstrations means SIX individual
trajectories: THREE complete A/B pairs.** Never split one start group or its
session across TRAIN and evaluation; never count its frames as independent N.

| Start group | Intended later use | Goal order | Current requirement |
|---|---|---|---|
| train_01 | Two training-candidate trajectories | A, B | Register a fresh left-side open-room start; validate this pair first |
| train_02 | Two training-candidate trajectories | B, A | Fresh right-side pose and heading, separate session |
| eval_01 | Two reserved evaluation trajectories | A, B | Distinct position/depth and heading; freeze the whole group before training |

All numeric poses await user getpos/screenshot/collision evidence. Provisional
diversity gate: ≥128 units between prospective groups and from exposed starts,
and ≥5° orientation difference between new groups. These checks reduce obvious
duplicates; one chamber/one held-out group cannot establish broad generalization.

Exact commands and acceptance checks belong in the
[first-pair procedure](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md).
The command generator does not execute gameplay. Later-group release requires
intact first-pair review and all three registered poses. Raw captures remain
pending; intended collection role is not training approval.

## Capability requirements

| Priority / purpose | Recording scenarios and diversity | Required labels/instrumentation | Existing coverage and missing data | First quantity and effort | Dependency / preparation target / validation |
|---|---|---|---|---|---|
| P0 movement reliability | Same frozen checkpoint on test11/test12 under matched decoder settings | Applied/predicted actions, focus, camera response, frame health | Decoder tools exist; clean live benefit unproven | Existing 12-attempt screen, 12 minutes action cap; allow 30–45 minutes with setup | Before interpreting new failures; Nov 15–20 if user available; preflight/capture validity, per-map completion, jitter/overshoot review |
| P1 visual goal following | Common start, different visible A/B targets; three distinct start groups above | Requested goal, persistent source crop/bounds/hashes, frame phases/ownership, pose/plan/map/session identities, waypoint event | Two exposed development pairs; zero approved fresh TRAIN/evaluation pairs | First 2 clips only; then 4 more after review. First pair 15–30 minutes setup/recording +20–40 labels/review | Nov 15–16 first pair, Nov 16–18 rest conditional on passing; intact sources, both visible, correct events, paired-view/crop checks |
| P2 pathfinding/route diversity | Two safe detours around a genuinely blocking obstacle to the SAME destination; later separate layout | Route ID, persistent goal, obstruction geometry and outcome | Same-exit repeats do not establish alternate routes | First 2 routes ×1 clip; 1–2 hours including review after variant build | After goal-choice feasibility; Nov 23–Dec 6 provisional; both routes reach same target, whole-layout/start split |
| P3 progress/obstruction | Commanded movement with and without displacement, camera-only turns, waiting, useful detours | Trusted displacement/goal progress or reviewed outcome windows; synchronized timestamps | Existing video/actions lack trustworthy physical progress labels | First 6 short windows across 3 situations plus matched controls; 2–3 hours setup/review | Verified telemetry labels kept outside policy; Nov 23–Dec 6; negative controls, freshness, uncertainty and held-out starts |
| P3 recovery/goal resumption | Goal visible before wall-facing state; escape then resume it; wandering/re-entry negatives | Stable goal ID/crop, failure/escape/resumption phases, full trajectory and final outcome | Reviewed recoveries retain old roles/exposure; no clean goal-resumption cohort | First 2 complete recovery clips +2 negative windows; 1–2 hours | After goal and progress label feasibility; Dec 1–13; require sustained useful progress, not wall departure alone |
| P4 PPO evaluation readiness | Visible target, wall detour, unfavorable pose, corners and varied starts | Repeatable reset, fresh terminal reasons, trusted reward/progress, invalid flags and throughput | Reset helper exists; no validated RL environment/reward pipeline | First 3 scenario types ×3 reset checks; 2–4 hours supervised setup | Dec 7–20 conditional on control/progress gates; bounded durations, exploit checks, measured environment/reset speed, no automatic PPO |
| P5 button/cube interactions | Button approach/activation and cube pickup/carry/place across distinct map variants | Use press/release, object state/outcome, subgoal phases, independent mechanism events | Inert landmark approaches are not positive interaction demonstrations | First 3 episodes per training variant; 2–4 hours/family including pilot review. Only then consider 40–60 successes +10–20 development | Map preparation Nov 23–Dec 13, data Dec–Jan conditional on navigation; hand-check events, actual pickup/place evidence and reserved layouts |
| P6 causal/subgoal reasoning | Matched interventions: relevant/irrelevant button, cube placement, failed mechanisms | Intervention/outcome pairs, observed state and sequence boundaries; separate inferred relationships | No verified causal cohort in this audit | First 2 interventions/mechanism on 2 variants; 3–5 hours design/review | Jan 4–24 only after stable interactions; counterexamples and mechanism/layout holdouts; defer if core demo at risk |

Every effort estimate includes setup uncertainty and is unmeasured unless stated
otherwise. Measure recording, labeling and validation separately on feasibility
batches before expanding. Short goal clips provisionally need 10–20 minutes each
for first-pass labels, with independent review in addition.

## Schedule and parallel preparation

1. Nov 15–16: user registers the first NEW pose and records only its A/B pair.
   Codex reviews all frames/actions/events, prepares source-linked crops and phase
   labels, and validates without assigning training membership.
2. Nov 16–18, only after passing: register/freeze the remaining starts/sessions;
   collect four more trajectories. Allow 30–60 minutes setup/collection and
   40–80 minutes labels/review. Total six-clip budget: 45–90 and 60–120 minutes.
3. Nov 18–20, conditional: review split/provenance, explicit user approval and
   source compatibility, then prepare the bounded selected-vs-constant target
   experiment. GPU cost profiling/training require separate authorization.
4. Nov–Jan: prepare obstacle/recovery and interaction variants in dependency order.
   The user can author/hand-test maps while Codex prepares offline interfaces.
5. Feb 15–28: preserve the intended feature/evaluation freeze; March is reporting,
   reproducibility and demo buffer. The older November interaction dates are at
   risk; reduce scope rather than claiming unachieved milestones.

The critical path is safe distinct starts → capture/annotation validity → approved
independent data → bounded learning → actual destination reaching. Do not replace
missing goal diversity with more repeats of the same exit route. PPO and planning
are conditional extensions, not reasons to delay a simple puzzle demonstration.

## Collection instructions and storage

For each capability use a fresh named start and logical session, safe routes and
complete trajectories. Route diversity holds the destination fixed; goal choice
changes destination from the SAME start. Recovery keeps its original objective
through escape. Progress collection includes stationary/camera-only controls.
Interaction clips must include the actual press or pickup/carry/place outcome.
Causal examples change one mechanism/intervention at a time.

Immediate paired raw data goes under
`model_attempts/goal_choice_pilot_pending/<planned_session>/<A_or_B>/` with result
files and immutable metadata. Other future scenarios remain under
`model_attempts/goal_navigation_pending/<capability>/<session>/`. Keep separate
versioned ledgers, descriptors, review records and proposed split manifests.
Directory names and successful events grant no eligibility.

Reject invalid capture, spent setup triggers, ambiguous ownership, wrong-goal
success, missing frames/actions, mismatched start/config or invented visual labels.
Preserve partials for diagnosis. FOV remains unmeasured: keep it unchanged and
review comparable views; no numeric camera-match claim is justified. Privileged
telemetry stays outside policy inputs. Existing strict frame-gap failures remain
an open timing issue; never promote their invalid recordings.
