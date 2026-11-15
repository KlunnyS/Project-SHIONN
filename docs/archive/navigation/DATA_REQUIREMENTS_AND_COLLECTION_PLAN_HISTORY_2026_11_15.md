> Historical record. Use [the current navigation guide](../../NAVIGATION.md) for operational instructions.

# Data requirements and collection plan

Planning revision: 2026-11-14. Dates and quantities below are provisional budgets,
not evidence of sufficiency. March 2027 remains a delivery reference. Read this
plan at the start of each learning iteration; update coverage, gaps, cost and
next collection gate before building a new trainer. Preserve all frozen splits.

## Verified starting point

The reviewed route-expansion split contains seven TRAIN and two development
clips. TRAIN covers six test1 routes and one test9 route; the bounded visual
screen establishes two chamber-specific exits and zero validated alternative-goal
pairs. One annotation-only ledger has 145 rows and twelve visible marker centers.
It supplies no training eligibility. Five November 12 recordings remain
quarantined. Existing recovery data can support off-route visual/action review;
it lacks trusted frame-aligned destination/progress labels. See
`GOAL_ANNOTATION_READINESS_2026_11_14.md` for evidence and limitations.

## Dependency-ordered preparation schedule

| Capability / purpose | Scenarios and diversity | Labels and existing coverage / gap | Initial quantity and effort | Priority, dependency and preparation target | Validation |
|---|---|---|---|---|---|
| Movement control: reliable steering without retraining | Same frozen policy on test11/test12, matched decoder settings; autonomous scored attempts | Predicted/applied actions, camera response, focus and frame health; decoder infrastructure exists, live comparison pending | Existing 12-attempt argmax/expected screen, 12 minutes policy time plus resets; budget 30–45 minutes | P0; before interpreting new policy failures; Nov 14–20 | Bidirectional preflight, capture-valid attempts, hashes, per-map completion/turning/jitter/wall stalls |
| Goal conditioning: follow a supplied objective | One common start with two visible, independently reachable intermediate waypoints; later different starts/layouts | Requested goal ID, frame-relative target center/visibility, ownership and start identity; no valid pairs exist | First two trajectories only, 30–60 minutes setup/collection plus 20–40 minutes annotation/review; then planned 8 TRAIN + 8 held-out trajectories and 6 live starts | P1; verified reset and projection supplier; feasibility Nov 14–22, expanded data only after pair approval | Hash provenance, whole-start/session splits, manual visual review, annotation validator and paired-goal gate |
| Multiple routes: detour robustness to a fixed destination | Same goal around left/right sides of a safe obstacle, then separate layout | Route ID, fixed goal ID, obstacle/layout/start identity; current repeated exit routes do not establish alternatives | First 2 routes × 1 demonstration; budget 1–2 hours including review; scale only on failure analysis | P2 preparation; after two-goal feasibility; Nov 23–Dec 6 | Both routes genuinely differ and reach same goal; separate alternate routes from alternate goals |
| Progress/obstruction prediction: distinguish ineffective actions | Commanded movement with displacement, obstruction, pure camera turn, waiting, detour temporarily away from goal | Verified pose/displacement or independently reviewed outcome windows, goal distance only where trustworthy; existing videos lack trusted physical progress | First 6 short windows across 3 situations plus matched successful controls; budget 2–3 hours setup/labels | P3; telemetry/reset and goal definitions; prepare Nov 23–Dec 6 before auxiliary-head work | Timestamp alignment, telemetry freshness, held-out start groups, camera-motion negative controls, label uncertainty |
| Recovery and goal resumption | Same goal before/after wall contact, unfavorable orientation and detour; include escape followed by wrong-way wandering as negatives | Original goal, failure/escape/resumption windows, ownership and terminal outcome; reviewed recoveries usable only in existing roles, no clean goal holdout implied | First 2 recovery-to-goal clips and 2 reviewed negative windows; budget 1–2 hours; retain full trajectories | P3; goal pair and progress label feasibility; Dec 1–13 | Escape plus sustained goal progress, no immediate re-entry, full-episode split and exposed-data checks |
| PPO navigation evaluation: reward and reset integrity | Visible target, wall detour, unfavorable pose, corners, alternate routes, varied starts | Scenario registry, terminal reason, trusted progress, invalid-capture and reset logs; validated RL adapter/reward absent | First 3 scenario types × 3 reset checks; then 6 held-out starts with 2 goals; 2–4 hours supervised setup, throughput measured live later | P4; validated control/reset/reward first; Dec 7–20 readiness decision, no PPO launch scheduled | Pose/state repeatability, bounded episodes, fresh events, reward-exploit tests, measured environment FPS/reset cost |
| Button/cube: reusable interactions | Approach button; pickup, carry, place cube; paired training variants and distinct development/evaluation layouts | Object/goal IDs, use press/release, pickup/drop/contact/door outcomes and subgoal sequence; navigation clips do not supply positive interaction labels | First 3 episodes per training variant; only then provisional 40–60 successes/family + 10–20 development; pilot 2–4 hours/family, expansion cost measured afterward | P5 preparation can run alongside navigation; map review Nov 23–Dec 13, interaction data Dec–Jan depending on navigation gate | Hand-test map events, action timing, actual pickup/placement video, untouched evaluation layout and whole-session separation |
| Causal reasoning/subgoal planning: learn dependencies | Button controls door vs irrelevant button; cube placement changes access; failed interactions and alternate mechanisms | Intervention/outcome pairs, observed object state, sequence boundaries, causal hypothesis vs observed effect; none verified in present audit | First 2 matched interventions per mechanism on 2 variants; budget 3–5 hours design/review, no large batch | P6; stable interactions and outcome signals; Jan 4–24 feasibility, defer complexity if core demo at risk | Counterexamples, no hard-coded chamber relation, mechanism/layout holdouts, full provenance |

First-pass goal labeling estimate is 10–20 minutes per short clip, plus independent
review; it was not measured as a human productivity benchmark. All other efforts
are provisional planning allowances. Measure setup, recording, annotation and
validation separately on each feasibility batch before requesting expansion.

## Critical path and parallel preparation

Reliable input and capture → repeatable starts with explicit destination geometry
→ valid two-goal pair → goal-aware trainer ingestion → bounded goal-use pilot.
Progress labels and full recovery trajectories can be prepared while trainer
integration is written, after the first pair passes. The user can identify safe
waypoint regions and author/hand-test interaction variants while Codex implements
offline tooling. Do not collect a large batch before validating the protocol.

By Nov 22 decide whether paired collection works. If not, spend the next iteration
on the verified reset/projection fault. By Dec 6 decide whether goal-dependent
behavior is measurable; reduce scope if it is not. Keep Feb 15–28 for frozen
assessment and March for reporting/demo buffer. PPO and planning are conditional
research stages, not promises required to ship a simple puzzle demonstration.
These evidence-dependent dates refine the older March plan, whose Nov button
schedule is now at risk; do not pretend that deadline was met.

## User collection contract for every row

Before recording, create a development-only scenario card under
`data/annotations/navigation_collection_pending/` describing map/hash, start
position/orientation source, named goals, expected behavior, ownership, and split
intent. Never invent engine coordinates from an image. Store new raw material
under `model_attempts/goal_navigation_pending/<capability>/<session>/`; directory
names and successful terminal events grant no training eligibility. Original
recordings stay unchanged. Keep per-attempt requested-goal/start records beside
raw artifacts, then create a separate schema-valid annotation ledger.

For route diversity, use the same selected destination and demonstrate opposite
safe obstacle detours. For progress, include explicit stationary and camera-only
controls as well as moving/stalled windows. For recovery, begin at a registered
unfavorable pose, keep the same goal throughout, and continue after escaping until
the goal or episode cap. For interactions, use the named object at a registered
approach pose, demonstrate the complete press or pickup/carry/place/outcome and
label actual events rather than intended effects. For causal examples, change one
mechanism/intervention while preserving the comparison conditions. For PPO
readiness, inspect/reset scenarios and log outcomes; do not substitute human
recordings for a validated environment adapter.

Initially justify only the feasibility quantities in the table. Record exact
verified poses and orientations on scenario cards before user execution; no
numeric poses are available from this offline audit. Reject missing/stale frames,
input errors, ambiguous ownership, incomplete actions or unverified target labels.
Keep failures/partials diagnostic-only. A reviewer must approve integrity and
membership before any cache or training manifest includes new data. Future
holdouts must be reserved before they guide model or protocol selection.

Detailed immediate procedure: `TWO_WAYPOINT_COLLECTION_PROTOCOL.md`.

## Visual-only chamber revision — 2026-11-14

Use `GOAL_CHOICE_CHAMBER_SPEC.md` and `diagrams/goal_choice_v1_layout.svg` for
Puzzle Maker construction. They supersede custom-colored-circle/triangle asset
requirements. A pedestal-button area and stationary-cube area are passive visual
landmarks; no interaction or second exit is required. The optional separate
`shionn_waypoints.nut` emits evaluator-only A/B events after the existing script;
current shared parser/recorder do not consume waypoint success yet. Verify event
routing before automatic scores. Existing recordings contain no validated data
from this new room; first pair remains development-only and no membership changes.

For the first two demonstrations preserve initial target crop/source image hash,
bounds, goal ID and per-frame image-space labels. Engine coordinates, angles,
velocity, trigger state and distance are teacher/evaluator-only, never inference
features. A crop embedding/tracker is a future distinct architecture, not an
implemented reinterpretation of old waypoint weights. Preserve the selected crop
through occlusion; identity cannot be recovered from two identical invisible
waypoint vectors alone.

| Stage | Additional demonstrations/labels and instrumentation | Initial quantity, effort, diversity, dependency and date | Risk / acceptance |
|---|---|---|---|
| v1 goal choice | A/B from identical registered view; crop and frame labels; player-only A/B triggers independent of exit | 2 development clips only; 1–3 hours build/compile plus prior collection/label estimates; Nov 14–22; both visible and independently reachable | Stable cube, unconnected button, reset/capture/event checks and distinct early actions; no automatic scaling |
| v2 obstruction | One selected visual target behind a simple obstruction; blocked/direct and successful detour windows; evaluator-only displacement/progress | First 2 clips; 1–2 hours review/collection after variant build; Nov 23–Dec 6; new start/layout groups | Direct route truly blocked, visual goal identity retained, no camera-motion-as-progress labels |
| v3 recovery | Goal observed before unfavorable pose; escape then full return to goal; failed/wandering negatives; phase and goal-resumption labels | First 2 successes plus 2 negative windows; Dec 1–13 after v1/telemetry gates; vary wall face/heading | Wall departure alone is insufficient; preserve full trajectory and old recovery exclusions |
| v4 route diversity | Two paths to one goal, route IDs and chosen target crop, independent evaluation layouts | First 2 routes; Dec 7–20 if prior gates pass; vary obstacle/start/layout, not just appearance | Different routes are not different-goal evidence; matched progress/reaching measures |
| v5 interaction | Button activation or cube carry/place only in a separate variant; intervention/state/subgoal labels, independent mechanism events | Three inspected pilots per training variant before expanding; Dec–Jan contingent on navigation | Inert v1 landmarks are not positive interaction data; causal claims require interventions and counterexamples |

User can prepare the Puzzle Maker room, stock landmarks and screenshot/crop
choices while Codex prepares parser/evaluator and annotation loading. Export,
compile, hand-validation and known start state are critical-path work. Do not let
PPO or crop-encoder implementation delay the first valid development pair.

## Pair-review update — 2026-11-15

See `GOAL_CHOICE_PAIR_REVIEW_2026_11_15.md`. Both complete clips were reviewed;
281 annotation-only frame rows validate. Idle ranges A[0,111), B[0,29), goal
steering begins around A130/B44 after a common forward approach. Neither shows
a settled stop. Receive-time uncertainty and approximate landmark-center labels
are explicit. Both originals remain development_pending, not approved for training.

Revised smallest split-feasibility proposal: six NEW trajectories, four proposed
TRAIN across two new start groups and two held-out from a third distinct group,
with separate sessions. Existing pair stays exposed development. This is a smaller
within-room diagnostic than the earlier 8+8 plan; it cannot support broad
navigation generalization claims. Do not collect yet: first finalize visual-crop
labels, start/config evidence and split-aware ingestion. Proposed Nov 15–17 offline
preparation, Nov 18–22 supervised six-clip preparation (45–90 min collection/setup,
60–120 min labels/review), Nov 23–26 controlled profiling/pilot only with explicit
authorization. A later multi-layout cohort remains necessary. Need a new protocol
if terminal stopping/dwell rather than region entry is required. Existing strict
frame-gap check is unresolved; passed pair has no skipped/reused IDs, but invalid
captures must not be promoted. All numbers/dates are planning assumptions.

### Visual target preparation, 2026-11-15

Region entry was explicitly accepted as success. Native-resolution persistent
button/cube crops and hashed descriptors now exist, plus bounded offline loading
that verifies sources, preserves full history and masks initial waiting. These
are development-only artifacts; no learned visual-goal adapter/trainer is ready.
Follow `GOAL_CHOICE_VISUAL_TARGET_PREPARATION_2026_11_15.md`: register one new
interior start first (screenshot/getpos), then validate its A/B pair before the
proposed six-clip expansion. Current pair remains exposed development, not TRAIN.

### Registered interior pair reviewed — 2026-11-15

Both region-entry demonstrations now have a full frame/action/event review and
160 annotation-only rows. Final measured XYZ/yaw match and pitch differs by
0.000038°; both share the config hash. Native button/cube target crops are saved.
After waiting masks, 50/51 frames remain. Keep both pairs exposed development;
no train/evaluation split exists. See `GOAL_CHOICE_INTERIOR_01_REVIEW_2026_11_15.md`.
Next is bounded CPU crop-interface/batching work, not another recording request.
Fresh split groups and map/config evidence must be fixed before any expansion.

## Visual-goal software milestone completed — 2026-11-15

The optional shared-trunk crop encoder, persistent goal readout, inference/reset
and checkpoint path, sequence dataset, training-loss/epoch helper and CPU
counterfactual/preparation tools are implemented. See
`VISUAL_GOAL_IMPLEMENTATION_2026_11_15.md` for exact tensor contracts and commands.
Current pairs remain annotation_only/development_pending and fail TRAIN selection.
Future training approval is a separate versioned file pinning video/actions/
metadata hashes and explicit authorization; never edit original capture metadata
to make it eligible. No real approval or training was performed.

Dependency queue:

| Work | State | Evidence / remaining requirement |
|---|---|---|
| Crop encoder, recurrent readout, persistent inference | Implemented, CPU tested | Separate architecture; baseline-equivalent zero adapter; reset/checkpoint tests |
| Episode goal loading, loss masks, provenance and split gates | Implemented, CPU tested | Both real pairs loaded; development optimization denied; source/crop checks |
| Bounded training/evaluation infrastructure | Implemented, not trained | Existing BC objectives; matching constant-crop control; no automatic jobs |
| Counterfactual and preparation commands | Implemented, CPU only | Same-state sensitivity screen; approved TRAIN+validation preparation gate |
| Dataset expansion | User-dependent | First register new start groups/sessions; proposed four TRAIN + two held-out clips after review |
| GPU cost profiling and actual pilot | User-dependent | Explicit authorization, eligible data and frozen manifest needed |
| Live waypoint navigation screen | User-dependent execution; further integration deferred | Only prepare after a viable candidate and approved protocol; generic exit score is insufficient |

User collection planning remains the small six-clip feasibility proposal, not a
large batch: two new training start groups × A/B, one separately reserved held-out
start group × A/B. Existing exposed pairs remain development only. First validate
one new pair before completing the six. Record map hash, capture/FOV/control config,
registered pose and configuration hash with each future pair. Budget 45–90 minutes
setup/collection and 60–120 minutes review/labels; quantities and effort are
provisional. No broad generalization claim is possible from a single held-out
start in one chamber. Prepare new layouts, occlusion and recovery only after
within-room goal use is measured. Keep the March 2027 scope/February freeze intact.

## Fresh paired-goal collection matrix — 2026-11-15

Authoritative next procedure: `PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md`.
The proposed FOUR training-candidate and TWO held-out demonstrations are individual
trajectories: TWO train-candidate A/B pairs plus ONE reserved A/B pair (six clips,
three whole start groups). Not six pairs. Never count frames as independent trials.

Versioned plan: `configs/goal_choice_collection_plan_v1.json`. train_01 is a fresh
left-side start, train_02 a fresh right-side start, eval_01 a separate position/depth
and heading reserved before training. All actual numeric poses are pending user
getpos/visibility/collision evidence. Provisional diversity gates: at least 128
engine units between new groups and from exposed doorway/interior starts; at
least 5° camera difference between new groups. These gates reduce near duplicates
but do not establish unseen-layout generalization. The minimum held-out N is
one group/two trajectories. No existing development episode is reused/promoted.

Ready offline work completed: append-only pose registration, whole-group/session
plan validation, first-pair-only command generation, map/start/capture hashes in
metadata, raw console setup-event guard (including split lines), pair integrity
and visual-annotation gate, source-linked crops and immutable review fingerprints.
Reserved collection intent is enforced by the visual dataset: heldout_reserved
cannot enter TRAIN, and train_candidate cannot be called validation/regression.
Existing non-collection defaults remain available. No real training approval exists.

First new pair only: target Nov 15–16, 15–30 minutes setup/recording plus 20–40
minutes labeling/review. After passing, remaining four trajectories: Nov 16–18,
30–60 minutes setup/recording and 40–80 minutes labels/review. Nov 18–20 final
review/experiment preparation only if prior gates pass. These are provisional,
user-dependent dates, not automatic jobs. GPU training requires separate approval
and profiling; February freeze/March delivery remains the planning reference.

The user can select/measure the first new start and preserve its screenshot while
Codex completes the subsequent integrity/crop/phase review. Do not record the full
batch before first_pair_review status passed. Later command release additionally
requires all three starts registered and intact first-pair evidence. Same FOV must
be maintained manually; current metadata explicitly records it as not_measured.
