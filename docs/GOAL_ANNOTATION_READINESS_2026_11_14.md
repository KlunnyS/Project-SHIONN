# Goal annotation and evaluation readiness — 2026-11-14

> Dated snapshot. Current status: [navigation guide](NAVIGATION.md). Current collection commands: [paired-goal protocol](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md). Later implementation may supersede statements below.

Status: offline investigation in progress; no training authorization or membership change.

## Source selection

Selected `episode_20261105_161823_785804`, TRAIN in
`data/annotations/route_expansion_split_20261109.json`, 145 decoded frames.
Prior review establishes human ownership and recovery through stairs toward an exit.
Leading waiting frames 0–24 remain in the raw recording and must remain in an
annotation-only ledger; they are excluded from existing action training targets.

Initial visual review at 12-frame intervals confirms a wall-facing start,
stair traversal, a circular doorway and a corridor beyond it. The intended
objective is inferred from the human route and prior review. Visible doorway
pixels are observations; they do not establish an unseen world position.
Dense frame review is now complete for this episode. The annotation-only ledger
is `data/annotations/goal_feasibility_annotation_only_20261114.json`. It contains
145 rows and twelve manually estimated marker centers (133–144). Earlier rows
have no verifiable marker-center projection. Partial unidentified fragments are
not treated as visible centers. Coordinates were read at 320×180, with about
five review pixels uncertainty; no interpolation or world coordinates were used.

## Initial diversity evidence

The reviewed split contains seven TRAIN and two development episodes; all but
one TRAIN episode use dataset_test1, with one dataset_test9 TRAIN example.
Repeated white-wall and pitched-down starts in test1 lead through the same
stairs and exit; these provide recovery variation, not demonstrated alternate
goal choice. The two dark-wall examples are development holdouts and remain so.
Counts of semantic waypoints require visual review of the other TRAIN clips.
No paired different-goal examples have yet been established.

## Evaluation interface findings

`wrapper.py` provides console command submission and map load with `getpos`
polling; its readiness check recognizes a `setpos` response but does not verify
an expected pose. `portal_assets/scripts/vscripts/shionn_events.nut` emits
chamber_ready, goal_reached and episode_failed, and exposes ResetEpisodeSignals.
The goal event is chamber-wide and has no destination identity. These are useful
building blocks but do not yet prove repeatable starts or per-waypoint success.
Any planned pose reset must also verify velocity, game state and event-buffer
isolation; this investigation has not executed console commands or gameplay.

## Completed diversity screen and readiness decision

All seven TRAIN clips were decoded at five equally spaced frames each, including
both endpoints. This is a bounded semantic screen, not an exhaustive object
inventory. Six test1 clips converge on the same stair-top doorway/corridor; one
test9 clip reaches its own chamber doorway/corridor. Thus two chamber-specific
exit destinations are evidenced. Test1 additionally supports candidate stair
landing and doorway waypoints; test9 supports an approach-to-door waypoint.
These are points along demonstrated routes, not observed alternative destination
choices. We cannot justify an exhaustive count of all possible waypoints from
this screen. Shared corridor art does not make the two exits independent goals
from a shared starting observation.

The white-wall starts are comparable observations, but all demonstrated routes
pursue the same exit. No reviewed TRAIN start has evidence of two different
requested goals. Counterfactual examples therefore number zero established
pairs. Relabeling frames with contradictory hypothetical goals would create
unverified action targets. Annotating more repeats cannot cure this route-choice
limitation. The two recovery development episodes remain held out, and the five
November 12 quarantined recordings contribute no eligible data here.

Decision: **not ready for meaningful goal-conditioned training**. Label feasibility
is established, but goal choice, repeatable starts, upstream target projection,
and trainer ingestion remain prerequisites. The current adapter does not locate
waypoints; future live tests must declare the projection supplier (human or
verified perception) and report this limitation explicitly. An invisible waypoint
maps to zero location plus target kind; the stable external ID is not an embedded
world coordinate. A goal hidden from the beginning cannot be distinguished by
this input alone. Start the first pilot with both alternatives visible and a
supplied projection, then test temporary occlusion separately.

## Reproducible evaluation-start design (proposed, not live verified)

1. Freeze map asset hashes, event-script hash, checkpoint SHA-256, full model and
   decoder configuration, capture/control FPS, resolution, seed and run command.
   Freeze a start registry before collecting candidate scores. Each entry needs
   map, start ID, source provenance, position XYZ, pitch/yaw/roll, initial velocity,
   object/door state, two destination IDs and destination geometry. These values
   must be obtained from verified engine telemetry or map assets, never images.
2. Reload the map for every attempt, await the current load acknowledgement and
   player spawn, release held inputs, clear old event messages, and reset episode
   signals. Prefer authored spawn entities; otherwise validate engine setpos and
   setang via the existing command channel in a supervised user run. Cheat or
   permission requirements and persistence must be documented before use.
3. Verify observed pose against the registered pose (proposed tolerances: 1 engine
   unit position, 0.5 degree orientation, speed below 1 unit/s over 0.5 seconds).
   Abort on missing telemetry, stale responses, collision, object-state mismatch
   or failed readiness acknowledgement. The current wrapper does not enforce
   these checks; command submission alone is not proof of reset.
4. Run existing focus/confinement, bidirectional mouse preflight and settle checks.
   Reapply and verify the registered start after preflight because preflight moves
   the camera. Reset policy history and recurrent memory, set the requested goal,
   then start recording/scoring only after the final pose verification.
5. Use a fixed baseline start and preregistered safe perturbations (proposed ±8
   units horizontal and ±10 degrees yaw, verified collision-free). Pair every
   start/perturbation and goal across models; counterbalance order. Splits belong
   to entire start groups, including perturbations, source sessions and routes.
6. Define each destination as a verified map-space region or authored named
   trigger with a goal ID. For a waypoint, require player position inside the
   registered region for 0.5 seconds; fix its bounds before evaluation. Existing
   chamber goal_reached events may score exits only after independently checking
   which trigger emits them; they cannot score arbitrary waypoint choice. Record
   wrong destination entries, distance-to-goal change using trusted telemetry,
   completion time, stalls, timeout at 60 seconds, death and out-of-bounds.
   Keep telemetry used for scoring outside policy inputs.
7. An attempt is valid only with verified pose/state, checkpoint/config identities,
   event attribution and continuous healthy observations/inputs. Missing or stale
   frames, capture/input faults, focus/preflight failures invalidate the attempt
   and stop the batch; retain diagnostic partials. Never count invalid captures
   as navigation failures or silently replace them. Preserve raw video, actions,
   goal updates, timestamps, reset verification and result reasons.

The current event script has no run token or destination payload. A future small
interface change should attach these identities or validate a fresh event epoch;
old event-buffer text is insufficient. No brittle keyboard reset automation is
proposed. The protocol is a design and requires supervised calibration before
scores can establish anything about navigation.

## Annotation effort and validation

The selected clip lasts about six seconds at 24 FPS. All 145 frames were reviewed
on dense sheets; twelve visible centers were placed manually, and source video,
actions and metadata SHA-256 values were computed. Existing source ownership is
human; intention is retrospectively inferred, not observed metadata. Original
observations, actions and split membership remain unchanged.

Planning estimate: 10–20 minutes per similarly short episode for visual review,
point placement, provenance checks and validator pass; budget 20–40 minutes for
independent review or ambiguous transitions. This is an estimate, not a timed
human annotation benchmark. At 24 FPS, longer clips scale with visible frames and
occlusions rather than just clip duration. Sixteen short pilot trajectories imply
roughly 3–6 hours for first-pass labels, plus independent review and collection.
This single mostly-hidden target is a feasibility example, not an efficient or
sufficient training cohort.

Validation command (offline):

```bash
.venv/bin/python -m models.imitation.goals data/annotations/goal_feasibility_annotation_only_20261114.json
```

Result: valid=true, episode_count=1, errors=[]. The same ledger with
`--require-counterfactual-pairs` must fail, because annotation-only records cannot
satisfy the paired-goal gate. Validation proves structural consistency, not visual
truth, provenance-file matching or training eligibility. Source file hashes were
separately computed when creating the ledger. No code changes were required for
this feasibility exercise and no general training benchmark was rerun.

## Smallest next user action

First select **one safe shared start with two visibly distinct reachable waypoint
regions** in an existing chamber, and identify both destinations. Do not collect
another same-exit recovery batch. A supervised setup check must register and
verify that pose and both regions through game telemetry/map assets. Then obtain
one human trajectory to each target from that same verified start as a collection
feasibility pair, with explicit requested-goal metadata. This pair is development
material, not a clean evaluation holdout. Do not begin the full collection until
the pair shows valid capture, reproducible reset and clearly different actions.

After that check, the existing pilot plan calls for eight TRAIN trajectories
(four starts × two goals), eight counterfactual held-out trajectories from four
other starts, and six separate live evaluation starts. Whole-start/session split
rules and existing test11/test12 exclusions apply. The current annotation-only
ledger grants none of these memberships. Training commands and GPU cost estimates
remain premature until data and trainer ingestion are verified; the existing
500-update frozen-base adapter proposal remains a planning budget only.
