# First fresh paired visual-goal collection

## Counts and roles

Four training-candidate demonstrations means FOUR INDIVIDUAL TRAJECTORIES, not
four pairs. Two held-out demonstrations means TWO INDIVIDUAL TRAJECTORIES.
Total: six trajectories = three complete A/B pairs = three start groups.
No frame is an independent evaluation example. The minimum held-out cohort is
one start group/two trajectories, so results are exploratory within this room,
not a statistically convincing or unseen-layout navigation result.

The two exposed pairs (doorway and interior_01; four episodes total) remain
outside this fresh split. Their source-video hashes and start IDs/poses are
excluded by the versioned collection plan. No existing membership is changed.

| Group | Intended later role | Registered start | Heading/position requirement | Goal order | Logical session |
|---|---|---|---|---|---|
| train_01 | training candidate, pending review/approval | goal_choice_train_01 | new left-side open-floor position; both landmarks visible; choose a modest heading change | A then B | goal_choice_train01_session_01 |
| train_02 | training candidate, pending review/approval | goal_choice_train_02 | new right-side position, different heading | B then A | goal_choice_train02_session_01 |
| eval_01 | reserved evaluation candidate, whole pair | goal_choice_eval_01 | different depth/lateral position and heading; freeze before training | A then B | goal_choice_eval01_session_01 |

Actual numeric poses are NOT registered yet. They require user getpos evidence,
collision/visibility checks and absence of trigger overlap. We do not invent safe
coordinates from the screenshots. The plan's provisional minimum is 128 engine
units between new start groups and from either exposed start; new group camera
orientations must differ by at least 5 degrees. These rules reject renamed nearby
starts; they do not prove semantic/layout independence. If safe positions cannot
meet them while both goals stay visible, revise the design explicitly before
collection rather than silently weakening checks. Both demonstrations in a group
use exactly the same registered position/orientation and config bytes.

## Step 1: register the FIRST new start only

1. In Portal 2, reload goal_choice_v1 yourself. Walk to an open-floor position on
   the left side, away from both waypoint volumes and at least 128 units from the
   previous doorway/interior starts. Keep BOTH cube and floor button visible.
   Do not enter A/B during setup: those trigger_once volumes must remain unused.
2. Keep a level view and choose a modest different heading. About yaw 85° is a
   proposed first-group direction, NOT a required measured value. Later groups
   need different headings; do not keep all three at yaw 90°. Do not change FOV.
3. Run `getpos` in the game console and save a screenshot of this view. Copy the
   entire printed setpos/setang line. The episode timer may expire during setup;
   that is okay for registering a pose because each recording reloads the map.
4. In a project terminal run the registration command below, replacing ONLY the
   quoted example placeholder with the exact printed line. Do not use the old
   interior_01 coordinates. This tool is OFFLINE: it writes a new registry and
   never connects to the game.

```bash
.venv/bin/python paired_goal_collection.py register \
  --group train_01 \
  --pose 'PASTE THE COMPLETE setpos ...;setang ... LINE HERE' \
  --output configs/goal_choice_pilot_starts_v1.json
```

The placeholder is intentionally not a usable coordinate. Alternatively send the
screenshot/getpos line to Codex for registration. Existing registries are never
overwritten. A too-close pose is rejected before output is written. The original
configs/registered_starts_v1.json remains unchanged.

5. Send/inspect the starting screenshot to confirm both landmarks and safe routes.
   Registration does not certify collision safety, goal visibility, FOV or trigger
   overlap. The recorder later restores/verifies the pose and rejects setup
   waypoint/terminal events. If registration fails, select a genuinely different
   safe point and repeat; do not force an exposed pose through with another ID.

## Step 2: collect A, then B — ONLY after the new start is checked

Portal 2 must be running with netconsole. Use the active desktop terminal and
confirm DP-1 is still the intended output. The actual installed BSP must exist at
the --map-file path below. If your installation differs, use its real BSP path;
never hash a stale SDK copy. Do not change the compiled map or scripts mid-pair.

The exact A command, after Step 1 creates the named registry:

```bash
.venv/bin/python record_recovery.py \
  --map goal_choice_v1 \
  --registered-start goal_choice_train_01 \
  --start-config configs/goal_choice_pilot_starts_v1.json \
  --collection-plan configs/goal_choice_collection_plan_v1.json \
  --collection-group train_01 \
  --collection-session goal_choice_train01_session_01 \
  --map-file "/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/portal2/maps/goal_choice_v1.bsp" \
  --requested-waypoint A \
  --manual-focus --session-controls \
  --max-seconds 20 --fps 24 --width 1920 --height 1080 --output DP-1 \
  --episodes-root model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01/A \
  --result-file model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01/A/result.json
```

Wait for restored-start verification. Return focus without changing the view or
holding movement keys. Press P when prompted. Wait for STARTING RECORDING, then
walk to A, the RIGHT floor-button approach region. Do not activate the button.
Region entry ends recording automatically. Do not press K to mark success.
If capture/start/event checks fail, stop and review the error; failed raw artifacts
remain diagnostic, and no training approval is issued.

Then the B command reloads the map and restores the identical start:

```bash
.venv/bin/python record_recovery.py \
  --map goal_choice_v1 \
  --registered-start goal_choice_train_01 \
  --start-config configs/goal_choice_pilot_starts_v1.json \
  --collection-plan configs/goal_choice_collection_plan_v1.json \
  --collection-group train_01 \
  --collection-session goal_choice_train01_session_01 \
  --map-file "/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/portal2/maps/goal_choice_v1.bsp" \
  --requested-waypoint B \
  --manual-focus --session-controls \
  --max-seconds 20 --fps 24 --width 1920 --height 1080 --output DP-1 \
  --episodes-root model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01/B \
  --result-file model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01/B/result.json
```

Walk to B, the LEFT cube approach region, without pickup. Keep only this first
pair for now. Two 20-second caps mean at most 40 seconds of recorded policy-free
human action; allow 15–30 minutes for start registration, checks, setup and review.

For exact commands generated from the registered config rather than copying:

```bash
.venv/bin/python paired_goal_collection.py commands \
  --group train_01 \
  --start-config configs/goal_choice_pilot_starts_v1.json
```

This PRINTS commands; it never executes them.

## Step 3: bounded first-pair validation

Run the read-only integrity check:

```bash
.venv/bin/python review_paired_goal_capture.py \
  --start-config configs/goal_choice_pilot_starts_v1.json \
  --root model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01 \
  --output model_attempts/goal_choice_pilot_pending/goal_choice_train01_session_01/integrity_review_v1.json
```

Expected successful first-stage status:
`integrity_passed_visual_annotation_review_pending`. This is NOT training
approval or release of the rest of the batch. Send Codex the result. The tool
checks correct single A/B events, aligned receive time, valid capture, frame/action
counts, source-frame IDs, finite actions, absence of interactions/jumps, final
stable poses, shared start/registry/map/capture identities, frozen plan/session
provenance and exclusion of exposed/duplicate source videos.

Codex then reviews the COMPLETE two videos/actions, compares initial views,
labels waiting/onset/approach boundaries and selects one persistent native RGB
crop per episode. Use annotation_only with the planned start_group_id and logical
source_session_id. Include top-level waypoint_goal_ids mapping A/B to distinct
visual IDs, e.g. goal_choice_v1_A and goal_choice_v1_B. Every frame must be labeled;
keep original observations/actions. Only the landmark crop is a model input;
pose/trigger/map telemetry is setup/scoring-only. Initial crop must come from
that episode's frame zero and be tied to its video/frame/crop hashes. If the target
is not visible at capture start, reject/reposition this first-pair scenario.

The current fixed-goal visual dataset supports no goal changes within an episode.
It verifies native crop pixels, source metadata/map/ownership and hashes, and
keeps waiting as context with masked loss. A/B label consistency is checked against
actual recorded requested_waypoint, not just filenames. Approximate per-frame
landmark centers remain feasibility labels, not invisible trigger projections.

With reviewed labels/crops and a written visual review, Codex reruns the check
using --annotations, --targets and --visual-review, writing a NEW review path.
Only status passed releases future commands. The review binds annotations to all
three actual capture hashes and resolved source paths; passing never grants
training membership. FOV is explicitly unmeasured in current metadata: leave the
in-game setting unchanged and inspect matched initial views; do not claim numeric
FOV equality from resolution/pose alone. Hand visibility checks remain necessary.

## Remaining matrix, after the first pair PASSES

Register train_02 into a NEW registry version using --base for the previous
registry, then eval_01 into another new version, never editing the first start.
Register and freeze all three positions/headings before requesting full-stage
commands. Reserve eval_01/session as an entire group; do not use its trajectories,
crops or results for model fitting or protocol tuning. Quality review/annotation
must remain separate from candidate selection. Any evaluation used for tuning
becomes development and needs a new holdout under an explicit protocol change.

After first_pair_review_v1.json is actually passed, the command generator accepts
--stage full --first-pair-review PATH for train_02 and eval_01. It requires all
three registered starts, intact evidence and unchanged first-start pose; the
recorder checks later BSP/capture settings against the first reviewed pair.
These commands are deliberately not released before those conditions hold.

No new trajectories are currently training-approved. Four candidate TRAIN + two
reserved evaluation clips are a minimal data-contract pilot, not adequate proof
of generalization. Keep complete groups/sessions together, use trajectories/group
counts rather than frames as independent N, and retain a later distinct-layout
cohort. Explicit user approval and the existing versioned training-approval
contract are required before gradients. GPU cost remains unmeasured.

## Effort and schedule

Provisional: Nov 15–16 first registration/pair, 15–30 minutes setup/collection and
20–40 minutes labels/review. Only after it passes, Nov 16–18 register/reserve the
remaining groups and collect four trajectories, 30–60 minutes setup/collection and
40–80 minutes labels/review. Nov 18–20 review all membership/provenance and prepare
the bounded experiment; no training is scheduled automatically. Full six-clip
budget remains about 45–90 minutes setup/collection and 60–120 minutes annotation.
Dates are planning targets; first-pair faults take precedence over collecting more.
Critical dependencies: safe visible new poses, healthy capture and unspent triggers,
complete crop/phase labels, reserved group identities and explicit approval.

## Implementation validation

The new offline tools and setup guard are CPU/mocked-tested; no game or collection
was executed by Codex. Numerical candidate poses used by tests are synthetic and
were NOT published as safe map coordinates. Existing raw captures, registered
interior_01 config, splits and deployed checkpoints remain unchanged. Current
blocker is the user's actual NEW pose/visibility evidence, not missing software.

Final offline validation: 489 regression tests passed in 23.692 seconds; CLI help
and Python compilation checks passed. Independent read-only review passed the
source-linkage, setup-event, role-reservation and first-pair release gates. No
live map/reset/recording action was executed. The remaining manual FOV/visibility
and collision checks cannot be certified by these tests.
