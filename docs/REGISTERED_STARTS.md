# Registered interior starts

Implemented 2026-11-15; offline-tested only. No live game movement or recording
was performed by Codex. Original commands retain doorway/current-state behavior.

## Contract

`record_recovery.py --registered-start NAME` implies --load-map. It loads the
specified map through the existing bounded chamber-ready handshake, releases
policy-held actions, restores setpos/setang and verifies two fresh pose samples.
It then prepares input/camera monitoring and waits for the user. Immediately
before test monitoring or episode writing it verifies again, and discards pending
preparation input. Failed verification raises an error and writes a failed result
without starting a video/CSV episode. Existing capture invalidation remains active.

Configuration schema is shionn_registered_starts_v1. The initial entry in
`configs/registered_starts_v1.json` is goal_choice_interior_01, map goal_choice_v1,
position [320,26.443790,64.031250], angles [0.001461,90,0]. These values came from
the user's screenshot/getpos; actual restoration still requires supervised check.
A/B must use this same entry/configuration and the same capture/control settings.

Fresh getpos replies are bracketed by unique whole-line markers with a two-second
per-query limit and bounded 64-KiB buffer. Echoed/stale/partial/ambiguous responses
cannot authorize recording. Position tolerance is Euclidean distance ≤1 engine
unit; wrapped per-axis camera tolerance ≤0.5°. Two readings separated by 0.1s
must move ≤0.25 units. This is a stability proxy, not measured engine velocity.
Restoration uses at most two attempts, then fails. Every query is bounded; worst
case setup may take tens of seconds, so the existing 60-second chamber timer still
matters. No timeout suppression or expansion is automatic.

The precise development mechanism is setpos/setang, using sv_cheats temporarily.
The existing cvar setting must be readable before positioning; its original value
is restored in finally and read back. If restoration cannot be verified, recording
is refused; the error may require checking the cvar manually. No keyboard movement
sequence or guessed spatial offset is used. The player should leave all movement
controls untouched during positioning and verification; release_policy_actions
cannot forcibly release physically held keys.

Metadata records start identity, config SHA-256/path, requested pose, tolerances,
two measured restore samples, attempt count and final pre-recording verification.
All pose telemetry is setup/evaluator-only and never an input to the visual policy.
The helper is reusable by later benchmarks/reset adapters, but is not automatically
wired into run_model_sequence or PPO. Map/object/door state and actual velocity
are not certified by pose alone; map reload plus chamber validation remain required.
No candidate/training membership is approved by a successful start check.

## First user-run A command

Portal 2 must already be running with netconsole and healthy desktop capture.
Run from the active Wayland desktop terminal. The recorder loads the map itself.

```bash
.venv/bin/python record_recovery.py \
  --map goal_choice_v1 \
  --registered-start goal_choice_interior_01 \
  --requested-waypoint A \
  --manual-focus --session-controls \
  --max-seconds 20 --output DP-1 \
  --episodes-root model_attempts/goal_navigation_pending/interior_01/waypoint_A \
  --result-file model_attempts/goal_navigation_pending/interior_01/waypoint_A/result.json
```

Wait for the restored-start message. Return focus to the game without changing
camera or position. Press P when prompted; wait for STARTING RECORDING, then walk
to A (RIGHT floor button) without pressing it. Correct region entry stops recording.
If verification fails, do not manually compensate during the run: inspect the
error and retry from a fresh invocation after correcting the actual fault.
Keep this first new-start pair development_pending for protocol review.

## B command, after A review

```bash
.venv/bin/python record_recovery.py \
  --map goal_choice_v1 \
  --registered-start goal_choice_interior_01 \
  --requested-waypoint B \
  --manual-focus --session-controls \
  --max-seconds 20 --output DP-1 \
  --episodes-root model_attempts/goal_navigation_pending/interior_01/waypoint_B \
  --result-file model_attempts/goal_navigation_pending/interior_01/waypoint_B/result.json
```

Walk to B (LEFT cube) without pickup. Do not use K to finalize success; K interrupts
and invalidates a clip. Wrong waypoint/final exit remains a distinct failure.

## Register another start

1. Choose an unused start ID and safe stationary position in the desired chamber.
2. Read getpos yourself, save the screenshot and source/provenance. Use actual
   setpos/setang values; coordinates are setup evidence, never visual labels.
3. Copy the config to a NEW versioned filename, preserving historical config bytes.
   Add an entry with unique id, map, three-number position/angles and provenance.
4. Use --start-config PATH --registered-start ID with the appropriate --map.
   Map mismatch, missing/duplicate IDs and nonfinite coordinates are rejected.
5. Hand-check the first restored view, collision safety and unchanged object state.
   Compare metadata/config fingerprints for A/B and review before approving any split.

Existing default-spawn commands still work by omitting --registered-start. Random
start generators may select a registered entry later, but must log the realized
configuration and satisfy the same checks. No randomization is implemented here.

## Validation

Twelve focused offline tests cover named configs/CLI defaults, map mismatch,
nonfinite coordinates, fresh chunked marker replies, timeout/command failure,
tolerance/wrapped angles, two-sample stability, successful retry and cleanup,
bounded failure, metadata/waypoint integration and final-pose failure refusal.
Full regression suite results are reported with the implementation. Live behavior,
getpos cvar formatting and positioning permission must still be checked by user
execution. Existing strict camera-gap failures remain a separate known issue.

### Live startup settling correction — 2026-11-15

User's first attempt failed with 47.294-unit/0.522-degree error, while subsequent
manual getpos showed the exact requested position and 0.372-degree yaw error.
This is consistent with transient positioning/load settling, though intermediate
pose telemetry was not logged. Verification now polls within its original
two-second deadline for two consecutive in-tolerance stable samples. It does not
relax spatial/angular limits or extend episode timeouts. Persistent mismatch still
fails and now includes measured XYZ/angles. Thirteen focused tests pass, including
transient mismatch followed by stable success and persistent instability rejection.
