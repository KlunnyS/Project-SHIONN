> Historical record. Use [the current navigation guide](../../NAVIGATION.md) for operational instructions.

# Two-waypoint collection feasibility protocol

Status: preparation only, 2026-11-14. No gameplay or collection has been executed.
Goal: validate one shared start and two different intermediate destinations. One
chamber exit is sufficient; the two targets can be safe floor/landing waypoints.

1. On the desktop, choose an existing safe navigation chamber with two visible,
   independently reachable floor regions left and right from the same start.
   Neither target may require pickup, portals or visiting the other first. Select
   distinctive visual landmarks whose projected centers can be labeled. Avoid
   two points on the same forced corridor. Name them `waypoint_A` and `waypoint_B`.
2. Before recording, register the map hash, start XYZ and pitch/yaw from verified
   game telemetry/map assets, stationary state and both destination regions.
   Register initial door/object state too. A screenshot alone does not verify
   world pose. Use the reset-verification protocol in
   `GOAL_ANNOTATION_READINESS_2026_11_14.md`. If no repeatable reset is available,
   stop at setup diagnosis; do not label approximate manual starts as paired.
3. Start Portal 2 yourself and confirm capture output with `wf-recorder -L` in
   the active desktop terminal. Verify the game console connection and both
   physical turning directions. Follow existing focus/capture safeguards. A
   remote shell needs the actual desktop environment, not guessed display IDs.
4. Use existing manual human collection, one trajectory per target. The command
   below is a template: replace the map placeholder with the selected installed
   chamber. Do not use shell angle brackets. The recovery recorder does not load
   or reset the map and does not set requested-goal metadata automatically.

```bash
.venv/bin/python record_recovery.py \
  --map SELECTED_MAP \
  --manual-focus --session-controls \
  --max-seconds 20 \
  --output DP-1 \
  --episodes-root model_attempts/goal_navigation_pending/two_waypoint/waypoint_A \
  --result-file model_attempts/goal_navigation_pending/two_waypoint/waypoint_A/result.json
```

Use the actual capture output. Review the recorder's printed start/stop controls
before acting. At the verified common start, record human movement toward A;
continue to the defined region and finish under the recorder's supported
termination behavior. A 20-second partial can support waypoint review if capture
is valid; a chamber terminal event is not required to establish waypoint reaching.
Do not force a successful episode label. Restore and verify the identical start,
then repeat with both output paths changed to `waypoint_B`. Avoid extra practice
recordings or multiple repeats until the first pair is reviewed. Timeout-disable
is omitted because it changes chamber behavior and requires verified script
acknowledgement; use a chamber whose normal limit permits the short clip.

5. For each recording create a sidecar scenario card containing start ID, exact
   verified pose/orientation and verification log, goal ID/definition, intended
   destination (declared before movement), session ID, source map hash, action
   owner `human`, outcome and any assistance. Save it beside the result file;
   do not alter original actions or fabricate recorder metadata.
6. Review metadata `capture_valid`, diagnostic reasons, video decoding and
   continuity, input/action timestamps, short taps, and actual target entry.
   `invalid/` recordings are diagnostic only. No source is training-approved by
   placement under pending paths. Preserve raw waiting frames and complete
   trajectories; subsequent sampling decisions belong in an explicit manifest.
7. Produce a separate `shionn_goal_annotations_v1` ledger covering every frame.
   Use `target_kind: waypoint`, stable requested goal ID, human action owner,
   normalized x/y in [-1,1] with top-left (-1,-1) and bottom-right (1,1), and null
   coordinates when no projection can be verified. Record manual uncertainty and
   distinguish observed landmarks from inferred identity. Keep goal identity
   stable during occlusion; never carry stale image coordinates forward.
8. Hash video/actions/metadata and use `split_role: development` for both clips,
   the same verified start_group_id, and distinct requested_goal_id values.
   This is exposed collection feasibility material, never a clean final holdout.
   Use the existing ledger as a structural example, not as a source of labels.

```bash
.venv/bin/python -m models.imitation.goals PATH_TO_PAIR_LEDGER.json --require-counterfactual-pairs
```

9. Manual acceptance requires valid capture, genuinely comparable starts, two
   distinct target regions, different meaningful early actions, both destinations
   reached, correct frame-level projections and ownership, and no split leakage.
   The validator checks structure and declared groups; it does not prove actual
   pose equivalence, target truth or source-file hashes. Independently inspect
   these. If a target is invisible from the start, the present vector cannot
   convey its direction; choose another pair for this first test.
10. Stop after this pair and review it with Codex. Only then define the expanded
    whole-start/session split and prepare trainer ingestion. No training command
    is authorized or ready from this feasibility procedure.

## Implemented waypoint recording — 2026-11-15

User-verified chamber mapping: A = RIGHT floor-button area; B = LEFT cube area.
User reported both waypoint printouts plus ready/exit/timeout working. Use
SignalWaypointA()/SignalWaypointB() in Hammer; quoted parameters failed in the
observed setup. Same position was reported across two reloads:
(320,-96,64.03125), yaw approximately 90, pitch approximately zero, with deviations
under 0.5 degrees. This verifies two observed resets, not an automated pose gate.

The recorder now supports `--requested-waypoint A|B`. This is separate from the
visual goal annotation and does not feed engine signals into a neural policy.
Metadata records requested_waypoint, human ownership, development_pending role,
training_approved=false and received events with monotonic receive timestamps.
Timestamped trigger receipt is not exact render exposure time. Stop outcomes:
waypoint_reached (requested), wrong_waypoint (other), wrong_destination (final
exit first), timeout, and existing invalid/interrupted safeguards. Unknown waypoint
payloads invalidate captures. Healthy failures are retained as failures, not
expert successes. No dataset membership or annotation approval is automatic.

First collect A only: reload goal_choice_v1 yourself, stay at the verified start,
then run from the active desktop terminal:

```bash
.venv/bin/python record_recovery.py \
  --map goal_choice_v1 \
  --requested-waypoint A \
  --manual-focus --session-controls \
  --max-seconds 20 \
  --output DP-1 \
  --episodes-root model_attempts/goal_navigation_pending/two_waypoint/waypoint_A \
  --result-file model_attempts/goal_navigation_pending/two_waypoint/waypoint_A/result.json
```

Return focus to the game without changing the view; press P after the printed
preparation delay and wait for the recording-start message. Walk toward the
RIGHT floor button, stop in its trigger region and do not press/use it. Recording
stops on the waypoint event. Do not press K to finalize a successful demonstration:
K is an interruption and invalidates the clip. The chamber timeout continues from
map start; if preparation exceeds it, reload and retry, never relabel the result.
Manual-focus skips automatic mouse checks: verify physical turning separately
before the paired recording and then reload to restore the start.

Review A's result/metadata/video first. For B, reload to the same start and change
--requested-waypoint and both path suffixes from A to B, then walk LEFT toward the
cube without pickup. No auto-reset, map-launch, actual-pose verification or crop
tracking was introduced. Goal crops and complete frame annotations remain separate
required work after capture. This first pair stays development-only.

### Automatic map load

Add `--load-map` to record_recovery.py to load --map through the existing bounded
chamber-ready handshake after console connection, before input/camera preparation.
Portal 2 must already be running. Without this flag, existing current-state
recovery behavior is unchanged. Manual P still controls recording start.

A user capture failed with frame IDs 231→233. This is a latest-camera sampling
skip, not evidence that manually loading the map caused a fault. Automatic loading
does not change the strict frame-gap safeguard. That check needs a separate timing
audit before relaxing it; preserve the invalid recording for diagnosis.

### Registered interior start

Use `REGISTERED_STARTS.md` for the new interior_01 commands and verification
contract. This optional feature restores a versioned pose after reload and checks
it again before recording; omitting it preserves the original workflow. Actual
live restoration remains to be user-verified. No new dataset membership is implied.
