# Expert-action replay diagnostic — 2026-11-12

Purpose: establish whether a successful human trajectory can be reproduced by
recorded actions through the policy's actual input path, then compare raw mouse
labels with source-bin encoded/decoded labels. Neural policy weights are not
used for replay. This is a reset/input/action-representation diagnostic, not a
model gameplay score or a new training dataset.

## One fresh source demonstration

Portal must already be running with console port8020 in the current Hyprland
session. Use the actual installed BSP path below; it must identify the map loaded
by this game installation, not a VMF or an unrelated copy. The recording command
runs one attempt, fingerprints pose/map/settings/bin reference, and preserves
raw MP4/CSV. It uses preflight, reload and3s settling before capture. Source and
replay both suppress the chamber timer for this diagnostic.

```bash
env XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 DISPLAY=:1 \
  .venv/bin/python record_dataset.py \
  --oracle-source \
  --map dataset_test1 \
  --episodes 1 \
  --duration 30 \
  --fps 24 \
  --output DP-1 \
  --no-launch \
  --oracle-map-file '/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/portal2/maps/dataset_test1.bsp' \
  --episodes-root model_attempts/expert_oracle_20261112/sources_pending
```

Leave keyboard/mouse untouched through preparation and **ORACLE WAIT**. Move only
after **ORACLE GO**, then demonstrate the complete route from spawn through the
exit using normal mouse movement. Do not crouch or pause: policy submission does
not implement crouch, and a pause breaks the timing oracle. K or Ctrl+C cancels;
an incomplete attempt stays available for diagnosis but is rejected for replay.
This is one attempt, with no automatic failure retry. Source review comes before
interpreting the comparison.

The source must have a successful outcome, matched video/CSV length, ordered
finite capture/input timestamps, no discarded input windows, initial72idle
frames, subsequent human action, and unchanged raw/source/map hashes. Loader
bounds the source to60s/1440frames. Legacy demonstrations with unrecorded initial
pose are rejected. Leading idle frames are preserved, not trimmed or repaired.

## Replay after source review

Replace EPISODE_PATH with the completed folder printed by the recorder:

```bash
.venv/bin/python run_expert_replay.py --episode EPISODE_PATH
```

This default only validates source files and prints the two planned commands.
To execute the reviewed diagnostic on the desktop:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 DISPLAY=:1 \
  .venv/bin/python run_expert_replay.py --episode EPISODE_PATH --execute
```

The wrapper owns one persistent virtual mouse, runs raw first, and runs binned
only if raw reaches the goal. All eight binary actions stay identical. The
binned condition changes only dx/dy via the deployed source bin codec. Both use
`Portal2Controller.apply_action`: one relative mouse pulse per24Hz tick. The
older interpolating `play_csv` method is not used. Each replay independently
reloads/preflights/settles, verifies position within0.5game units and each wrapped
angle within0.5degrees, and compares sensitivity/m_yaw/m_pitch/m_rawinput. Source
and map fingerprints are rechecked before and after each attempt.

Focus/confinement, preflight and Esc/Ctrl+C remain active. Unexpected reset,
source change, failed pose/settings check, timing drift beyond125ms or catch-up
burst aborts the replay. EOF adds only12explicit zero-action release ticks
(0.5s), then records `expert_replay_exhausted` if no terminal event arrived.
This is not a model timeout or expert success. No action cap or decision-rate
change is permitted. Replay logs have `policy_kind=expert_replay`, no neural
checkpoint attribution, non-expert flags and action-lookup timing instead of
neural inference timing. Replay video is not human-labelled training data.

Reports go to fresh `model_attempts/expert_oracle_20261112/replays/oracle_*`
folders and remain pending review. No checkpoint is written or promoted.

## Interpretation

- Raw fails: investigate reset, input semantics/gain, capture timing, source
  recording and scene differences. Binned replay is skipped; no quantization
  conclusion follows.
- Raw succeeds, binned fails: representation distortion is a hypothesis to
  confirm under repeated/reversed order, not an established causal result.
- Both succeed: this trajectory is reproducible under the codec; learned action
  selection and coverage remain separate issues.

Matching pose is not full world-state equivalence. This first protocol is for
static navigation test1, not moving-object/puzzle state restoration. BSP paths
are declared installed assets and require verifying the actual game's mount.
Snapshot timestamps are not exact hardware-event/exposure times. These limits
prevent interpreting a failed open-loop replay as proof of an architecture bug.

## Engineering validation

Independent read-only review passed after fixing full idle-prefix validation
and post-attempt source revalidation. Final regression suite:398tests passed.
Mocked integration verifies no neural policy load, pose rejection before first
replay action, terminal cleanup, unexpected-reset abort, and exact source/tail
submission counts. Tests do not exercise a live desktop. Actual console-settings
parsing, pose matching, capture and expert-route reproduction remain unverified
until the user-operated demonstration and replay.
