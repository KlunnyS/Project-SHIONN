# Handoff observation-context calibration — 2026-11-10

## Purpose

The assisted recovery recorder currently stops the policy process before starting the human recorder. This leaves roughly 10–11 seconds without recorded observations at takeover. The optional context observer is intended to record an unlabelled camera stream across both subprocesses and that startup interval. This is a capture-continuity experiment; it does not repair timestamps, add action labels, or establish that the stream is suitable for training.

## Current implementation

`run_recovery_session.py --record-context` starts a separate Wayland observer before each model attempt, keeps it alive through human recovery, then finalizes it before counting the correction. `--no-model-video` avoids also encoding the model runner's video; model action diagnostics and the human recorder's own aligned video/actions remain enabled. The context directory contains `context.mp4`, `frames.jsonl`, `metadata.json`, and orchestration-only `links.jsonl`. It has no `actions.csv`, declares `expert_targets=false`, `training_approved=false`, and must never be loaded as expert targets.

The read-only audit command checks capture lifecycle, decoded video/frame index count, timestamp continuity, model/human provenance, and coverage of the handoff interval. A passing audit is structural only: the observer is an independent camera stream, so scene alignment and useful image quality still need human visual review. Do not merge frames with the human episode or train on this context stream.

## One live calibration

Do this only while Portal 2 is already running in the active Wayland desktop and you can operate its keyboard/mouse. SSH-only access without a way to perform the physical P takeover and human recovery is not sufficient. The command does not launch the game. It uses the deployed checkpoint and one TRAIN chamber, and writes to fresh pending directories.

From the repository root, in the desktop user's shell (set the session variables if this is an SSH shell):

```bash
env XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 DISPLAY=:1 \
  .venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --target-corrections 1 \
  --max-model-attempts 3 \
  --model-seconds 60 \
  --recovery-seconds 0 \
  --record-context \
  --no-model-video \
  --output DP-1 \
  --episodes-root model_attempts/targeted_navigation_20261109/context_calibration_episodes_pending \
  --session-root model_attempts/targeted_navigation_20261109/context_calibration_sessions
```

1. Let the policy run. At a clear wrong-route/wall-recovery opportunity, before simply waiting for its timeout, press **P on the physical keyboard** to hand off.
2. Leave keyboard and mouse untouched during the runner's camera preflight. Wait for the human recorder's **STARTING RECORDING** message before moving the view or navigating.
3. Demonstrate recovery toward the intended route and continue through the chamber exit. This creates the actual human-labelled correction; the context observer itself remains unlabelled.
4. Use **R** only to discard/retry the chamber, **K** to stop the session, and **Esc/Ctrl+C** for emergency stop. Do not run a second calibration automatically if capture fails.

`--recovery-seconds 0` leaves the human recorder without a code-imposed episode timer. The run stops after one successful correction or three model attempts. If there is no representative recovery opportunity, stop with K rather than inventing a correction.

## Review before further collection

Use the session path printed at startup. Find the successful numbered attempt's `observation_context` directory in `session.jsonl`, its model `attempt_*.jsonl`, and the human episode path from the matching recovery row. Then run:

```bash
.venv/bin/python audit_observation_context.py \
  --directory PATH_TO_ATTEMPT/observation_context \
  --model-log PATH_TO_ATTEMPT/attempt_....jsonl \
  --human-episode PATH_TO_HUMAN_EPISODE
```

Inspect `context.mp4` around the model-to-human transition and compare it with the human episode video. Confirm the camera view is current and visually continuous, the handoff interval is covered, and the human clip contains the intended off-route recovery through the exit. A passing structural audit alone does not approve the episode or context for training. If context capture fails or visual continuity is poor, retain the files for diagnosis and do not start batch collection or correction training.

## Validation performed before calibration

- Focused mocked session and context tests: 23 passed.
- Independent code review confirmed default behavior remains unchanged, test mode rejects real context capture, the observer owns a separate process group, and capture finalization precedes correction counting.
- No live Wayland dual-capture, encode-throughput, display alignment, or gameplay test has been performed by the agent. Those properties remain unverified until this one user-operated calibration.
