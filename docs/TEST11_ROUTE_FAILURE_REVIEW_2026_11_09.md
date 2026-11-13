# Test11 route failure review — 2026-11-09

## Cohort and observations

Source sequence: `model_attempts/sequences/startup_settle_validation_20261109/sequence_20261109_125621_643215`. Model is the unchanged deployed v5 feedforward checkpoint. All3test11attempts are capture-valid timeouts. Their first five applied forward actions are all1; startup initiation is not the current failure.

Qualitative review samples span each video at0/2/4/6/8/10/12/15/18/22/27/32/38/44/50/58seconds. These are coarse video-time observations, not precise collisions or automatically generated expert targets.

| Attempt | Observed behavior |
|---|---|
| 1 | Reaches lower room; faces wall around6–8s; revisits upper stairs/unmarked door. Signed exit appears around32s but is not reached. No idle run>=1s. |
| 3 | Similar early lower-room approach and wall contact; repeatedly revisits upper stairs/unmarked door. No idle run>=1s. |
| 5 | Turns toward dark corner/stair region; wanders, then outputs all-zero actions from21.94to59.97policy elapsed seconds.888distinct capture frame IDs occur in the22–59s idle interval. |

The map build sheet describes spawn at level1, descending to level0and taking the lower exit branch. Repeated travel back toward upper stairs is consistent with losing that route, but does not prove a symbolic goal-recognition or memory defect. Seeing a door in an image is not evidence the network recognizes its purpose. All-zero output plus continued received frames supports a policy stall rather than a capture pipeline freeze; it does not measure exact compositor exposure timing.

Metadata/raw hashes and file links: `data/annotations/test11_route_failure_windows_20261109.json`. Test11remains validation-only; the failed autonomous inputs are not expert corrections and must not become gradient targets.

## Decision

No more broad baseline repetitions or generic training now. Preserve deployed model and the corrected startup protocol. The earlier correction and recurrent-readout experiments did not establish sufficient transfer or regression safety.

Next useful human demonstrations should combine recovery with sustained progress to the correct exit, including different starting views and a flat/hidden-exit route so every correction does not end by ascending stairs. Use TRAIN chambers only: test1for established off-route/wall examples and test9for the authored wall-avoidance/hidden-exit route. This is a small coverage pilot, not a declaration that those maps guarantee the missing behavior. Do not force bad states or prescribed actions if the model completes a chamber naturally.

With direct game control available, the following existing recorder session can collect3new corrections, separately quarantined:

```fish
.venv/bin/python run_recovery_session.py \
  --checkpoint models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt \
  --map dataset_test1 \
  --map dataset_test9 \
  --target-corrections 3 \
  --max-model-attempts 8 \
  --model-seconds 60 \
  --recovery-seconds 0 \
  --output DP-1 \
  --episodes-root model_attempts/targeted_navigation_20261109/route_episodes_pending \
  --session-root model_attempts/targeted_navigation_20261109/route_collection_sessions
```

This is NOT an SSH-only task: P is read from physical devices, and human recovery needs observable game control whose actions the recorder captures. Confirm physical mouse/capture readiness first. At a natural route loss, pressPbefore60s, hold still through the handoff, wait for STARTING RECORDING, then recover toward the correct route and continue through the exit. Back away when useful, not mechanically in every scene. Rdiscards/retries, Kstops. Successful model-only attempts retry the same chamber; maps advance after a successful human correction, so if no suitable failure appears do not manufacture one or keep retrying indefinitely—stop and report coverage unavailable.

All existing six reviewed clips keep their roles. New episodes need timing/semantic review and a declared split before any new training. Earlier dark-wall holdouts are reviewed development diagnostics; adding similar-view TRAIN examples changes the generalization question and must be stated explicitly. No new training runner or gameplay was launched by this review.
