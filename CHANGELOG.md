# Changelog

All notable changes to Project SHIONN are documented in this file.

## Unreleased

### Added

- Add supervised recovery sessions: P hands off a live model failure to a
  correction recording without a map reload, and successful corrections
  advance to the next training chamber with linked attempt logs.
- Keep the configured model time limit while suppressing chamber timeout
  during supervised sessions; reacquire input focus and leave human
  correction recording unlimited by default.
- Add physical Escape pause/resume and K stop controls to normal, recovery, and
  low-level recording. Paused intervals emit no frames/actions or duration time;
  manual stops preserve partial episodes under `interrupted`.
- Add a training operations runbook covering the current v5 resume point,
  systemd resource containment, monitoring, file-descriptor requirements,
  checkpoint retention, and documentation maintenance.
- Add the 13.2M-parameter residual v5 imitation policy with GroupNorm, SiLU,
  squeeze/excitation, dropout, and a spatial 3x5 visual bottleneck.
- Add whole-chamber validation through repeatable `--holdout-map` arguments.
- Add temporally consistent brightness and contrast augmentation, optional
  label-corrected horizontal flips, label smoothing, gradient clipping, and
  configurable AdamW weight decay.
- Add tests for whole-chamber split isolation and mirrored action labels.
- Add regression coverage for sample-weighted partial batches, persisted
  scheduler/early-stopping state, and legacy patience recovery from metrics.
- Add opt-in live mouse-delta limiting with separate raw/applied JSONL actions,
  plus sustained `policy_freeze`, `wall_stuck`, and `turn_loop` event records.
- Log physical mouse movement, Portal 2 camera angles, and Hyprland cursor/focus
  during recovery handoff; allow real correction runs without model-attempt MP4s
  to isolate encoding-related differences from test mode.
- Recognize Portal 2's fullscreen Gamescope wrapper as a focus target while
  leaving Gamescope's forced relative mouse grab untouched.
- Refresh Gamescope's intermittent mouse grab at recovery handoff by briefly
  switching to an empty workspace and back before the human focus check.
- Re-focus Portal 2 before a new recovery-session model attempt if the prior
  handoff left another window active.

### Changed

- Recenter Portal 2's confined pointer at recovery handoff and on a weak-input
  retry; require physical left/right motion to produce proportional camera yaw
  before recording, including in manual-focus mode.
- Delay recovery recording by three seconds after Escape resumes Portal 2, so
  menu frames and transition inputs are omitted from the saved clip.
- Poll physical mouse movement across the recovery focus wait and retry input
  capture once when no camera movement is detected.
- Make R discard and restart the current supervised chamber, K stop the whole
  session, and model time limits retry the same chamber; temporarily unbind and
  restore Portal 2's R binding while each phase is active.
- Replace virtual `MOUSE4` focus clicks with an XWayland XTEST left click that
  temporarily unbinds and restores `MOUSE1`; verify the physical mouse turns
  the camera before starting a recovery clip, and reject camera snaps.
- Add manual-focus and no-save test modes for supervised recovery handoff.
- Wait for the new chamber-ready event before a recovery-session model attempt
  instead of accepting a stale position from the previous chamber.
- Consolidate local, Fish, and SSH model launchers through one Bash entry point;
  require an explicit checkpoint and let the runner detect the Wayland output.
- Simplify dependency setup/check scripts without changing their permission or
  installation flags, and allow Hammer++'s directory to be configured.
- Update the README and Program Guide for the 1,300-episode frozen split.
- Add purpose descriptions and import-source comments across the Python modules,
  tests, and manual diagnostics without changing their behavior.
- Reject the experimental soft mouse-response limiter after a controlled live
  comparison reduced steering, increased freeze events, and lowered success on
  held-out chambers; retain uncapped mouse output as the supported default.
- Complete the isolated balanced-data v5 candidate after four epochs and
  early stopping, retaining `dataset_test11` and `dataset_test12` as full
  holdouts; promotion remains pending same-evaluation expert and live
  comparisons.
- Activate focused XWayland input with unbound `MOUSE4` instead of Portal 2's
  zoom-bound `MOUSE3`, preventing live runs from starting with a train/inference
  field-of-view mismatch. A live `dataset_test12` smoke attempt confirmed
  unzoomed input activation and reached the goal with zero focus losses.
- Pass the mouse cap through sequence runs and record the clean 2026-10-04
  five-chamber diagnostic result and next controlled comparison.
- Record the completed v5 periodic-checkpoint cleanup and invalidate the
  zoom-affected 2026-09-30 live sequence as an authoritative benchmark.
- Consolidate project documentation around one source of truth per subject:
  shorten the README, merge the script cheat sheet into the CLI reference,
  move chamber setup into a maintained authoring guide, and fold reusable
  mouse/recovery diagnostics into the benchmark workflow.
- Remove the superseded v1 imitation build plan and the experiment-specific
  mouse policy guide after preserving their still-current operational content.
- Reject the initial windowed v5 live benchmark after its videos exposed
  desktop chrome and a missing viewmodel; the corrected fullscreen/equipped
  sequence reached the goal on held-out `dataset_test12` and timed out on
  `dataset_test1` and `dataset_test11`.
- Make `--keep-focused` prepare a clean Hyprland fullscreen visual state and
  equip the portal gun after each map load, recording that state in attempt
  metadata before inference begins.
- Use a validation-driven learning-rate reduction followed by two-epoch early
  stopping, batch-32 validation, and 10,000-step periodic checkpoints by
  default; sample-weight epoch metrics remain comparable across batch sizes.
- Persist scheduler and early-stopping state in checkpoints, recover legacy
  patience progress from epoch metrics, and avoid an extra epoch when a resumed
  checkpoint has already met the stopping condition.
- Complete the current v5 holdout training run at epoch 6/global step 494,292;
  early stopping retained epoch 2 as the best validation checkpoint.
- Make the binned mouse head, unweighted binary losses, and a 3x
  jump-positive weight the behavior-cloning defaults.
- Use CUDA channels-last tensors, fused AdamW, persistent data workers, and
  cuDNN benchmarking where supported.
- Strengthen checkpoint resume validation for model, optimizer, dataset,
  holdout, sampling, and augmentation settings.
- Reduce default early-stopping patience from five unimproved epochs to two.
- Update the documented dataset status to 850 demonstrations across 12
  chambers and describe the v5 training and evaluation workflow.

### Removed

- Remove the unused one-off helpers `scripts/legacy/run_sequence.py`,
  `scripts/legacy/example_usage.py`, and `scripts/legacy/hammerpp_notas.sh`.
  The supported recorder, runner, replay primitive, and Hammer++ launcher remain.

### Compatibility

- Keep exact v1/v2, v3, and v4-binned network implementations available so
  previously trained checkpoints remain loadable for inference.
