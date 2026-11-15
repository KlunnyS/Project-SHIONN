# Persistent visual target preparation — 2026-11-15

> Dated snapshot. Current status: [navigation guide](NAVIGATION.md). Current collection commands: [paired-goal protocol](PAIRED_VISUAL_GOAL_COLLECTION_2026_11_15.md). Later implementation may supersede statements below.

User approved region entry as success. No settled-stop label is required; do not
invent stop actions. The existing two clips remain development_pending and their
annotation ledger remains annotation_only. No training or gameplay was run.

## Implemented

`models/imitation/visual_targets.py` verifies original video/action/metadata
SHA-256 values before decoding, requires capture_valid=true, checks bounded
frame/action counts and initial-wait boundaries, and returns full RGB frame
history, existing action semantics and a separate loss mask. It rejects TRAIN
access unless explicit ledger approval and source metadata permit it. It is an
offline helper, not integrated into train_bc or a recurrent batching pipeline;
use it on a schema-validated ledger. It does not generate membership approval,
truncate episodes, share hidden states or convert engine telemetry to features.

Persistent native-resolution PNG crops and descriptors are under
`data/annotations/visual_targets/goal_choice_pair_v1/`. A depicts the right floor
button; B depicts the left cube. Both were visually inspected after extraction.
Descriptors include native/source dimensions and bounds, source-video and decoded
frame hashes, crop hashes, fixed frame-zero identity, development-only status and
initial-wait loss-mask count. The PNG is stored losslessly; RGB interpretation
applies after decoding, while OpenCV internally uses BGR. Crop context includes
some surrounding floor; landmark appearance/position shortcuts still need tests.
The two crops are visual target evidence, not learned goal embeddings. Model
integration, encoder preprocessing choice and counterfactual inference remain
pending. Existing 11-value adapter/checkpoints were not reinterpreted.

Reproduction (choose a NEW output directory; existing evidence is never replaced):

```bash
.venv/bin/python -m models.imitation.visual_targets \
  data/annotations/goal_choice_visual_annotations_v1_20261115.json \
  --output-dir /tmp/shionn_goal_targets_recheck
```

Initial-wait losses are disabled for A[0,111) and B[0,29). All 177/104 frames
remain available as history. Later batching must avoid letting unequal waiting
lengths identify the requested goal and use matched burn-in/window sampling.

## Small collection plan, not yet a recording request

Keep the current default-spawn pair as exposed development material. Register
THREE new starts inside the open room, with both targets visible and no collision:
two proposed training start groups, one proposed evaluation start group. Use
separate sessions per split and reserve the evaluation group before using its
images to tune the model. No automatic membership change is authorized. Two
trajectories per start (A/B) gives six proposed new clips. This is the minimum
split feasibility study, not unseen-layout evaluation; fresh layouts remain later.

For each start, use a stable, level forward view and record its actual getpos
position/angles and map hash. Register requested goal before movement and verify
the reset for both demonstrations. Use the same FPS/resolution/output/preparation
settings. No use, pickup, jump or portals. First correct waypoint entry ends
recording; other-waypoint or final-exit-first is failure. Waiting remains recorded
but is explicitly annotated, not trained as navigation. Validate every clip before
accepting it into the proposed role. Keep the original captures and all invalids.

Do not collect all six until one new registered start pair passes reset/capture
and label review. Prior estimate: 45–90 minutes setup/collection and 60–120 minutes
labeling/review for the six-clip stage; measure actual time on the first pair.
Dependencies: per-clip start/config evidence, crop-aware batching/approval checks,
source compatibility audit and fixed counterfactual protocol. Nov 15–17 tooling,
Nov 18–22 supervised preparation remains provisional; no automatic training date.

## Exact next user action

Load the chamber yourself and walk through the entrance into the open room.
Stop at a candidate new start where the doorway no longer fills the view and
both landmarks remain visible. Level the camera, take a screenshot and run getpos.
Send that screenshot and the setpos/setang output. This registers a candidate,
not a training clip or clean test start. Do not record a batch yet. Avoid visiting
waypoints during setup; reload before any actual collection so trigger_once
volumes and timeout state are fresh.

## Validation

Six focused tests cover wait-mask/history preservation, training rejection,
tampered source hash, missing frames, bounded frame budget, phase ranges,
persistent crop hashes and overwrite prevention. The real pair was hash-verified
and decoded for crop extraction; originals were not edited. Full suite results
are reported in the final response. No GPU profiling/training or live inference
was performed. The strict frame-gap safeguard remains an open timing audit item.
