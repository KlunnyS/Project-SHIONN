# Persistent visual-goal implementation — 2026-11-15

## Scope and status

The current authorized independent work is implemented: optional crop encoder,
CNN/GRU readout, inference/checkpoint support, episode-level sequence loading,
source/split approval gates, imitation-loss/epoch integration, counterfactual
screen and preparation commands. No actual-data optimizer updates, GPU work,
gameplay, collection, checkpoint promotion or membership change were performed.
The models are not trained and navigation improvement is unmeasured.

Root owned all writes. A read-only explorer checked existing contracts and a
fresh read-only verifier checked model/data/evaluation correctness. The verifier
found goal-transition and renamed-source leakage holes; both were fixed with
regression tests, along with source map/ownership consistency. Final independent
recheck passed. This is engineering evidence, not autonomous goal-use evidence.

## Architecture and input contracts

`models/imitation/visual_goal.py` adds architecture
`shionn_imitation_v5_gru_visual_goal_binned`. Existing feedforward, GRU and
11-value coordinate architectures remain separately loadable. RGB crops resize
to 96×96 with INTER_AREA and uint8/255 normalization. Four repeats of crop RGB
use the existing 12-channel scene trunk, avoiding another large visual network.
The trunk's 512 features project to a configurable embedding (default 128).
A configurable readout adapter combines the scene GRU state and persistent goal
embedding, with zero-initialized residual action logits. Warm-start construction
strictly loads/fixes the source base; no source parameter is changed. Feature
gating remains independently configurable for the scene representation.

Shapes: scene [B,T,12,180,320], crop [B,3,96,96], encoded goal [B,E], hidden
[1,B,H], valid/loss masks [B,T], existing factored action logits [B,T,C]. Model
methods accept normalized floating tensors. Loader/inference APIs consume uint8
RGB. Camera observations remain BGR only at the existing screenshot boundary.
Pixel RGB crops, not goal IDs/coordinates/triggers/velocity/distance, enter the
new model. The goal ID is bookkeeping. No changes to action semantics.

`PolicyInference.set_visual_goal(crop_rgb, goal_id=...)` encodes once under
no-grad and stores a detached target. Scene updates do not re-encode it; explicit
goal replacement does. reset_memory clears frame/hidden state but retains the
goal, while reset clears both. Checkpoints must specify the exact
VISUAL_GOAL_CONFIG, dimensions and memory config; strict state loading catches
incompatible shapes. Cached embeddings assume model weights stay unchanged,
which is the inference contract. No online updates or automatic visual target
selection/tracking are implemented. This is goal following infrastructure, not
visual localization, learned goal selection or puzzle reasoning.

The original coordinate interface is preserved. It still cannot distinguish two
initially invisible waypoint identities by itself. The crop interface carries
appearance through occlusion, but whether the network can use it is an experiment.

## Data path and training integration

`visual_goal_dataset.py` uses the existing annotation schema and per-episode target
descriptors. It verifies source video/actions/metadata SHA-256, actual capture
validity, map and human ownership, crop hash, source frame hash/shape and exact
crop pixels against native source bounds. It rejects unsupported within-episode
goal transitions rather than silently applying an old crop. ID-based session/start
split checks are supplemented with video-hash checks so renaming the same source
cannot create a holdout. Source metadata and target descriptors must explicitly
permit training; annotation-only/development data cannot enter the optimizer path.
Frozen test11/test12 remain excluded from gradients.

No original capture, annotations or cache membership was changed. Every sample
keeps one episode-level crop, correct four-frame stacks and episode-local padding.
Waiting remains available for context; loss_mask excludes it and burn-in. Chunk
hidden state initializes independently, with no-grad burn-in and detached hidden
state before the supervised segment. Tail padding cannot advance memory. No
augmentation is applied: future image flips must transform scene, goal and actions
consistently, so baseline augmentation is not silently reused on only one input.
Total decoded frames default to a bounded 4000; this is small-pilot infrastructure,
not a replacement for the large mmap training pipeline.

`visual_goal_training.py` provides:

- warm_start_visual_policy: CPU-default strict source conversion in memory, frozen
  base, source hash and distinct target config. Historical source training/data
  fields do not masquerade as a new run's membership.
- visual_goal_batch_loss: existing binary/mouse losses on the valid supervised
  segment; separate selected-crop and constant-black-crop control modes.
- run_visual_goal_epoch: bounded 1–200 batches, no optimizer for evaluation, and
  optimizer refusal unless the loader is a validated TRAIN visual dataset.

The caller owns optimizer/checkpoint/update budgets and must use save_checkpoint
with actual run membership and configuration. There is no automatic training job
or promotion. A rehearsal mixture requires explicit compatible annotated records;
old unannotated navigation clips cannot silently become crop-conditioned data.
A GPU/mixed-precision path is not benchmarked; current implementation uses FP32.

## Preparation and counterfactual commands

No current goal-training dataset is approved. The preparation command intentionally
rejects our exposed development/annotation-only ledgers. Once a separately reviewed
TRAIN+validation manifest and approved target descriptors exist:

```bash
.venv/bin/python prepare_visual_goal_pilot.py \
  --annotations APPROVED_TRAIN_AND_VALIDATION_LEDGER.json \
  --targets APPROVED_TARGET_DESCRIPTORS.json \
  --source models/imitation/checkpoints/runs/temporal_readout_comparison_v1/gru/endpoint.pt \
  --output /tmp/shionn_visual_goal_pilot_preparation.json
```

This validates paired goals and whole-start/split provenance and builds a model
on CPU without updates. It writes only a new preparation manifest, never a model.
All uppercase paths are future inputs, not currently available data. No training
command is supplied that would bypass this gate.

`evaluate_visual_goal_counterfactual.py` is a bounded CPU-only read-only screen for
a future visual-goal checkpoint. It evaluates up to 16 chunks by default (hard
maximum 64), swaps two persistent crops with identical observations/history/state,
and reports per-head distribution total variation, argmax-change rates and signed
expected mouse dx. Episode crop sources, checkpoint/ledger/descriptor hashes and
truncation status are recorded. It is not a reaching score or expert-correct
counterfactual label. Observed A/B histories diverge, so only matched initial
views/declared decision windows support direction interpretation. Both human
routes may correctly start straight; opposite steering is not required everywhere.

```bash
.venv/bin/python evaluate_visual_goal_counterfactual.py \
  --checkpoint FUTURE_VISUAL_GOAL_CHECKPOINT.pt \
  --annotations data/annotations/goal_choice_interior_01_annotations_v1_20261115.json \
  --targets data/annotations/visual_targets/goal_choice_interior_01_v1/targets.json \
  --max-chunks 16 \
  --output /tmp/shionn_visual_goal_counterfactual.json
```

This command requires a compatible checkpoint; none has been trained in this work.
PolicyInference supports the new architecture programmatically. The live gameplay
runner has no visual-goal selection/scoring CLI yet; do not pass this architecture
to a live command that cannot provide its goal. A later supervised live comparison
also needs requested-waypoint scoring, not the generic chamber-exit metric.

## Evidence, resources and remaining experiments

Real data checks: doorway pair loads 10 chunks/141 supervised frames; interior
pair loads 8 chunks/101 supervised frames. Both reject TRAIN selection. Original
capture bytes are hash-verified; unchanged episodes remain development_pending.

Pinned recurrent source SHA-256:
338906faa5328ccbe6ace33b946b6adce1775c2cb5487ed2cecb237fd2a1f41b.
A real CPU no-update smoke check loaded its 512-unit GRU, built the shared crop
encoder and evaluated one interior chunk with both goals. Setup+screen took
4.916 seconds with two CPU threads on this host. Zero adapter means mean mouse
probability variation=0, as expected; this is no evidence of learned goal use.
Trainable added parameters: 153902, about 0.59 MiB FP32 weights or 2.35 MiB including
weights, gradients and Adam moments, excluding base/activations/data. Actual GPU
runtime/peak VRAM and training throughput remain unmeasured. Do not extrapolate
this CPU inference measurement to training or environment throughput.

Tests cover strict crop inputs, baseline equivalence, batch/step consistency,
masked states, gradient flow and burn truncation, checkpoint load/reset persistence,
crop identity and source leakage, transitions, source metadata mismatch, wait/tail
masks, optimizer membership refusal, constant-target control and source warm-start
identity. Tiny deterministic test-head changes establish that the computation can
use a goal; they are not learned behavior. Full regression result is reported in
the consolidated response. No broad training/generalization claims follow.

Remaining experimental requirements: fresh approved paired data with whole-start
TRAIN/held-out separation; map/FOV/control provenance; source exposure audit;
bounded matched selected-vs-constant experiment; held-out counterfactual and actual
reaching evaluation; later occlusion/recovery/alternate-layout evidence. Live
runner and waypoint-scoring integration must be prepared before live execution.
Strict frame-gap timing behavior remains separately unresolved, not disabled.

## Immutable approval records

Raw capture metadata remains unchanged even if the user later explicitly approves
new data for training. For sources originally marked training_approved=false, the
loader additionally requires a separate versioned approval document with schema
`shionn_visual_goal_training_approvals_v1`. Each entry must name episode_id,
training_approved=true, matching video_sha256/actions_sha256/metadata_sha256 and a
nonempty authorization_ref identifying the explicit review decision. TRAIN role
and target-descriptor approval remain independently necessary. Pass this document
to `prepare_visual_goal_pilot.py --training-approvals PATH`. Its SHA-256 is saved
in the preparation manifest. No real approval document was created in this work;
synthetic tests prove mismatched source/action hashes are refused and raw metadata
is not rewritten. Claimed human approval is an audit record, not an automatic
permission grant or a replacement for source review.
