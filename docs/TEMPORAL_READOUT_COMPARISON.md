# SSH-only temporal readout comparison

## Purpose and scope

`run_temporal_readout_comparison.py` prepares a small comparison using existing reviewed data while human recovery collection is unavailable. It does not replace the targeted navigation investigation or establish that memory solves wandering.

This is a **frozen visual encoder / new action-readout pilot**, not end-to-end recurrent policy training. The deployed v5 visual encoder is identical and frozen in all arms, including evaluation behavior (encoder dropout disabled). Source action heads are deliberately discarded in ALL trained arms and replaced by identical seeded heads. Comparing pretrained feedforward heads against newly initialized GRU heads would be unfair. Original deployed-head results are recorded as an untrained reference only, not a matched arm.

Three matched arms:

1. Feedforward linear heads over512visual features.
2. GRU512memory plus the same initially seeded action heads.
3. The identical GRU architecture reset at every observation, to distinguish persistent state from extra readout capacity/nonlinearity. This is a diagnostic control, not an exported live policy.

Four-frame visual stacking and existing action semantics remain in every arm. Optional post-encoder gating is disabled. Existing squeeze-excitation in the common deployed encoder remains unchanged. No PPO, world model, new game interaction or per-frame weight adaptation is involved.

## Data and sampling

Reuse frozen v3 original TRAIN membership1100episodes, navigation validation200episodes/39769usable frames, four reviewed human TRAIN corrections/550frames and two reviewed development holdouts/272frames. The holdouts never enter optimizer targets or selection. They are already visually reviewed diagnostics on one familiar chamber, not untouched final or unseen-layout tests. Source encoder ancestry is preserved separately from current gradient membership.

All arms receive the same contiguous episode-local chunks from SequenceDataset:16supervised positions plus16preceding burn-in positions, batch2chunks,500successful float32updates. Every fifth chunk draw is a human TRAIN correction; other chunks use chamber-balanced original TRAIN draws. The resulting nominal80/20CHUNK allocation gives actual12347base and3103human supervised frames (15450total), not exactly80/20frames because of tails. Exact episode exposure and schedule hash are recorded. Original raw recovery trajectories are preserved; there is no concatenation across handoffs.

Padding cannot update recurrent state or contribute loss. GRU burn-in has no gradients, is detached before supervision, and each independently sampled chunk starts with zero state. Feedforward/reset controls receive the same supervised targets. No state passes between unrelated batches. Training loss keeps source binary weights, mouse bins and label smoothing; LR0.001/fresh AdamW is shared across the newly initialized readouts, with source weight decay and gradient clip1. This LR is for new heads/memory, not an instruction to fine-tune the pretrained encoder. No augmentation or AMP is used in this pilot. All metadata reflects those settings.

The common encoder computes required TRAIN context and complete evaluation features once, into CPU RAM. Full shape remains[B,T,12,180,320] at extraction; readouts see[B,T,512]. The actual plan requires63248unique usable frame features:22657base,39769navigation,550human TRAIN and272human holdout. Source features are float32 and checksummed; every arm uses the same bank. Uninitialized/unrequested rows are excluded from its fingerprint. There are no persisted feature-cache files or changes to original datasets/splits.

## Evaluation and interpretation

Record each arm's initial AND final full navigation/correction scores. Initial predictions need not match: identical head tensors act on different feature representations. Evaluation carries GRU state through each complete usable expert episode and resets before the next. Training reconstructs only16frames of burn-in, so full-episode evaluation can expose a context-length mismatch. Leading-idle trimming follows the original data contract; these are expert-trajectory replay scores, not closed-loop game results.

Report per-episode/equal-episode and frame-weighted metrics, mouse direction/magnitude, idle periods, backward recall/false positives, ordinary navigation and startup outputs. Real-valued loss and nonfinite checks run before argmax. The synthetic delayed-cue positive control checks that persistent GRU can retain information where reset inference cannot; it is not game generalization evidence.

A GRU gain over feedforward alone could reflect capacity; check the reset-control comparison. A gain over reset control is still limited by dataset diversity and sequence/training settings. Do not interpret head-only failure as proof that end-to-end GRU learning cannot work. Do not promote a model from offline accuracy or this pilot alone. Targeted human backward-recovery recordings and autonomous game evaluation remain later user tasks.

## Safety and checkpoints

Default command is plan-only and creates no model output:

```fish
.venv/bin/python run_temporal_readout_comparison.py
```

Strict checks pin the deployed source, sidecar/payload, reviewed correction hashes and4/2roles, frozen base membership, code and deterministic chunk schedule. End-of-run guards also verify encoder and populated feature-bank hashes. No input files or existing checkpoints are modified. Existing output directories are rejected.

The two deployable-format endpoints are feedforward and persistent GRU with the existing architecture IDs and full encoder weights. Their config records frozen encoder, newly initialized heads, source ancestry, actual exposure, zero augmentation and GRU dimensions. Generic optimizer resume is blocked because the readout-only optimizer is incompatible with full-model training. Both remain experimental and unpromoted.

The reset control saves `diagnostic_state.pt`, deliberately missing the normal checkpoint contract; PolicyInference/load_checkpoint reject it. It is a reproducibility artifact, not a stateful GRU candidate.

All three arms/evaluations/guards/final hashes must finish before the two endpoints are published. `comparison_complete=true` in report.json is authoritative. Partial staging/publication after interruption is not a successful result. No automatic retry, extension, checkpoint selection or promotion.

## Runtime, hardware and execution

Synthetic RTX4060 smoke checks used ten random-feature/zero-target updates per arm, not real demonstration targets. Frozen encoder timing on128existing observations measured about293frames/s with four CPU threads. Readout-stage peak allocated VRAM was roughly68MiB(feedforward),107MiB(GRU),108MiB(reset control); these figures exclude the larger feature-extraction peak and CPU mmap/cache residency. Cold ten-update timings are not a comparative throughput benchmark. Actual feature-encoding time, peak allocated memory and arm training times are recorded by the real run.

Estimated5–12minutes overall; shared13minute soft deadline and15minute outer ceiling. Encoding/data access dominates this estimate. Use the following from the repository root over SSH; Portal2 and Wayland access are not needed:

```fish
systemd-run --user \
  --unit=shionn-temporal-readout-v1 \
  --collect \
  -p WorkingDirectory="$PWD" \
  -p LimitNOFILE=8192 \
  -p MemoryHigh=16G \
  -p MemoryMax=20G \
  -p MemorySwapMax=2G \
  -p RuntimeMaxSec=15min \
  -p TimeoutStopSec=1s \
  "$PWD/.venv/bin/python" -u run_temporal_readout_comparison.py --execute
```

Monitor:

```fish
journalctl --user -u shionn-temporal-readout-v1 -f
```

Stop:

```fish
systemctl --user stop shionn-temporal-readout-v1
```

Output: `models/imitation/checkpoints/runs/temporal_readout_comparison_v1/report.json`. Preserve partial outputs and review rather than overwriting/restarting. Keep fingerprinted source/code/data unchanged while running. No real comparison training was launched during preparation.

## Verification

All306regression tests passed, including eight focused temporal-readout tests. Tests cover identical new heads/encoder, raw-pixel versus cached-feature equivalence, reset control behavior, episode-local masks/tails and identical target exposure, real synthetic readout backward/update with invalid padding labels, frozen encoder integrity, delayed-cue positive control, staged completion, diagnostic artifact rejection and final-hash timeout. Independent read-only review passes after fixing exported augmentation provenance. GPU smoke tests are synthetic; real runtime, convergence and gameplay effects remain unverified until evaluated.

## Completed pilot — 2026-11-06

Report: `models/imitation/checkpoints/runs/temporal_readout_comparison_v1/report.json`. All3arms completed500updates on identical15450supervised frames, comparison_complete true. All artifact hashes match. Elapsed316.513s;271.393s was shared feature encoding. Actual readout training took5.34sFF,8.41sGRU,4.15sreset-GRU; this does not include shared extraction/evaluation. Readout peak allocated GPU memory68.2/118.3/107.2MiB. Frozen encoder and feature-bank checks passed.

Held-out whole-trajectory equal-episode keyboard agreement: deployed reference69.47%, newFF66.92%, GRU66.16%, reset-GRU65.05%. Backward recall8/40deployed,4/40FF,12/40GRU,8/40reset. False backward outputs2/0/1/0respectively. Thus persistent memory adds some backward-action retention relative to controls, but does not establish better overall recovery, goal finding or gameplay.

First predicted backward occurs at offsets3/3forFF and2/3forbothGRUandreset-GRU (two held-out clips). Expert backing begins at0. The persistent-state gain is mainly after onset; neither recurrent model fixes stationary initiation. The holdouts remain a tiny reviewed development set.

Navigation pooled keyboard agreement84.604%deployed,84.880%FF,84.425%GRU,84.015%reset. Pooled horizontal mouse MAE6.292deployed,6.823FF,6.902GRU,6.992reset. All3fail the source-relative5%horizontal MAE regression guard. FFalso fails opening idle; reset additionally fails pooled keyboard and opening idle. GRUopening idle157/1600improves over source175/1600, but does not override steering regression. These source-relative checks are eligibility diagnostics, not a causal comparison between identical trained priors; all new arms intentionally reset heads.

Keep original deployed checkpoint. No gameplay gains or promotion, and no extended end-to-end/longer-budget training follows automatically. The result is limited to frozen features, one seed and500updates; it is not evidence that end-to-end recurrent learning cannot help. The unresolved targeted correction coverage remains the next data requirement when human desktop control is available. While SSH-only, do not launch the assisted recorder or manufacture expert labels from autonomous actions.
