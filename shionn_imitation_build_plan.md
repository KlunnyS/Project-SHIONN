# SHIONN — Imitation Learning Model: Build Plan

This document specifies the network architecture, data pipeline, and training
setup for the Stage 1–2 imitation-learning model. It is written to be
actionable directly — either followed manually or handed to an implementation
agent as a build spec.

---

## 0. Scope

Pure behavior cloning (BC): supervised learning from recorded human gameplay.
No reinforcement learning, no value head, no reward signal in this phase.
Input is FHD (1920×1080) recorded footage; the network itself trains on a
downsampled version of each frame — these are two separate resolutions, kept
deliberately distinct throughout this document.

---

## 1. Recording Configuration

| Parameter | Value |
|---|---|
| Game render resolution | 1920×1080 (Full HD) |
| Capture resolution | 1920×1080, matches render |
| Capture rate | 24 Hz (fixed tick loop) |
| Video storage | H.264 or H.265, moderate CRF (~18–23) — visually near-lossless but not literally lossless, to keep multi-hour sessions on disk manageably |
| Action log tick rate | **24 Hz — must match capture rate exactly.** If the existing recorder's sync loop is hardcoded to 20 Hz from earlier setup, update it to 24 Hz here, or frame/action rows will drift out of alignment over a long recording session. |
| Action log format | unchanged: `frame_idx, move_w, move_a, move_s, move_d, jump, use, fire_left, fire_right, mouse_dx, mouse_dy` |

Recording at full render resolution is intentional: it preserves the option
to re-derive different training resolutions later, or review footage at full
quality, without re-recording. The network never sees this resolution
directly — see §2.

---

## 2. Preprocessing Pipeline (offline, before training)

**Do not decode and resize 1080p frames on the fly during training** — this
will bottleneck the data loader regardless of GPU. Preprocess once, cache the
result.

1. Decode each episode's `video.mp4`.
2. Resize every frame from 1920×1080 → **320×180** (exact 1/6 scale — clean
   integer division, no aspect-ratio distortion, no letterboxing needed).
3. Store as a compact on-disk format for fast random access during training
   — e.g. a memory-mapped `.npy`/`.npz` array per episode, or an LMDB/WebDataset
   shard. Avoid re-decoding video during training entirely.
4. Keep `actions.csv` as-is; index it by the same `frame_idx` used in the
   cached frame array so a `(frame_idx)` lookup retrieves both in sync.

**Why 320×180, not 128×128 or raw 1080p:** this is the resolution decision
that operationalizes your teacher's "keep enough detail" guidance without
requiring raw-resolution input. At 320×180 fine detail relevant to portal
placement (reticle position, wall-surface edges) is still meaningfully
preserved, while staying computationally tractable at reasonable batch sizes
on both the prototyping and full-training hardware tiers below. Treat this as
a tunable hyperparameter, not a fixed law — if training reveals the model is
detail-starved, this is the first knob to revisit (e.g. step up to 384×216,
an exact 1/5 scale).

---

## 3. Network Architecture

### 3.1 Design intent

Per your teacher's guidance: favor **width (channel count) over depth**, and
avoid a single aggressive early downsample (the classic Atari-style 8×8
stride-4 first layer). Instead, downsample gently across several **normal**
3×3 convolutions, widening channels as spatial resolution shrinks. This is a
shallow, wide CNN — closer to an early-VGG-style stem than a deep ResNet —
which keeps detail alive longer into the network at the cost of more compute
per layer, which is the intended tradeoff here.

### 3.2 Input

4 stacked frames (for implicit motion/velocity information, unchanged from
the earlier plan), each 320×180 RGB, channel-stacked:

```
Input tensor shape: (batch, 12, 180, 320)   # 4 frames × 3 channels
```

### 3.3 Trunk (shared)

| Layer | Op | Output shape (H×W) | Channels |
|---|---|---|---|
| Stem | Conv 5×5, stride 2, pad 2 + ReLU | 90×160 | 64 |
| Block 1a | Conv 3×3, stride 1, pad 1 + ReLU | 90×160 | 64 |
| Block 1b | Conv 3×3, stride 2, pad 1 + ReLU | 45×80 | 128 |
| Block 2a | Conv 3×3, stride 1, pad 1 + ReLU | 45×80 | 128 |
| Block 2b | Conv 3×3, stride 2, pad 1 + ReLU | 23×40 | 256 |
| Block 3a | Conv 3×3, stride 1, pad 1 + ReLU | 23×40 | 256 |
| Block 3b | Conv 3×3, stride 2, pad 1 + ReLU | 12×20 | 384 |
| Pool | AdaptiveAvgPool2d → 4×4 | 4×4 | 384 |
| Flatten | — | — | 6144 |
| FC | Linear → ReLU | — | 1024 |
| FC | Linear → ReLU | — | 512 |

Total: 7 conv layers (deliberately shallow), widening 64 → 384 channels
(deliberately wide), 4 total downsampling stages (stem + 3 blocks) instead of
one aggressive stride-4 opener. This is the concrete architecture matching
your teacher's "wide channels, normal/gentle convolution" instruction.

### 3.4 Action heads (off the shared 512-d trunk output)

Unchanged from the earlier factored action-space design — no single softmax
over all actions, no explicit "do nothing" category (it's the zero-state
across heads):

```
move_w, move_a, move_s, move_d       → Linear(512, 2) each   [binary]
jump, use, fire_left, fire_right     → Linear(512, 2) each   [binary]
mouse_dx, mouse_dy                   → Linear(512, 4)        [Gaussian: mean + log-std, 2 each]
```

No value head — this is pure BC, not actor-critic (that's introduced later
at Stage 3, reusing this same trunk).

### 3.5 Parameter / memory estimate

Roughly 8–10M parameters in the trunk (dominated by the two FC layers, not
the convs — flag this: if parameter count needs trimming later, reduce the
6144→1024 FC width first, not the conv channels, to preserve the
detail-retention property the wide convs exist for).

Activation memory is the real constraint, not parameter count. At batch
size 16 with this architecture at 320×180 input, expect on the order of a
few GB of activation memory during training with float32 — see §5 for
per-tier batch size guidance.

---

## 4. Loss Function

```
loss = Σ CrossEntropy(binary_head_logits, human_action)   # 8 binary heads
     + NLL(mouse_gaussian, human_mouse_delta)             # or MSE on the mean, as a simpler v1
```

Sum (not average) across heads for v1 — revisit per-head loss weighting only
if training reveals one head (typically mouse) dominating or being starved
relative to the others.

---

## 5. Hardware Tiers

### Tier 1 — RTX 4060 (8GB), prototyping and pipeline validation

Purpose: confirm the data pipeline, loss curves, and overfitting-on-a-small-batch
sanity check are all correct **before** spending rented GPU time.

| Setting | Value |
|---|---|
| Batch size | 8–16 |
| Precision | Mixed precision (`torch.cuda.amp`), not full float32 |
| Resolution | 320×180 as specified — if this doesn't fit even at batch 8 with AMP, drop to batch 4 before dropping resolution |
| Goal | Loss decreasing on a small (pilot, ~20–30 min) dataset; no crashes; no NaNs |

### Tier 2 — Rented GPU (24GB+, e.g. RTX 4090 / A100), full training runs

| Setting | Value |
|---|---|
| Batch size | 32–64 (scale to fill VRAM with AMP) |
| Precision | Mixed precision, still — no reason to drop it just because VRAM is available |
| Resolution | 320×180 default; 384×216 is the next step up if Tier 1 validation suggests more detail is needed |
| Goal | Full dataset training runs, hyperparameter sweeps |

**Do not rent GPU time until Tier 1 has validated the pipeline end-to-end**
on a small pilot batch — this is the same "prove the pipe before scaling"
principle used for the environment wrapper earlier in this project.

---

## 6. Train / Validation Split

Split **by episode, not by frame.** Frames within an episode are highly
correlated (adjacent frames of the same attempt); a frame-level random split
leaks near-duplicate information between train and validation and will
produce misleadingly good validation numbers. Hold out entire episodes —
ideally including at least one held-out episode per chamber, not just
held-out episodes concentrated in one chamber.

---

## 7. Data Augmentation — caution

**Do not apply horizontal flip augmentation naively.** Flipping the frame
without also negating `mouse_dx` and swapping `move_a`↔`move_d` in the
corresponding action row silently teaches the model an inverted control
scheme half the time. If flip augmentation is wanted later, it must
transform the action row in lockstep with the frame — treat this as a
deliberate future addition, not a default for v1.

Safe for v1: mild brightness/contrast jitter only, if any augmentation is
used at all. Given dataset size is currently the bigger constraint than
overfitting risk, skipping augmentation entirely for the first training run
is a reasonable default.

---

## 8. File Structure Additions

```
SHIONN/
├── data/
│   ├── recordings/          # raw video.mp4 + actions.csv per episode (existing)
│   └── datasets/
│       └── cached_frames/   # NEW: preprocessed 320x180 arrays, one per episode
│
├── models/
│   └── imitation/
│       ├── network.py       # NEW: trunk + factored heads, per §3
│       ├── dataset.py       # NEW: episode-level train/val split, frame-stack sampling
│       └── train_bc.py      # NEW: training loop, loss per §4, AMP per §5
```

---

## 9. Build Order

1. Write the offline preprocessing script (§2): video → cached 320×180 arrays,
   indexed by `frame_idx` alongside `actions.csv`.
2. Write `dataset.py`: episode-level split, samples a 4-frame stack + the
   corresponding action row for a given index.
3. Write `network.py`: exact architecture from §3 — trunk + 8 binary heads +
   1 Gaussian mouse head.
4. Write `train_bc.py`: loss per §4, AMP enabled, Tier 1 batch size, logging
   per-head loss separately (so a struggling head is visible immediately,
   not hidden inside a summed loss).
5. **Validate on Tier 1 (4060) first**, on the existing pilot dataset — confirm
   loss decreases, no NaNs, no crashes, and per-head losses all move
   sensibly.
6. Only after Tier 1 validation passes: record the full dataset (2–8+ hours),
   move to Tier 2 (rented GPU) for the real training run.

---

## 10. Open Decisions to Revisit Later (not v1 blockers)

- Whether to bump input resolution to 384×216 if 320×180 proves
  detail-starved (first check: is the model specifically failing at portal
  placement precision, or at something else? — resolution only helps the
  former).
- Whether per-frame siamese encoding (shared small encoder run on each of
  the 4 frames independently, then concatenated) outperforms the current
  channel-stacking approach — a reasonable v2 experiment, not a v1
  requirement.
- Per-head loss weighting, if one head's loss curve visibly dominates or
  stalls relative to the others once real training begins.
