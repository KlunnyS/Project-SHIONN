# Architecture enhancement assessment and first increment

## Current implementation

The v5 policy already implements residual CNN blocks, GroupNorm/LayerNorm, SiLU, squeeze-excitation channel gates, spatial 3x5 pooling and a 512-feature projection (`models/imitation/network.py`). The baseline is already a gated residual CNN; adding residual blocks again is unnecessary.

Inputs are four cached RGB frames, `[B,12,180,320]`, normalized by the existing network. Binary controls have independent two-logit heads. Camera control supports Gaussian outputs or independent discrete dx/dy bins. Existing datasets, frozen episode splits, augmentation, weighted frame sampling, AdamW, CUDA AMP, checkpoint sidecars and live capture/evaluation are reusable.

Temporal support currently means four-frame stacking, not recurrent memory. Training samples individual frame stacks, often shuffled/weighted. No implemented PPO learner, value head or Gym environment adapter was found in the Python code. RL is a future integration dependency, not a functioning pipeline to which a GRU can simply be attached.

## Implemented increment: optional final-feature gating

`--feature-gate none` preserves the baseline. `--feature-gate observation --mouse-head binned` selects `shionn_imitation_v5_residual_binned_feature_gate` in the existing imitation trainer and inference loader. Gaussian gating is explicitly rejected for this first experiment.

The new module takes `[B,512]`, applies Linear(512,64), SiLU, Linear(64,512), and multiplies its input by twice the sigmoid of the result. The final linear layer is initialized to zero, so the gate starts as an exact identity. Scales lie in (0,2), allowing learned suppression and amplification. Added parameter count: 66,112 (about 0.5% of the documented v5 size). Its actual latency/VRAM overhead must be measured, not inferred from parameter count.

This is observation-conditioned post-encoder modulation, distinct from the existing spatial channel gates. It has no recurrent state, previous-action input, new dependencies or action-space changes. Learning uses existing imitation losses. It is an ablation candidate, not a demonstrated behavioral improvement.

Existing architecture IDs/classes remain loadable. Gated checkpoints have their own ID and strict state dict. No automatic checkpoint conversion is provided; baseline weights cannot be resumed as gated weights with stale Adam state. The fixed gate shape is encoded by the architecture ID. Existing pinned recovery-probe helpers remain baseline-only; do not use them to train this variant. Changes to core source files also invalidate historical experiments' exact code fingerprints, as intended; do not rewrite those fingerprints to bypass guards.

## Validation and reproducible comparison

Functional tests cover identical baseline/gated initial outputs for the same seed, identity initialization, finite gradients and a learnable gate update, baseline and gated checkpoint/inference round trips, reset behavior and rejection of unsupported Gaussian gating. This comparison tests integration, not task improvement.

For a future behavioral comparison, run the existing trainer twice with identical explicit data/split, seed, binned targets, sampling, augmentation, optimizer settings, batch sizes, fixed update budget and validation cadence. Change only `--feature-gate none` versus `--feature-gate observation`, with separate empty output directories. Use fresh initialization with the same seed; identity tests verify shared initial parameters. Record software versions, source hashes, exact command, parameter count, elapsed compute and peak VRAM. The gated constructor consumes extra RNG: reset CPU/CUDA execution RNG to the same state after construction, and isolate sampler/draw generators, before training so augmentation/dropout exposure is matched. Same-seed commands alone do not guarantee that matching. Repeat across seeds after a pilot. Never compare unrelated historical training runs as the causal control.

Do not run an unrestricted default epoch schedule to test this addition. Before a user-executed training comparison, prepare a bounded runner with the same number of optimizer updates and full final evaluation per arm, plus a predeclared shared runtime ceiling; account for initialization, validation and hashing. Existing bounded probe helpers are intentionally pinned and require an explicit separate design for new architectures.

Evaluate frozen imitation metrics, per-chamber completion and sustained navigation, with user-run gameplay only. At 24Hz the full inference/control pipeline has about41.7ms per frame; profile end-to-end and model-only latency separately. Measure CUDA peak allocated/reserved memory and system RAM. AMP correctness and throughput on the target GPU remain unmeasured. No training or live game experiment was launched for this increment.

## Next increments, in order

1. **GRU imitation end to end.** Reuse the visual encoder and action heads. Add a versioned sequence dataset returning contiguous `[B,T,12,180,320]`, targets and valid/start masks. Never cross episode boundaries. Begin with fixed short chunks and configurable burn-in; initially budget B=1–2 and T=8–16, then measure8GB VRAM feasibility. CNN activations across time dominate memory; frozen-encoder feature caching can be an explicit separate experiment, not an equivalent substitute for end-to-end training.
2. **Explicit recurrent contract.** Proposed features `[B,T,512]`, GRU hidden `[1,B,H]`, initial H=256; project outputs to existing heads. Expose reset/step/sequence interfaces, zero state on new episodes, reloads, restarts and handoffs, detach at truncation boundaries, and preserve ordering and padding masks in losses. Burn-in reconstructs state for sampled chunks; do not carry state between unrelated batches. Checkpoint learned weights/optimizer and sequence configuration, not a global live episode state. Define same-episode resume separately. Previous-action conditioning must align action t-1 with observation t and distinguish teacher-forced expert actions from executed policy actions.
3. **Memory-specific tests before performance claims.** Test reset isolation, sequence-vs-step equivalence in eval mode, masking, burn-in, truncated gradient boundaries and save/load. Include a controlled observation-aliasing task and compare matched frame-stacked baselines. Memory can help partial observability but does not guarantee route planning.
4. **RL prerequisites and recurrent PPO.** Define the game environment adapter, terminal versus timeout handling, action distributions/log probabilities, rewards, value head and recurrent rollout storage. PPO minibatches must preserve sequences and initial state, apply masks and bootstrap truncations correctly. SB3-Contrib's RecurrentPPO uses LSTM, so it is a reference rather than a drop-in GRU/mixed-action solution. Preserve current control semantics; evaluate binned action distribution first.
5. **Later isolated experiments.** Bounded causal temporal attention with masks, position encoding and resets; previous-action/memory-conditioned gates; goal-conditioned policy before a hierarchical controller; auxiliary latent next-state prediction conditioned on actions, with a stop-gradient target or another explicitly tested anti-collapse design. Predictive loss is not evidence that planning works. Only investigate planning after prediction usefulness is measured.
6. **Adaptation.** Recurrent state is fast context, not weight learning. Keep slow updates offline in versioned candidates with replay, regression testing and rollback. Limited online adapter updates require explicit budgets and evaluation; no per-frame uncontrolled updates. LLM weight editing remains a separate backlog research track (#34).

No architectural migration or new recording format is required for this implemented gate. GRU sequence sampling can reuse cached episodes, but changes the training contract and requires its own implementation and validation increment.

## Research references

- PyTorch GRU shape/state contract: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.GRU.html
- SB3-Contrib recurrent PPO and episode-start handling: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html

## Phase3 implementation update — 2026-11-05

Optional GRU sequence imitation training and persistent inference are now implemented. See [RECURRENT_IMITATION.md](RECURRENT_IMITATION.md) for configuration, tensor/state contracts, supported sampling, compatibility, tests and synthetic hardware measurements. The earlier ordered increments remain the design rationale; GRU end-to-end integration is no longer merely proposed. Behavioral efficacy, recurrent PPO and later speculative components remain unimplemented or unevaluated.
