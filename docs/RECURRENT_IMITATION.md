# Optional recurrent imitation learning

Implemented architecture: `shionn_imitation_v5_gru_binned`. The original feedforward default, Gaussian policies, cached episode format and existing checkpoint IDs remain unchanged. No old checkpoints are automatically converted.

## Configuration

Use the existing `python -m models.imitation.train_bc` entry point with:

```
--temporal-memory gru --gru-hidden-size 256
--sequence-length 8 --burn-in 4
--mouse-head binned --sampling uniform
--batch-size 1 --validation-batch-size 1
```

These are option fragments, not a request to start training. Supply the normal frozen split/data settings, separate output directory and a reviewed budget before executing. `--feature-gate observation` independently enables the existing post-encoder gate; default `none` disables it. Current residual blocks retain their existing squeeze-excitation gates in either case.

The first recurrent implementation deliberately rejects Gaussian output, chamber-balanced sampling, opening boosts other than1 and correction draw fractions. These frame-level sampling promises require separate chunk exposure semantics. Extra registered TRAIN episodes can still be included with uniform chunk sampling. Class weights and mouse bins are calculated from the original TRAIN frames before sequence wrapping.

Start with batch1 for both training and validation on8GB hardware. Existing global batch defaults are preserved and are not recommended recurrent defaults. No throughput or gameplay improvement follows from this integration.

## Sequence and state contracts

- Cached frames/actions and frozen episode splits are reused without conversion. SequenceDataset wraps an already split BehaviorCloningDataset.
- A sample contains fixed capacity `[burn_in+sequence_length,12,180,320]` uint8 frames, `[T,10]` targets and `[T]` boolean validity. Samples never cross episodes. Four-frame history also remains episode-local.
- Supervised chunks are nonoverlapping from the first usable frame onward. Short tails are right-padded. Missing preceding burn-in is left-padded. Padding cannot change recurrent state or contribute loss. Leading-idle trimming follows the original dataset contract; real history preceding a usable frame still follows the original four-frame stacking behavior.
- Training starts hidden state at zero for every independently sampled chunk, reconstructs limited context with no-grad burn-in, detaches that state and backpropagates through only the supervised segment. No state survives between shuffled batches. This is independent-chunk truncated BPTT, not full-episode persistent-state training. Information farther back than burn-in is not reconstructed.
- Feature inputs `[B,T,512]` feed a one-layer GRU. State is `[1,B,H]`. forward returns `(action_logits,state)` with `[B,T,C]` logits; step accepts `[B,12,180,320]` and returns `[B,C]` logits plus state. Inputs to the network remain normalized to0–1 by caller; encoder normalization stays unchanged.
- Burn-in has no loss; padding is excluded before applying existing binary and binned mouse losses. Brightness, contrast and flip decisions are shared across each sequence, including its context; flips update all corresponding strafe/dx labels.
- Inference owns detached hidden state per PolicyInference instance. reset clears frame history and hidden state. Existing runner reset calls cover preflight reload, chamber readiness, focus loss and termination. Restart/handoff ends the invocation; the next invocation owns fresh state. External callers must call reset whenever their environment resets. Each prediction advances memory once.

## Checkpoint and resume

The checkpoint's `memory` contract records GRU type, hidden size, feature gate, sequence length and burn-in. Resume validates it alongside architecture/preprocessing/action targets. Checkpoints contain learned weights and optimizer/progress, not an active gameplay hidden state. Loading for inference begins with empty memory. Changing dimensions/gating or chunk settings requires a new experiment, not silent resume. Baseline checkpoints remain usable through the old loaders.

Existing pinned feedforward recovery probes are not recurrent trainers. Offline callers that invoke policy directly must understand recurrent tuple output and ordered state management; use the recurrent epoch evaluator or public PolicyInference API rather than assuming dict-only forward output.

## Verification and benchmarks

Focused tests exercise ordering, episode isolation, burn-in/tail masks (including invalid padded labels), step/sequence equivalence, hidden shapes, gradient truncation, actual recurrent imitation updates, gated/ungated checkpoint roundtrips, session isolation and resets. A synthetic trainer CLI run exercises checkpoint construction and rejects changed chunk settings on resume. Existing regression tests remain required.

`benchmark_recurrent.py --device cpu|cuda` is a bounded synthetic benchmark (default B1,T4,H256). It reports parameter counts, model-only step latency, one float32 synthetic squared-logit optimizer update and peak CUDA memory. It does not represent steady-state training throughput or real imitation convergence. Five inference timings and a cold optimizer update are noisy; sequential versus batched CNN execution and first-use overhead prevent inferring a training speed advantage. No capture/control overhead is measured. No checkpoints or dataset writes occur.

On this machine (RTX4060, PyTorch2.14.0+cu130), one illustrative run measured baseline/GRU model-only latency2.59/2.72ms, float32 peak allocated421.6/426.5MiB and reserved502/538MiB at B1,T4. Synthetic one-update rates10.7/28.8frames/s are cold measurements, not a controlled speedup claim. CPU single-thread latency115.9/101.8ms was similarly a tiny noisy sample. Reports: `/tmp/shionn-gru-cuda-benchmark.json` and `/tmp/shionn-gru-cpu-benchmark.json`. Production batch sizes, longer chunks, burn-in and augmentation need separate profiling.

A separate real-loss synthetic CUDA autocast/GradScaler update with gated GRU verifies mixed-precision training, finite loss and changed GRU weights at initial gradient scale128. The default65536 scale skipped the first synthetic update after overflow; this is not a successful optimizer update. The trainer retains the existing adaptive GradScaler behavior; its global_step counter counts attempted batches (including scaler-skipped updates), so fixed successful-update comparisons need explicit skip accounting. No dataset training or gameplay was performed.

## Remaining work

Run matched, bounded training comparisons and memory-dependent evaluation before any performance claim. Consider longer-context state reconstruction, chunk sampling balances and exact mid-episode resume as separate experiments. PPO, world models, hierarchical policies and online weight adaptation are outside this increment.

## Bounded SSH-only comparison

A separate [temporal-readout pilot](TEMPORAL_READOUT_COMPARISON.md) now compares feedforward, persistent GRU and reset-GRU readouts using an identical frozen deployed encoder and matched newly initialized heads. It preserves this module's existing end-to-end sequence-training capability, but does not run end-to-end training or prove gameplay benefit. The pilot defaults to planning and requires explicit user execution.
