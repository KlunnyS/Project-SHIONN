"""Synthetic real-loss/latency/parity checks; no dataset, game or model writes."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import run_ordered_gru_comparison as runner
from models.imitation.mouse_bins import MouseBins
from models.imitation.network import BinnedImitationPolicy


def inference_parity(model, frames, arm, bins):
    """Separate default CUDA rounding from state/order correctness over512steps."""
    original_conv = torch.backends.cudnn.allow_tf32
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    modes = {}
    try:
        for mode in ('runtime_defaults', 'ieee_float32_check'):
            if mode == 'ieee_float32_check':
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
            optimized, native = [], []
            hidden = native_hidden = None
            with torch.inference_mode():
                for _ in range(16):
                    out, hidden = runner.forward_chunk(model, frames, arm, hidden)
                    optimized.append(out)
                    for frame in frames:
                        pixels = frame[None].contiguous(memory_format=torch.channels_last)
                        if arm == 'feedforward':
                            one = model(pixels)
                        else:
                            one, native_hidden = model.step(pixels, None if arm == 'gru_reset' else native_hidden)
                        native.append(one)
                actual = {k: torch.cat([o[k] for o in optimized], 1) for k in optimized[0]}
                expected = {k: torch.cat([o[k] for o in native], 0)[None] for k in native[0]}
                errors = {k: float((actual[k] - expected[k]).abs().max()) for k in actual}
                state_error = float((hidden - native_hidden).abs().max()) if arm == 'gru' else None
                mismatches = int(np.any(runner.decode(actual, bins) != runner.decode(expected, bins), axis=1).sum())
                if mismatches:
                    raise AssertionError(f'{arm}/{mode}: {mismatches} native action mismatches')
                if mode == 'ieee_float32_check':
                    for k in actual:
                        torch.testing.assert_close(actual[k], expected[k], rtol=1e-4, atol=1e-5)
                    if arm == 'gru':
                        torch.testing.assert_close(hidden, native_hidden, rtol=1e-4, atol=1e-5)
                modes[mode] = {'different_action_rows': mismatches,
                    'max_logit_absolute_error': max(errors.values()), 'max_state_absolute_error': state_error,
                    'conv_allow_tf32': torch.backends.cudnn.allow_tf32,
                    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    finally:
        torch.backends.cudnn.allow_tf32 = original_conv
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
    return {'observations_per_mode': 512, 'modes': modes,
            'ieee_numeric_tolerances': {'rtol': 1e-4, 'atol': 1e-5},
            'runtime_numeric_error_reported_not_hidden': True}


def profile(arm, state, cfg, bins, device):
    model = runner.build_model(arm, state, cfg, device)
    optimizer = runner.optimizer_for(model, cfg)
    model.train()
    frames = torch.rand(32, 12, 180, 320, device=device)
    targets = torch.zeros(32, 10, device=device)
    targets[:, 0] = 1
    valid = torch.ones(32, dtype=torch.bool, device=device)
    weights = torch.tensor(cfg['target_processing']['binary_class_weights'], device=device)
    hidden = None
    seconds = []
    before = runner.temporal.tensor_hash(model.trunk.state_dict())
    torch.cuda.reset_peak_memory_stats()
    for step in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if hidden is not None:
            hidden = hidden.detach()
        out, hidden = runner.forward_chunk(model, frames, arm, hidden, valid)
        loss, _ = runner.parent.base.behavior_cloning_loss({k: v[0, valid] for k, v in out.items()},
            targets[valid], weights, mouse_bins=bins, label_smoothing=.01)
        if not torch.isfinite(loss):
            raise FloatingPointError('Synthetic loss is nonfinite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if step:
            seconds.append(time.perf_counter() - started)
    if before == runner.temporal.tensor_hash(model.trunk.state_dict()):
        raise AssertionError('Synthetic encoder did not update')
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    del hidden, out, loss
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        hidden = None
        for _ in range(16):
            out, hidden = runner.forward_chunk(model, frames, arm, hidden, valid)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - started
    parity = inference_parity(model, frames, arm, bins)
    return {'synthetic_updates': 4, 'timed_warm_updates': 3,
        'mean_update_seconds': statistics.mean(seconds), 'median_update_seconds': statistics.median(seconds),
        'peak_allocated_mib': allocated, 'peak_reserved_mib': reserved,
        'synthetic_eval_frames': 512, 'synthetic_eval_frames_per_second': 512 / eval_seconds,
        'encoder_changed': True, 'cuda_native_inference_parity': parity}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Synthetic CUDA benchmark requires CUDA; no fallback')
    values = np.column_stack([np.arange(-100, 101), np.arange(100, -101, -1)]).astype(np.float32)
    bins = MouseBins.fit(values, 15)
    cfg = {'target_processing': {'mouse_bins': bins.config, 'binary_class_weights': [[1., 1.]] * 8},
           'training': {'weight_decay': .0001, 'label_smoothing': .01}}
    source = BinnedImitationPolicy(*bins.class_counts())
    state = source.state_dict()
    del source
    results = {}
    for arm in runner.ARMS:
        results[arm] = profile(arm, state, cfg, bins, device)
        print(arm, json.dumps(results[arm]), flush=True)
        torch.cuda.empty_cache()
    report = {'scope': 'Synthetic pixels/labels and scratch weights only; no dataset, source checkpoint, gameplay or saved model.',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(), 'sequence_length': 32,
        'batch_episode_streams': 1, 'precision': 'float32',
        'benchmark_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(), 'results': results,
        'limitations': ['Three warm timed updates per arm are a small model-only sample.',
            'Real loading, full action metrics and episode scheduling are excluded.',
            'Exact action parity is checked in both modes; tight numeric parity is checked separately with TF32 disabled. Runtime defaults are restored, not changed by the runner.',
            'No gameplay or demonstration convergence evidence.']}
    Path('/tmp/shionn-ordered-gru-synthetic-profile.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
