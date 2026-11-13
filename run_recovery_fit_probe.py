"""Plan a bounded memorization check on existing recovery training starts.

Training requires --execute. This is a fitting diagnostic, not a candidate
policy or generalization benchmark. No checkpoint is saved; no game or input
APIs are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import run_recovery_probe as base
from models.imitation.checkpoint import load_checkpoint
from models.imitation.dataset import BehaviorCloningDataset, discover_cached_episodes
from models.imitation.frozen_split import load_frozen_split
from models.imitation.mouse_bins import MouseBins

SOURCE = Path('models/imitation/checkpoints/runs/recovery_history_probe_v1/endpoint.pt')
OUTPUT = Path('model_attempts/diagnostics/recovery_fit_probe_v1')


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class RecoveryStarts(Dataset):
    """Snapshot one first active target per episode, using newest RGB only."""
    def __init__(self, dataset):
        positions = {pair: i for i, pair in enumerate(dataset.index)}
        self.items, self.records = [], []
        for e, name in enumerate(dataset.episode_names):
            frame_idx = dataset.first_action_frames[e]
            if (e, frame_idx) not in positions:
                raise ValueError(f'No active start for {name}')
            frames, target = dataset[positions[e, frame_idx]]
            if not np.isfinite(target).all() or not np.isin(target[:8], [0, 1]).all() or not np.any(target):
                raise ValueError(f'Invalid/non-active start target for {name}')
            self.items.append((np.tile(frames[-3:], (4, 1, 1)), target.copy()))
            self.records.append({'episode': name, 'frame_idx': int(frame_idx)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def encoded_targets(targets, bins):
    dx, dy = bins.encode(torch.as_tensor(targets[:, 8:10], dtype=torch.float32))
    return np.column_stack((targets[:, :8].astype(np.int64), dx.numpy(), dy.numpy()))


def validate_samples(dataset, bins):
    labels = encoded_targets(np.stack([y for _, y in dataset]), bins)
    seen, digest = {}, hashlib.sha256()
    for (frames, target), label, record in zip(dataset, labels, dataset.records):
        key = hashlib.sha256(frames.tobytes()).hexdigest()
        if key in seen and not np.array_equal(label, seen[key]):
            raise ValueError('Identical stationary input has conflicting encoded targets')
        seen[key] = label
        digest.update(json.dumps(record, sort_keys=True).encode())
        digest.update(frames.tobytes())
        digest.update(target.tobytes())
    return digest.hexdigest()


def balanced_draws(size, count, seed):
    generator = torch.Generator().manual_seed(seed)
    draws = []
    while len(draws) < count:
        draws.extend(torch.randperm(size, generator=generator).tolist())
    return np.asarray(draws[:count], dtype=np.int64)


def prepare(args, budget=None):
    metadata = json.loads(args.source.with_suffix('.json').read_text())
    config = metadata['config']
    base.validate_source(config)
    if config['training'].get('recovery_probe') and metadata.get('training_state', {}).get('probe_status') != 'complete_fixed_endpoint':
        raise ValueError('Source probe endpoint is incomplete')
    split = load_frozen_split(args.split_manifest)
    manifests = discover_cached_episodes(args.recovery_cache)
    split.verify_cache(manifests, 'recovery')
    if len(manifests) != 31 or any(split.role(m, 'recovery') != 'train' for m in manifests):
        raise ValueError('Requires exactly the 31 frozen recovery TRAIN episodes')
    if config['training'].get('split_sha256') != split.digest:
        raise ValueError('Source split differs')
    if not {m.name for m in manifests} <= set(config['training']['training_episodes']):
        raise ValueError('Recovery episodes were not in the source training membership')
    if args.output.resolve().is_relative_to(args.recovery_cache.resolve()):
        raise ValueError('Output must stay outside the training cache')
    dataset = RecoveryStarts(BehaviorCloningDataset(manifests))
    fingerprint = validate_samples(dataset, MouseBins(config['target_processing']['mouse_bins']))
    draws = balanced_draws(len(dataset), args.steps * args.batch_size, args.seed)
    counts = np.bincount(draws, minlength=len(dataset))
    plan = {
        'mode': 'execute' if args.execute else 'plan_only_no_training',
        'purpose': 'Intentional training-set memorization; not generalization or promotion',
        'source': str(args.source.resolve()), 'source_sha256': file_hash(args.source),
        'source_sidecar_sha256': file_hash(args.source.with_suffix('.json')),
        'source_progress': {k: metadata.get(k, 0) for k in ('epoch', 'global_step', 'step_in_epoch')},
        'helper_sha256': file_hash(base.__file__), 'output': str(args.output.resolve()),
        'code_sha256': {str(Path(__file__).resolve().parent / p): file_hash(Path(__file__).resolve().parent / p)
                       for p in ('run_recovery_fit_probe.py', 'run_recovery_probe.py',
                                 'models/imitation/train_bc.py', 'models/imitation/network.py',
                                 'models/imitation/dataset.py', 'models/imitation/mouse_bins.py',
                                 'models/imitation/frozen_split.py', 'models/imitation/checkpoint.py')},
        'libraries': {'torch': str(torch.__version__), 'numpy': np.__version__},
        'split_sha256': split.digest, 'samples_sha256': fingerprint,
        'samples': dataset.records, 'train_starts': len(dataset),
        'history': 'Newest RGB repeated in all four history slots',
        'targets': 'Unchanged source eight binary actions and two mouse-bin classes',
        'assessment_loaded': False, 'assessment_used_for_gradient_or_selection': False,
        'steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'seed': args.seed, 'sample_draws': len(draws),
        'draws_per_start_min_max': [int(counts.min()), int(counts.max())],
        'draw_sequence_sha256': hashlib.sha256(draws.tobytes()).hexdigest(),
        'augmentation': 'Disabled brightness/contrast/flip for fixed-input fit diagnostic',
        'dropout_loss_weights_bins_smoothing': 'Preserved from source',
        'fresh_optimizer': True, 'precision': 'float32_no_amp_skips',
        'completion': 'All fixed updates and complete before/after scoring on the same training starts',
        'fit_goal': 'All 31 encoded action vectors correct; failure at this budget does not prove impossibility',
        'shared_soft_budget_seconds': args.time_budget_seconds,
        'estimated_runtime_minutes': [1, 3], 'external_hard_cap': 'RuntimeMaxSec=5min',
        'checkpoint_saved': False, 'automatic_retry_or_promotion': False,
    }
    if budget:
        budget.check()
    return SimpleNamespace(args=args, config=config, dataset=dataset, draws=draws, plan=plan)


def score(model, dataset, config, device, budget, batch_size):
    bins = MouseBins(config['target_processing']['mouse_bins'])
    predicted, expected = [], []
    model.eval()
    with torch.inference_mode():
        for frames, targets in base.CheckedLoader(DataLoader(dataset, batch_size=batch_size), budget):
            x = frames.to(device=device, dtype=torch.float32).div_(255)
            if device.type == 'cuda':
                x = x.contiguous(memory_format=torch.channels_last)
            output = model(x)
            if not all(torch.isfinite(v).all() for v in output.values()):
                raise FloatingPointError('Non-finite diagnostic predictions')
            predicted.extend(np.column_stack([output[k].argmax(1).cpu().numpy()
                             for k in (*base.BINARY_ACTION_COLUMNS, 'mouse_dx_logits', 'mouse_dy_logits')]))
            expected.extend(targets.numpy())
    budget.check()
    p, target = np.asarray(predicted), np.asarray(expected)
    if len(target) != len(dataset):
        raise RuntimeError('Incomplete diagnostic scoring cohort')
    e = encoded_targets(target, bins)
    decoded_mouse = np.column_stack([np.rint(np.asarray(bins.representatives(axis))[p[:, 8+axis]]) for axis in range(2)])
    expected_mouse = np.column_stack([np.rint(np.asarray(bins.representatives(axis))[e[:, 8+axis]]) for axis in range(2)])
    correct = np.all(p == e, axis=1)
    return {'complete': True, 'samples': len(target), 'full_encoded_correct': int(correct.sum()),
            'fit_goal_met': bool(correct.all()), 'full_encoded_accuracy': float(correct.mean()),
            'binary_exact': float(np.all(p[:, :8] == e[:, :8], axis=1).mean()),
            'mouse_class_accuracy': [float((p[:, 8+i] == e[:, 8+i]).mean()) for i in range(2)],
            'predicted_idle': int((np.all(p[:, :8] == 0, axis=1) & np.all(decoded_mouse == 0, axis=1)).sum()),
            'dx_mae': float(np.abs(decoded_mouse[:, 0]-target[:, 8]).mean()),
            'target_quantization_dx_mae': float(np.abs(expected_mouse[:, 0]-target[:, 8]).mean()),
            'backward_correct': int(((p[:, 2] == 1) & (e[:, 2] == 1)).sum()),
            'backward_expected': int((e[:, 2] == 1).sum()),
            'per_start': [{**record, 'expected_classes': label.tolist(), 'predicted_classes': pred.tolist(),
                           'correct': bool(ok)} for record, label, pred, ok in zip(dataset.records, e, p, correct)]}


def execute(prepared, budget):
    args, cfg, plan, dataset = prepared.args, prepared.config, prepared.plan, prepared.dataset
    if args.output.exists():
        raise FileExistsError('Existing output refused; no automatic retry')
    for path, expected in ((args.source, plan['source_sha256']),
                           (args.source.with_suffix('.json'), plan['source_sidecar_sha256']),
                           (base.__file__, plan['helper_sha256'])):
        if file_hash(path) != expected:
            raise ValueError('Source/helper changed during preparation')
    if validate_samples(dataset, MouseBins(cfg['target_processing']['mouse_bins'])) != plan['samples_sha256']:
        raise ValueError('Prepared samples changed')
    if hashlib.sha256(prepared.draws.tobytes()).hexdigest() != plan['draw_sequence_sha256']:
        raise ValueError('Prepared draw sequence changed')
    if any(file_hash(path) != expected for path, expected in plan['code_sha256'].items()):
        raise ValueError('Diagnostic implementation changed during preparation')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    budget.check()
    args.output.mkdir(parents=True, exist_ok=False)
    progress = {'arm': 'stationary_training_fit', 'steps_completed': 0}
    report = {'plan': plan, 'status': 'running', 'diagnostic_complete': False,
              'progress': progress, 'script_sha256': file_hash(__file__)}

    def write_report():
        report['elapsed_seconds'] = round(budget.clock() - budget.started, 3)
        tmp = args.output/'report.tmp.json'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(args.output/'report.json')

    write_report()
    try:
        source = load_checkpoint(args.source, device='cpu')
        if source['config'] != cfg:
            raise ValueError('Source binary config differs')
        if {k: source.get(k, 0) for k in plan['source_progress']} != plan['source_progress']:
            raise ValueError('Source binary progress differs from sidecar')
        model = base.fresh_model(source['model_state_dict'], cfg, device, args.seed)
        del source
        report['before'] = score(model, dataset, cfg, device, budget, args.batch_size)
        write_report()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=cfg['training']['weight_decay'], fused=device.type == 'cuda')
        fit_cfg = copy.deepcopy(cfg)
        fit_cfg['training']['augmentation'] = {'brightness': 0., 'contrast': 0., 'horizontal_flip_probability': 0.}
        loader = DataLoader(Subset(dataset, prepared.draws.tolist()), batch_size=args.batch_size, shuffle=False)
        report['training'] = base.train_probe_steps(model, loader, optimizer, device, fit_cfg, budget, progress)
        if progress['steps_completed'] != args.steps:
            raise RuntimeError('Fixed update count not reached')
        report['after'] = score(model, dataset, cfg, device, budget, args.batch_size)
        if file_hash(args.source) != plan['source_sha256']:
            raise RuntimeError('Source changed during diagnostic')
        budget.check()
        report.update(status='complete_diagnostic_only', diagnostic_complete=True)
    except base.BudgetExceeded as error:
        report.update(status='budget_exhausted_incomplete', reason=str(error))
    except BaseException as error:
        report.update(status='interrupted_or_failed_incomplete', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        progress['consumed_draws'] = progress['steps_completed'] * args.batch_size
        write_report()
    print(f"Fit diagnostic: {report['status']}; {args.output/'report.json'}", flush=True)
    return 0 if report['diagnostic_complete'] else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--recovery-cache', type=Path, default=Path('data/datasets/recovery_cached_frames'))
    parser.add_argument('--split-manifest', type=Path, default=Path('data/splits/frozen_v3.json'))
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--time-budget-seconds', type=float, default=240)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 400 or not 1 <= args.batch_size <= 8 or args.seed < 0:
        parser.error('At most 400 updates, batch at most 8 and nonnegative seed required')
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 5e-5:
        parser.error('Learning rate must be positive, finite and at most 5e-5')
    if not math.isfinite(args.time_budget_seconds) or not 0 < args.time_budget_seconds <= 240:
        parser.error('Shared budget must be positive, finite and at most 240 seconds')
    return args


def main(argv=None):
    args = parse_args(argv)
    budget = base.Budget(args.time_budget_seconds) if args.execute else None
    prepared = prepare(args, budget)
    print(json.dumps(prepared.plan, indent=2), flush=True)
    return execute(prepared, budget) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
