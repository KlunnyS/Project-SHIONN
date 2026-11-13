"""Plan or explicitly train one bounded stationary-recovery replay candidate.

One fixed endpoint, fresh optimizer, full before/after evaluation. No game/input
APIs, automatic promotion, assessment gradients or assessment checkpoint selection.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import run_recovery_probe as base
import run_recovery_fit_probe as fit
from run_recovery_history_probe import cache_fingerprint
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BehaviorCloningDataset, discover_cached_episodes
from models.imitation.mouse_bins import MouseBins

OUTPUT = Path('models/imitation/checkpoints/runs/v5_stationary_replay_v1')
BATCH_SIZE, STATIC_PER_BATCH = 8, 2


class ReplayBatches(Dataset):
    """Exactly two stationary starts and six regular frames per loader batch."""
    def __init__(self, regular, starts, regular_draws, start_draws):
        if len(start_draws) % 2 or len(regular_draws) != len(start_draws) * 3:
            raise ValueError('Replay requires exactly two starts and six regular draws per batch')
        self.regular, self.starts = regular, starts
        self.regular_draws, self.start_draws = regular_draws, start_draws

    def __len__(self):
        return len(self.start_draws) * 4

    def __getitem__(self, index):
        batch, slot = divmod(index, BATCH_SIZE)
        if slot < STATIC_PER_BATCH:
            frames, target = self.starts[int(self.start_draws[batch * 2 + slot])]
            return frames, target, True
        frames, target = self.regular[int(self.regular_draws[batch * 6 + slot - 2])]
        return frames, target, False


def code_fingerprint():
    root = Path(__file__).resolve().parent
    paths = ['run_stationary_replay.py', 'run_recovery_probe.py', 'run_recovery_fit_probe.py',
             'run_recovery_history_probe.py', 'models/imitation/checkpoint.py',
             'models/imitation/dataset.py', 'models/imitation/frozen_split.py',
             'models/imitation/mouse_bins.py', 'models/imitation/network.py',
             'models/imitation/train_bc.py']
    return {str(root / p): fit.file_hash(root / p) for p in paths}


def prepare(args, budget=None):
    metadata = json.loads(args.source.with_suffix('.json').read_text())
    cfg = metadata['config']
    base.validate_source(cfg)
    if cfg['training'].get('recovery_probe') and metadata.get('training_state', {}).get('probe_status') != 'complete_fixed_endpoint':
        raise ValueError('Source probe endpoint must be complete')
    data = base.load_data(args, cfg, budget)
    train, validation, assessment, groups, split = data
    paths = {g: str(p.resolve()) for g, p in (
        ('base', args.base_cache), ('recovery', args.recovery_cache),
        ('proactive_turn', args.proactive_cache), ('assessment', args.assessment_cache))}
    if any(args.output.resolve().is_relative_to(Path(p)) for p in paths.values()):
        raise ValueError('Output must stay outside all data caches')
    recovery = BehaviorCloningDataset(discover_cached_episodes(args.recovery_cache))
    if len(recovery.episode_names) != 31 or set(recovery.episode_names) != {n for n, g in groups.items() if g == 'recovery'}:
        raise ValueError('Requires exactly the 31 frozen recovery TRAIN episodes')
    starts, assessment_starts = fit.RecoveryStarts(recovery), fit.RecoveryStarts(assessment)
    bins = MouseBins(cfg['target_processing']['mouse_bins'])
    start_hash = fit.validate_samples(starts, bins)
    weights, _ = base.recovery_opening_weights(train, groups)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), args.steps * 6, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    regular_draws = np.asarray(list(sampler), dtype=np.int64)
    start_draws = fit.balanced_draws(len(starts), args.steps * 2, args.seed + 1)
    labels = np.array([groups.get(train.episode_names[e], 'base') for e, _ in train.index])
    counts = {g: int((labels[regular_draws] == g).sum()) for g in ('base', 'recovery', 'proactive_turn')}
    exposures = np.bincount(start_draws, minlength=len(starts))
    plan = {
        'mode': 'execute' if args.execute else 'plan_only_no_training',
        'source': str(args.source.resolve()), 'source_sha256': fit.file_hash(args.source),
        'source_sidecar_sha256': fit.file_hash(args.source.with_suffix('.json')),
        'source_progress': {k: metadata.get(k, 0) for k in ('epoch', 'global_step', 'step_in_epoch')},
        'code_sha256': code_fingerprint(), 'libraries': {'torch': str(torch.__version__), 'numpy': np.__version__},
        'output': str(args.output.resolve()), 'cache_paths': paths,
        'cache_fingerprint': cache_fingerprint(paths, 2**63-1),
        'split_sha256': split.digest, 'train_episodes': len(train.episode_names),
        'validation_episodes': len(validation.episode_names), 'validation_frames': len(validation),
        'assessment_episodes': len(assessment.episode_names), 'assessment_frames': len(assessment),
        'assessment_used_for_gradient_or_checkpoint_selection': False,
        'stationary_starts': starts.records, 'stationary_samples_sha256': start_hash,
        'steps': args.steps, 'batch_size': BATCH_SIZE, 'static_per_batch': STATIC_PER_BATCH,
        'learning_rate': args.learning_rate, 'seed': args.seed,
        'regular_group_expected_fractions': {'base': .9, **base.GROUP_FRACTIONS},
        'total_expected_group_fractions': {'base': .675, 'recovery': .2875, 'proactive_turn': .0375},
        'regular_draw_counts': counts, 'regular_draws': len(regular_draws),
        'stationary_draws': len(start_draws), 'start_exposures_min_max': [int(exposures.min()), int(exposures.max())],
        'regular_draws_sha256': hashlib.sha256(regular_draws.tobytes()).hexdigest(),
        'stationary_draws_sha256': hashlib.sha256(start_draws.tobytes()).hexdigest(),
        'augmentation': 'Source augmentation on regular frames only; stationary starts unchanged',
        'fresh_optimizer': True, 'training_precision': 'float32_no_amp_skips',
        'evaluation': 'Fresh complete before/after validation and six assessment clips, plus stationary-start class scores',
        'endpoint_selection': 'Fixed final update; no assessment selection or automatic promotion',
        'validation_screen': {'binary_exact_max_drop': .005, 'mouse_mae_max_ratio': 1.05,
                              'first8_idle_max_additional_fraction': .01},
        'shared_soft_budget_seconds': args.time_budget_seconds,
        'estimated_total_runtime_minutes': [6, 10], 'external_hard_cap': 'RuntimeMaxSec=14min',
        'no_automatic_retry_or_extension': True,
    }
    if budget:
        budget.check()
    return SimpleNamespace(args=args, cfg=cfg, data=data, starts=starts, assessment_starts=assessment_starts,
                           regular_draws=regular_draws, start_draws=start_draws, labels=labels, plan=plan)


def train_steps(model, loader, optimizer, device, cfg, budget, progress):
    model.train()
    bins = MouseBins(cfg['target_processing']['mouse_bins'])
    weights = torch.tensor(cfg['target_processing']['binary_class_weights'], dtype=torch.float32, device=device)
    total_loss, samples = 0., 0
    for frames, targets, stationary in base.CheckedLoader(loader, budget):
        frames = frames.to(device=device, dtype=torch.float32).div_(255)
        if device.type == 'cuda':
            frames = frames.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device)
        regular = ~stationary.to(device=device, dtype=torch.bool)
        aug = cfg['training']['augmentation']
        augmented, augmented_targets = base.augment_batch(
            frames[regular], targets[regular], brightness=aug['brightness'], contrast=aug['contrast'],
            horizontal_flip_probability=aug['horizontal_flip_probability'])
        frames[regular], targets[regular] = augmented, augmented_targets
        optimizer.zero_grad(set_to_none=True)
        loss, _ = base.behavior_cloning_loss(model(frames), targets, weights, mouse_bins=bins,
                                             label_smoothing=cfg['training']['label_smoothing'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite candidate training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'] or float('inf'),
                                       error_if_nonfinite=True)
        optimizer.step()
        progress['steps_completed'] += 1
        total_loss += loss.item() * len(targets)
        samples += len(targets)
        if progress['steps_completed'] % 100 == 0:
            print(f"stationary replay: {progress['steps_completed']} updates", flush=True)
        budget.check()
    return {'loss': total_loss / samples, 'samples': samples, 'precision': 'float32_no_amp_skips'}


def evaluate(model, prepared, device, budget):
    cfg, (_, validation, assessment, _, _) = prepared.cfg, prepared.data
    result = {}
    for name, dataset in (('validation', validation), ('assessment', assessment)):
        print(f'Evaluating full {name} ({len(dataset):,} frames)...', flush=True)
        result[name] = base.evaluate(model, dataset, device, cfg['target_processing'], budget, 32,
                                     cfg['training']['label_smoothing'])
    for name, dataset in (('training_starts', prepared.starts), ('assessment_starts', prepared.assessment_starts)):
        result[name] = fit.score(model, dataset, cfg, device, budget, BATCH_SIZE)
    return result


def validation_screen(before, after, limits):
    b, a = before['segments']['all'], after['segments']['all']
    opening_b, opening_a = before['segments']['first8'], after['segments']['first8']
    checks = {'binary_exact': a['exact_binary_match'] >= b['exact_binary_match'] - limits['binary_exact_max_drop'],
              **{f'mouse_{axis}_mae': a['mouse_mae'][axis] <= b['mouse_mae'][axis] * limits['mouse_mae_max_ratio']
                 for axis in ('dx', 'dy')},
              'opening_idle': opening_a['predicted_idle_frames'] / opening_a['frames'] <=
                              opening_b['predicted_idle_frames'] / opening_b['frames'] + limits['first8_idle_max_additional_fraction']}
    checks = {name: bool(value) for name, value in checks.items()}
    return {'checks': checks, 'passed': all(checks.values()),
            'meaning': 'Navigation imitation guard versus warm-start source; not a game success or promotion gate'}


def endpoint_config(prepared):
    cfg = copy.deepcopy(prepared.cfg)
    plan = prepared.plan
    cfg['training'].update(sampling='stationary-replay-mixed', learning_rate=plan['learning_rate'],
                           batch_size=BATCH_SIZE, seed=plan['seed'],
                           correction_sampling_fraction=0.,
                           correction_group_fractions={'recovery': .2875, 'proactive_turn': .0375},
                           stationary_replay={k: plan[k] for k in ('static_per_batch', 'regular_group_expected_fractions',
                                                                 'total_expected_group_fractions', 'stationary_samples_sha256')})
    cfg['training']['recovery_probe'] = {
        'arm': 'stationary_replay_candidate', 'source': plan['source'], 'source_sha256': plan['source_sha256'],
        'source_progress': plan['source_progress'], 'optimizer_steps': plan['steps'],
        'fresh_optimizer': True, 'generic_resume_supported': False,
        'assessment_used_for_gradient_or_selection': False,
    }
    return cfg


def execute(prepared, budget):
    args, plan, cfg = prepared.args, prepared.plan, prepared.cfg
    if args.output.exists():
        raise FileExistsError('Output exists; prior runs are never overwritten')
    for path, expected in [(args.source, plan['source_sha256']),
                           (args.source.with_suffix('.json'), plan['source_sidecar_sha256']), *plan['code_sha256'].items()]:
        if fit.file_hash(path) != expected:
            raise ValueError('Source/code changed during preparation')
    for values, key in ((prepared.regular_draws, 'regular_draws_sha256'), (prepared.start_draws, 'stationary_draws_sha256')):
        if hashlib.sha256(values.tobytes()).hexdigest() != plan[key]:
            raise ValueError('Prepared draws changed')
    if fit.validate_samples(prepared.starts, MouseBins(cfg['target_processing']['mouse_bins'])) != plan['stationary_samples_sha256']:
        raise ValueError('Prepared stationary samples changed')
    if cache_fingerprint(plan['cache_paths'], 2**63-1) != plan['cache_fingerprint']:
        raise ValueError('Caches changed during preparation')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    budget.check()
    args.output.mkdir(parents=True, exist_ok=False)
    progress = {'steps_completed': 0, 'status': 'evaluating_source'}
    report = {'plan': plan, 'status': 'running', 'candidate_complete': False, 'progress': progress}

    def write_report():
        report['elapsed_seconds'] = round(budget.clock() - budget.started, 3)
        tmp = args.output/'report.tmp.json'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(args.output/'report.json')

    write_report()
    try:
        source = load_checkpoint(args.source, device='cpu')
        if source['config'] != cfg or {k: source.get(k, 0) for k in plan['source_progress']} != plan['source_progress']:
            raise ValueError('Source binary config/progress differs from sidecar')
        model = base.fresh_model(source['model_state_dict'], cfg, device, args.seed)
        del source
        report['before'] = evaluate(model, prepared, device, budget)
        write_report()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=cfg['training']['weight_decay'], fused=device.type == 'cuda')
        train = prepared.data[0]
        dataset = ReplayBatches(train, prepared.starts, prepared.regular_draws, prepared.start_draws)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        progress['status'] = 'training'
        write_report()
        report['training'] = train_steps(model, loader, optimizer, device, cfg, budget, progress)
        if progress['steps_completed'] != args.steps:
            raise RuntimeError('Fixed optimizer budget did not complete')
        progress['status'] = 'evaluating_candidate'
        write_report()
        report['after'] = evaluate(model, prepared, device, budget)
        report['validation_screen'] = validation_screen(report['before']['validation'], report['after']['validation'], plan['validation_screen'])
        if fit.file_hash(args.source) != plan['source_sha256'] or cache_fingerprint(plan['cache_paths'], 2**63-1) != plan['cache_fingerprint']:
            raise ValueError('Source/data changed during execution')
        budget.check()
        # Final name is absent until fixed updates and all evaluation complete.
        save_checkpoint(args.output/'pending.pt', model=model, optimizer=optimizer, epoch=0,
                        global_step=progress['steps_completed'], step_in_epoch=progress['steps_completed'],
                        best_val_loss=report['after']['validation']['loss'], config=endpoint_config(prepared),
                        training_state={'probe_status': 'complete_fixed_endpoint'})
        budget.check()
        (args.output/'pending.pt').replace(args.output/'candidate.pt')
        (args.output/'pending.json').replace(args.output/'candidate.json')
        report['candidate_sha256'] = fit.file_hash(args.output/'candidate.pt')
        progress['status'] = 'complete'
        report.update(status='complete_no_automatic_promotion', candidate_complete=True)
    except base.BudgetExceeded as error:
        progress['status'] = 'budget_exhausted_incomplete'
        report.update(status='budget_exhausted_incomplete', reason=str(error))
    except BaseException as error:
        progress['status'] = 'interrupted_or_failed_incomplete'
        report.update(status='interrupted_or_failed_incomplete', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        n = progress['steps_completed']
        progress['consumed_stationary_draws'] = n * 2
        progress['consumed_regular_draw_counts'] = {g: int((prepared.labels[prepared.regular_draws[:n*6]] == g).sum())
                                                   for g in ('base', 'recovery', 'proactive_turn')}
        write_report()
    print(f"Candidate: {report['status']}; report: {args.output/'report.json'}", flush=True)
    return 0 if report['candidate_complete'] else 2


def parse_args(argv=None):
    defaults = base.parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source', type=Path, default=fit.SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    for name in ('split_manifest', 'base_cache', 'recovery_cache', 'proactive_cache', 'assessment_cache'):
        parser.add_argument('--'+name.replace('_','-'), type=Path, default=getattr(defaults, name))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time-budget-seconds', type=float, default=720)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 2000 or not 0 <= args.seed < 2**63-2:
        parser.error('At most 2000 updates and a bounded nonnegative seed required')
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 5e-5:
        parser.error('Learning rate must be positive finite and at most 5e-5')
    if not math.isfinite(args.time_budget_seconds) or not 0 < args.time_budget_seconds <= 720:
        parser.error('Shared budget must be positive finite and at most 720 seconds')
    return args


def main(argv=None):
    args = parse_args(argv)
    budget = base.Budget(args.time_budget_seconds) if args.execute else None
    prepared = prepare(args, budget)
    print(json.dumps(prepared.plan, indent=2), flush=True)
    return execute(prepared, budget) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
