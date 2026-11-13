"""Plan-only whole-episode FF/GRU comparison; no game APIs or automatic retries."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import run_route_expansion as parent
import run_temporal_readout_comparison as temporal
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BINARY_ACTION_COLUMNS
from models.imitation.mouse_bins import MouseBins
from models.imitation.network import BINNED_ARCHITECTURE_VERSION
from models.imitation.recurrent import RECURRENT_ARCHITECTURE_VERSION
from models.imitation import recovery_transfer as rt

ARMS = ('feedforward', 'gru', 'gru_reset')
LENGTH = 32
REFERENCE = Path('models/imitation/checkpoints/runs/route_expansion_v1/report.json')
OUTPUT = Path('models/imitation/checkpoints/runs/ordered_gru_route_v1')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class EpisodeStream:
    """Ordered active-window indices; original dataset supplies same-episode history."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.offsets = []
        offset = 0
        for e, (frames, actions) in enumerate(dataset.episodes):
            first = dataset.first_action_frames[e]
            if not 0 <= first < len(frames) or len(actions) != len(frames):
                raise ValueError('Empty or misaligned episode')
            count = len(frames) - first
            if dataset.index[offset:offset + count] != [(e, f) for f in range(first, len(frames))]:
                raise ValueError('Dataset is not consecutive episode-local data')
            self.offsets.append((offset, count))
            offset += count
        if offset != len(dataset.index):
            raise ValueError('Unexpected dataset targets')

    def chunks(self, episode, length=LENGTH):
        offset, count = self.offsets[episode]
        for start in range(0, count, length):
            yield list(range(offset + start, offset + min(start + length, count)))

    def batch(self, indices, length=LENGTH):
        if not 0 < len(indices) <= length:
            raise ValueError('Invalid chunk capacity')
        items = [self.dataset[i] for i in indices]
        shape = np.asarray(items[0][0]).shape
        x = np.zeros((length, *shape), dtype=np.uint8)
        y = np.zeros((length, 10), dtype=np.float32)
        valid = np.zeros(length, dtype=bool)
        for j, (frame, action) in enumerate(items):
            x[j], y[j], valid[j] = frame, action, True
        return x, y, valid


def make_schedule(streams, target_frames=16000, seed=0):
    """Complete episodes; greedily balance base-map frame mass and human share."""
    if target_frames < 1:
        raise ValueError('Target budget must be positive')
    rng = np.random.default_rng(seed)
    for stream in streams.values():
        if any(m not in {f'dataset_test{i}' for i in range(1, 11)} for m in stream.dataset.episode_maps):
            raise ValueError('Schedule contains a non-TRAIN map')
    by_map = defaultdict(list)
    for e, name in enumerate(streams['base'].dataset.episode_maps):
        by_map[name].append(e)
    if not by_map or not streams['human'].offsets:
        raise ValueError('Both TRAIN pools required')
    queues = {}
    def draw(key, options):
        if not queues.get(key):
            queues[key] = list(rng.permutation(options))
        return int(queues[key].pop())
    total = human = 0
    map_frames = Counter({m: 0 for m in by_map})
    schedule = []
    while total < target_frames:
        role = 'human' if human < .2 * total else 'base'
        if role == 'human':
            episode = draw('human', range(len(streams['human'].offsets)))
        else:
            minimum = min(map_frames.values())
            candidates = sorted(m for m in by_map if map_frames[m] == minimum)
            mapname = candidates[int(rng.integers(len(candidates)))]
            episode = draw(mapname, by_map[mapname])
        count = streams[role].offsets[episode][1]
        schedule.append((role, episode))
        total += count
        if role == 'human':
            human += count
        else:
            map_frames[streams[role].dataset.episode_maps[episode]] += count
    return schedule


def exposure(streams, schedule):
    groups, episodes, maps, categories = Counter(), Counter(), Counter(), Counter()
    steps = 0
    for role, e in schedule:
        ds = streams[role].dataset
        count = streams[role].offsets[e][1]
        groups[role] += count
        episodes[ds.episode_names[e]] += count
        maps[ds.episode_maps[e]] += count
        category = getattr(ds, 'episode_categories', {}).get(ds.episode_names[e], 'original_navigation' if role == 'base' else 'original_recovery')
        categories[category] += count
        steps += math.ceil(count / LENGTH)
    return {'supervised_frames': sum(groups.values()), 'frames_by_group': dict(groups),
            'frames_by_episode': dict(episodes), 'frames_by_map': dict(maps), 'frames_by_category': dict(categories),
            'optimizer_updates_per_arm': steps, 'episode_occurrences': len(schedule)}


def build_model(arm, state, cfg, device):
    model = temporal.build_model(arm, state, cfg, 0)
    model.trunk.requires_grad_(True)
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    return model


def forward_chunk(model, frames, arm, hidden=None, valid=None):
    """Normalized frames [T,12,H,W]; same batched encoder in every arm.

    Return logits [1,T,C], state [1,1,512] or None. Invalid padding is
    excluded from encoding; the shared readout masks persistent state updates.
    """
    if arm not in ARMS or frames.ndim != 4:
        raise ValueError('Unknown arm or invalid frame shape')
    if valid is None:
        valid = torch.ones(len(frames), dtype=torch.bool, device=frames.device)
    if valid.dtype != torch.bool or valid.shape != (len(frames),) or not bool(valid.any()):
        raise ValueError('Invalid or empty chunk mask')
    indices = valid.nonzero(as_tuple=False).flatten()
    pixels = frames[valid]
    if pixels.device.type == 'cuda':
        pixels = pixels.contiguous(memory_format=torch.channels_last)
    encoded = model.trunk(pixels)
    features = encoded.new_zeros((len(frames), encoded.shape[-1])).index_copy(0, indices, encoded)
    return temporal.readout(model, features[None], arm, hidden, valid[None])


def optimizer_for(model, cfg):
    encoder = list(model.trunk.parameters())
    encoder_ids = {id(v) for v in encoder}
    readout = [v for v in model.parameters() if id(v) not in encoder_ids]
    return torch.optim.AdamW([{'params': encoder, 'lr': 1e-5},
                             {'params': readout, 'lr': 1e-3}],
                            weight_decay=cfg['training']['weight_decay'])


def train(model, arm, p, device, budget, progress):
    model.train()
    optimizer = optimizer_for(model, p.parent.cfg)
    bins = MouseBins(p.parent.cfg['target_processing']['mouse_bins'])
    weights = torch.tensor(p.parent.cfg['target_processing']['binary_class_weights'], device=device)
    count = 0
    loss_sum = 0.
    trace = hashlib.sha256()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for occurrence, (role, e) in enumerate(p.schedule):
        hidden = None  # Reset even if the same episode was sampled again.
        stream = p.streams[role]
        for indices in stream.chunks(e):
            budget.check()
            x, y, mask = stream.batch(indices)
            x = torch.from_numpy(x).to(device, dtype=torch.float32).div_(255)
            y, mask = torch.from_numpy(y).to(device), torch.from_numpy(mask).to(device)
            if hidden is not None:
                hidden = hidden.detach()
            optimizer.zero_grad(set_to_none=True)
            logits, hidden = forward_chunk(model, x, arm, hidden, mask)
            selected = {k: v[0, mask] for k, v in logits.items()}
            loss, _ = parent.base.behavior_cloning_loss(selected, y[mask], weights,
                mouse_bins=bins, label_smoothing=p.parent.cfg['training']['label_smoothing'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ordered training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            progress['steps_completed'] += 1
            n = len(indices)
            count += n
            loss_sum += float(loss.detach()) * n
            trace.update(json.dumps([occurrence, role, indices], separators=(',', ':')).encode())
            if progress['steps_completed'] % 100 == 0:
                print(f'{arm}: {progress["steps_completed"]} updates, {count} targets', flush=True)
            budget.check()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return optimizer, {'loss': loss_sum / count, 'supervised_frames': count,
        'target_trace_sha256': trace.hexdigest(), 'precision': 'float32_no_amp_skips',
        'elapsed_seconds': seconds, 'supervised_frames_per_second': count / seconds,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None}


def decode(logits, bins):
    binary = np.stack([logits[k][0].argmax(-1).cpu().numpy() for k in BINARY_ACTION_COLUMNS], axis=1)
    mouse = np.stack([np.rint(np.asarray(bins.representatives(axis))[
        logits[key][0].argmax(-1).cpu().numpy()]) for axis, key in enumerate(('mouse_dx_logits', 'mouse_dy_logits'))], axis=1)
    return np.column_stack((binary, mouse))


def evaluate_dataset(model, arm, stream, cfg, device, budget, openings_only=False):
    model.eval()
    bins = MouseBins(cfg['target_processing']['mouse_bins'])
    weights = torch.tensor(cfg['target_processing']['binary_class_weights'], device=device)
    metrics = {k: parent.base.ActionMetrics() for k in ('all', 'first8')}
    idle, expert_idle = Counter(), Counter()
    records = []
    total_loss = 0.
    ds = stream.dataset
    with torch.inference_mode():
        for e, name in enumerate(ds.episode_names):
            if e and e % 20 == 0:
                print(f'{arm}: evaluated {e}/{len(ds.episode_names)} episodes', flush=True)
            hidden = None
            expected, predicted = [], []
            chunks = stream.chunks(e, 8 if openings_only else LENGTH)
            for indices in chunks:
                budget.check()
                x, y, valid = stream.batch(indices, 8 if openings_only else LENGTH)
                x = torch.from_numpy(x).to(device, dtype=torch.float32).div_(255)
                y, valid = torch.from_numpy(y).to(device), torch.from_numpy(valid).to(device)
                logits, hidden = forward_chunk(model, x, arm, hidden, valid)
                if not all(torch.isfinite(v).all() for v in logits.values()):
                    raise FloatingPointError('Nonfinite ordered evaluation logits')
                loss, _ = parent.base.behavior_cloning_loss({k: v[0, valid] for k, v in logits.items()},
                    y[valid], weights, mouse_bins=bins, label_smoothing=cfg['training']['label_smoothing'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite ordered evaluation loss')
                expected.extend(y[valid].cpu().numpy())
                predicted.extend(decode(logits, bins)[valid.cpu().numpy()])
                total_loss += float(loss) * len(indices)
                if openings_only:
                    break
            a, b = np.asarray(expected), np.asarray(predicted)
            records.append({'episode': name, 'map': ds.episode_maps[e], 'metrics': rt.episode_metrics(a, b),
                            **({'targets': a.tolist(), 'predictions': b.tolist()} if openings_only or ds is not getattr(stream, 'navigation_dataset', None) else {})})
            for j, (target, action) in enumerate(zip(a, b)):
                for segment in ('all', 'first8') if j < 8 else ('all',):
                    metrics[segment].add(target, action)
                    idle[segment] += int(not action.any())
                    expert_idle[segment] += int(not target.any())
    total = sum(r['metrics']['frames'] for r in records)
    budget.check()
    return {'complete': True, 'frames': total, 'loss': total_loss / total,
        'episodes': len(records), 'records': records, 'aggregate': rt.aggregate([r['metrics'] for r in records]),
        'segments': {k: {**v.report(), 'predicted_idle_frames': idle[k], 'expert_idle_frames': expert_idle[k]} for k, v in metrics.items()}}


def evaluate(model, arm, p, device, budget, initial=False):
    roles = ('calibration', 'human_train', 'human_held') if initial else ('navigation', 'human_train', 'human_held')
    result = {}
    for role in roles:
        print(f'{arm}: evaluating {role} ({"initial" if initial else "endpoint"})', flush=True)
        result[role] = evaluate_dataset(model, arm, p.eval_streams[role], p.parent.cfg, device, budget, openings_only=role == 'calibration')
    records = result['human_train']['records']
    result['human_train_categories'] = {category: rt.aggregate([r['metrics'] for r in records if p.plan['human_categories'].get(r['episode']) == category]) for category in sorted(set(p.plan['human_categories'].values()))}
    return result


def verify_source_reference(reference, plan):
    ignore = {'mode', 'soft_budget_seconds'}
    if not reference.get('paired_complete') or reference.get('status') != 'complete_no_automatic_promotion':
        raise ValueError('Source baseline pair incomplete')
    if {k: v for k, v in reference['plan'].items() if k not in ignore} != {k: v for k, v in plan.items() if k not in ignore}:
        raise ValueError('Source baseline provenance differs')
    if not reference['before']['navigation']['complete'] or reference['before']['navigation']['frames'] != 39769:
        raise ValueError('Source baseline evaluation incomplete')
    for arm in ('control', 'correction'):
        entry = reference['arms'][arm]
        if entry.get('status') != 'complete' or entry.get('steps_completed') != 1000 or not entry['evaluation']['navigation']['complete']:
            raise ValueError('Reference arm incomplete')


def target_trace(streams, schedule):
    digest = hashlib.sha256()
    for occurrence, (role, e) in enumerate(schedule):
        for indices in streams[role].chunks(e):
            digest.update(json.dumps([occurrence, role, indices], separators=(',', ':')).encode())
    return digest.hexdigest()


def development_screen(report):
    arms = report['arms']
    gru = arms['gru']['after']['human_held']['aggregate']
    reference = report['source_baseline']['development_holdout']['aggregate']
    score = gru['episode_mean']['binary_exact']
    checks = {'navigation': arms['gru']['navigation_screen']['passed'],
        'over_feedforward_by_002': score >= arms['feedforward']['after']['human_held']['aggregate']['episode_mean']['binary_exact'] + .02,
        'over_reset_by_002': score >= arms['gru_reset']['after']['human_held']['aggregate']['episode_mean']['binary_exact'] + .02,
        'source_heldout_agreement': score >= reference['episode_mean']['binary_exact'],
        'backward_true_positive': gru['backward_tp'] >= 8,
        'backward_false_positive': gru['false_backward'] <= 2}
    return {'checks': checks, 'passed': all(checks.values()),
            'meaning': 'Fixed development eligibility screen; not gameplay performance or automatic promotion'}


def prepare(args, budget):
    parent_args = SimpleNamespace(execute=False, steps=1000, seed=0,
        time_budget_seconds=args.time_budget_seconds, base_cache=Path('data/datasets/cached_frames'),
        output=args.output, device=args.device)
    if args.output.resolve().is_relative_to(REFERENCE.parent.resolve()):
        raise ValueError('Output overlaps historical reference run')
    p = parent.prepare(parent_args, budget)
    reference = json.loads(REFERENCE.read_text())
    verify_source_reference(reference, p.plan)
    p.corrections.episode_categories = {name: p.plan['new_episodes'].get(name, 'original_recovery') for name in p.corrections.episode_names}
    streams = {'base': EpisodeStream(p.regular), 'human': EpisodeStream(p.corrections)}
    schedule = make_schedule(streams, args.target_frames)
    actual = exposure(streams, schedule)
    if not actual['frames_by_group'].get('base') or not actual['frames_by_group'].get('human') or not set(p.corrections.episode_names) <= set(actual['frames_by_episode']):
        raise ValueError('Requested budget omits TRAIN corrections; increase before executing')
    if actual['supervised_frames'] > 20000:
        raise ValueError('Whole-episode target budget exceeds20000; revise before executing')
    # Choose calibration episodes before their predictions, independent of schedule.
    names = set()
    for m in sorted(set(p.regular.episode_maps)):
        group = [n for n, mapname in zip(p.regular.episode_names, p.regular.episode_maps) if mapname == m]
        names.update(sorted(group, key=lambda n: hashlib.sha256(('ordered-gru-calibration-v1:' + n).encode()).hexdigest())[:10])
    # A view retains original episode arrays/history but rebuilds consecutive indices.
    calibration = subset_dataset(p.regular, names)
    eval_streams = {'navigation': EpisodeStream(p.navigation), 'human_train': streams['human'],
                    'human_held': EpisodeStream(p.held), 'calibration': EpisodeStream(calibration)}
    eval_streams['navigation'].navigation_dataset = p.navigation
    helper_files = [Path(__file__), Path('run_temporal_readout_comparison.py'), Path('run_targeted_navigation_correction.py'), REFERENCE]
    plan = {'mode': 'execute' if args.execute else 'plan_only_no_training', 'input_context': p.plan,
        'input_sha256': {str(f): parent.file_hash(f) for f in helper_files}, 'source': str(p.source),
        'input_context_note': 'Parent source/data verification only; its frame schedule is NOT trained by this runner',
        'schedule': schedule, 'schedule_sha256': json_hash(schedule), 'target_trace_sha256': target_trace(streams, schedule), 'exposure': actual,
        'calibration_episodes': sorted(names), 'arms': list(ARMS), 'sequence_length': LENGTH,
        'state_policy': 'ordered_episode_state_carry_detach_each_chunk',
        'human_context_limit': 'Human clips start with zero state; unrecorded handoff gap and autonomous prefix are not joined or labelled expert', 'initialization': 'identical_source_encoder_and_NEW_seeded_heads; GRUs_identical',
        'encoder_frozen': False, 'optimizer_lrs': {'encoder': 1e-5, 'heads_memory': 1e-3},
        'precision': 'float32_no_amp_skips', 'augmentation': 'none', 'seed': 0, 'cpu_threads': 4,
        'libraries': {'torch': str(torch.__version__), 'numpy': np.__version__},
        'human_categories': p.corrections.episode_categories,
        'source_ancestry': p.cfg['training']['training_episodes'],
        'train_episodes': sorted(actual['frames_by_episode']), 'source_reference': str(REFERENCE),
        'soft_budget_seconds': args.time_budget_seconds, 'estimated_minutes': [12, 20],
        'external_hard_cap_minutes': 30, 'automatic_promotion_or_retry': False}
    budget.check()
    return SimpleNamespace(args=args, parent=p, streams=streams, eval_streams=eval_streams,
                           schedule=schedule, plan=plan, reference=reference)


def subset_dataset(dataset, names):
    result = copy.copy(dataset)
    selected = [e for e, n in enumerate(dataset.episode_names) if n in names]
    result.episodes = [dataset.episodes[e] for e in selected]
    result.first_action_frames = [dataset.first_action_frames[e] for e in selected]
    result.episode_names = [dataset.episode_names[e] for e in selected]
    result.episode_maps = [dataset.episode_maps[e] for e in selected]
    result.index = [(e, f) for e, (frames, _) in enumerate(result.episodes)
                    for f in range(result.first_action_frames[e], len(frames))]
    return result


def guard(p, budget):
    parent.guard(p.parent, budget)
    for file, digest in p.plan['input_sha256'].items():
        if parent.file_hash(file) != digest:
            raise ValueError('Ordered experiment code/reference changed')
    if json_hash(p.schedule) != p.plan['schedule_sha256'] or exposure(p.streams, p.schedule) != p.plan['exposure'] or target_trace(p.streams, p.schedule) != p.plan['target_trace_sha256']:
        raise ValueError('Ordered target exposure changed')
    budget.check()


def endpoint_config(p, arm):
    cfg = copy.deepcopy(p.parent.cfg)
    cfg['architecture'] = BINNED_ARCHITECTURE_VERSION if arm == 'feedforward' else RECURRENT_ARCHITECTURE_VERSION
    cfg['memory'] = None if arm == 'feedforward' else {'type': 'gru', 'hidden_size': 512,
        'feature_gate': False, 'sequence_length': LENGTH, 'burn_in': 0}
    count = p.plan['exposure']['supervised_frames']
    cfg['training'].update(learning_rate=1e-3, batch_size=1, encoder_frozen=False,
        lr_scheduler={'name': 'none'}, train_extra_cache_dirs=[], start_sampling_boost=1.,
        optimizer_parameter_group_lrs={'encoder': 1e-5, 'heads_memory': 1e-3},
        sampling='ordered_whole_episode_tbptt', training_episodes=p.plan['train_episodes'],
        augmentation={'brightness': 0., 'contrast': 0., 'horizontal_flip_probability': 0.},
        correction_sampling_fraction=p.plan['exposure']['frames_by_group']['human'] / count,
        correction_group_fractions={}, correction_episode_groups={},
        split_id='ordered-gru-route-v1', split_sha256=p.plan['input_sha256'][str(Path(__file__))],
        ordered_gru_comparison={'heads_reset': True, 'state_policy': p.plan['state_policy'],
            'sequence_length': LENGTH, 'source_sha256': parent.SOURCE_SHA,
            'exposure': p.plan['exposure'], 'schedule_sha256': p.plan['schedule_sha256']},
        recovery_probe={'generic_resume_supported': False, 'comparison_report_required': True,
            'arm': arm, 'parent_training_episodes': p.plan['source_ancestry'],
            'held_out_corrections': p.parent.plan['held_out_corrections']})
    return cfg


def execute(p, budget):
    device = torch.device(p.args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    guard(p, budget)
    source = load_checkpoint(p.parent.source, device='cpu')
    if source['config'] != p.parent.cfg or any(source.get(k, 0) != v for k, v in p.parent.plan['source_progress'].items()):
        raise ValueError('Source payload/sidecar differ')
    output = p.args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {'plan': p.plan, 'status': 'running', 'comparison_complete': False, 'arms': {},
              'source_baseline': p.reference['before']}
    def write():
        report['elapsed_seconds'] = round(budget.clock() - budget.started, 3)
        temp = output / 'report.tmp.json'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(output / 'report.json')
    write()
    try:
        for arm in ARMS:
            budget.check()
            model = build_model(arm, source['model_state_dict'], p.parent.cfg, device)
            progress = {'status': 'initial_evaluation', 'steps_completed': 0,
                'initial_encoder_sha256': temporal.tensor_hash(model.trunk.state_dict()),
                'initial_head_sha256': temporal.tensor_hash({k: v for k, v in model.state_dict().items() if 'head' in k})}
            if arm != 'feedforward':
                progress['initial_memory_sha256'] = temporal.tensor_hash(model.memory.state_dict())
            if progress['initial_encoder_sha256'] != temporal.tensor_hash({k[6:]: v for k, v in source['model_state_dict'].items() if k.startswith('trunk.')}):
                raise ValueError('Initial encoder differs from source')
            for prior in report['arms'].values():
                if prior['initial_head_sha256'] != progress['initial_head_sha256']:
                    raise ValueError('Initial heads differ')
                if arm == 'gru_reset' and prior.get('initial_memory_sha256') != progress['initial_memory_sha256'] and 'initial_memory_sha256' in prior:
                    raise ValueError('Initial memory differs')
            report['arms'][arm] = progress
            progress['before'] = evaluate(model, arm, p, device, budget, initial=True)
            write()
            torch.manual_seed(0)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(0)
            progress['status'] = 'training'
            optimizer, metrics = train(model, arm, p, device, budget, progress)
            progress['training'] = metrics
            if progress['steps_completed'] != p.plan['exposure']['optimizer_updates_per_arm'] or metrics['supervised_frames'] != p.plan['exposure']['supervised_frames'] or metrics['target_trace_sha256'] != p.plan['target_trace_sha256']:
                raise ValueError('Incomplete or unequal ordered targets')
            progress['after'] = evaluate(model, arm, p, device, budget)
            progress['navigation_screen'] = parent.validation_screen(p.reference['before']['navigation'], progress['after']['navigation'],
                {'binary_exact_max_drop': .005, 'mouse_mae_max_ratio': 1.05, 'first8_idle_max_additional_fraction': .01})
            progress['final_encoder_sha256'] = temporal.tensor_hash(model.trunk.state_dict())
            folder = output / arm
            folder.mkdir()
            if arm == 'gru_reset':
                torch.save({'diagnostic_only': True, 'reason': 'Reset every timestep; not a live policy',
                    'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'config': endpoint_config(p, arm)}, folder / 'diagnostic_state.pt')
            else:
                save_checkpoint(folder / 'pending.pt', model=model, optimizer=optimizer, epoch=0,
                    global_step=progress['steps_completed'], best_val_loss=progress['after']['navigation']['loss'],
                    config=endpoint_config(p, arm), training_state={'comparison_report_required': True, 'probe_status': 'complete_fixed_endpoint'})
            progress['status'] = 'evaluated_pending_comparison'
            del model, optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            write()
        for key in ('initial_encoder_sha256', 'initial_head_sha256'):
            if len({v[key] for v in report['arms'].values()}) != 1:
                raise ValueError('Initial encoder/heads differ between arms')
        if report['arms']['gru']['initial_memory_sha256'] != report['arms']['gru_reset']['initial_memory_sha256']:
            raise ValueError('GRU initialization differs')
        if len({v['training']['target_trace_sha256'] for v in report['arms'].values()}) != 1:
            raise ValueError('Realized target traces differ')
        report['development_screen'] = development_screen(report)
        guard(p, budget)
        hashes = {arm: parent.file_hash(output / arm / ('diagnostic_state.pt' if arm == 'gru_reset' else 'pending.pt')) for arm in ARMS}
        budget.check()
        for arm in ('feedforward', 'gru'):
            for ext in ('pt', 'json'):
                (output / arm / f'pending.{ext}').replace(output / arm / f'endpoint.{ext}')
        budget.check()
        for arm in ARMS:
            report['arms'][arm].update(status='complete', checkpoint_sha256=hashes[arm])
        report.update(status='complete_no_automatic_promotion', comparison_complete=True)
        write()
    except BaseException as error:
        for arm in ('feedforward', 'gru'):
            for ext in ('pt', 'json'):
                endpoint = output / arm / f'endpoint.{ext}'
                if endpoint.exists():
                    endpoint.replace(output / arm / f'pending.{ext}')
        report.update(status='incomplete', comparison_complete=False, error=f'{type(error).__name__}: {error}')
        write()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--target-frames', type=int, default=16000)
    parser.add_argument('--time-budget-seconds', type=float, default=1440)
    args = parser.parse_args(argv)
    if not 1 <= args.target_frames <= 16000 or not 0 < args.time_budget_seconds <= 1440:
        parser.error('Maximum16000 requested target frames/arm and1440 seconds; no automatic extension')
    torch.set_num_threads(4)
    budget = parent.base.Budget(args.time_budget_seconds)
    print('Preparing ordered experiment inputs; no optimizer starts without --execute.', file=sys.stderr, flush=True)
    p = prepare(args, budget)
    if args.execute:
        execute(p, budget)
    else:
        print(json.dumps(p.plan, indent=2))


if __name__ == '__main__':
    main()
