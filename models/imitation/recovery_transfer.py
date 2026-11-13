"""Experiment-specific recovery split enforcement and offline imitation metrics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .dataset import BehaviorCloningDataset, discover_cached_episodes
from .frozen_split import load_frozen_split

ROLES = {'train': 'train_recovery_episodes', 'recovery_validation': 'validation_recovery_episodes',
         'diagnostic_only': 'diagnostic_only_episodes'}


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if p.get('version') != 1 or p.get('experiment_id') != 'recovery-transfer-development-v1':
        raise ValueError('Unsupported recovery experiment')
    if file_hash(p['parent_frozen_split']) != p['parent_frozen_split_sha256']:
        raise ValueError('Parent split changed')
    seen = set()
    for role, field in ROLES.items():
        names = p[field]
        expected = {'train': 22, 'recovery_validation': 9, 'diagnostic_only': 6}[role]
        if len(names) != expected or len(set(names)) != len(names) or seen & set(names):
            raise ValueError('Recovery partitions must be unique, disjoint and complete')
        seen.update(names)
    records = p['records']
    if len(p['reserved_recovery_maps']) != 3 or set(p['reserved_recovery_maps']) != {'dataset_test2','dataset_test5','dataset_test8'}:
        raise ValueError('Reserved map grouping differs from version1')
    if len(records) != 37 or {r['episode'] for r in records} != seen:
        raise ValueError('Records differ from declared partitions')
    if len({r['episode'] for r in records}) != 37 or p['window_frames'] != 48:
        raise ValueError('Duplicate records or changed window length')
    for r in records:
        if r['role'] not in ROLES or r['episode'] not in p[ROLES[r['role']]]:
            raise ValueError('Record role differs from partition')
        if r['group'] != ('assessment' if r['role'] == 'diagnostic_only' else 'recovery'):
            raise ValueError('Record cache group differs')
        if r['group'] == 'recovery' and ((r['map'] in p['reserved_recovery_maps']) != (r['role'] == 'recovery_validation')):
            raise ValueError('Reserved recovery map crosses partitions')
        start, end = r['offline_window']
        if not r['offline_window_end_exclusive'] or start != r['first_active_frame'] or end-start != 48 or start < 0 or end > r['frame_count']:
            raise ValueError('Invalid recovery window bounds')
    return p


def validate_training_membership(protocol, config):
    """May be called for any future warm start, before optimizer construction."""
    training = set(config['training']['training_episodes'])
    forbidden = set(protocol['validation_recovery_episodes']) | set(protocol['diagnostic_only_episodes'])
    overlap = training & forbidden
    if overlap:
        raise ValueError(f'Warm start already trained on reserved episodes: {sorted(overlap)}')


def exposure(protocol, config):
    names = config.get('training',{}).get('training_episodes')
    if names is None:
        return {'status':'unknown','recovery_validation_exposed':None,'meaning':'Missing training membership'}
    training = set(names)
    result = {role: sorted(training & set(protocol[field])) for role, field in ROLES.items()}
    return {'overlap_by_role': result, 'status':'recorded_overlap' if result['recovery_validation'] else 'no_recorded_overlap',
            'recovery_validation_exposed': bool(result['recovery_validation']),
            'meaning': 'Recorded episode membership only; full ancestry not established; not proof of unseen pixels or layouts'}


def load_context(protocol, base_cache, recovery_cache, assessment_cache):
    split = load_frozen_split(Path(protocol['parent_frozen_split']))
    base = discover_cached_episodes(Path(base_cache));split.verify_cache(base, 'base')
    recovery = discover_cached_episodes(Path(recovery_cache));split.verify_cache(recovery, 'recovery')
    diagnostic = discover_cached_episodes(Path(assessment_cache))
    if {m.name for m in recovery} != set(protocol['train_recovery_episodes'] + protocol['validation_recovery_episodes']):
        raise ValueError('Recovery cache membership changed')
    if len(diagnostic) != 6 or {m.name for m in diagnostic} != set(protocol['diagnostic_only_episodes']):
        raise ValueError('Diagnostic cache membership changed')
    if set(protocol['diagnostic_only_episodes']) & {name for _, name in split.episodes}:
        raise ValueError('Diagnostics overlap frozen data')
    if any(split.role(m,'recovery') != 'train' for m in recovery):
        raise ValueError('Parent recovery role must be TRAIN')
    datasets = {'recovery': BehaviorCloningDataset(recovery), 'assessment': BehaviorCloningDataset(diagnostic)}
    for record in protocol['records']:
        ds = datasets[record['group']];e = ds.episode_names.index(record['episode'])
        if (ds.episode_maps[e] != record['map'] or len(ds.episodes[e][0]) != record['frame_count']
                or ds.first_action_frames[e] != record['first_active_frame']):
            raise ValueError('Cached episode differs from protocol')
        m = (recovery if record['group']=='recovery' else diagnostic)[e]
        if json.loads(m.frames_path.with_suffix('.json').read_text()).get('source_fps') != 24:
            raise ValueError('Recovery scoring requires recorded 24 FPS')
    base_train = [m for m in base if split.role(m,'base') == 'train']
    validation = [m for m in base if split.role(m,'base') == 'validation']
    return {'datasets':datasets, 'base_train':base_train, 'navigation_validation':validation,
            'recovery_train':[m for m in recovery if m.name in protocol['train_recovery_episodes']],
            'recovery_validation':[m for m in recovery if m.name in protocol['validation_recovery_episodes']]}


def training_partitions(protocol, context, source=None):
    """Return only allowed manifests, requiring an eligible source before training."""
    source = Path(source or protocol['source'])
    config = json.loads(source.with_suffix('.json').read_text())['config']
    validate_training_membership(protocol, config)
    if source.resolve() != Path(protocol['source']).resolve() or file_hash(source) != protocol['source_sha256'] or file_hash(source.with_suffix('.json')) != protocol['source_sidecar_sha256']:
        raise ValueError('This experiment requires its recorded clean initial source')
    if set(config['training']['training_episodes']) != {m.name for m in context['base_train']}:
        raise ValueError('Source base training membership differs')
    if set(config['training']['validation_episodes']) != {m.name for m in context['navigation_validation']}:
        raise ValueError('Source navigation validation membership differs')
    if config['architecture'] != protocol['source_architecture'] or len(context['base_train']) != protocol['source_training_episode_count']:
        raise ValueError('Source architecture/count differs')
    if {m.name for m in context['recovery_train']} != set(protocol['train_recovery_episodes']) or {m.name for m in context['recovery_validation']} != set(protocol['validation_recovery_episodes']):
        raise ValueError('Training loader crossed the declared recovery partition')
    return context['base_train'] + context['recovery_train'], context['recovery_validation']


def episode_metrics(expected, predicted, fps=24):
    e, p = np.asarray(expected), np.asarray(predicted)
    if e.shape != p.shape or e.ndim != 2 or e.shape[1] != 10 or not len(e) or not np.isfinite(e).all() or not np.isfinite(p).all() or fps <= 0:
        raise ValueError('Need complete finite aligned action arrays')
    idle = np.all(p == 0, axis=1);longest=run=0
    for value in idle:
        run = run+1 if value else 0;longest=max(longest,run)
    result = {'frames':len(e), 'idle_fraction':float(idle.mean()), 'longest_idle_seconds':longest/fps,
              'binary_exact':float(np.all(e[:,:8]==p[:,:8],axis=1).mean()),
              'mouse_mae':np.abs(e[:,8:]-p[:,8:]).mean(axis=0).tolist(),
              'backward_tp':int(((e[:,2]!=0)&(p[:,2]!=0)).sum()),
              'backward_expected':int((e[:,2]!=0).sum()),
              'false_backward':int(((e[:,2]==0)&(p[:,2]!=0)).sum())}
    for axis, i in [('dx',8),('dy',9)]:
        significant=np.abs(e[:,i])>=3;nonzero=e[:,i]!=0
        result[axis]={'significant_frames':int(significant.sum()),
                      'sign_correct':int((np.sign(e[significant,i])==np.sign(p[significant,i])).sum()),
                      'sign_accuracy':float((np.sign(e[significant,i])==np.sign(p[significant,i])).mean()) if significant.any() else None,
                      'expert_nonzero_frames':int(nonzero.sum()),
                      'nonzero_predicted_zero':int((nonzero & (p[:,i]==0)).sum())}
    return result


def aggregate(records):
    """Equal-episode means plus denominator-weighted counts, without success claims."""
    if not records:return {'episodes':0}
    result={'episodes':len(records),'frames':sum(x['frames'] for x in records),
            'episode_mean':{k:float(np.mean([r[k] for r in records])) for k in ['idle_fraction','longest_idle_seconds','binary_exact']},
            'episode_mouse_mae':np.mean([r['mouse_mae'] for r in records],axis=0).tolist(),
            'frame_weighted':{k:float(np.average([r[k] for r in records],weights=[r['frames'] for r in records])) for k in ['idle_fraction','binary_exact']}}
    tp=sum(r['backward_tp'] for r in records);den=sum(r['backward_expected'] for r in records)
    result.update(backward_tp=tp,backward_expected=den,backward_recall=tp/den if den else None,
                  false_backward=sum(r['false_backward'] for r in records))
    supported=[r['backward_tp']/r['backward_expected'] for r in records if r['backward_expected']]
    result['episode_backward_recall']=float(np.mean(supported)) if supported else None
    result['backward_supported_episodes']=len(supported)
    result['frame_weighted']['mouse_mae']=np.average([r['mouse_mae'] for r in records],axis=0,weights=[r['frames'] for r in records]).tolist()
    for axis in ['dx','dy']:
        support=sum(r[axis]['significant_frames'] for r in records);correct=sum(r[axis]['sign_correct'] for r in records)
        nonzero=sum(r[axis]['expert_nonzero_frames'] for r in records);zeros=sum(r[axis]['nonzero_predicted_zero'] for r in records)
        supported=[r[axis]['sign_accuracy'] for r in records if r[axis]['sign_accuracy'] is not None]
        result[axis]={'significant_frames':support,'sign_correct':correct,'frame_sign_accuracy':correct/support if support else None,
                      'episode_sign_accuracy':float(np.mean(supported)) if supported else None,
                      'supported_episodes':len(supported),'expert_nonzero_frames':nonzero,
                      'nonzero_predicted_zero':zeros,'nonzero_zero_rate':zeros/nonzero if nonzero else None}
    return result
