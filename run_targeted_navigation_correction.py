"""Plan-only by default: matched bounded base rehearsal versus human corrections.

No game APIs. No automatic training, checkpoint promotion, retry or resume.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import run_recovery_probe as base
from run_recovery_fit_probe import file_hash, balanced_draws
from run_recovery_history_probe import cache_fingerprint
from run_turn_onset_comparison import score_windows
from run_stationary_replay import validation_screen
from models.imitation.dataset import BehaviorCloningDataset, discover_cached_episodes
from models.imitation.frozen_split import load_frozen_split
from models.imitation.train_bc import partition_frozen_training
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.preprocess import ACTION_COLUMNS, preprocess_bgr_frame, read_actions

PROTOCOL = Path('data/annotations/targeted_navigation_correction_experiment_v1.json')
OUTPUT = Path('models/imitation/checkpoints/runs/targeted_navigation_correction_v1')
SOURCE_SHA = '2ac674293279370b3fd37aadae7914abc2ede84f2941298ded24c054701c2483'


class HumanDataset(BehaviorCloningDataset):
    """Decode reviewed human-only episodes in RAM, preserving original history."""
    def __init__(self, records, budget):
        self.episodes, self.index, self.episode_names, self.episode_maps = [], [], [], []
        self.first_action_frames, self.frame_stack, self.leading_idle_frames = [], 4, 0
        for e, rec in enumerate(records):
            budget.check()
            path = Path(rec['path'])
            for name, sha in rec['hashes'].items():
                if file_hash(path / name) != sha:
                    raise ValueError('Raw correction hash changed')
            metadata = json.loads((path / 'metadata.json').read_text())
            if metadata.get('source') != 'model_recovery' or metadata.get('map') != 'dataset_test1' or metadata.get('source_attempt') != rec['source_attempt']:
                raise ValueError('Correction ownership/map/source mismatch')
            log = [json.loads(line) for line in Path(rec['source_attempt']).read_text().splitlines()]
            if not any(r.get('type') == 'stop' and r.get('reason') == 'recovery_handoff' for r in log):
                raise ValueError('Missing model handoff')
            rows = read_actions(path / 'actions.csv')
            actions = np.array([[float(r[k]) for k in ACTION_COLUMNS] for r in rows], dtype=np.float32)
            if not np.isfinite(actions).all() or not np.isin(actions[:, :8], [0, 1]).all():
                raise ValueError('Invalid human actions')
            if any(int(r['frame_idx']) != i for i,r in enumerate(rows)):
                raise ValueError('Action frame indexing differs')
            capture = cv2.VideoCapture(str(path / 'video.mp4'))
            frames = []
            try:
                while True:
                    budget.check()
                    ok, frame = capture.read()
                    if not ok: break
                    frames.append(preprocess_bgr_frame(frame))
            finally:
                capture.release()
            if len(frames) != len(rows):
                raise ValueError('Correction video/action length mismatch')
            first = self._first_action_frame(actions)
            if rec['usable_window'] != [first, len(rows)] or first == len(rows):
                raise ValueError('Reviewed usable window differs')
            self.episodes.append((np.stack(frames), actions))
            self.episode_names.append(rec['episode']); self.episode_maps.append(rec['map'])
            self.first_action_frames.append(first); self.leading_idle_frames += first
            self.index.extend((e,i) for i in range(first,len(rows)))


def validate_review(protocol):
    if file_hash(protocol['correction_review_manifest']) != protocol['review_manifest_sha256']:
        raise ValueError('Review manifest changed')
    review = json.loads(Path(protocol['correction_review_manifest']).read_text())
    records = review['episodes']; ids = [r['episode'] for r in records]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError('Requires six unique reviewed episodes')
    if any(r['status'] != 'approved_for_bounded_pilot_with_documented_limits' or r['action_owner'] != 'human' or r['map'] != 'dataset_test1' for r in records):
        raise ValueError('Unapproved/nonhuman correction')
    train = [r for r in records if r['role'] == 'train']
    held = [r for r in records if r['role'] == 'development_holdout']
    if len(train) != 4 or len(held) != 2 or {r['episode'] for r in train} != set(protocol['train_corrections']) or {r['episode'] for r in held} != set(protocol['held_out_corrections']):
        raise ValueError('Correction split differs')
    if {r['view_group'] for r in train} & {r['view_group'] for r in held}:
        raise ValueError('Starting-view groups cross split')
    return train, held


class MatchedBatches(Dataset):
    def __init__(self, regular, corrections, shared, extra, correction_draws, arm):
        self.regular, self.corrections = regular, corrections
        self.shared, self.extra, self.correction_draws, self.arm = shared, extra, correction_draws, arm
        if len(shared) % 8 or len(extra)*4 != len(shared) or len(extra) != len(correction_draws):
            raise ValueError('Invalid fixed 8+2 draw schedule')
    def __len__(self): return len(self.extra)*5
    def __getitem__(self, i):
        batch, slot = divmod(i,10)
        if slot < 8: return self.regular[int(self.shared[batch*8+slot])]
        j = batch*2+slot-8
        return (self.corrections[int(self.correction_draws[j])] if self.arm == 'correction'
                else self.regular[int(self.extra[j])])


def digest_draws(p):
    return {k: hashlib.sha256(getattr(p,k).tobytes()).hexdigest() for k in ('shared','extra','correction_draws')}


def prepare(args, budget):
    protocol = json.loads(PROTOCOL.read_text())
    train_records, held_records = validate_review(protocol)
    source = Path(protocol['source'])
    if protocol['source_sha256'] != SOURCE_SHA or file_hash(source) != SOURCE_SHA:
        raise ValueError('Pinned deployed source differs')
    sidecar = json.loads(source.with_suffix('.json').read_text()); cfg = sidecar['config']
    base.validate_source(cfg)
    if cfg.get('memory') is not None:
        raise ValueError('Feedforward source required')
    split_path = Path('data/splits/frozen_v3.json'); split = load_frozen_split(split_path)
    if cfg['training'].get('split_sha256') != split.digest:
        raise ValueError('Source frozen split differs')
    manifests = discover_cached_episodes(args.base_cache); split.verify_cache(manifests,'base')
    train, val = partition_frozen_training(manifests,split)
    if len(train) != 1100 or len(val) != 200 or not {m.name for m in train} <= set(cfg['training']['training_episodes']):
        raise ValueError('Original data membership differs')
    if set(protocol['train_corrections']+protocol['held_out_corrections']) & set(cfg['training']['training_episodes']):
        raise ValueError('Source already exposed to new corrections')
    regular, navigation = BehaviorCloningDataset(train), BehaviorCloningDataset(val)
    corrections, held = HumanDataset(train_records,budget), HumanDataset(held_records,budget)
    if len(corrections) != 550 or len(held) != 272:
        raise ValueError('Reviewed frame counts differ')
    weights = torch.from_numpy(regular.chamber_sampling_weights())
    def sample(n,seed):
        return np.array(list(WeightedRandomSampler(weights,n,replacement=True,generator=torch.Generator().manual_seed(seed))),dtype=np.int64)
    p = SimpleNamespace(args=args, cfg=cfg, source=source, regular=regular, navigation=navigation,
        corrections=corrections, held=held, shared=sample(args.steps*8,args.seed),
        extra=sample(args.steps*2,args.seed+1), correction_draws=balanced_draws(len(corrections),args.steps*2,args.seed+2))
    if args.output.exists(): raise FileExistsError('Output exists; no overwrite/retry')
    protected = [args.base_cache.resolve(),source.parent.resolve()] + [Path(r['path']).resolve() for r in train_records+held_records]
    if any(args.output.resolve().is_relative_to(path) for path in protected):
        raise ValueError('Output overlaps protected input directory')
    paths = [PROTOCOL,Path(protocol['correction_review_manifest']),source,source.with_suffix('.json'),split_path]
    for r in train_records+held_records:
        paths.extend(Path(r['path'])/n for n in r['hashes'])
        paths.append(Path(r['source_attempt']))
    paths.extend(Path(n) for n in ('run_targeted_navigation_correction.py','run_recovery_probe.py','run_recovery_fit_probe.py','run_recovery_history_probe.py','run_turn_onset_comparison.py','run_stationary_replay.py','models/imitation/checkpoint.py','models/imitation/dataset.py','models/imitation/network.py','models/imitation/recurrent.py','models/imitation/train_bc.py','models/imitation/preprocess.py','models/imitation/mouse_bins.py','models/imitation/frozen_split.py','models/imitation/recovery_transfer.py'))
    p.plan = {'mode':'execute' if args.execute else 'plan_only_no_training', 'source':str(source),
        'source_progress':{k:sidecar.get(k,0) for k in ('epoch','global_step','step_in_epoch')},
        'input_sha256':{str(path):file_hash(path) for path in paths},'base_cache':str(args.base_cache),
        'base_cache_fingerprint':cache_fingerprint({'base':str(args.base_cache)},2**63-1),
        'correction_draws_per_episode':{name:sum(1 for i in p.correction_draws if corrections.episode_names[corrections.index[int(i)][0]]==name) for name in corrections.episode_names},
        'draw_sha256':digest_draws(p),'base_train_episodes':[m.name for m in train],
        'train_corrections':protocol['train_corrections'],'held_out_corrections':protocol['held_out_corrections'],
        'navigation_validation_episodes':[m.name for m in val], 'correction_frames':550,'held_frames':272,
        'steps_per_arm':args.steps,'batch_size':10,'learning_rate':1e-5,'seed':args.seed,
        'arms':{'control':'10 base frames','correction':'8 shared base +2 human TRAIN correction frames'},
        'draws':'Private fixed generators; same eight base draws per update in both arms; no static-history replacement',
        'precision':'float32_no_amp_skips','fresh_optimizer':True,
        'augmentation':'Source augmentation applied identically by slot; all labels remain human/base expert actions',
        'evaluation':'Full39769-frame navigation and all550TRAIN/272development-holdout correction frames before/after',
        'soft_budget_seconds':args.time_budget_seconds,'estimated_minutes':[12,18],
        'hard_cap':'20min external systemd cap','automatic_promotion_or_retry':False,
        'libraries':{'torch':str(torch.__version__),'numpy':np.__version__,'opencv':cv2.__version__}}
    budget.check()
    return p


def guard(p,budget):
    for path,sha in p.plan['input_sha256'].items():
        if file_hash(path) != sha: raise ValueError('Pinned input/source/code changed')
        budget.check()
    if cache_fingerprint({'base':str(p.args.base_cache)},2**63-1) != p.plan['base_cache_fingerprint']:
        raise ValueError('Base cache changed')
    if digest_draws(p) != p.plan['draw_sha256']: raise ValueError('Draws changed')
    budget.check()


def evaluate(model,p,device,budget):
    result={'navigation':base.evaluate(model,p.navigation,device,p.cfg['target_processing'],budget,32,p.cfg['training']['label_smoothing'])}
    for name,ds in [('train_corrections',p.corrections),('development_holdout',p.held)]:
        windows={n:[ds.first_action_frames[e],len(ds.episodes[e][0])] for e,n in enumerate(ds.episode_names)}
        result[name]=score_windows(model,ds,windows,p.cfg,device,budget)
    return result


def execute(p,budget):
    args=p.args; device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    guard(p,budget)
    source=load_checkpoint(p.source,device='cpu')
    if source['config'] != p.cfg or any(source.get(k,0)!=v for k,v in p.plan['source_progress'].items()):
        raise ValueError('Source payload/sidecar differ')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'plan':p.plan,'status':'running','paired_complete':False,'arms':{}}
    def write():
        report['elapsed_seconds']=round(budget.clock()-budget.started,3)
        tmp=args.output/'report.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'report.json')
    write()
    try:
        model=base.fresh_model(source['model_state_dict'],p.cfg,device,args.seed)
        report['before']=evaluate(model,p,device,budget);del model;write()
        for arm in ('control','correction'):
            progress={'arm':arm,'steps_completed':0,'status':'training'};report['arms'][arm]=progress;write()
            model=base.fresh_model(source['model_state_dict'],p.cfg,device,args.seed)
            # Reset execution RNG after identical source construction; evaluation
            # and data sampling never perturb one arm's training stream.
            torch.manual_seed(args.seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
            settings=p.cfg['training']
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=settings['weight_decay'])
            ds=MatchedBatches(p.regular,p.corrections,p.shared,p.extra,p.correction_draws,arm)
            loader=DataLoader(ds,batch_size=10,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
            progress['training']=base.train_probe_steps(model,loader,optimizer,device,p.cfg,budget,progress)
            if progress['steps_completed']!=args.steps:raise RuntimeError('Incomplete training')
            progress['evaluation']=evaluate(model,p,device,budget)
            progress['navigation_screen']=validation_screen(report['before']['navigation'],progress['evaluation']['navigation'],
                {'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01})
            cfg=copy.deepcopy(p.cfg)
            cfg['training'].update(learning_rate=1e-5,batch_size=10,seed=args.seed,sampling='targeted-navigation-matched',
                correction_sampling_fraction=.2 if arm=='correction' else 0.,correction_group_fractions={},correction_episode_groups={},
                training_episodes=p.plan['base_train_episodes']+(p.plan['train_corrections'] if arm=='correction' else []),
                split_id='targeted-navigation-correction-v1',split_sha256=p.plan['input_sha256'][str(PROTOCOL)],
                recovery_probe={'generic_resume_supported':False,'fresh_optimizer':True,'arm':arm,
                    'source_sha256':SOURCE_SHA,'paired_report_required':True,'parent_training_episodes':p.cfg['training']['training_episodes'],
                    'held_out_corrections':p.plan['held_out_corrections']})
            folder=args.output/arm
            save_checkpoint(folder/'pending.pt',model=model,optimizer=optimizer,epoch=0,global_step=args.steps,
                best_val_loss=progress['evaluation']['navigation']['loss'],config=cfg,
                training_state={'probe_status':'complete_fixed_endpoint','paired_report_required':True})
            progress['status']='evaluated_pending_pair';del model,optimizer;write();budget.check()
        guard(p,budget)
        hashes={arm:file_hash(args.output/arm/'pending.pt') for arm in report['arms']}
        budget.check()
        for arm in report['arms']:
            for ext in ('pt','json'):(args.output/arm/f'pending.{ext}').replace(args.output/arm/f'endpoint.{ext}')
            report['arms'][arm].update(status='complete',checkpoint_sha256=hashes[arm])
        budget.check();report.update(status='complete_no_automatic_promotion',paired_complete=True);write()
    except BaseException as error:
        report.update(status='incomplete',error=f'{type(error).__name__}: {error}',paired_complete=False);write();raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--time-budget-seconds',type=float,default=1080)
    parser.add_argument('--base-cache',type=Path,default=Path('data/datasets/cached_frames'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args(argv)
    if not 1<=args.steps<=1000 or not 0<args.time_budget_seconds<=1080:parser.error('Maximum1000 steps/arm and1080 seconds')
    budget=base.Budget(args.time_budget_seconds);p=prepare(args,budget)
    if args.execute:execute(p,budget)
    else:print(json.dumps(p.plan,indent=2))

if __name__=='__main__':main()
