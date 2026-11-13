"""Isolated 1000-update paired correction experiment; default is plan-only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_route_expansion as prior
from run_opening_rehearsal import opening_indices
from run_recovery_fit_probe import balanced_draws, file_hash
from models.imitation.checkpoint import load_checkpoint, save_checkpoint

base = prior.base
validation_screen = prior.validation_screen
evaluate = prior.evaluate
PROTOCOL = Path('docs/reviews/recovery_correction_plan_20261112.json')
SOURCE_SHA = prior.SOURCE_SHA
OUTPUT = Path('models/imitation/checkpoints/runs/reviewed_recovery_pair_20261112')


class PairedRehearsal(prior.MatchedBatches):
    def __init__(self, p, arm):
        super().__init__(p.regular, p.corrections, p.shared, p.extra,
                         p.correction_draws, arm, p.control, p.control_draws)
        self.openings, self.opening_draws = p.openings, p.opening_draws
    def __getitem__(self, i):
        batch, slot = divmod(i, 10)
        if slot == 7:
            return self.regular[self.openings[int(self.opening_draws[batch])]]
        return super().__getitem__(i)


def reviewed_records(protocol):
    if file_hash(Path(protocol['new_review'])) != protocol['new_review_sha256']:
        raise ValueError('New review changed')
    review = json.loads(Path(protocol['new_review']).read_text())
    selected = []
    for rec in review['episodes']:
        if rec['selection'] != 'approved_for_bounded_feedforward_training':
            continue
        if rec['action_owner'] != 'human' or not rec['context_audit']['passed']:
            raise ValueError('Unreviewed correction')
        selected.append({'path':rec['episode'], 'episode':Path(rec['episode']).name,
            'map':rec['map'], 'source_attempt':rec['context_audit']['model_log'],
            'hashes':rec['hashes'], 'usable_window':rec['usable_window']})
    cal = next(Path(p) for p in protocol['new_training_episodes'] if Path(p).name == 'episode_20261112_134935_742233')
    meta = json.loads((cal/'metadata.json').read_text())
    selected.append({'path':str(cal), 'episode':cal.name, 'map':meta['map'],
        'source_attempt':meta['source_attempt'], 'hashes':protocol['calibration_hashes'],
        'usable_window':protocol['calibration_window']})
    if len(selected)!=5 or {r['path'] for r in selected} != set(protocol['new_training_episodes']):
        raise ValueError('Reviewed membership differs')
    return selected


def prepare(args, budget):
    protocol = json.loads(PROTOCOL.read_text())
    if file_hash(Path(protocol['previous_split'])) != protocol['previous_split_sha256']:
        raise ValueError('Previous split changed')
    if protocol['steps_per_arm'] != 1000 or protocol['seed'] != 0 or protocol['source_sha256'] != SOURCE_SHA:
        raise ValueError('Bounded source/settings differ')
    p = prior.prepare(args, budget)
    parent = copy.copy(p); parent.plan = copy.deepcopy(p.plan)
    new_records = reviewed_records(protocol)
    old_review = json.loads(Path(protocol['previous_split']).read_text())
    old_records = [r for r in old_review['episodes'] if r['role']=='train']
    new_ds = prior.HumanDataset(new_records, budget)
    if len(new_ds)!=806: raise ValueError('New frame count differs')
    protected=[Path(r['path']).resolve() for r in new_records]
    if any(args.output.resolve().is_relative_to(path) for path in protected):
        raise ValueError('Output overlaps a correction')
    p.parent_guard = parent
    p.control = p.corrections
    p.corrections = prior.HumanDataset(old_records+new_records,budget)
    if len(p.control)!=801 or len(p.corrections)!=1607 or len(p.held)!=272:
        raise ValueError('Reviewed pool counts differ')
    if set(p.corrections.episode_names) & set(p.held.episode_names):
        raise ValueError('Held-out leakage')
    p.control_draws = balanced_draws(801,2000,2)
    p.correction_draws = balanced_draws(1607,2000,2)
    p.openings = opening_indices(p.regular)
    if len(p.openings)!=8800: raise ValueError('Opening pool differs')
    p.opening_draws = balanced_draws(8800,1000,4)
    paths=[PROTOCOL,Path(protocol['new_review']),Path(__file__),Path('run_opening_rehearsal.py')]
    for r in new_records:
        paths.extend(Path(r['path'])/name for name in r['hashes'])
        paths.append(Path(r['source_attempt']))
    p.plan = copy.deepcopy(parent.plan)
    p.plan['input_sha256'].update({str(path):file_hash(path) for path in paths})
    p.plan.update(purpose='Previous versus expanded recovery data with matched startup rehearsal',
        control_corrections=p.control.episode_names, train_corrections=p.corrections.episode_names,
        control_correction_frames=801,correction_frames=1607,
        arms={'control':'7base+1opening+2previous-human','correction':'7base+1opening+2expanded-human'},
        human_weighting='Frame-balanced within each usable pool',
        draw_sha256=prior.digest_draws(p), opening_draw_sha256=hashlib.sha256(p.opening_draws.tobytes()).hexdigest(),
        opening_pool=p.openings, steps_per_arm=1000,
        hard_cap='30min external systemd cap', soft_budget_seconds=1620,
        new_episodes={**parent.plan['new_episodes'],**{r['episode']:'reviewed_recovery_20261112' for r in new_records}},
        comparison_configuration=str(PROTOCOL),
        checkpoint_resume='Unsupported; fixed endpoints require completed paired report')
    # Replace stale summaries inherited from the previous experiment.
    p.plan['control_draws_per_episode'] = {name:sum(p.control.episode_names[p.control.index[int(i)][0]]==name for i in p.control_draws) for name in p.control.episode_names}
    p.plan['correction_draws_per_episode'] = {name:sum(p.corrections.episode_names[p.corrections.index[int(i)][0]]==name for i in p.correction_draws) for name in p.corrections.episode_names}
    p.plan['backward_targets'] = {label:sum(int(ds.episodes[ds.index[int(i)][0]][1][ds.index[int(i)][1],2]) for i in draws) for label,ds,draws in [('control',p.control,p.control_draws),('correction',p.corrections,p.correction_draws)]}
    p.plan['evaluation']='Full39769navigation; all1607TRAIN and272development correction targets before/after'
    p.plan['draws']='Same7 base and1 opening draws in each batch; only human pool differs'
    return p


def guard(p,budget):
    prior.guard(p.parent_guard,budget)
    for path,sha in p.plan['input_sha256'].items():
        if file_hash(Path(path)) != sha: raise ValueError('Pinned experiment input changed')
        budget.check()
    if prior.digest_draws(p)!=p.plan['draw_sha256']:
        raise ValueError('Paired draws changed')
    if hashlib.sha256(p.opening_draws.tobytes()).hexdigest()!=p.plan['opening_draw_sha256'] or p.openings!=p.plan['opening_pool']:
        raise ValueError('Opening draws changed')
    budget.check()


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
            ds=PairedRehearsal(p,arm)
            loader=DataLoader(ds,batch_size=10,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
            progress['training']=base.train_probe_steps(model,loader,optimizer,device,p.cfg,budget,progress)
            if progress['steps_completed']!=args.steps:raise RuntimeError('Incomplete training')
            progress['evaluation']=evaluate(model,p,device,budget)
            progress['navigation_screen']=validation_screen(report['before']['navigation'],progress['evaluation']['navigation'],
                {'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01})
            cfg=copy.deepcopy(p.cfg)
            cfg['training'].update(learning_rate=1e-5,batch_size=10,seed=args.seed,sampling='reviewed-recovery-paired-opening-rehearsal', opening_rehearsal_fraction=.1, opening_rehearsal_window=8,
                correction_sampling_fraction=.2,correction_group_fractions={},correction_episode_groups={},
                training_episodes=p.plan['base_train_episodes']+(p.plan['train_corrections'] if arm=='correction' else p.plan['control_corrections']),
                split_id='reviewed-recovery-pair-20261112',split_sha256=p.plan['input_sha256'][str(PROTOCOL)],
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
        # A failed final publication is never a completed pair. Restore any
        # partially renamed files so incomplete runs expose no endpoints.
        for arm in report['arms']:
            for ext in ('pt','json'):
                endpoint=args.output/arm/f'endpoint.{ext}'
                if endpoint.exists(): endpoint.replace(args.output/arm/f'pending.{ext}')
        report.update(status='incomplete',error=f'{type(error).__name__}: {error}',paired_complete=False);write();raise
    return report



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args(argv)
    args.steps=1000; args.seed=0; args.time_budget_seconds=1620
    args.base_cache=Path('data/datasets/cached_frames')
    budget=base.Budget(1620)
    p=prepare(args,budget)
    if args.execute: execute(p,budget)
    else: print(json.dumps(p.plan,indent=2))

if __name__=='__main__':main()
