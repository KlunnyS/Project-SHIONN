"""Fixed-budget turn rehearsal against a hash-verified completed control."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import run_reviewed_recovery_pair as pair
from run_recovery_fit_probe import balanced_draws, file_hash
from models.imitation.checkpoint import load_checkpoint, save_checkpoint

REFERENCE=pair.OUTPUT/'report.json'
OUTPUT=Path('models/imitation/checkpoints/runs/reviewed_turn_rehearsal_20261112')


class TurnRehearsal(pair.PairedRehearsal):
    def __init__(self,parent,indices,draws):
        super().__init__(parent,'correction')
        self.indices,self.draws=indices,draws
        if len(draws)*10!=len(self):raise ValueError('Requires one turn target per batch')
    def __getitem__(self,i):
        batch,slot=divmod(i,10)
        if slot==9:return self.corrections[self.indices[int(self.draws[batch])]]
        return super().__getitem__(i)


def verify_reference(reference,plan):
    if reference.get('status')!='complete_no_automatic_promotion' or not reference.get('paired_complete'):
        raise ValueError('Reference pair incomplete')
    if {k:v for k,v in reference['plan'].items() if k!='mode'}!={k:v for k,v in plan.items() if k!='mode'}:
        raise ValueError('Reference source/data/draws/settings differ')
    arm=reference['arms']['correction']
    if arm['status']!='complete' or arm['steps_completed']!=1000 or not arm['evaluation']['navigation']['complete']:
        raise ValueError('Reference control incomplete')


def prepare(args,budget):
    parent_args=SimpleNamespace(execute=False,steps=1000,seed=0,time_budget_seconds=1620,
        base_cache=Path('data/datasets/cached_frames'),output=args.output,device=args.device)
    parent=pair.prepare(parent_args,budget)
    reference=json.loads(REFERENCE.read_text());verify_reference(reference,parent.plan)
    checkpoint=REFERENCE.parent/'correction/endpoint.pt'
    if file_hash(checkpoint)!=reference['arms']['correction']['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    if args.output.resolve().is_relative_to(REFERENCE.parent.resolve()):raise ValueError('Output overlaps reference')
    protocol=json.loads(pair.PROTOCOL.read_text())
    names={Path(p).name for p in protocol['new_training_episodes']}
    indices=[i for i,(e,f) in enumerate(parent.corrections.index)
             if parent.corrections.episode_names[e] in names and f-parent.corrections.first_action_frames[e]<8]
    if len(indices)!=40:raise ValueError('Expected40 reviewed turn targets')
    draws=balanced_draws(40,1000,5)
    selected=[{'episode':parent.corrections.episode_names[parent.corrections.index[i][0]],
               'frame':parent.corrections.index[i][1]} for i in indices]
    for i in indices:
        e,f=parent.corrections.index[i]
        if abs(parent.corrections.episodes[e][1][f,8])<3:raise ValueError('Selected target is not a horizontal turn')
    files=[Path(__file__),REFERENCE,checkpoint,checkpoint.with_suffix('.json'),
           Path('docs/reviews/recovery_turn_fit_probe_20261112/report.json')]
    diagnostic=json.loads(files[-1].read_text())
    if not diagnostic.get('complete') or diagnostic.get('weights_exported') is not False:
        raise ValueError('Fitting diagnostic incomplete')
    plan={'mode':'execute' if args.execute else 'plan_only','purpose':'Concentrated turn sampling at unchanged total correction fraction and update budget',
          'parent_plan':parent.plan,'reference_report':str(REFERENCE),'reference_arm':'correction',
          'input_sha256':{str(p):file_hash(p) for p in files},'turn_examples':selected,
          'draw_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
          'source':str(parent.source),'steps':1000,'seed':0,'batch_size':10,'learning_rate':1e-5,
          'target_counts':{'original_base':7000,'original_openings':1000,'uniform_human':1000,'new_first8_human':1000},
          'changed_factor':'Replace slot9 uniform human with a reviewed first8-frame turn target; all other slots/draws/settings identical',
          'augmentation':'Original source settings, unlike the fitting diagnostic',
          'evaluation':'Full navigation and unchanged complete correction/development datasets; TRAIN fit is descriptive only',
          'soft_budget_seconds':840,'external_hard_cap':'15min','automatic_promotion_or_retry':False}
    budget.check()
    return SimpleNamespace(args=args,parent=parent,indices=indices,draws=draws,plan=plan,reference=reference)


def guard(p,budget):
    pair.guard(p.parent,budget)
    for path,digest in p.plan['input_sha256'].items():
        if file_hash(Path(path))!=digest:raise ValueError('Reference/diagnostic/code changed')
        budget.check()
    if hashlib.sha256(p.draws.tobytes()).hexdigest()!=p.plan['draw_sha256']:raise ValueError('Turn draws changed')
    selected=[{'episode':p.parent.corrections.episode_names[p.parent.corrections.index[i][0]],
               'frame':p.parent.corrections.index[i][1]} for i in p.indices]
    if selected!=p.plan['turn_examples']:raise ValueError('Turn membership changed')
    budget.check()


def execute(p,budget):
    args=p.args;parent=p.parent;device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    guard(p,budget);source=load_checkpoint(parent.source,device='cpu')
    if source['config']!=parent.cfg or any(source.get(k,0)!=v for k,v in parent.plan['source_progress'].items()):raise ValueError('Source payload differs')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'plan':p.plan,'status':'running','comparison_complete':False,'steps_completed':0,
            'source_baseline':p.reference['before'],'reused_control':p.reference['arms']['correction']}
    def write():
        report['elapsed_seconds']=budget.clock()-budget.started
        tmp=args.output/'report.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'report.json')
    write()
    try:
        model=pair.base.fresh_model(source['model_state_dict'],parent.cfg,device,0)
        torch.manual_seed(0)
        if device.type=='cuda':torch.cuda.manual_seed_all(0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=parent.cfg['training']['weight_decay'])
        loader=DataLoader(TurnRehearsal(parent,p.indices,p.draws),batch_size=10,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0))
        report['arm']='turn_rehearsal'
        report['training']=pair.base.train_probe_steps(model,loader,optimizer,device,parent.cfg,budget,report)
        if report['steps_completed']!=1000:raise RuntimeError('Incomplete training')
        report['evaluation']=pair.evaluate(model,parent,device,budget)
        report['navigation_screen']=pair.validation_screen(p.reference['before']['navigation'],report['evaluation']['navigation'],
            {'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01})
        cfg=copy.deepcopy(parent.cfg)
        cfg['training'].update(learning_rate=1e-5,batch_size=10,seed=0,sampling='reviewed-turn-rehearsal',
            training_episodes=parent.plan['base_train_episodes']+parent.plan['train_corrections'],
            correction_sampling_fraction=.2,correction_group_fractions={},correction_episode_groups={},
            opening_rehearsal_fraction=.1,opening_rehearsal_window=8,turn_rehearsal_fraction=.1,turn_rehearsal_window=8,
            split_id='reviewed-turn-rehearsal-20261112',split_sha256=parent.plan['input_sha256'][str(pair.PROTOCOL)],
            recovery_probe={'generic_resume_supported':False,'fresh_optimizer':True,'source_sha256':pair.SOURCE_SHA,
                'comparison_report_required':True,'reference_report':str(REFERENCE),'held_out_corrections':parent.plan['held_out_corrections'],
                'parent_training_episodes':parent.cfg['training']['training_episodes']})
        save_checkpoint(args.output/'pending.pt',model=model,optimizer=optimizer,epoch=0,global_step=1000,
            best_val_loss=report['evaluation']['navigation']['loss'],config=cfg,
            training_state={'probe_status':'complete_fixed_endpoint','comparison_report_required':True})
        guard(p,budget);digest=file_hash(args.output/'pending.pt');budget.check()
        for ext in ('pt','json'):(args.output/f'pending.{ext}').replace(args.output/f'endpoint.{ext}')
        budget.check();report.update(status='complete_no_automatic_promotion',comparison_complete=True,checkpoint_sha256=digest);write()
    except BaseException as error:
        for ext in ('pt','json'):
            endpoint=args.output/f'endpoint.{ext}'
            if endpoint.exists():endpoint.replace(args.output/f'pending.{ext}')
        report.update(status='incomplete',comparison_complete=False,error=f'{type(error).__name__}: {error}');write();raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args(argv)
    budget=pair.base.Budget(840);p=prepare(args,budget)
    if args.execute:execute(p,budget)
    else:print(json.dumps(p.plan,indent=2))

if __name__=='__main__':main()
