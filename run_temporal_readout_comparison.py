"""Plan a frozen-encoder FF/GRU/reset-GRU pilot; execution is explicit.

No gameplay, end-to-end encoder training, automatic retry or promotion.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
import run_targeted_navigation_correction as parent
from models.imitation.network import BinnedImitationPolicy, BINNED_ARCHITECTURE_VERSION
from models.imitation.recurrent import RecurrentBinnedPolicy, RECURRENT_ARCHITECTURE_VERSION, SequenceDataset
from models.imitation.dataset import BINARY_ACTION_COLUMNS
from models.imitation.mouse_bins import MouseBins
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.recovery_transfer import episode_metrics, aggregate

OUTPUT=Path('models/imitation/checkpoints/runs/temporal_readout_comparison_v1')
ARMS=('feedforward','gru','gru_reset')
LENGTH,BURN,BATCH=16,16,2


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(arm,state,cfg,seed):
    """All arms share pretrained encoder tensors and identical NEW action heads."""
    counts=MouseBins(cfg['target_processing']['mouse_bins']).class_counts()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100)
        heads=nn.ModuleDict({k:nn.Linear(512,2) for k in BINARY_ACTION_COLUMNS})
        dx,dy=nn.Linear(512,counts[0]),nn.Linear(512,counts[1])
        torch.manual_seed(seed+200)
        model=BinnedImitationPolicy(*counts) if arm=='feedforward' else RecurrentBinnedPolicy(*counts,hidden_size=512,feature_gate=False)
        if arm=='feedforward':model.load_state_dict(state)
        else:model.trunk.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith('trunk.')})
        model.binary_heads.load_state_dict(heads.state_dict())
        model.mouse_dx_head.load_state_dict(dx.state_dict());model.mouse_dy_head.load_state_dict(dy.state_dict())
    model.trunk.requires_grad_(False);model.trunk.eval()
    return model


def readout(model,features,arm,hidden=None,valid=None):
    """Features [B,T,512]; hidden [1,B,512]; masks exclude state changes."""
    b,t,_=features.shape
    if arm=='feedforward':z=features;hidden=None
    elif arm=='gru_reset':
        z,_=model.memory(features.reshape(b*t,1,-1))
        z=z.reshape(b,t,-1);hidden=None
    elif valid is None:
        z,hidden=model.memory(features,hidden)
    else:
        if hidden is None:hidden=features.new_zeros(1,b,model.hidden_size)
        output=[]
        for i in range(t):
            _,candidate=model.memory(features[:,i:i+1],hidden.contiguous())
            hidden=torch.where(valid[:,i][None,:,None],candidate,hidden)
            output.append(hidden[0])
        z=torch.stack(output,1)
    result={k:head(z) for k,head in model.binary_heads.items()}
    result['mouse_dx_logits']=model.mouse_dx_head(z);result['mouse_dy_logits']=model.mouse_dy_head(z)
    return result,hidden


def chunk_indices(seq,index):
    offset,first,start,end=seq.chunks[int(index)]
    return np.array([offset+f-first if first<=f<end else -1
        for f in range(start-seq.burn_in,start+seq.sequence_length)],dtype=np.int64)


def make_schedule(datasets,steps,seed):
    sequences={k:SequenceDataset(d,LENGTH,BURN) for k,d in datasets.items() if k in ('base','human_train')}
    result=[]
    generators={k:torch.Generator().manual_seed(seed+i+1) for i,k in enumerate(sequences)}
    for role,seq in sequences.items():
        counts={}
        maps=[]
        for offset,first,start,end in seq.chunks:
            e=seq.base.index[offset][0];m=seq.base.episode_maps[e];maps.append(m);counts[m]=counts.get(m,0)+1
        weights=torch.tensor([1/counts[m] for m in maps],dtype=torch.double)
        n=sum(('human_train' if i%5==4 else 'base')==role for i in range(steps*BATCH))
        draws=list(WeightedRandomSampler(weights,n,replacement=True,generator=generators[role])) if n else []
        generators[role]=iter(draws)
    for i in range(steps*BATCH):
        role='human_train' if i%5==4 else 'base'
        result.append((role,int(next(generators[role]))))
    return sequences,result


class FeatureBank:
    """CPU float32 features; extract only selected TRAIN context plus full eval."""
    def __init__(self,datasets):
        self.datasets=datasets
        self.features={k:np.empty((len(d),512),np.float32) for k,d in datasets.items()}
        self.ready={k:np.zeros(len(d),dtype=bool) for k,d in datasets.items()}
    def populate(self,encoder,requirements,device,budget):
        encoder.eval()
        with torch.inference_mode():
            for role,indices in requirements.items():
                indices=sorted(set(indices))
                print(f'Encoding {role}: {len(indices):,} frames...',flush=True)
                for off in range(0,len(indices),32):
                    budget.check();ix=indices[off:off+32]
                    x=torch.from_numpy(np.stack([self.datasets[role][i][0] for i in ix])).to(device,dtype=torch.float32).div_(255.)
                    if device.type=='cuda':x=x.contiguous(memory_format=torch.channels_last)
                    z=encoder(x).cpu().numpy()
                    if z.shape!=(len(ix),512) or not np.isfinite(z).all():raise ValueError('Invalid frozen features')
                    self.features[role][ix]=z;self.ready[role][ix]=True
    def fingerprint(self):
        h=hashlib.sha256()
        for role in sorted(self.features):
            h.update(role.encode());h.update(self.ready[role].tobytes())
            h.update(self.features[role][self.ready[role]].tobytes())
        return h.hexdigest()
    def chunk(self,role,seq,index):
        ix=chunk_indices(seq,index);valid=ix>=0
        if not self.ready[role][ix[valid]].all():raise ValueError('Missing planned features')
        features=np.zeros((len(ix),512),np.float32);targets=np.zeros((len(ix),10),np.float32)
        features[valid]=self.features[role][ix[valid]]
        ds=self.datasets[role]
        targets[valid]=np.stack([ds.episodes[ds.index[i][0]][1][ds.index[i][1]] for i in ix[valid]])
        return features,targets,valid


def exposure(sequences,schedule):
    counts={};episodes={};h=hashlib.sha256()
    for role,i in schedule:
        ix=chunk_indices(sequences[role],i);valid=ix>=0
        counts[role]=counts.get(role,0)+int(valid[BURN:].sum())
        e=sequences[role].base.index[int(ix[valid][0])][0];name=sequences[role].base.episode_names[e]
        episodes[name]=episodes.get(name,0)+int(valid[BURN:].sum())
        h.update(json.dumps([role,i,ix.tolist()]).encode())
    return {'schedule_sha256':h.hexdigest(),'supervised_frames_by_group':counts,'supervised_frames_by_episode':episodes,'sequences':len(schedule)}


def train(model,arm,p,bank,device,budget,progress):
    started=time.perf_counter()
    model.train();model.trunk.eval()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001,weight_decay=p.parent.cfg['training']['weight_decay'])
    weights=torch.tensor(p.parent.cfg['target_processing']['binary_class_weights'],dtype=torch.float32,device=device)
    bins=MouseBins(p.parent.cfg['target_processing']['mouse_bins']);total=count=0
    for off in range(0,len(p.schedule),BATCH):
        budget.check();items=[bank.chunk(role,p.sequences[role],i) for role,i in p.schedule[off:off+BATCH]]
        x=torch.from_numpy(np.stack([a for a,b,c in items])).to(device)
        y=torch.from_numpy(np.stack([b for a,b,c in items])).to(device)
        mask=torch.from_numpy(np.stack([c for a,b,c in items])).to(device)
        opt.zero_grad(set_to_none=True);hidden=None
        with torch.no_grad():
            if arm=='gru':_,hidden=readout(model,x[:,:BURN],arm,valid=mask[:,:BURN])
        if hidden is not None:hidden=hidden.detach()
        out,_=readout(model,x[:,BURN:],arm,hidden,mask[:,BURN:])
        valid=mask[:,BURN:];n=int(valid.sum())
        loss,_=parent.base.behavior_cloning_loss({k:v[valid] for k,v in out.items()},y[:,BURN:][valid],weights,
            mouse_bins=bins,label_smoothing=p.parent.cfg['training']['label_smoothing'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite readout loss')
        loss.backward();nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True);opt.step()
        progress['steps_completed']+=1;count+=n;total+=float(loss.detach())*n
        if progress['steps_completed']%100==0:print(arm,progress['steps_completed'],'updates',flush=True)
        budget.check()
    if any(v.grad is not None for v in model.trunk.parameters()):raise ValueError('Encoder received gradient')
    elapsed=time.perf_counter()-started
    return opt,{'loss':total/count,'supervised_frames':count,'precision':'float32_no_amp_skips',
        'elapsed_seconds':elapsed,'supervised_frames_per_second':count/elapsed,
        'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None}


def evaluate(model,arm,bank,device,budget):
    bins=MouseBins(model._target['mouse_bins']);result={};model.eval()
    for role in ('navigation','human_train','human_held'):
        ds=bank.datasets[role];records=[];loss_sum=0.;allmetric=parent.base.ActionMetrics();firstmetric=parent.base.ActionMetrics();idle=0;first_idle=0;total=0;first_total=0
        for e,name in enumerate(ds.episode_names):
            ix=[i for i,(ep,f) in enumerate(ds.index) if ep==e]
            if not ix:continue
            if not bank.ready[role][ix].all():raise ValueError('Incomplete evaluation features')
            predictions=[];hidden=None
            targets=np.stack([ds.episodes[e][1][ds.index[i][1]] for i in ix])
            with torch.inference_mode():
                for off in range(0,len(ix),128):
                    budget.check();x=torch.from_numpy(bank.features[role][ix[off:off+128]]).to(device)[None]
                    out,hidden=readout(model,x,arm,hidden)
                    if not all(torch.isfinite(v).all() for v in out.values()):raise FloatingPointError('Nonfinite evaluation logits')
                    y=torch.from_numpy(targets[off:off+128]).to(device)
                    loss,_=parent.base.behavior_cloning_loss({k:v[0] for k,v in out.items()},y,torch.tensor(model._target['binary_class_weights'],device=device),mouse_bins=bins,label_smoothing=getattr(model,'_smoothing',.01))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite evaluation loss')
                    loss_sum+=float(loss)*len(y)
                    binary=np.stack([out[k][0].argmax(1).cpu().numpy() for k in BINARY_ACTION_COLUMNS],1)
                    axes=np.stack([[bins.decode(axis,int(j)) for j in out[key][0].argmax(1).cpu()] for axis,key in enumerate(('mouse_dx_logits','mouse_dy_logits'))],1)
                    predictions.extend(np.column_stack([binary,axes]).tolist())
            pred=np.array(predictions)
            metrics=episode_metrics(targets,pred)
            r={'episode':name,'map':ds.episode_maps[e],'metrics':metrics}
            if role!='navigation':r.update(targets=targets.tolist(),predictions=predictions)
            records.append(r)
            for j,(a,b) in enumerate(zip(targets,pred)):
                allmetric.add(a,b);idle+=int(not b.any());total+=1
                if j<8:firstmetric.add(a,b);first_idle+=int(not b.any());first_total+=1
        result[role]={'complete':True,'loss':loss_sum/max(1,total),'frames':total,'episodes':len(records),'records':records,
            'aggregate':aggregate([r['metrics'] for r in records]),
            'segments':{'all':{**allmetric.report(),'predicted_idle_frames':idle},'first8':{**firstmetric.report(),'predicted_idle_frames':first_idle}}}
    budget.check();return result


def prepare(args,budget):
    if not 1<=args.steps<=500:raise ValueError('Maximum500updates per arm')
    parent_args=SimpleNamespace(execute=False,steps=1000,seed=0,time_budget_seconds=780,base_cache=Path('data/datasets/cached_frames'),output=args.output,device=args.device)
    pp=parent.prepare(parent_args,budget)
    datasets={'base':pp.regular,'navigation':pp.navigation,'human_train':pp.corrections,'human_held':pp.held}
    sequences,schedule=make_schedule(datasets,args.steps,0)
    requirements={k:set() for k in datasets}
    for role,i in schedule:requirements[role].update(j for j in chunk_indices(sequences[role],i) if j>=0)
    for role in ('navigation','human_train','human_held'):requirements[role].update(range(len(datasets[role])))
    extra={str(Path(__file__)):parent.file_hash(__file__)}
    plan={'mode':'execute' if args.execute else 'plan_only_no_training','purpose':'Frozen pretrained encoder, NEW matched action readouts; not end-to-end or gameplay comparison',
        'source':str(pp.source),'source_progress':pp.plan['source_progress'],'parent_input_sha256':pp.plan['input_sha256'],'extra_sha256':extra,
        'base_cache_fingerprint':pp.plan['base_cache_fingerprint'],'arms':list(ARMS),'encoder_frozen_eval':True,'old_action_heads_discarded_in_all_arms':True,
        'optional_feature_gate':False,'hidden_size':512,'steps_per_arm':args.steps,'batch_sequences':BATCH,'sequence_length':LENGTH,'burn_in':BURN,
        'learning_rate':.001,'seed':0,'augmentation':'Disabled in all arms; common float32 frozen features','precision':'float32_no_amp_skips',
        'data_exposure':exposure(sequences,schedule),'feature_frames':{k:len(v) for k,v in requirements.items()},
        'libraries':pp.plan['libraries'],'torch_threads':4,
        'source_ancestry':pp.cfg['training']['training_episodes'],'train_episodes':pp.plan['base_train_episodes']+pp.plan['train_corrections'],
        'held_out_corrections':pp.plan['held_out_corrections'],'soft_budget_seconds':780,'hard_cap':'15min external','estimated_minutes':[5,12],
        'endpoint_policy':'Fixed endpoint, no selection or promotion. Reset-GRU is diagnostic-only and cannot be loaded as a policy checkpoint.'}
    budget.check();return SimpleNamespace(args=args,parent=pp,datasets=datasets,sequences=sequences,schedule=schedule,requirements=requirements,plan=plan)


def guard(p,budget):
    parent.guard(p.parent,budget)
    for path,sha in p.plan['extra_sha256'].items():
        if parent.file_hash(path)!=sha:raise ValueError('Comparison code changed')
    if exposure(p.sequences,p.schedule)!=p.plan['data_exposure']:raise ValueError('Sequence schedule changed')
    budget.check()


def execute(p,budget):
    device=torch.device(p.args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    guard(p,budget);source=load_checkpoint(p.parent.source,device='cpu')
    if source['config']!=p.parent.cfg or any(source.get(k,0)!=v for k,v in p.plan['source_progress'].items()):raise ValueError('Source payload differs')
    p.args.output.mkdir(parents=True,exist_ok=False)
    report={'plan':p.plan,'status':'running','comparison_complete':False,'arms':{}}
    def write():
        report['elapsed_seconds']=round(budget.clock()-budget.started,3)
        tmp=p.args.output/'report.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p.args.output/'report.json')
    write()
    try:
        encoder_model=parent.base.fresh_model(source['model_state_dict'],p.parent.cfg,device,0)
        encoder_sha=tensor_hash(encoder_model.trunk.state_dict());bank=FeatureBank(p.datasets)
        encoding_started=budget.clock()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        bank.populate(encoder_model.trunk,p.requirements,device,budget)
        report['feature_encoding_seconds']=budget.clock()-encoding_started
        report['feature_bank_sha256']=bank.fingerprint()
        report['feature_array_capacity_mib']=sum(a.nbytes for a in bank.features.values())/2**20
        report['encoding_peak_allocated_mib']=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None
        encoder_model._smoothing=p.parent.cfg['training']['label_smoothing']
        encoder_model._target=p.parent.cfg['target_processing']
        report['deployed_reference_only']=evaluate(encoder_model,'feedforward',bank,device,budget)
        del encoder_model;write()
        for arm in ARMS:
            budget.check();model=build_model(arm,source['model_state_dict'],p.parent.cfg,0).to(device);model._target=p.parent.cfg['target_processing'];model._smoothing=p.parent.cfg['training']['label_smoothing']
            if tensor_hash(model.trunk.state_dict())!=encoder_sha:raise ValueError('Initial encoder differs')
            heads={k:v for k,v in model.state_dict().items() if 'head' in k};headsha=tensor_hash(heads)
            progress={'steps_completed':0,'status':'initial_evaluation','initial_head_sha256':headsha,
                'parameters':sum(v.numel() for v in model.parameters()),'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad)};report['arms'][arm]=progress
            progress['before']=evaluate(model,arm,bank,device,budget);write()
            torch.manual_seed(0)
            if device.type=='cuda':torch.cuda.manual_seed_all(0)
            opt,metrics=train(model,arm,p,bank,device,budget,progress);progress['training']=metrics
            if metrics['supervised_frames']!=sum(p.plan['data_exposure']['supervised_frames_by_group'].values()):raise ValueError('Target exposure differs')
            if progress['steps_completed']!=p.args.steps:raise ValueError('Incomplete training')
            progress['after']=evaluate(model,arm,bank,device,budget)
            if tensor_hash(model.trunk.state_dict())!=encoder_sha:raise ValueError('Frozen encoder changed')
            progress['encoder_sha256']=encoder_sha
            cfg=copy.deepcopy(p.parent.cfg);cfg['memory']=None if arm=='feedforward' else {'type':'gru','hidden_size':512,'feature_gate':False,'sequence_length':LENGTH,'burn_in':BURN}
            cfg['architecture']=BINNED_ARCHITECTURE_VERSION if arm=='feedforward' else RECURRENT_ARCHITECTURE_VERSION
            cfg['training'].update(learning_rate=.001,batch_size=BATCH,sampling='matched-temporal-chunks',training_episodes=p.plan['train_episodes'],encoder_frozen=True,
                readout_comparison={'heads_reset':True,'sequence_length':LENGTH,'burn_in':BURN,'source_sha256':p.plan['parent_input_sha256'][str(p.parent.source)],'encoder_sha256':encoder_sha,'data_exposure':p.plan['data_exposure']},
                split_id='temporal-readout-comparison-v1',split_sha256=p.plan['parent_input_sha256'][str(parent.PROTOCOL)],
                augmentation={'brightness':0.,'contrast':0.,'horizontal_flip_probability':0.},
                correction_sampling_fraction=p.plan['data_exposure']['supervised_frames_by_group'].get('human_train',0)/sum(p.plan['data_exposure']['supervised_frames_by_group'].values()),correction_group_fractions={},correction_episode_groups={},
                recovery_probe={'generic_resume_supported':False,'comparison_report_required':True,'arm':arm,'heads_reset':True,'parent_training_episodes':p.plan['source_ancestry']})
            folder=p.args.output/arm;folder.mkdir()
            if arm=='gru_reset':
                torch.save({'diagnostic_only':True,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'reason':'Reset-every-step control; not an inference checkpoint'},folder/'diagnostic_state.pt')
            else:
                save_checkpoint(folder/'pending.pt',model=model,optimizer=opt,epoch=0,global_step=p.args.steps,best_val_loss=progress['after']['navigation']['loss'],config=cfg,
                    training_state={'probe_status':'complete_fixed_endpoint','comparison_report_required':True})
            progress['status']='evaluated_pending_comparison';del model,opt;write();budget.check()
        if len({a['initial_head_sha256'] for a in report['arms'].values()})!=1:raise ValueError('Initial action heads differ')
        guard(p,budget)
        if bank.fingerprint()!=report['feature_bank_sha256']:raise ValueError('Frozen feature bank changed')
        hashes={arm:parent.file_hash(p.args.output/arm/('diagnostic_state.pt' if arm=='gru_reset' else 'pending.pt')) for arm in ARMS}
        budget.check()
        for arm in ARMS:
            if arm!='gru_reset':
                for ext in ('pt','json'):(p.args.output/arm/f'pending.{ext}').replace(p.args.output/arm/f'endpoint.{ext}')
            report['arms'][arm].update(status='complete',artifact_sha256=hashes[arm])
        budget.check();report.update(status='complete_no_automatic_promotion',comparison_complete=True);write()
    except BaseException as error:
        report.update(status='incomplete',comparison_complete=False,error=f'{type(error).__name__}: {error}');write();raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv);torch.set_num_threads(4)
    budget=parent.base.Budget(780);p=prepare(args,budget)
    if args.execute:execute(p,budget)
    else:print(json.dumps(p.plan,indent=2))

if __name__=='__main__':main()
