"""Read-only TRAIN-calibrated backward threshold diagnostic; does not edit policies."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from torch.utils.data import WeightedRandomSampler
from models.imitation.dataset import BehaviorCloningDataset,discover_cached_episodes
from models.imitation.frozen_split import load_frozen_split
from models.imitation.train_bc import partition_frozen_training
from models.imitation.inference import PolicyInference
from run_targeted_navigation_correction import HumanDataset
from run_recovery_probe import Budget


def main():
    budget=Budget(240)
    split=load_frozen_split(Path('data/splits/frozen_v3.json'));train,_=partition_frozen_training(discover_cached_episodes(Path('data/datasets/cached_frames')),split);base=BehaviorCloningDataset(train)
    indices=list(WeightedRandomSampler(torch.from_numpy(base.chamber_sampling_weights()),10000,replacement=True,generator=torch.Generator().manual_seed(8105)))
    review=json.load(open('data/annotations/targeted_navigation_pilot_20261105.json'))
    new=HumanDataset([r for r in review['episodes'] if r['role']=='train'],budget)
    held=HumanDataset([r for r in review['episodes'] if r['role']=='development_holdout'],budget)
    old=BehaviorCloningDataset(discover_cached_episodes(Path('data/datasets/recovery_cached_frames')))
    backward=[i for i,(e,f) in enumerate(old.index) if old.episodes[e][1][f,2]==1]
    starts=[]
    for e,(_,a) in enumerate(old.episodes):
     lo=old.first_action_frames[e];s=np.flatnonzero(a[lo:lo+48,2])
     if len(s):starts.append(next(i for i,p in enumerate(old.index) if p==(e,lo+int(s[0]))))
    sets={'base_calibration':(base,indices),'new_train':(new,list(range(len(new)))),'old_backward_train':(old,backward),'old_first_backward_train':(old,starts),'held':(held,list(range(len(held))))}
    output={'scope':'Global backward threshold feasibility; no weights/decoder changed. Selection uses TRAIN calibration only; held reviewed diagnostics not selection.','sample_seed':8105,'sampled_base_frames':10000,'models':{}}
    for name,path in [('source','models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt'),('rehearsal','models/imitation/checkpoints/runs/backward_rehearsal_v1/endpoint.pt')]:
     p=PolicyInference(Path(path),device='cuda');result={}
     for group,(ds,ix) in sets.items():
      scores=[];targets=[]
      for off in range(0,len(ix),32):
       budget.check();items=[ds[i] for i in ix[off:off+32]]
       inp=torch.from_numpy(np.stack([a for a,b in items])).to('cuda',dtype=torch.float32).div_(255)
       with torch.inference_mode():out=p.policy(inp)
       scores.extend(torch.softmax(out['move_s'],1)[:,1].cpu().tolist());targets.extend(int(t[2]) for a,t in items)
      result[group]={'scores':scores,'targets':targets}
     def stats(group,threshold):
      z=result[group];s=np.array(z['scores']);y=np.array(z['targets'],bool);pred=s>threshold
      return {'tp':int((pred&y).sum()),'positive':int(y.sum()),'fp':int((pred&~y).sum()),'negative':int((~y).sum())}
     result['at_default']={g:stats(g,.5) for g in sets}
     output['models'][name]=result
     print(name,result['at_default'],flush=True)
    # Equal false-positive budget from unchanged source on TRAIN-only base calibration.
    limit=output['models']['source']['at_default']['base_calibration']['fp']
    for name,result in output['models'].items():
     def stat(g,thr):
      s=np.array(result[g]['scores']);y=np.array(result[g]['targets'],bool);p=s>thr
      return {'tp':int((p&y).sum()),'positive':int(y.sum()),'fp':int((p&~y).sum()),'negative':int((~y).sum())}
     thresholds=sorted(set([.5,1.]+result['base_calibration']['scores']))
     viable=[t for t in thresholds if stat('base_calibration',t)['fp']<=limit]
     # Most permissive valid threshold maximizes all positive recalls, independent
     # of held-out targets; tie convention uses strict greater-than probabilities.
     threshold=min(viable)
     result['calibrated']={'threshold':threshold,'base_false_positive_limit':limit,'metrics':{g:stat(g,threshold) for g in sets}}
     print(name,'calibrated',result['calibrated'],flush=True)
    Path('/tmp/shionn-backward-threshold-audit.json').write_text(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
