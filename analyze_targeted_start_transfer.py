"""Read-only GPU diagnostic for the completed targeted recovery experiment."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from models.imitation.dataset import BehaviorCloningDataset,discover_cached_episodes
from models.imitation.inference import PolicyInference
from models.imitation.recovery_transfer import episode_metrics,aggregate
from models.imitation.mouse_bins import MouseBins
from run_targeted_navigation_correction import HumanDataset
from run_recovery_probe import Budget
from run_recovery_fit_probe import balanced_draws


def main():
    report_path=Path('models/imitation/checkpoints/runs/targeted_navigation_20261105_195411/report.json')
    prior=json.loads(report_path.read_text());review=json.load(open('data/annotations/targeted_navigation_pilot_20261105.json'))
    budget=Budget(180)
    sets={role:HumanDataset([r for r in review['episodes'] if r['role']==role],budget) for role in ('train','development_holdout')}
    old=BehaviorCloningDataset(discover_cached_episodes(Path('data/datasets/recovery_cached_frames')))
    sets['old_backward_source_exposed']=old
    result={'purpose':'Read-only initial48 recovery action coverage and normal/static history diagnostic; no optimization','prior_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'models':{},'coverage':{}}
    for role,ds in sets.items():
     records=[]
     for e,name in enumerate(ds.episode_names):
      start=ds.first_action_frames[e];targets=ds.episodes[e][1][start:start+48]
      if role.startswith('old') and not targets[:,2].any():continue
      records.append({'episode':name,'map':ds.episode_maps[e],'first_active':start,'backward_frames':int(targets[:,2].sum()),'left_frames':int((targets[:,8]<=-3).sum()),'right_frames':int((targets[:,8]>=3).sum())})
     result['coverage'][role]=records
    for name,path in [('source','models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt'),('control','models/imitation/checkpoints/runs/targeted_navigation_20261105_195411/control/endpoint.pt'),('correction','models/imitation/checkpoints/runs/targeted_navigation_20261105_195411/correction/endpoint.pt')]:
     policy=PolicyInference(Path(path),device='cuda');model=policy.policy;bins=policy.mouse_bins;records=[]
     for role,ds in sets.items():
      positions={pair:i for i,pair in enumerate(ds.index)}
      for c in result['coverage'][role]:
       e=ds.episode_names.index(c['episode']);start=c['first_active'];items=[ds[positions[e,i]] for i in range(start,min(start+48,len(ds.episodes[e][0])))]
       for history in ['normal','repeated_newest']:
        predictions=[]
        for off in range(0,len(items),8):
         budget.check();array=np.stack([v[0] for v in items[off:off+8]])
         if history=='repeated_newest':array=np.tile(array[:,-3:],(1,4,1,1))
         inp=torch.from_numpy(array).to('cuda',dtype=torch.float32).div_(255.)
         with torch.inference_mode():out=model(inp)
         binary=np.stack([out[k].argmax(1).cpu().numpy() for k in ['move_w','move_a','move_s','move_d','jump','use','fire_left','fire_right']],1)
         axes=np.stack([[bins.decode(a,int(i)) for i in out[k].argmax(1).cpu()] for a,k in enumerate(['mouse_dx_logits','mouse_dy_logits'])],1)
         predictions.extend(np.column_stack([binary,axes]).tolist())
        targets=np.array([v[1] for v in items]);pred=np.array(predictions)
        records.append({'episode':c['episode'],'role':role,'history':history,'metrics':episode_metrics(targets,pred),'targets':targets.tolist(),'predictions':pred.tolist()})
     result['models'][name]={'checkpoint':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'records':records}
     for role in sets:
      for hist in ['normal','repeated_newest']:
       a=aggregate([r['metrics'] for r in records if r['role']==role and r['history']==hist]);print(name,role,hist,'S',a['backward_tp'],'/',a['backward_expected'],'binary',round(a['episode_mean']['binary_exact']*100,1),'idle',round(a['episode_mean']['idle_fraction']*100,1),flush=True)
     del policy,model
    Path('/tmp/shionn-targeted-start-audit.json').write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
