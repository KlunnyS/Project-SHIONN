"""Read-only route-expansion opening exposure and TRAIN calibration analysis.

Default: reproduce completed-run draw coverage. --inference additionally reads
three checkpoints on CUDA; no optimizer, game APIs or checkpoint writes.
"""
import argparse

def coverage():
    import json,hashlib,collections
    from pathlib import Path
    import numpy as np
    import torch
    from torch.utils.data import WeightedRandomSampler
    from models.imitation.dataset import BehaviorCloningDataset,discover_cached_episodes
    from models.imitation.frozen_split import load_frozen_split
    from models.imitation.train_bc import partition_frozen_training
    r=json.loads(Path('models/imitation/checkpoints/runs/route_expansion_v1/report.json').read_text());assert r.get('paired_complete') and all(v.get('steps_completed')==1000 for v in r['arms'].values());plan=r['plan'];split=load_frozen_split(Path('data/splits/frozen_v3.json'));man=discover_cached_episodes(Path(plan['base_cache']));split.verify_cache(man,'base');train,val=partition_frozen_training(man,split);ds=BehaviorCloningDataset(train)
    assert ds.episode_names==plan['base_train_episodes']
    w=torch.from_numpy(ds.chamber_sampling_weights());draw=np.array(list(WeightedRandomSampler(w,8000,replacement=True,generator=torch.Generator().manual_seed(plan['seed']))),dtype=np.int64)
    assert hashlib.sha256(draw.tobytes()).hexdigest()==plan['draw_sha256']['shared']
    counts=collections.Counter();windows={}
    for width in [1,8,24]:
     selected=[int(i) for i in draw if ds.index[int(i)][1]-ds.first_action_frames[ds.index[int(i)][0]]<width]
     ep=collections.Counter(ds.episode_names[ds.index[i][0]] for i in selected)
     windows[str(width)]={'draws':len(selected),'unique_frames':len(set(selected)),'episodes_covered':len(ep),'episodes_not_covered':len(train)-len(ep),'by_map':dict(collections.Counter(ds.episode_maps[ds.index[i][0]] for i in selected))}
    cal=[]
    for mapname in sorted({m.map_name for m in train}):
     group=sorted((m for m in train if m.map_name==mapname),key=lambda m:hashlib.sha256(('opening-train-calibration-v1:'+m.name).encode()).hexdigest())[:10]
     cal.extend(m.name for m in group)
    out={'report':'models/imitation/checkpoints/runs/route_expansion_v1/report.json','shared_draw_hash_verified':True,'base_frames':len(ds),'base_episodes':len(train),'base_draws_per_arm':8000,'opening_windows_frames':windows,'calibration_selection':'Fixed SHA256 ranking by episode name per TRAIN map; ten whole episodes/map, before inspecting their predictions. First eight active frames each.','calibration_train_episodes':cal,'calibration_frames':len(cal)*8,'validation_episodes_in_calibration':sorted(set(cal)&{m.name for m in val}),'scope':'Exposure audit only. No gradients; zero sampled starts is not proof of causal forgetting.'}
    Path('data/annotations/route_expansion_train_opening_coverage_20261109.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='calibration_train_episodes'},indent=2))

def inference():
    import json,hashlib
    from pathlib import Path
    import numpy as np
    import torch
    from models.imitation.dataset import BehaviorCloningDataset,discover_cached_episodes,BINARY_ACTION_COLUMNS
    from models.imitation.checkpoint import load_checkpoint
    from models.imitation.mouse_bins import MouseBins
    import run_recovery_probe as base
    coverage=json.loads(Path('data/annotations/route_expansion_train_opening_coverage_20261109.json').read_text());ids=set(coverage['calibration_train_episodes']);ds=BehaviorCloningDataset([m for m in discover_cached_episodes(Path('data/datasets/cached_frames')) if m.name in ids]);idx=[i for i,(e,f) in enumerate(ds.index) if f-ds.first_action_frames[e]<8];assert len(idx)==800
    paths={'source':'models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt',**{a:f'models/imitation/checkpoints/runs/route_expansion_v1/{a}/endpoint.pt' for a in ['control','correction']}};out={}
    torch.set_num_threads(4);device=torch.device('cuda');assert torch.cuda.is_available()
    reference=json.loads(Path('models/imitation/checkpoints/runs/route_expansion_v1/report.json').read_text())
    for name,path in paths.items():
        expected_sha=reference['plan']['input_sha256'][path] if name=='source' else reference['arms'][name]['checkpoint_sha256']
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_sha
        ck=load_checkpoint(Path(path),device='cpu');cfg=ck['config'];model=base.fresh_model(ck['model_state_dict'],cfg,device,0).eval();bins=MouseBins(cfg['target_processing']['mouse_bins']);records=[]
        for start in range(0,len(idx),32):
            indices=idx[start:start+32];items=[ds[i] for i in indices];x=torch.from_numpy(np.stack([a for a,b in items])).to(device,dtype=torch.float32).div_(255).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():pred=model(x)
            binary=np.stack([pred[k].argmax(1).cpu().numpy() for k in BINARY_ACTION_COLUMNS],1);mouse=np.stack([np.rint(np.asarray(bins.representatives(axis))[pred[k].argmax(1).cpu().numpy()]) for axis,k in enumerate(['mouse_dx_logits','mouse_dy_logits'])],1);actions=np.concatenate([binary,mouse],1);probs=pred['move_w'].softmax(1)[:,1].cpu().numpy()
            for i,(_,y),a,prob in zip(indices,items,actions,probs):
                ep,f=ds.index[i];records.append({'episode':ds.episode_names[ep],'age':f-ds.first_action_frames[ep],'expected':y.tolist(),'predicted':a.tolist(),'w_probability':float(prob)})
        out[name]={'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'frames':len(records),'predicted_idle':sum(not any(r['predicted']) for r in records),'false_idle':sum(not any(r['predicted']) and any(r['expected']) for r in records),'w_false_negative':sum(r['expected'][0]==1 and r['predicted'][0]==0 for r in records),'records':records}
        print(name,{k:v for k,v in out[name].items() if k!='records'},flush=True);del model,ck;torch.cuda.empty_cache()
    result={'scope':'Read-only fixed TRAIN calibration; not validation or gameplay and not gradients.','coverage_manifest':'data/annotations/route_expansion_train_opening_coverage_20261109.json','arms':out};Path('data/annotations/route_expansion_train_opening_predictions_20261109.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', action='store_true')
    args = parser.parse_args()
    coverage()
    if args.inference:
        inference()
