"""Synthetic bounded model-only benchmark; no game, dataset or checkpoint writes."""
import argparse
import json
import time
import torch
from models.imitation.network import BinnedImitationPolicy
from models.imitation.recurrent import RecurrentBinnedPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--sequence-length',type=int,default=4)
    parser.add_argument('--hidden-size',type=int,default=256)
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--feature-gate',action='store_true')
    args=parser.parse_args()
    if min(args.sequence_length,args.hidden_size,args.iterations)<1:parser.error('Dimensions/iterations must be positive')
    torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    report={'scope':'Synthetic model-only, batch1, random normalized inputs, float32; no gameplay benefit measured','device':str(device),'torch':torch.__version__,'sequence_length':args.sequence_length,'results':{}}
    for name in ('baseline','gru'):
        torch.manual_seed(0)
        model=(BinnedImitationPolicy(15,15) if name=='baseline' else RecurrentBinnedPolicy(15,15,args.hidden_size,args.feature_gate)).to(device).eval()
        frame=torch.rand(1,12,180,320,device=device)
        with torch.inference_mode():
            state=None
            for _ in range(2):
                if name=='gru':_,state=model.step(frame,state)
                else:model(frame)
            sync();start=time.perf_counter()
            for _ in range(args.iterations):
                if name=='gru':_,state=model.step(frame,state)
                else:model(frame)
            sync();latency=(time.perf_counter()-start)*1000/args.iterations
        del state
        model.train();optimizer=torch.optim.AdamW(model.parameters())
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        sync();start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if name=='gru':out,_=model(frame[:,None].expand(-1,args.sequence_length,-1,-1,-1))
        else:out=model(frame.expand(args.sequence_length,-1,-1,-1))
        loss=sum(v.float().square().mean() for v in out.values());loss.backward();optimizer.step()
        sync();seconds=time.perf_counter()-start
        report['results'][name]={'parameters':sum(p.numel() for p in model.parameters()),'step_latency_ms':latency,'synthetic_training_frames_per_second':args.sequence_length/seconds,'one_update_seconds':seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else None}
        del model,optimizer,frame,out,loss
        if device.type=='cuda':torch.cuda.empty_cache()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
