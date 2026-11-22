"""Fixed P33 CNN training/inference; use idle CUDA, otherwise CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import numpy as np
import torch
from train_hourly_sequence import HourlyCNN,moments,digest

_last_gpu_scan=-1e9
_last_gpu_busy=True
def busy():
    global _last_gpu_scan,_last_gpu_busy
    now=time.monotonic()
    if now-_last_gpu_scan<3: return _last_gpu_busy
    result=_scan_gpu()
    _last_gpu_scan=now;_last_gpu_busy=result
    return result

def _scan_gpu():
    try:
        value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=8)
        desktop={'explorer.exe','crossdeviceresume.exe','msedgewebview2.exe','predatorsense.exe','vgtray.exe',
                 'brave.exe','chatgpt.exe','dwm.exe','shellexperiencehost.exe','startmenuexperiencehost.exe'}
        own=False
        for line in value.splitlines():
            parts=line.split(',',1)
            if len(parts)!=2 or not parts[0].strip().isdigit(): continue
            if int(parts[0])==os.getpid(): own=True;continue
            if Path(parts[1].strip()).name.lower() not in desktop: return True
        if not own:
            load=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=8)
            utilization,memory=[float(v.strip()) for v in load.splitlines()[0].split(',')]
            if utilization>=15 or memory>=1024: return True
        return False
    except (OSError,subprocess.SubprocessError): return True

def device():
    return torch.device('cuda:0' if torch.cuda.is_available() and not busy() else 'cpu')

def predict(model,cube,static,where):
    model=model.to(where).eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(cube[i:i+128]).to(where),
                    torch.from_numpy(static[i:i+128]).to(where))).cpu().numpy() for i in range(0,len(cube),128)])

def train(path,seed):
    folder=path.parent
    with np.load(ML/'artifacts/p33_hourly_sequence/inputs/ab.npz') as a:
        cube,static,y=a['cube'],a['static'],a['y'].astype(np.float32)
    where=device(); fallback=False
    started=time.monotonic()
    while True:
        torch.manual_seed(seed)
        model=HourlyCNN(*moments(cube.transpose(0,2,1).reshape(-1,20)),*moments(static)).to(where)
        opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.005)
        dyn,st,label=[torch.from_numpy(a).to(where) for a in (cube,static,y)]
        restart=False; losses=[]
        for epoch in range(8):
            if where.type=='cuda' and busy():
                restart=True; break
            model.train(); total=0.
            for batch in torch.randperm(len(y)).split(128):
                batch=batch.to(where)
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(dyn[batch],st[batch]),label[batch])
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); opt.step(); total+=float(loss.detach().cpu())*len(batch)
            losses.append(total/len(y))
        if not restart: break
        del model,opt,dyn,st,label
        torch.cuda.empty_cache();where=torch.device('cpu');fallback=True
    model=model.cpu().eval()
    loaded=torch.jit.script(model);loaded.save(str(path))
    np.testing.assert_array_equal(predict(loaded,cube[:32],static[:32],torch.device('cpu')),
                                  predict(model,cube[:32],static[:32],torch.device('cpu')))
    receipt=dict(seed=seed,epochs=8,policy='plain',training_stays=len(y),deaths=int(y.sum()),
                 device=str(where),cpu_fallback=fallback,torch=torch.__version__,seconds=time.monotonic()-started,
                 training_loss=losses,weights_sha256=digest(path),input_sha256=digest(ML/'artifacts/p33_hourly_sequence/inputs/ab.npz'))
    (folder/'cnn_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(f'P33 seed {seed}: 8 fixed epochs on {where}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','infer','infer-list']);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--packet',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if a.mode=='fit': train(a.model,a.seed)
    elif a.mode=='infer':
        where=device();model=torch.jit.load(str(a.model),map_location='cpu')
        with np.load(a.packet) as data:
            values=predict(model,data['cube'],data['static'],where)
        if not (np.isfinite(values)&(values>=0)&(values<=1)).all(): raise ValueError('Invalid CNN probabilities')
        np.save(a.output,values)
    else:
        model=torch.jit.load(str(a.model),map_location='cpu')
        for job in json.loads(a.packet.read_text(encoding='utf-8')):
            target=Path(job['output'])
            if target.exists(): continue
            # Keep CNN inference consistent with the five completed CPU fits;
            # the slower KEDGN comparator may use the verified idle GPU.
            where=torch.device('cpu')
            with np.load(job['input']) as data: values=predict(model,data['cube'],data['static'],where)
            assert (np.isfinite(values)&(values>=0)&(values<=1)).all()
            np.save(target,values)
            print(f'CNN {a.model.parent.name}/{target.parent.name}: {where}',flush=True)

if __name__=='__main__': main()
