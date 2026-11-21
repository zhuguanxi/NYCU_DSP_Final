"""Verify trained A/B weights with the original CPU and dense GPU forwards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_kedgn_ab as runtime


def main(root,device):
    frozen=json.loads((root/'frozen.json').read_text(encoding='utf-8'))
    rows,data=runtime.records(root/'inputs/ab.npz')
    selected=list(np.linspace(0,len(rows)-1,7,dtype=int))
    empty=np.flatnonzero(data['lengths']==0)
    selected.append(int(empty[0]) if len(empty) else 0)
    selected=np.asarray(selected)
    embeddings=torch.load(runtime.AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location='cpu')
    gpu_embeddings=embeddings.to(device)
    errors=[]
    for seed in range(5):
        folder=root/f'final_seed_{seed}'
        path=folder/'weights.pt'
        if runtime.sha(path)!=frozen['seeds'][str(seed)]['weights_sha256']:
            raise ValueError('Frozen trained weight changed')
        with np.load(folder/'statistics.npz') as source:
            stats=tuple(source[k] for k in ('mf','sf','ms','ss'))
        tensors=runtime.tensorize(rows[selected],data['y'][selected],stats)
        original=runtime.model_for(torch.device('cpu'),seed,dense=False)
        optimized=runtime.model_for(device,seed,dense=True)
        state=torch.load(path,map_location='cpu')
        original.load_state_dict(state)
        optimized.load_state_dict(state)
        original.eval(); optimized.eval()
        with torch.no_grad():
            expected=runtime.forward(original,tensors,slice(None),embeddings,torch.device('cpu'))
            actual=runtime.forward(optimized,tensors,slice(None),gpu_embeddings,device).cpu()
        torch.testing.assert_close(expected,actual,rtol=2e-4,atol=2e-5)
        errors.append(dict(seed=seed,max_abs_logit_difference=float((expected-actual).abs().max())))
        print(f'Trained frozen seed {seed}: original CPU and optimized GPU agree.',flush=True)
        del original,optimized,state,tensors
    runtime.write_json(root/'frozen_runtime_verification.json',dict(seeds=errors,samples=8,
                       data='A/B probe only, including zero-observation case',model_fits=0,
                       frozen_sha256=runtime.sha(root/'frozen.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=runtime.ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    main(args.output,torch.device(args.device))
