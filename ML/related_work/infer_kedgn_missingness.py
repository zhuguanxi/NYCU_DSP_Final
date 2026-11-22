"""Frozen GPU KEDGN predictions under preregistered full-channel masks."""

import json
from pathlib import Path

import numpy as np
import torch

import run_kedgn_ab as runtime

ROOT=runtime.ML/'artifacts/p55_kedgn_missingness'
AB=runtime.ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121'
CHANNELS=('GCS','Lactate','MAP','Mg','Urine','MechVent')
CASES=(('clean',()),*((f'full_{x}',(x,)) for x in CHANNELS),('full_all_six',CHANNELS))


def main():
    device=torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('Requested GPU unavailable')
    ROOT.mkdir(exist_ok=True)
    author=json.loads((runtime.P53/'inference.json').read_text())
    final=json.loads((AB/'frozen.json').read_text())
    inputs=json.loads((runtime.P53/'inputs.json').read_text())
    columns=inputs['dynamic_channels']
    rows,unused=runtime.records(runtime.P53/'set_c_portable.npz')
    del unused
    with np.load(runtime.P53/'set_c_labels.npz') as saved: ids,y=saved['ids'],saved['y']
    embeddings=torch.load(runtime.AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location=device)
    hashes={}
    sources={}
    for version in ('set_a','ab'):
        oldroot=runtime.P53 if version=='set_a' else AB
        with np.load(oldroot/'set_c_predictions.npz') as original: clean=original['probabilities'].copy()
        for name,channels in CASES:
            destination=ROOT/f'{version}_{name}.npz'
            if destination.exists():
                hashes[destination.name]=runtime.sha(destination)
                print(f'Retaining {version}/{name}',flush=True)
                continue
            selected=[columns.index(key) for key in channels]
            changed=np.asarray([bool(selected) and np.any(row['arr'][:row['length'],selected]>0) for row in rows])
            positions=np.flatnonzero(changed)
            if not channels: positions=np.arange(len(rows))
            positions=positions[np.argsort([rows[i]['length'] for i in positions],kind='stable')]
            values=clean.copy()
            for seed in range(5):
                if version=='set_a':
                    matches=[(Path(p),h) for p,h in author['checkpoints_sha256'].items() if Path(p).stem.endswith(f'_{seed}')]
                    weight,digest=matches[0]
                    with np.load(runtime.P53/'training_statistics.npz') as stats:
                        scaling=tuple(stats[k] for k in ('mf','stdf','ms','ss'))
                else:
                    folder=AB/f'final_seed_{seed}'
                    weight=folder/'weights.pt'
                    digest=final['seeds'][str(seed)]['weights_sha256']
                    with np.load(folder/'statistics.npz') as stats:
                        scaling=tuple(stats[k] for k in ('mf','sf','ms','ss'))
                if runtime.sha(weight)!=digest: raise ValueError('Frozen model changed')
                sources[str(weight)]=digest
                model=runtime.model_for(device,seed)
                model.load_state_dict(torch.load(weight,map_location=device))
                for start in range(0,len(positions),64):
                    idx=positions[start:start+64]
                    chunk=[]
                    for position in idx:
                        row=dict(rows[position])
                        if selected:
                            row['arr']=row['arr'].copy()
                            row['arr'][:,selected]=0
                        chunk.append(row)
                    tensors=runtime.tensorize(np.asarray(chunk,dtype=object),y[idx],scaling)
                    values[seed,idx]=runtime.predict(model,tensors,embeddings,device)
                del model
            if not channels:
                np.testing.assert_allclose(values,clean,atol=2e-6,rtol=2e-5)
            if not (np.isfinite(values)&(values>=0)&(values<=1)).all(): raise ValueError('Invalid probability')
            temporary=destination.with_suffix('.pending.npz')
            np.savez_compressed(temporary,ids=ids,y=y,probabilities=values)
            temporary.replace(destination)
            hashes[destination.name]=runtime.sha(destination)
            print(f'{version}/{name}: five frozen seeds complete; changed stays={len(positions)}',flush=True)
    for path,digest in sources.items():
        if runtime.sha(path)!=digest: raise ValueError('Weight changed during inference')
    runtime.write_json(ROOT/'kedgn_manifest.json',dict(predictions=hashes,model_sources_sha256=sources,
                       model_fits=0,device=str(device),masking='Zero channel values; preserve times and original lengths',
                       input_sha256=runtime.sha(runtime.P53/'set_c_portable.npz'),
                       inference_code_sha256=runtime.sha(Path(__file__))))


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    main()
