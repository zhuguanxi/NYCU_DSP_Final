"""All five frozen A+B KEDGN members on identical hour masks; zero fits."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import run_kedgn_ab as runtime
from p60_seed_worker import device,busy

ROOT=runtime.ML/'artifacts/p61_progressive_block_missingness'
AB=runtime.ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121'

def main():
    manifest=json.loads((ROOT/'inputs.json').read_text())
    frozen=json.loads((AB/'frozen.json').read_text())
    inputs=json.loads((runtime.P53/'inputs.json').read_text())
    channels=inputs['dynamic_channels']
    rows,original=runtime.records(runtime.P53/'set_c_portable.npz')
    with np.load(runtime.P53/'set_c_labels.npz') as data: ids,y=data['ids'],data['y']
    with np.load(AB/'set_c_predictions.npz') as data: clean=data['probabilities'].copy()
    scaling=[];sources={}
    for seed in range(5):
        folder=AB/f'final_seed_{seed}';weight=folder/'weights.pt'
        assert runtime.sha(weight)==frozen['seeds'][str(seed)]['weights_sha256']
        sources[str(weight.relative_to(runtime.ML))]=runtime.sha(weight)
        with np.load(folder/'statistics.npz') as s: scaling.append(tuple(s[k] for k in ('mf','sf','ms','ss')))
    for stat in scaling[1:]:
        for a,b in zip(scaling[0],stat): np.testing.assert_array_equal(a,b)
    embeddings=torch.load(runtime.AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location='cpu')
    probe=runtime.tensorize(rows[:64],y[:64],scaling[0])
    for seed in range(5):
        where=device();model=runtime.model_for(where,seed)
        model.load_state_dict(torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=where))
        test=runtime.predict(model,probe,embeddings.to(where),where)
        np.testing.assert_allclose(test,clean[seed,:64],atol=2e-6,rtol=2e-5)
        del model
    receipts={}
    for case in manifest['cases']:
        folder=ROOT/case['name'];target=folder/'kedgn_predictions.npz'
        receipt=folder/'kedgn_receipt.json'
        if receipt.exists():
            meta=json.loads(receipt.read_text());assert runtime.sha(target)==meta['sha256']
            receipts[case['name']]=meta;continue
        tick=time.monotonic()
        with np.load(folder/'input.npz') as p:
            np.testing.assert_array_equal(p['ids'],ids);np.testing.assert_array_equal(p['y'],y);hours=p['hours']
        assert runtime.sha(folder/'input.npz')==case['files']['input.npz']
        if case['design'] in ('clean','full'):
            old={'clean':'clean','gcs_full':'full_GCS','six_full':'full_all_six'}[case['name']]
            source=runtime.ML/f'artifacts/p55_kedgn_missingness/ab_{old}.npz'
            with np.load(source) as olddata:
                np.testing.assert_array_equal(olddata['ids'],ids);values=olddata['probabilities'].copy()
            note=dict(reused_endpoint_sha256=runtime.sha(source),devices=['cached P55'])
        else:
            selected=[channels.index(c) for c in case['channels']]
            modified=[];positions=[]
            for i,row in enumerate(rows):
                hour=(row['time'][:row['length'],0]//60).astype(int).clip(0,47)
                mask=hours[i,hour]
                if not np.any(row['arr'][:row['length']][np.ix_(mask,selected)]>0): continue
                changed=dict(row);changed['arr']=row['arr'].copy()
                changed['arr'][np.ix_(np.flatnonzero(mask),selected)]=0
                positions.append(i);modified.append(changed)
            idx=np.asarray(positions,dtype=int)
            tensors=runtime.tensorize(np.asarray(modified,dtype=object),y[idx],scaling[0])
            values=clean.copy();devices=[]
            for seed in range(5):
                where=device();model=runtime.model_for(where,seed)
                model.load_state_dict(torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=where))
                # Tensorization is shared across seeds; training statistics are identical.
                if where.type=='cuda' and busy():
                    del model;torch.cuda.empty_cache();where=torch.device('cpu');model=runtime.model_for(where,seed)
                    model.load_state_dict(torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=where))
                for start in range(0,len(idx),256):
                    if where.type=='cuda' and busy():
                        del model;torch.cuda.empty_cache();where=torch.device('cpu');model=runtime.model_for(where,seed)
                        model.load_state_dict(torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=where))
                    chunk=tuple(t[start:start+256] for t in tensors)
                    with torch.no_grad(): values[seed,idx[start:start+256]]=runtime.predict(model,chunk,embeddings.to(where),where,batch=256)
                devices.append(str(where));del model
            note=dict(devices=devices,changed_stays=len(idx),tensorization_shared=True)
        assert (np.isfinite(values)&(values>=0)&(values<=1)).all()
        np.savez_compressed(target,ids=ids,y=y,probabilities=values)
        meta=dict(sha256=runtime.sha(target),input_sha256=case['files']['input.npz'],
                  seconds=time.monotonic()-tick,model_fits=0,**note)
        runtime.write_json(receipt,meta);receipts[case['name']]=meta
        print(f'KEDGN five members: {case["name"]} in {meta["seconds"]:.1f}s; {note["devices"]}',flush=True)
    assert all(runtime.sha(runtime.ML/file)==digest for file,digest in sources.items())
    runtime.write_json(ROOT/'kedgn_evaluation.json',dict(cases=receipts,model_sources_sha256=sources,
       model_fits=0,all_seeds_retained=True,input_manifest_sha256=runtime.sha(ROOT/'inputs.json')))

if __name__=='__main__':
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    main()
