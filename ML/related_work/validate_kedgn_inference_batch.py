"""Select an inference memory batch by A/B output equivalence and runtime only."""
from pathlib import Path
import time
import numpy as np
import torch
import run_kedgn_ab as runtime
from p60_seed_worker import busy

AB=runtime.ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121'
ROOT=runtime.ML/'artifacts/p61_progressive_block_missingness'

def main():
    if busy() or not torch.cuda.is_available(): raise RuntimeError('GPU is occupied; no batch benchmark launched')
    where=torch.device('cuda:0')
    rows,source=runtime.records(AB/'inputs/ab.npz');rows=rows[:1024]
    with np.load(AB/'final_seed_0/statistics.npz') as s: scaling=tuple(s[k] for k in ('mf','sf','ms','ss'))
    tensors=runtime.tensorize(rows,source['y'][:1024],scaling)
    model=runtime.model_for(where,0)
    model.load_state_dict(torch.load(AB/'final_seed_0/weights.pt',map_location=where))
    embeddings=torch.load(runtime.AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location=where)
    results=[];reference=None
    for batch in (256,1024,512):
        torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
        try:
            values=runtime.predict(model,tensors,embeddings,where,batch=batch)
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            if reference is None: reference=values.copy()
            error=float(np.max(np.abs(values-reference)))
            np.testing.assert_allclose(values,reference,atol=2e-6,rtol=2e-5)
            results.append(dict(batch=batch,seconds=seconds,max_abs_error=error,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,valid=True))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();results.append(dict(batch=batch,valid=False,reason='GPU memory limit'))
        print(results[-1],flush=True)
    selected=min((r for r in results if r['valid']),key=lambda r:r['seconds'])['batch']
    runtime.write_json(ROOT/'inference_batch_validation.json',dict(dataset='P12 A/B first 1024 fixed rows',
       selected_batch=selected,selection='Runtime among numerically equivalent inference batches; no AP/labels used',
       results=results,new_model_fits=0,input_sha256=runtime.sha(AB/'inputs/ab.npz'),
       weights_sha256=runtime.sha(AB/'final_seed_0/weights.pt')))
    print(f'Validated inference batch {selected}; no training or test-based selection.',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    main()
