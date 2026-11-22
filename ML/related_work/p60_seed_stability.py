"""Retrain fixed full A/B pipelines at five prespecified seeds; no test selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import train_gcs_routed_temporal as route
from train_missingness_augmentation import ab_inputs,fit_augmented,mask_observed,SEED as AUG_SEED
from prepare import OUT,sha

ROOT=OUT/'p60_five_seed_stability'
SEEDS=(42,43,44,45,46)
MODELS=('frozen_full','frozen_elastic','gcs_only_full','gcs_only_elastic','p26','p28','p33')

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def cnn(folder,seed):
    receipt=folder/'cnn_receipt.json'
    if receipt.exists():
        value=read(receipt)
        assert value['seed']==seed and sha(folder/'sequence.pt')==value['weights_sha256']
        return value
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'}
    env.pop('PYTHONPATH',None)
    subprocess.run([str(OUT/'related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe'),
        str(Path(__file__).with_name('p60_seed_worker.py')),'fit','--model',str(folder/'sequence.pt'),'--seed',str(seed)],check=True,env=env)
    return read(receipt)

def main():
    ROOT.mkdir(exist_ok=True)
    frozen=ROOT/'frozen.json'
    if frozen.exists(): raise FileExistsError(frozen)
    inputs=[OUT/'features_AB.parquet',OUT/'labels_AB.csv',OUT/'folds3.csv',
       OUT/'p22_temporal_ab_pilot/features_AB_temporal.parquet',OUT/'p33_hourly_sequence/inputs/ab.npz',
       OUT/'p26_gcs_routed_temporal/manifest.json',OUT/'p28_robust_ensemble/manifest.json',
       OUT/'p33_hourly_sequence/fitted/manifest.json',ML/'reports/P60_five_seed_stability_protocol.md']
    for name in ('Full','Elastic'): inputs.append(OUT/f'final_models/{name}.joblib')
    source={str(p.relative_to(ML)):sha(p) for p in inputs}
    start=ROOT/'protocol.json'
    protocol=dict(seeds=SEEDS,models=MODELS,training='Full A+B only',fixed_epochs=8,
                  no_new_selection=True,augmentation_seed_fixed=AUG_SEED+1000,source_sha256=source)
    if start.exists(): assert read(start)==json.loads(json.dumps(protocol))
    else: write(start,protocol)
    x,y,folds=ab_inputs()
    timing=pd.read_parquet(OUT/'p22_temporal_ab_pilot/features_AB_temporal.parquet')
    assert timing.index.equals(x.index)
    originals={n:joblib.load(OUT/f'final_models/{n}.joblib') for n in ('Full','Elastic')}
    masked,changed,_=mask_observed(x,{'GCS':60},AUG_SEED+1000)
    receipts={}
    for seed in SEEDS:
        folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True)
        final=folder/'receipt.json'
        if final.exists():
            receipt=read(final)
            assert receipt['seed']==seed
            for file,digest in receipt['model_sha256'].items(): assert sha(folder/file)==digest
            receipts[str(seed)]=receipt;continue
        timings=[]
        for name in ('Full','Elastic'):
            template,metadata,columns=originals[name]
            pipe=clone(template).set_params(model__random_state=seed,model__device='cpu',model__n_jobs=2)
            tick=time.monotonic();pipe.fit(x.loc[:,columns],y)
            joblib.dump((pipe,columns),folder/f'{name}.joblib')
            augmented,seconds=fit_augmented(pipe,columns,x,y,masked,changed)
            joblib.dump((augmented,columns),folder/f'gcs_{name}.joblib')
            if seed==42:
                np.testing.assert_array_equal(pipe.predict_proba(x.loc[:,columns]),template.predict_proba(x.loc[:,columns]))
                previous=joblib.load(OUT/f'gcs_only_augmentation_AB/{name}.joblib')
                np.testing.assert_array_equal(augmented.predict_proba(x.loc[:,columns]),previous[0].predict_proba(x.loc[:,columns]))
            timings.append(dict(component=name,plain_seconds=time.monotonic()-tick-seconds,augmentation_seconds=seconds))
        factory=route.estimator
        try:
            route.estimator=lambda alg,rounds: factory(alg,rounds).set_params(random_state=seed,n_jobs=2)
            for tag in ('temporal','blind_lgb'):
                tick=time.monotonic();model=route.train_one(tag,x,timing,y,route.SHIFT_SEED+400)
                joblib.dump(model,folder/f'{tag}.joblib')
                if seed==42:
                    oldpath=OUT/('p22_temporal_ab_pilot/fitted/xgboost.joblib' if tag=='temporal' else f'p26_gcs_routed_temporal/{tag}.joblib')
                    before=joblib.load(oldpath)
                    native=route.matrix(x,x,timing)
                    np.testing.assert_array_equal(model.predict_proba(native),before.predict_proba(native))
                timings.append(dict(component=tag,seconds=time.monotonic()-tick))
        finally: route.estimator=factory
        neural=cnn(folder,seed)
        files=[f'{n}.joblib' for n in ('Full','Elastic','gcs_Full','gcs_Elastic','temporal','blind_lgb')]+['sequence.pt']
        receipt=dict(seed=seed,training_stays=len(y),deaths=int(y.sum()),fit_count=7,
                     source_sha256=source,model_sha256={p:sha(folder/p) for p in files},timings=timings,cnn=neural)
        write(final,receipt);receipts[str(seed)]=receipt
        print(f'Frozen all seven fitted components for seed {seed}',flush=True)
    assert all(sha(ML/p)==digest for p,digest in source.items())
    write(frozen,dict(protocol=protocol,seeds=receipts,fit_count=35,
                     selected_seed=None,selected_on_tests=False,all_seeds_retained=True))
    print('All five fixed pipelines frozen before new test evaluation.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
