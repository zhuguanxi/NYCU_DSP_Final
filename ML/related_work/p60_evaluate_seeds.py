"""Evaluate all frozen seeded pipelines on every prespecified case and clean Demo."""
import json
import os
from pathlib import Path
import subprocess
import sys
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from p60_seed_stability import ROOT,SEEDS,MODELS,read,write
from p61_missingness_inputs import ROOT as STRESS
from evaluate_gcs_routed_temporal import RoutedPredictor
from evaluate_robust_ensemble import Ensemble
from evaluate_predictive_boosting import no_category_warning
from evaluate_eicu_urine_mapping import urine_values
from pilot_temporal_features import time_summary
from prepare_hourly_sequence import hourly_cube,static_values
from prepare import OUT,sha

def validate_frozen():
    frozen=read(ROOT/'frozen.json')
    assert frozen['fit_count']==35 and frozen['selected_seed'] is None
    for seed in SEEDS:
        for file,digest in frozen['seeds'][str(seed)]['model_sha256'].items():
            assert sha(ROOT/f'seed_{seed}'/file)==digest
    for path,digest in frozen['protocol']['source_sha256'].items(): assert sha(ML/path)==digest
    return frozen

def prepare_demo():
    cohort=pd.read_parquet(OUT/'eicu_demo_external/cohort.parquet')
    rows=[]
    for mode in ('hourly','native'):
        folder=ROOT/f'eicu_{mode}';folder.mkdir(exist_ok=True)
        frame=pd.read_parquet(OUT/f'eicu_demo_external/features_{mode}.parquet')
        path=OUT/f'related_work/common_sequence_inputs/eicu_{mode}_events.parquet'
        events=pd.read_parquet(path)
        urine=events.loc[events.variable.eq('Urine')].rename(columns={
            'RecordID':'patientunitstayid','elapsed_minute':'minute','clean_value':'value'})
        update=urine_values(frame.index,urine);frame.loc[:,update.columns]=update
        frame.to_parquet(folder/'features.parquet');time_summary(events,frame.index).to_parquet(folder/'temporal.parquet')
        np.savez_compressed(folder/'input.npz',ids=frame.index.to_numpy(),y=cohort.y.to_numpy(),
                            cube=hourly_cube(events,frame.index),static=static_values(frame))
        rows.append(dict(name=f'eicu_{mode}',folder=str(folder),source_events_sha256=sha(path)))
    return rows

def main():
    frozen=validate_frozen();mask=read(STRESS/'inputs.json')
    cases=[dict(name=c['name'],folder=str(STRESS/c['name'])) for c in mask['cases']]
    for c in mask['cases']:
        for name,digest in c['files'].items(): assert sha(STRESS/c['name']/name)==digest
    cases+=prepare_demo()
    for seed in SEEDS:
        jobs=[dict(input=str(Path(c['folder'])/'input.npz'),output=str(Path(c['folder'])/f'cnn_seed_{seed}.npy')) for c in cases]
        packet=ROOT/f'cnn_jobs_{seed}.json';write(packet,jobs)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None)
        subprocess.run([str(OUT/'related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe'),
            str(Path(__file__).with_name('p60_seed_worker.py')),'infer-list','--model',str(ROOT/f'seed_{seed}/sequence.pt'),
            '--packet',str(packet)],env=env,check=True)
    rule=read(OUT/'p28_robust_ensemble/manifest.json')['rule']
    route=read(OUT/'p26_gcs_routed_temporal/manifest.json')['selected']
    pipelines=[]
    for seed in SEEDS:
        folder=ROOT/f'seed_{seed}'
        trees={tag:joblib.load(folder/f'{tag}.joblib') for tag in ('Full','Elastic','gcs_Full','gcs_Elastic')}
        routed=RoutedPredictor(route,{tag:joblib.load(folder/f'{tag}.joblib') for tag in ('temporal','blind_lgb')})
        pipelines.append((trees,routed,Ensemble(routed,trees['gcs_Elastic'],rule)))
    saved={}
    for case in cases:
        folder=Path(case['folder']);target=folder/'seed_predictions.npz'
        raw=pd.read_parquet(folder/'features.parquet');timing=pd.read_parquet(folder/'temporal.parquet')
        with np.load(folder/'input.npz') as data: ids,y=data['ids'],data['y']
        assert np.array_equal(ids,raw.index)
        scores=np.empty((5,len(MODELS),len(raw)))
        for s,(trees,routed,mixed) in enumerate(pipelines):
            for m,key in enumerate(('Full','Elastic','gcs_Full','gcs_Elastic')):
                pipe,cols=trees[key];scores[s,m]=no_category_warning(pipe,raw.loc[:,cols])
            joined=raw.join(timing)
            scores[s,4]=routed.predict_proba(joined)[:,1]
            scores[s,5]=mixed.predict_proba(joined)[:,1]
            scores[s,6]=.75*scores[s,5]+.25*np.load(folder/f'cnn_seed_{SEEDS[s]}.npy').astype(float)
        assert (np.isfinite(scores)&(scores>=0)&(scores<=1)).all()
        if case['name'] in ('clean','gcs_full','six_full'):
            oldname={'clean':'clean','gcs_full':'full_GCS','six_full':'full_all_six'}[case['name']]
            with np.load(OUT/'p55_kedgn_missingness/controls.npz') as old:
                row=list(old['scenarios']).index(oldname)
                np.testing.assert_allclose(scores[0,:6],old['predictions'][row,:6],rtol=0,atol=1e-12)
        elif case['name'].startswith('eicu_'):
            with np.load(OUT/'related_work/KEDGN/frozen_transfer_20261121/controls.npz') as old:
                for m,key in enumerate(MODELS[:6]):
                    np.testing.assert_allclose(scores[0,m],old[f'{case["name"]}_{key}'],rtol=0,atol=1e-12)
        np.savez_compressed(target,ids=ids,y=y,models=np.asarray(MODELS),seeds=np.asarray(SEEDS),predictions=scores)
        saved[case['name']]=dict(file=str(target.relative_to(ML)),sha256=sha(target))
        print(f'Evaluated all five seeds: {case["name"]}',flush=True)
    validate_frozen()
    write(ROOT/'evaluation.json',dict(cases=saved,model_fits=0,frozen_sha256=sha(ROOT/'frozen.json'),
           stress_inputs_sha256=sha(STRESS/'inputs.json'),selected_seed=None,all_seeds_retained=True))

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
