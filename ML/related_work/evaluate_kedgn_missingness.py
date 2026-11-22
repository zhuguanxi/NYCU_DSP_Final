"""Prepare common frozen controls, then compare KEDGN full-channel masking."""

import argparse
import json
import os
from pathlib import Path
import sys

ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_frozen_kedgn import ROOT as P53, COMMON, CONTROLS, result_row
from evaluate_eicu_confirmation import frozen_models, frozen_ensemble
from evaluate_gcs_routed_temporal import RoutedPredictor, checked_metrics
from evaluate_hourly_sequence import infer
from evaluate_predictive_boosting import benchmarks, no_category_warning, contrast_c
from evaluate_robust_ensemble import Ensemble
from pilot_temporal_features import time_summary
from prepare import OUT,sha
from prepare_hourly_sequence import hourly_cube,static_values,hidden_channels
from stress_missingness_c import load_cohort,mask_plan,apply_masks,score_metrics,CHANNELS
from train_predictive_boosting import from_json,verify_sources

ROOT=OUT/'p55_kedgn_missingness'
SCENARIOS=(('clean',()),*((f'full_{x}',(x,)) for x in CHANNELS),('full_all_six',CHANNELS))


def prepare():
    ROOT.mkdir(exist_ok=True)
    target=ROOT/'controls.npz'
    if target.exists(): raise FileExistsError(target)
    route,_,gcs,extras,protected=frozen_models()
    rule,_,_=frozen_ensemble(protected)
    controls=benchmarks()
    elastic,_,columns=joblib.load(OUT/'final_models/Elastic.joblib')
    elastic['model'].set_params(device='cpu',n_jobs=2)
    controls['frozen_elastic']=elastic,columns
    raw,y,_=load_cohort()
    positions,plan=mask_plan(raw)
    events=pd.read_parquet(COMMON/'p12_c_events.parquet')
    timing=time_summary(events,raw.index)
    cube=hourly_cube(events,raw.index)
    routed=RoutedPredictor(route.rule,route.models,raw,timing)
    mixed=Ensemble(routed,gcs,rule)
    altered=[raw if not channels else apply_masks(raw,{k:positions[k] for k in channels},plan,'full')
             for _,channels in SCENARIOS]
    worker=ROOT/'cnn_inference'
    worker.mkdir(exist_ok=True)
    seq_path=OUT/'p33_hourly_sequence/fitted/sequence.pt'
    old={name:from_json(OUT/f'{folder}/evaluation/summary.json') for name,folder in
         [('p26','p26_gcs_routed_temporal'),('p28','p28_robust_ensemble'),('p33','p33_hourly_sequence')]}
    if sha(seq_path)!=old['p33']['training']['model_sha256']: raise ValueError('P33 weights changed')
    sequence=infer(worker,seq_path,cubes=cube,statics=static_values(raw),
                   hidden=np.stack([hidden_channels(raw,view) for view in altered]))
    predictions=np.empty((len(SCENARIOS),len(CONTROLS),len(y)),dtype=np.float64)
    reference=np.load(P53/'controls.npz',allow_pickle=False)
    rows=[]
    for i,((name,channels),frame) in enumerate(zip(SCENARIOS,altered)):
        values={tag:no_category_warning(pipe,frame.loc[:,cols]) for tag,(pipe,cols) in controls.items()}
        values['p26']=routed.predict_proba(frame)[:,1]
        values['p28']=mixed.predict_proba(frame)[:,1]
        values['p33']=.75*values['p28']+.25*sequence[i].astype(float)
        for j,tag in enumerate(CONTROLS):
            predictions[i,j]=values[tag]
            if name=='clean': np.testing.assert_array_equal(values[tag],reference[f'set_c_{tag}'])
            if tag in old:
                prior,=[row for row in old[tag]['set_c'] if row['scenario']==name]
                checked_metrics(y,values[tag],prior)
            rows.append(dict(scenario=name,model=tag,**score_metrics(y,values[tag])))
        print(f'Replayed and verified original controls: {name}',flush=True)
    verify_sources(protected)
    np.savez_compressed(target,ids=raw.index.to_numpy(),y=y,predictions=predictions,
                        scenarios=np.asarray([x for x,_ in SCENARIOS]),models=np.asarray(CONTROLS))
    from_json(P53/'control_sources.json')
    manifest=dict(source_sha256=protected,p33_sha256=sha(seq_path),controls_sha256=sha(target),
                  scenarios=[dict(name=n,channels=list(c)) for n,c in SCENARIOS],control_metrics=rows,
                  model_fits=0,stays=len(y),deaths=int(y.sum()))
    (ROOT/'controls_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def compare():
    target=ROOT/'summary.json'
    if target.exists(): raise FileExistsError(target)
    manifest=from_json(ROOT/'controls_manifest.json')
    verify_sources(manifest['source_sha256'])
    if sha(ROOT/'controls.npz')!=manifest['controls_sha256']: raise ValueError('Control predictions changed')
    kg=from_json(ROOT/'kedgn_manifest.json')
    with np.load(ROOT/'controls.npz') as source:
        ids,y,controls=source['ids'],source['y'],source['predictions']
    rows=[]
    points={}
    for i,(name,_) in enumerate(SCENARIOS):
        scores={tag:controls[i,j] for j,tag in enumerate(CONTROLS)}
        for version in ('set_a','ab'):
            path=ROOT/f'{version}_{name}.npz'
            if sha(path)!=kg['predictions'][path.name]: raise ValueError('KEDGN score changed')
            with np.load(path,allow_pickle=False) as predicted:
                np.testing.assert_array_equal(ids,predicted['ids'])
                np.testing.assert_array_equal(y,predicted['y'])
                score,metrics=result_row(y,predicted['probabilities'])
            tag=f'kedgn_{version}'
            scores[tag]=score
            rows.append(dict(scenario=name,model=tag,**metrics))
        points[name]=scores
    metric_rows=[]
    for name,scores in points.items():
        for tag,values in scores.items():
            metrics=score_metrics(y,values)
            clean=score_metrics(y,points['clean'][tag])
            metric_rows.append(dict(scenario=name,model=tag,**metrics,
                                    delta_from_clean={k:metrics[k]-clean[k] for k in metrics}))
    intervals={}
    for name in ('full_GCS','full_all_six'):
        selected=points[name]
        baseline={key:selected[key] for key in ('kedgn_set_a','frozen_full','p26','p33')}
        intervals[name]=contrast_c(y,baseline,{'kedgn_ab':selected['kedgn_ab']})['kedgn_ab']
        print(f'Primary paired comparison complete: {name}',flush=True)
    report=dict(stays=len(y),deaths=int(y.sum()),model_fits=0,
                scenarios=manifest['scenarios'],metrics=metric_rows,kedgn_seed_metrics=rows,
                primary_paired_ci=intervals,bootstrap=dict(replicates=2000,seed=20261013,unit='stay'),
                source_sha256={str(p.relative_to(OUT)):sha(p) for p in
                               (ROOT/'controls_manifest.json',ROOT/'kedgn_manifest.json')})
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    dest=ML/'reports/P55_kedgn_missingness_summary.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table=pd.DataFrame(metric_rows).pivot(index='model',columns='scenario',values='ap')
    print(table.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','compare'))
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    with threadpool_limits(limits=2): {'prepare':prepare,'compare':compare}[args.phase]()
