"""Evaluate P62 only after all A/B selections and final weights are frozen."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import importlib.metadata
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from ma_gbt_p12_inputs import encode,SCHEMA_FILE
from ma_matched_runtime import reliance
from train_ma_matched import ROOT,read,write,verify_sources
from prepare import OUT,sha
from stress_missingness_c import load_cohort,mask_plan,mask_draws,apply_masks,SCENARIOS,score_metrics,METRICS

ML=Path(__file__).resolve().parents[1]
EVAL=ROOT/'evaluation'
def binary(y,p):
    order=np.argsort(-p,kind='mergesort');target=y[order];sorted_p=p[order]
    last=np.r_[sorted_p[:-1]!=sorted_p[1:],True]
    tp=np.cumsum(target)[last].astype(float);fp=np.cumsum(1-target)[last].astype(float)
    positives=target.sum();negatives=len(y)-positives
    return np.asarray([np.sum(np.diff(np.r_[0.,tp])*tp/(tp+fp))/positives,
        np.trapz(np.r_[0.,tp/positives],np.r_[0.,fp/negatives]),np.mean((p-y)**2),np.mean(p[y==1]>=.5)])
def check_metrics():
    rng=np.random.default_rng(42)
    for n in (25,4000):
        y=rng.integers(2,size=n)
        for p in (rng.random(n),rng.integers(5,size=n)/4):
            np.testing.assert_allclose(binary(y,p),list(score_metrics(y,p).values()),atol=1e-12,rtol=0)
def frozen():
    data=read(ROOT/'frozen.json');protocol=read(ROOT/'protocol.json');verify_sources(protocol)
    assert sha(ROOT/'numeric_check.json')==protocol['numeric_check_sha256']
    assert data['training_only_AB'] and not data['test_accessed_for_selection']
    assert sha(ROOT/'selection.json')==data['selection_sha256'] and sha(ROOT/'protocol.json')==data['protocol_sha256']
    for r in data['fits']:assert sha(ROOT/'models'/f'{r["key"]}.joblib')==r['model_sha256']
    return data
def contrast_pairs(selections):
    labels={}
    for r in selections:
        label=f'{r["features"]}_{"gcs60" if r["augmentation"] else "plain"}_{r["policy"]}'
        from train_ma_matched import key_for
        labels[label]=key_for(r['features'],r['augmentation'],r['alpha'],'final')
    pairs={}
    for name in ('Full','Elastic'):
        for aug in ('plain','gcs60'):
            for policy in ('ap','rho95'):
                pairs[f'{name}_{aug}_{policy}_minus_off']=(f'{name}_{aug}_{policy}',f'{name}_{aug}_off')
        for policy in ('off','ap','rho95'):
            pairs[f'{name}_{policy}_gcs60_minus_plain']=(f'{name}_gcs60_{policy}',f'{name}_plain_{policy}')
    return labels,pairs
def primary_references(case):
    history=read(OUT/'p60_five_seed_stability/evaluation.json')['cases'][case]
    path=ML/history['file'];assert sha(path)==history['sha256']
    with np.load(path) as p:
        names=list(p['models']);scores=p['predictions'][0].copy();ids=p['ids'].copy();y=p['y'].copy()
    lookup={'XGB_Full':'frozen_full','XGB_Elastic':'frozen_elastic','XGB_GCS_Full':'gcs_only_full','XGB_GCS_Elastic':'gcs_only_elastic'}
    return ids,y,{label:scores[names.index(tag)] for label,tag in lookup.items()},path
def main():
    check_metrics();meta=frozen();labels,pairs=contrast_pairs(meta['selected']);schemas=read(SCHEMA_FILE)['schemas']
    EVAL.mkdir(exist_ok=True)
    models={r['key']:joblib.load(ROOT/'models'/f'{r["key"]}.joblib') for r in meta['fits']}
    featuresets={r['key']:r['features'] for r in meta['fits']}
    structures={}
    for key,model in models.items():
        depths=[];leaves=[]
        assert len(model.estimators_)==300
        for tree in model.estimators_.ravel():
            todo=[tree.root_];depth=0;leaf=0
            while todo:
                node=todo.pop();depth=max(depth,node.depth)
                if node.left_subtree is None:leaf+=1
                else:todo.extend([node.left_subtree,node.right_subtree])
            depths.append(depth);leaves.append(leaf)
        assert max(depths)<=3
        structures[key]=dict(trees=len(depths),maximum_depth=max(depths),mean_leaves=float(np.mean(leaves)))
    def predict(frame,include_rho=False):
        encoded={name:encode(frame,schemas[name]['final']) for name in ('Full','Elastic')}
        scores={};rhos={}
        for key,model in models.items():
            x,m=encoded[featuresets[key]];scores[key]=model.predict_proba(x)[:,1]
            if include_rho:rhos[key]=reliance(model,x,m)
        return {tag:scores[key] for tag,key in labels.items()},{tag:rhos[key] for tag,key in labels.items()} if include_rho else {}
    protected={};primary={};metricrows=[];stressrows=[]
    def primary_case(case,frame,y,ids):
        refids,refy,refs,path=primary_references(case)
        np.testing.assert_array_equal(ids,refids);np.testing.assert_array_equal(y,refy)
        protected[path.relative_to(ML).as_posix()]=sha(path)
        path=EVAL/f'{case}.npz'
        if path.exists():
            with np.load(path) as p:
                np.testing.assert_array_equal(p['ids'],ids);np.testing.assert_array_equal(p['y'],y)
                scores=dict(zip(p['models'].tolist(),p['predictions'].copy()))
            info=read(EVAL/f'{case}.json');assert info['frozen_sha256']==sha(ROOT/'frozen.json')
            rhos=info['rho']
        else:
            scores,rhos=predict(frame,True);scores.update(refs)
            np.savez_compressed(path,ids=ids,y=y,models=np.asarray(list(scores)),predictions=np.asarray(list(scores.values())))
            write(EVAL/f'{case}.json',dict(frozen_sha256=sha(ROOT/'frozen.json'),rho=rhos,predictions_sha256=sha(path)))
        for tag,p in scores.items():
            metricrows.append(dict(cohort=case,model=tag,stays=len(y),deaths=int(y.sum()),**score_metrics(y,p),rho=rhos.get(tag)))
        primary[case]=(y,scores);print('PRIMARY',case,'done',flush=True)
    raw,y,_=load_cohort();positions,plan=mask_plan(raw)
    primary_case('clean',raw,y,raw.index.to_numpy())
    for case,variables in [('gcs_full',('GCS',)),('six_full',('GCS','Lactate','MAP','Mg','Urine','MechVent'))]:
        primary_case(case,apply_masks(raw,{k:positions[k] for k in variables},plan,'full'),y,raw.index.to_numpy())
    cohort=pd.read_parquet(OUT/'eicu_demo_external/cohort.parquet')
    assert len(cohort)==782 and int(cohort.y.sum())==91
    for mode in ('hourly','native'):
        path=OUT/f'p60_five_seed_stability/eicu_{mode}/features.parquet';frame=pd.read_parquet(path)
        assert frame.index.equals(cohort.index);protected[path.relative_to(ML).as_posix()]=sha(path)
        primary_case(f'eicu_{mode}',frame,cohort.y.to_numpy(),frame.index.to_numpy())
    for scenario,variables,design in SCENARIOS:
        path=EVAL/f'stress_{scenario}.json'
        if path.exists():
            saved=read(path);assert saved['frozen_sha256']==sha(ROOT/'frozen.json');stressrows.extend(saved['rows']);continue
        rows=[];draws=mask_draws(positions,plan) if design=='matched' else [{}]
        for repeat,sample in enumerate(draws,1):
            frame=apply_masks(raw,{k:sample.get(k,positions[k]) for k in variables},plan,design)
            scores,_=predict(frame)
            for tag,p in scores.items():rows.append(dict(scenario=scenario,design=design,repeat=repeat,model=tag,**score_metrics(y,p)))
        write(path,dict(frozen_sha256=sha(ROOT/'frozen.json'),rows=rows));stressrows.extend(rows)
        print('STRESS',scenario,len(draws),'positions',flush=True)
    # The complete original P9 matrix is retained. Time-block P61 is a separate historical study.
    bootstrap={}
    for case,(truth,scores) in primary.items():
        path=EVAL/f'bootstrap_{case}.json'
        if path.exists():
            saved=read(path);assert saved['frozen_sha256']==sha(ROOT/'frozen.json');bootstrap[case]=saved['contrasts'];continue
        tags=list(scores);pred=np.asarray([scores[t] for t in tags]);samples=[]
        external=case.startswith('eicu');nboot=1000 if external else 2000
        rng=np.random.default_rng(42 if external else 20261013)
        groups=list(cohort.groupby('uniquepid',sort=False).indices.values()) if external else None
        for i in range(nboot):
            rows=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) if external else rng.integers(len(truth),size=len(truth))
            target=truth[rows]
            if target.min()==target.max():continue
            samples.append([binary(target,p[rows])[:3] for p in pred])
        draws=np.asarray(samples);assert len(draws)>=.95*nboot
        contrasts={}
        for label,(after,before) in pairs.items():
            difference=draws[:,tags.index(after)]-draws[:,tags.index(before)]
            point=binary(truth,scores[after])[:3]-binary(truth,scores[before])[:3]
            bounds=np.quantile(difference,[.025,.975],axis=0)
            contrasts[label]={m:dict(delta=float(point[j]),ci=bounds[:,j].tolist()) for j,m in enumerate(('ap','auroc','brier'))}
        for name in ('Full','Elastic'):
            for aug,ref in [('plain',f'XGB_{name}'),('gcs60',f'XGB_GCS_{name}')]:
                for policy in ('off','ap','rho95'):
                    after=f'{name}_{aug}_{policy}';diff=draws[:,tags.index(after)]-draws[:,tags.index(ref)]
                    point=binary(truth,scores[after])[:3]-binary(truth,scores[ref])[:3];bounds=np.quantile(diff,[.025,.975],axis=0)
                    contrasts[f'{after}_minus_{ref}']={m:dict(delta=float(point[j]),ci=bounds[:,j].tolist()) for j,m in enumerate(('ap','auroc','brier'))}
        saved=dict(frozen_sha256=sha(ROOT/'frozen.json'),unit='patient' if external else 'stay',seed=42 if external else 20261013,
                   repeats=nboot,valid_repeats=len(draws),contrasts=contrasts)
        write(path,saved);bootstrap[case]=contrasts;print('BOOTSTRAP',case,len(draws),'done',flush=True)
    for path,digest in protected.items():assert sha(ML/path)==digest
    frozen()
    summary=dict(complete=True,protocol=read(ROOT/'protocol.json'),selected=meta['selected'],cv_fits=meta['cv_fits'],final_fits=meta['final_fits'],
       primary=metricrows,stress=stressrows,paired=bootstrap,protected_evaluation_sources=protected,
       inference='fixed seed42 models; unadjusted paired intervals; no training/search uncertainty',
       original_xgb='P60 seed42 frozen models; current explicit urine alignment',frozen_sha256=sha(ROOT/'frozen.json'),
       evaluation_source_sha256=sha(Path(__file__)),model_structure=structures,
       runtime_versions={k:importlib.metadata.version(k) for k in ('numpy','scikit-learn','torch','numba','llvmlite')},
       test_selected=False,model_fits_in_evaluation=0)
    write(EVAL/'summary.json',summary)
    pd.DataFrame(metricrows).to_csv(EVAL/'primary_metrics.csv',index=False)
    pd.DataFrame(stressrows).to_csv(EVAL/'all_stress_metrics.csv',index=False)
    print('COMPLETE P62: primary cohorts, all original masks, paired intervals and true rho',flush=True)
if __name__=='__main__':main()
