"""Complete the prespecified P52/P50 Demo comparison with patient/site-block CIs."""

import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits

import evaluate_24h_landmark as landmark
import evaluate_24h_catboost_blend as p52
from evaluate_eicu_24h_augmentation import updated_risk
from evaluate_real_eicu_augmentation import partition,P26_SEED
from audit_eicu_hospital_variation import hospital_blocks
from evaluate_gcs_routed_temporal import checked_metrics
from train_gcs_routed_temporal import matrix,train_one
from train_predictive_boosting import from_json,verify_sources,AB_FEATURES
from prepare import OUT,sha
from stress_missingness_c import score_metrics

ROOT=OUT/'p57_demo_paired_completion'
P50=OUT/'p50_24h_eicu_augmentation'


def triple(y,p):
    return np.asarray([average_precision_score(y,p),roc_auc_score(y,p),brier_score_loss(y,p)])


def intervals(cohort,scores,groups,pairs):
    rng=np.random.default_rng(42)
    result={key:[] for key in pairs}
    y=cohort.y.to_numpy()
    for repeat in range(1000):
        drawn=rng.integers(len(groups),size=len(groups))
        idx=np.concatenate([groups[i] for i in drawn])
        truth=y[idx]
        if truth.min()==truth.max(): continue
        metrics={tag:triple(truth,values[idx]) for tag,values in scores.items()}
        for key,(new,old) in pairs.items(): result[key].append(metrics[new]-metrics[old])
    if any(len(values)<950 for values in result.values()): raise ValueError('Too few paired samples')
    return {key:{**dict(zip(('ap','auroc','brier'),np.quantile(values,[.025,.975],axis=0).T.tolist())),
                 'n':len(values)} for key,values in result.items()}


def main():
    ROOT.mkdir(exist_ok=True)
    destination=ROOT/'summary.json'
    if destination.exists(): raise FileExistsError(destination)
    source52=from_json(p52.OUTPUT/'manifest.json')
    old52=from_json(p52.OUTPUT/'evaluation.json')['eicu_demo']
    old50=from_json(P50/'summary.json')
    verify_sources(source52['source_sha256'])
    verify_sources(old50['source_sha256'])
    model_path=p52.OUTPUT/'catboost.joblib'
    if sha(model_path)!=source52['model_sha256']: raise ValueError('P52 weight changed')
    models,anchor=landmark.frozen()
    if anchor!=old50['route']: raise ValueError('P50 route differs from original')
    candidate=joblib.load(model_path)
    recipe=source52['recipes'][source52['selected']]
    cache=ROOT/'inputs_manifest.json'
    if cache.exists():
        prepared=from_json(cache)
        for name,digest in prepared['files_sha256'].items():
            if sha(ROOT/name)!=digest: raise ValueError('Cached Demo input changed')
        cohort=pd.read_parquet(ROOT/'cohort.parquet')
        inputs={mode:(pd.read_parquet(ROOT/f'{mode}_base.parquet'),pd.read_parquet(ROOT/f'{mode}_time.parquet'))
                for mode in ('hourly','native')}
    else:
        columns=pd.read_parquet(AB_FEATURES).columns
        cohort,inputs,_,_,_=landmark.demo_inputs_24h(columns)
        cohort.to_parquet(ROOT/'cohort.parquet')
        for mode,(base,timing) in inputs.items():
            base.to_parquet(ROOT/f'{mode}_base.parquet')
            timing.to_parquet(ROOT/f'{mode}_time.parquet')
        files=['cohort.parquet',*[f'{m}_{kind}.parquet' for m in ('hourly','native') for kind in ('base','time')]]
        prepared=dict(files_sha256={name:sha(ROOT/name) for name in files})
        landmark.save_json(cache,prepared)
    if len(cohort)!=1398 or int(cohort.y.sum())!=121: raise ValueError('Unexpected 24-hour cohort')
    risks={}
    for mode,(base,timing) in inputs.items():
        values,observed=p52.scores(models,anchor,candidate,recipe,base,base,timing)
        for tag,pred in values.items(): checked_metrics(cohort.y.to_numpy(),pred,old52['all_metrics'][mode][tag])
        risks[mode]={**values,'p50':np.full(len(cohort),np.nan)}
    folds,sizes,grouping=partition(cohort)
    cached_folds=all((ROOT/f'fold_{fold}.npz').exists() for fold in range(5))
    if not cached_folds:
        x,timing,y,_,protected=landmark.development_24h()
        if len(y)!=old50['a_b_stays'] or int(y.sum())!=old50['a_b_deaths']:
            raise ValueError('P50 A/B training population changed')
        extra_base,extra_time=inputs['hourly']
        extra=matrix(extra_base,extra_base,extra_time)
    tag=anchor['primary']
    for fold,(training,tested) in enumerate(folds):
        path=ROOT/f'fold_{fold}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['ids'],cohort.index.to_numpy()[tested])
                predicted={mode:saved[mode] for mode in risks}
        else:
            current=train_one(tag,x,timing,y,P26_SEED,extra=(extra[training],cohort.y.to_numpy()[training]))
            predicted={mode:updated_risk(risks[mode]['p49'][tested],current,tag,base.iloc[tested],
                        base.iloc[tested],time.iloc[tested]) for mode,(base,time) in inputs.items()}
            joblib.dump(current,ROOT/f'fold_{fold}.joblib')
            np.savez_compressed(path,ids=cohort.index.to_numpy()[tested],**predicted)
        for mode,values in predicted.items(): risks[mode]['p50'][tested]=values
        print(f'Replayed P50 grouped OOF fold {fold+1}/5',flush=True)
    if not cached_folds: verify_sources(protected)
    _,blocks,_,patients,metadata=hospital_blocks(cohort)
    apache,included,counts=landmark.demo_apache(cohort)
    if counts!=old52['cohort']: raise ValueError('APACHE shared cohort changed')
    rows={}
    pairs={'p52_vs_p50':('p52','p50'),'p52_vs_p49':('p52','p49')}
    for mode,scores in risks.items():
        for tag,values in scores.items():
            if tag=='p50': checked_metrics(cohort.y.to_numpy(),values,old50['demo']['all']['metrics'][mode]['augmented'])
        observed=inputs[mode][0]['GCS__count'].gt(0).to_numpy()
        np.testing.assert_array_equal(scores['p52'][~observed],scores['p49'][~observed])
        np.testing.assert_array_equal(scores['p50'][~observed],scores['p49'][~observed])
        patient=intervals(cohort,scores,patients,pairs)
        block=intervals(cohort,scores,blocks,pairs)
        for metric in ('ap','auroc'):
            np.testing.assert_allclose(patient['p52_vs_p49'][metric],old52['full_paired_p49'][mode][metric],rtol=0,atol=1e-12)
        common=cohort.loc[included]
        matched={tag:values[included] for tag,values in scores.items()}
        matched['apache_iva']=apache
        _,common_blocks,_,common_patients,common_meta=hospital_blocks(common)
        common_pairs={**pairs,'p52_vs_apache':('p52','apache_iva')}
        common_patient=intervals(common,matched,common_patients,common_pairs)
        common_block=intervals(common,matched,common_blocks,common_pairs)
        for metric in ('ap','auroc'):
            np.testing.assert_allclose(common_patient['p52_vs_apache'][metric],
                       old52['matched_paired_apache'][mode][metric],rtol=0,atol=1e-12)
        checked_metrics(common.y.to_numpy(),matched['p50'],old50['demo']['matched_apache']['metrics'][mode]['augmented'])
        rows[mode]=dict(metrics={tag:score_metrics(cohort.y.to_numpy(),v) for tag,v in scores.items()},
                       patient_ci=patient,hospital_block_ci=block,
                       matched=dict(metrics={tag:score_metrics(common.y.to_numpy(),v) for tag,v in matched.items()},
                                    patient_ci=common_patient,hospital_block_ci=common_block,grouping=common_meta),
                       gcs_strata={label:dict(stays=int(mask.sum()),deaths=int(cohort.y.to_numpy()[mask].sum()),
                                             metrics={tag:score_metrics(cohort.y.to_numpy()[mask],v[mask]) for tag,v in scores.items()})
                                   for label,mask in [('observed',observed),('missing',~observed)]})
        np.savez_compressed(ROOT/f'{mode}_predictions.npz',ids=cohort.index.to_numpy(),y=cohort.y.to_numpy(),**scores)
        print(f'Completed patient and site-block comparisons: {mode}',flush=True)
    expected={k:v for k,v in old50['raw_eicu_sha256'].items() if k!=landmark.APACHE_FILE.name}
    if landmark.source_hashes(landmark.DEMO)!=expected: raise ValueError('Demo sources changed')
    verify_sources(source52['source_sha256'])
    report=dict(stays=len(cohort),deaths=int(cohort.y.sum()),patient_grouping=metadata,matched_cohort=counts,
                model_replay_fits=5,selected_new_parameters=False,bootstrap=dict(replicates=1000,seed=42),
                results=rows,source_sha256={str(p.relative_to(OUT)):sha(p) for p in
                    (p52.OUTPUT/'manifest.json',p52.OUTPUT/'evaluation.json',P50/'summary.json')})
    landmark.save_json(destination,report)
    landmark.save_json(ML/'reports/P57_demo_paired_summary.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
