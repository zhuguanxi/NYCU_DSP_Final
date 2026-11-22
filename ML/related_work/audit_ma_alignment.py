"""Freeze all MA-GBT recipes and align their urine input with the current benchmark."""

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
from threadpoolctl import threadpool_limits

from run_ma_gbt_p12 import validate_upstream
from ma_gbt_p12_inputs import SCHEMA_FILE
from evaluate_ma_gbt_p12 import Predictor
from evaluate_gcs_routed_temporal import checked_metrics
from evaluate_eicu_urine_mapping import urine_values
from evaluate_missingness_augmentation import paired_intervals
from stress_missingness_c import load_cohort,mask_plan,apply_masks,CHANNELS,score_metrics
from prepare import OUT,sha
from train_predictive_boosting import from_json

ROOT=OUT/'p59_ma_gbt_alignment'
COMMON=OUT/'related_work/common_sequence_inputs'
VARIANTS=(('original_1000','ma_gbt_p12/fitted','ma_gbt_p12/evaluation/summary.json','ma'),
          ('original_10','ma_gbt_alpha10','ma_gbt_alpha10/evaluation/summary.json','ma10'),
          ('augmented_10','ma_gbt_alpha10_bilirubin','ma_gbt_alpha10_bilirubin/evaluation/summary.json','bilirubin10'))


def saved_metrics(row,version):
    suffix=dict(original_1000='ma',original_10='ma10',augmented_10='bilirubin10')[version]
    return {metric:row[f'{metric}_{suffix}'] for metric in ('ap','auroc','brier','sensitivity')}


def main():
    validate_upstream()
    ROOT.mkdir(exist_ok=True)
    destination=ROOT/'summary.json'
    if destination.exists(): raise FileExistsError(destination)
    schemas=from_json(SCHEMA_FILE)['schemas']
    cohort=pd.read_parquet(OUT/'eicu_demo_external/cohort.parquet')
    raw={mode:pd.read_parquet(OUT/f'eicu_demo_external/features_{mode}.parquet') for mode in ('hourly','native')}
    mapped={mode:frame.copy() for mode,frame in raw.items()}
    for mode,frame in mapped.items():
        if not frame.index.equals(cohort.index): raise ValueError('External row order differs')
        events=pd.read_parquet(COMMON/f'eicu_{mode}_events.parquet')
        urine=events.loc[events.variable.eq('Urine')].rename(columns={
                'RecordID':'patientunitstayid','elapsed_minute':'minute','clean_value':'value'})
        update=urine_values(frame.index,urine)
        frame.loc[:,update.columns]=update
    frame,y,_=load_cohort(); positions,plan=mask_plan(frame)
    views={'clean':frame,'full_GCS':apply_masks(frame,{'GCS':positions['GCS']},plan,'full'),
           'full_all_six':apply_masks(frame,{k:positions[k] for k in CHANNELS},plan,'full')}
    c_rows=[]; demo=[]; sources={str(SCHEMA_FILE.relative_to(OUT)):sha(SCHEMA_FILE)}
    for version,folder,previous,_ in VARIANTS:
        base=OUT/folder
        fit=from_json(base/'manifest.json')
        old=from_json(OUT/previous)
        for model in ('Full','Elastic'):
            path=base/f'{model}.joblib'
            if sha(path)!=fit['model_sha256'][model]: raise ValueError('Frozen MA model changed')
            pipe=Predictor(joblib.load(path),schemas[model]['final'])
            for scenario,altered in views.items():
                probabilities,rho=pipe.predict_and_rho(altered)
                expected,=[row for row in old['set_c'] if row['scenario']==scenario and row['model']==model]
                checked_metrics(y,probabilities,saved_metrics(expected,version))
                c_rows.append(dict(variant=version,model=model,scenario=scenario,
                                   metrics=score_metrics(y,probabilities),rho=float(rho)))
            before={}; after={}; reliance={}
            for mode in ('hourly','native'):
                before[mode],old_rho=pipe.predict_and_rho(raw[mode])
                after[mode],new_rho=pipe.predict_and_rho(mapped[mode])
                reliance[mode]=(old_rho,new_rho)
                expected,=[row for row in old['eicu'] if row['model']==model and row['mode']==mode]
                checked_metrics(cohort.y.to_numpy(),before[mode],saved_metrics(expected,version))
            bounds=paired_intervals(cohort,before,after)
            for mode in ('hourly','native'):
                metrics_before=score_metrics(cohort.y.to_numpy(),before[mode])
                metrics_after=score_metrics(cohort.y.to_numpy(),after[mode])
                demo.append(dict(variant=version,model=model,mode=mode,before=metrics_before,after=metrics_after,
                                 delta={k:metrics_after[k]-metrics_before[k] for k in metrics_before},
                                 paired_ci=bounds[mode],rho_before=float(reliance[mode][0]),rho_after=float(reliance[mode][1])))
            np.savez_compressed(ROOT/f'{version}_{model}_external.npz',ids=cohort.index.to_numpy(),
                                y=cohort.y.to_numpy(),**{f'before_{k}':v for k,v in before.items()},
                                **{f'after_{k}':v for k,v in after.items()})
            sources[str(path.relative_to(OUT))]=sha(path)
            print(f'Verified old scores and aligned urine: {version}/{model}',flush=True)
        sources[str((OUT/previous).relative_to(OUT))]=sha(OUT/previous)
        sources[str((base/'manifest.json').relative_to(OUT))]=sha(base/'manifest.json')
    result=dict(model_fits=0,stays=len(cohort),deaths=int(cohort.y.sum()),set_c=c_rows,eicu=demo,
                source_sha256=sources,bootstrap=dict(replicates=1000,seed=42,unit='patient'))
    for path,digest in sources.items():
        if sha(OUT/path)!=digest: raise ValueError('MA model source changed')
    for path in (destination,ML/'reports/P59_ma_alignment_summary.json'):
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
