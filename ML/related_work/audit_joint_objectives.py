"""Verify the same frozen P33 against Full under clean/masked/transfer settings."""

import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits

from prepare import OUT,sha
from stress_missingness_c import score_metrics
from train_predictive_boosting import from_json,verify_sources

ROOT=OUT/'p58_joint_objectives'
P55=OUT/'p55_kedgn_missingness'
P53=OUT/'related_work/KEDGN/frozen_transfer_20261121'


def metrics(y,p):
    return np.asarray([average_precision_score(y,p),roc_auc_score(y,p),brier_score_loss(y,p)])


def bounds(rows):
    return dict(zip(('ap','auroc','brier'),np.quantile(rows,[.025,.975],axis=0).T.tolist()))


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'summary.json').exists(): raise FileExistsError(ROOT/'summary.json')
    manifest=from_json(P55/'controls_manifest.json')
    verify_sources(manifest['source_sha256'])
    if sha(P55/'controls.npz')!=manifest['controls_sha256']: raise ValueError('P55 control changed')
    with np.load(P55/'controls.npz',allow_pickle=False) as data:
        y=data['y']; scenarios=data['scenarios'].tolist(); models=data['models'].tolist()
        points={case:{tag:data['predictions'][scenarios.index(case),models.index(tag)] for tag in ('frozen_full','p33')}
                for case in ('clean','full_GCS','full_all_six')}
    rows={case:dict(metrics={tag:score_metrics(y,p) for tag,p in risks.items()},
                    delta={k:score_metrics(y,risks['p33'])[k]-score_metrics(y,risks['frozen_full'])[k]
                           for k in score_metrics(y,risks['p33'])}) for case,risks in points.items()}
    sampled={case:[] for case in points}; retention={case:[] for case in points if case!='clean'}
    rng=np.random.default_rng(20261013)
    for repeat in range(2000):
        idx=rng.integers(len(y),size=len(y)); truth=y[idx]
        computed={case:{tag:metrics(truth,p[idx]) for tag,p in risks.items()} for case,risks in points.items()}
        for case,values in computed.items():
            sampled[case].append(values['p33']-values['frozen_full'])
            if case!='clean':
                retention[case].append((values['p33'][0]-computed['clean']['p33'][0])-
                                       (values['frozen_full'][0]-computed['clean']['frozen_full'][0]))
    for case in points:
        rows[case]['paired_ci']=bounds(sampled[case])
        if case!='clean':
            delta={tag:rows[case]['metrics'][tag]['ap']-rows['clean']['metrics'][tag]['ap'] for tag in points[case]}
            rows[case]['ap_change_from_clean']=delta
            rows[case]['retention_gain']=delta['p33']-delta['frozen_full']
            rows[case]['retention_gain_ci']=np.quantile(retention[case],[.025,.975]).tolist()
        print(f'Joint Set C evidence complete: {case}',flush=True)
    cohort=pd.read_parquet(OUT/'eicu_demo_external/cohort.parquet')
    ref=from_json(P53/'control_sources.json')
    if sha(P53/'controls.npz')!=ref['controls_sha256']: raise ValueError('Frozen external scores changed')
    verify_sources(ref['source_sha256'])
    groups=list(cohort.groupby('uniquepid',sort=False).indices.values())
    external={}
    with np.load(P53/'controls.npz',allow_pickle=False) as data:
        for mode in ('hourly','native'):
            np.testing.assert_array_equal(data[f'eicu_{mode}_ids'],cohort.index.to_numpy())
            np.testing.assert_array_equal(data[f'eicu_{mode}_y'],cohort.y.to_numpy())
            external[mode]={tag:data[f'eicu_{mode}_{tag}'] for tag in ('frozen_full','p28','p33')}
    collected={mode:{tag:[] for tag in ('frozen_full','p28')} for mode in external}
    rng=np.random.default_rng(42)
    truth_all=cohort.y.to_numpy()
    for repeat in range(1000):
        idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        truth=truth_all[idx]
        if truth.min()==truth.max(): continue
        for mode,risks in external.items():
            computed={tag:metrics(truth,p[idx]) for tag,p in risks.items()}
            for tag in ('frozen_full','p28'): collected[mode][tag].append(computed['p33']-computed[tag])
    demo={mode:dict(metrics={tag:score_metrics(truth_all,p) for tag,p in risks.items()},
                    paired_ci={tag:bounds(values) for tag,values in collected[mode].items()},
                    delta={tag:{k:score_metrics(truth_all,risks['p33'])[k]-score_metrics(truth_all,risks[tag])[k]
                                for k in score_metrics(truth_all,risks[tag])} for tag in ('frozen_full','p28')})
          for mode,risks in external.items()}
    result=dict(model_fits=0,set_c=dict(stays=len(y),deaths=int(y.sum()),scenarios=rows),
                eicu_demo=dict(stays=len(cohort),deaths=int(cohort.y.sum()),modes=demo),
                bootstrap=dict(set_c=dict(replicates=2000,seed=20261013,unit='stay'),
                               eicu=dict(replicates=1000,seed=42,unit='patient')),
                source_sha256={str(p.relative_to(OUT)):sha(p) for p in (P55/'controls_manifest.json',P53/'control_sources.json')})
    for destination in (ROOT/'summary.json',ML/'reports/P58_joint_objectives_summary.json'):
        destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({case:{'delta_ap':r['delta']['ap'],'paired_ci':r['paired_ci']['ap'],
                            'retention_gain':r.get('retention_gain'),'retention_ci':r.get('retention_gain_ci')}
                      for case,r in rows.items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
