"""Audit study requirements against actual scores, source hashes and grouped OOF records."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ML=Path(__file__).resolve().parents[1]
OUT=ML/'artifacts'
REPORT=ML/'reports'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sources(manifest,base):
    for name,expected in manifest.get('source_sha256',{}).items():
        path=Path(name)
        if not path.is_absolute(): path=base/path
        if sha(path)!=expected: raise ValueError(f'Changed experimental source: {path}')


def privacy(value):
    banned={'ids','y','probabilities','patientunitstayid','uniquepid','RecordID','raw_files_sha256'}
    if isinstance(value,dict):
        if banned&set(value): raise ValueError('Patient-level field in public summary')
        for x in value.values(): privacy(x)
    elif isinstance(value,list):
        for x in value: privacy(x)
    elif isinstance(value,float):
        if not np.isfinite(value): raise ValueError('Nonfinite public statistic')


def main():
    names={55:'P55_kedgn_missingness_summary.json',56:'P56_observation_mixture_summary.json',
           57:'P57_demo_paired_summary.json',58:'P58_joint_objectives_summary.json',59:'P59_ma_alignment_summary.json'}
    summaries={key:read(REPORT/name) for key,name in names.items()}
    for key,d in summaries.items():
        privacy(d)
        sources(d,ML if key==56 else OUT)
    d=summaries[55]
    assert d['stays']==4000 and d['deaths']==585 and d['model_fits']==0
    assert len(d['scenarios'])==8 and len(d['metrics'])==72
    meta=read(OUT/'p55_kedgn_missingness/kedgn_manifest.json')
    controls_manifest=read(OUT/'p55_kedgn_missingness/controls_manifest.json')
    sources(controls_manifest,OUT)
    assert sha(OUT/'p55_kedgn_missingness/controls.npz')==controls_manifest['controls_sha256']
    for file,digest in meta['predictions'].items():
        assert sha(OUT/'p55_kedgn_missingness'/file)==digest
    for file,digest in meta['model_sources_sha256'].items(): assert sha(file)==digest
    with np.load(OUT/'p55_kedgn_missingness/controls.npz') as data:
        y=data['y']
        ids=data['ids']
        for i,case in enumerate(data['scenarios']):
            for j,model in enumerate(data['models']):
                row,=[r for r in d['metrics'] if r['scenario']==case and r['model']==model]
                np.testing.assert_allclose(average_precision_score(y,data['predictions'][i,j]),row['ap'],atol=1e-14,rtol=0)
            for version in ('set_a','ab'):
                with np.load(OUT/f'p55_kedgn_missingness/{version}_{case}.npz') as kg:
                    np.testing.assert_array_equal(kg['ids'],ids)
                    np.testing.assert_array_equal(kg['y'],y)
                    assert kg['probabilities'].shape==(5,4000)
                    value=average_precision_score(y,kg['probabilities'].mean(axis=0,dtype=np.float64))
                row,=[r for r in d['metrics'] if r['scenario']==case and r['model']==f'kedgn_{version}']
                np.testing.assert_allclose(value,row['ap'],atol=1e-14,rtol=0)
    kg_root=OUT/'related_work/KEDGN/ab_dense_gpu_20261121'
    frozen=read(kg_root/'frozen.json')
    for seed,receipt in frozen['seeds'].items():
        for file,key in [('weights.pt','weights_sha256'),('statistics.npz','statistics_sha256')]:
            assert sha(kg_root/f'final_seed_{seed}'/file)==receipt[key]
    d=summaries[56]
    assert d['cnn_replay_fits']==3 and not d['test_data_used'] and not d['accepted']
    with np.load(OUT/'p56_observation_mixture/oof.npz') as data:
        y,folds=data['y'],data['folds']
        assert len(y)==8000 and int(y.sum())==1122
        values=.75*data['anchor']+.25*data['sequence']
        views=read(OUT/'p33_hourly_sequence/inputs/manifest.json')['views']
        for key in ('clean','full_gcs'):
            p=values[views.index(key)]
            value=np.mean([average_precision_score(y[folds==f],p[folds==f]) for f in range(3)])
            np.testing.assert_allclose(value,d['baseline'][key],atol=1e-12,rtol=0)
    cohort=pd.read_parquet(OUT/'p57_demo_paired_completion/cohort.parquet')
    assignment=np.full(len(cohort),-1)
    for fold in range(5):
        with np.load(OUT/f'p57_demo_paired_completion/fold_{fold}.npz') as data:
            positions=cohort.index.get_indexer(data['ids'])
            assert (positions>=0).all() and (assignment[positions]==-1).all()
            assignment[positions]=fold
    assert (assignment>=0).all()
    grouped=cohort.assign(validation_fold=assignment)
    for key in ('uniquepid','hospitalid'):
        assert grouped.groupby(key).validation_fold.nunique().max()==1
    assert summaries[57]['model_replay_fits']==5 and not summaries[57]['selected_new_parameters']
    for mode,row in summaries[57]['results'].items():
        assert row['gcs_strata']['missing']['metrics']['p49']==row['gcs_strata']['missing']['metrics']['p50']
        assert row['gcs_strata']['missing']['metrics']['p49']==row['gcs_strata']['missing']['metrics']['p52']
    joint=summaries[58]['set_c']['scenarios']
    assert all(r['paired_ci']['ap'][0]>0 for r in joint.values())
    assert all(joint[k]['retention_gain_ci'][0]>0 for k in ('full_GCS','full_all_six'))
    assert all(r['paired_ci']['frozen_full']['ap'][0]<0<r['paired_ci']['frozen_full']['ap'][1]
               for r in summaries[58]['eicu_demo']['modes'].values())
    assert summaries[59]['model_fits']==0 and len(summaries[59]['eicu'])==12 and len(summaries[59]['set_c'])==18
    provenance=read(REPORT/'資料來源核對.json')
    assert len(provenance['verified_files'])==12 and all(r['verified'] for r in provenance['verified_files'])
    for file in provenance['verified_files']:
        assert sha(ML/'dataset/EICU/eicu-crd-demo-2.0.1'/file['file'])==file['expected_sha256']
    evidence={
      'unmasked_accuracy':dict(experiments_complete=True,effect='supported on fixed Set C',evidence=['P33','P54','P58']),
      'missingness_robustness':dict(experiments_complete=True,effect='supported for prespecified Set C channel masks',evidence=['P9-P20','P55','P58']),
      'cross_dataset_transfer':dict(experiments_complete=True,effect='AP improvement remains uncertain on Demo',evidence=['P45','P52','P54','P57','P58','P59']),
      'same_primary_model':dict(experiments_complete=True,model='P33',selection='P56 retained original mixture on A/B',evidence=['P56','P58']),
      'external_augmentation':dict(experiments_complete=True,setting='patient/site-key-block OOF, separate from P12-only transfer',evidence=['P44-P45','P50','P57']),
      'literature_controls':dict(experiments_complete=True,reproduced=['MA-GBT','KEDGN'],other_papers='literature context; cancelled schedules not relabelled completed'),
      '24h_extension':dict(experiments_complete=True,setting='separate risk-set proxy and APACHE shared cohort',evidence=['P49-P52','P57']),
      'data_provenance':dict(experiments_complete=True,eicu_tables_verified=12,evidence=['資料來源核對.json']),
    }
    result=dict(scope='Experiments and research narrative only; Demo-only data confirmed by user on 2026-11-22',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                experiments_complete=True,independent_full_eicu_available=False,
                external_ap_improvement_confirmed=False,evidence=evidence,
                verified_summary_sha256={name:sha(REPORT/name) for name in names.values()},
                source_checks='Frozen weights/statistics, OOF reconstruction, held-out grouping, original metrics and publisher checksums',
                new_replay_fits=dict(p56_cnn=3,p57_external_fold=5),
                no_patient_data_in_public_summaries=True,
                limitations=['Repeatedly viewed Set C and Demo','Fixed-model bootstrap excludes refitting uncertainty',
                             'Unknown physical interpretation of released hospital keys','P12 24h outcome-time proxy',
                             'Architecture/representation/capacity differences across literature controls'])
    privacy(result)
    path=REPORT/'研究完成驗收.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified current experimental evidence and grouped OOF coverage. External AP gain remains unconfirmed.',flush=True)


if __name__=='__main__': main()
