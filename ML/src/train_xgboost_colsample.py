"""P65: change only per-tree feature sampling for Full and R1; A/B final fits only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,sys,time
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from xgboost import XGBClassifier
from r1_report_features import transform

ML=Path(__file__).resolve().parents[1]
ROOT=ML/'artifacts/p65_xgboost_colsample_09'
P63=ML/'artifacts/p63_report_five_seed'
SEEDS=(42,43,44,45,46)
MODELS=('full_col90','r1_paper_col90')
BASELINE={'full_col90':'frozen_full','r1_paper_col90':'r1_paper'}
R1_PARAMS=dict(n_estimators=100,max_depth=5,learning_rate=.1,tree_method='hist',device='cpu',
    n_jobs=4,objective='binary:logistic',eval_metric='logloss')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    pending.replace(path)
def relative(path):return Path(path).resolve().relative_to(ML).as_posix()
def canonical(value):
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    return value
def verify_sources(records):
    for path,digest in records.items():assert sha(ML/path)==digest,path
def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ML/'src/r1_report_features.py',ML/'src/prepare.py',
        ML/'reports/P65_colsample_protocol.md',ML/'artifacts/features_AB.parquet',ML/'artifacts/labels_AB.csv',
        ML/'artifacts/p60_five_seed_stability/frozen.json',ML/'artifacts/p60_five_seed_stability/seed_42/Full.joblib',
        P63/'training_inputs/r1_AB.parquet',P63/'training_inputs/r1_AB_metadata.json',
        ML/'related_work/R1_baseline/frozen.json']
    template,columns=joblib.load(ML/'artifacts/p60_five_seed_stability/seed_42/Full.joblib')
    params=template['model'].get_params()
    assert params['n_estimators']==300 and params['max_depth']==3 and params['learning_rate']==.05
    value=dict(stage='P65 fixed per-tree feature sampling',date='2026-11-28',seeds=list(SEEDS),models=list(MODELS),
        colsample_bytree=.9,other_parameters_fixed=True,row_subsampling=1.,training_only='P12 A+B 8000 stays /1122 deaths',
        parameter_search=False,selected_seed=None,test_selected=False,feature_schema_changed=False,
        training_device='CPU; preserve original hist implementation',
        Full_template_parameters=canonical(params),Full_input_columns=len(columns),R1_parameters=R1_PARAMS,R1_input_columns=148,
        augmentation='none; same as original Full and R1 baseline',new_formal_fits=10,
        summary='five individual metric values, arithmetic mean and sample SD ddof1; no cross-seed probability averaging',
        versions={k:version(k) for k in ('xgboost','scikit-learn','numpy','pandas','joblib')},python=sys.version,
        source_sha256={relative(p):sha(p) for p in files})
    path=ROOT/'protocol.json'
    if path.exists():assert read(path)==value
    else:write(path,value)
    verify_sources(value['source_sha256']);return value

def validate_frozen():
    protocol=read(ROOT/'protocol.json');verify_sources(protocol['source_sha256'])
    frozen=read(ROOT/'frozen.json');assert frozen['protocol_sha256']==sha(ROOT/'protocol.json')
    assert frozen['new_fits']==10 and not frozen['test_used'] and len(frozen['records'])==10
    assert {(r['model'],r['seed']) for r in frozen['records']}=={(n,s) for n in MODELS for s in SEEDS}
    for row in frozen['records']:assert sha(ML/row['file'])==row['weights_sha256']
    return frozen

def main():
    initialize()
    if (ROOT/'frozen.json').exists():validate_frozen();print('Retained ten verified frozen col90 fits.',flush=True);return
    x=pd.read_parquet(ML/'artifacts/features_AB.parquet')
    labels=pd.read_csv(ML/'artifacts/labels_AB.csv').set_index('RecordID').loc[x.index]
    y=labels.In_hospital_death;assert len(y)==8000 and int(y.sum())==1122
    assert labels.source_set.value_counts().to_dict()=={'A':4000,'B':4000}
    template,columns=joblib.load(ML/'artifacts/p60_five_seed_stability/seed_42/Full.joblib')
    reference=template['pre'].transform(x.loc[:,columns])
    r1meta=read(P63/'training_inputs/r1_AB_metadata.json')
    assert sha(P63/'training_inputs/r1_AB.parquet')==r1meta['features_sha256']
    r1raw=pd.read_parquet(P63/'training_inputs/r1_AB.parquet')
    np.testing.assert_array_equal(r1raw.index,x.index)
    r1x=transform(r1raw.astype('float64'),r1meta['means']);assert r1x.shape==(8000,148)
    records=[]
    for seed in SEEDS:
        for name in MODELS:
            path=ROOT/'jobs'/f'{name}_{seed}.json'
            if path.exists():
                record=read(path);assert record['protocol_sha256']==sha(ROOT/'protocol.json')
                assert sha(ML/record['file'])==record['weights_sha256'];records.append(record);continue
            started=time.monotonic()
            if name=='full_col90':
                model=clone(template).set_params(model__random_state=seed,model__colsample_bytree=.9)
                model.fit(x.loc[:,columns],y);actual=model['model']
                np.testing.assert_array_equal(model['pre'].transform(x.loc[:,columns]),reference)
                expected={**template['model'].get_params(),'random_state':seed,'colsample_bytree':.9}
                assert canonical(actual.get_params())==canonical(expected)
                saved=(model,columns)
            else:
                actual=XGBClassifier(**R1_PARAMS,random_state=seed,colsample_bytree=.9)
                actual.fit(r1x,y);saved=actual
            config=json.loads(actual.get_booster().save_config())['learner']
            params=config['gradient_booster']['tree_train_param']
            assert abs(float(params['colsample_bytree'])-.9)<1e-6
            assert all(float(params[k])==1 for k in ('colsample_bylevel','colsample_bynode','subsample'))
            assert int(config['generic_param']['seed'])==seed
            trees=actual.get_booster().get_dump(dump_format='json')
            assert len(trees)==(300 if name=='full_col90' else 100)
            weights=ROOT/'models'/f'{name}_{seed}.joblib';weights.parent.mkdir(exist_ok=True)
            joblib.dump(saved,weights)
            record=dict(model=name,seed=seed,new_fit=True,training_stays=8000,deaths=1122,
                colsample_bytree=.9,subsample=1.,actual_training_seed=seed,trees=len(trees),
                tree_structure_sha256=hashlib.sha256('\n'.join(trees).encode()).hexdigest(),
                file=relative(weights),weights_sha256=sha(weights),protocol_sha256=sha(ROOT/'protocol.json'),
                preprocessing_identical=True,seconds=time.monotonic()-started)
            write(path,record);records.append(record)
            print(json.dumps({'completed':len(records),'total':10,**{k:record[k] for k in ('model','seed','seconds')}}),flush=True)
    verify_sources(read(ROOT/'protocol.json')['source_sha256'])
    write(ROOT/'frozen.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),records=records,new_fits=10,
        test_used=False,all_weights_frozen_before_new_test_predictions=True))
    validate_frozen();print('Frozen both col90 models x five seeds; A/B-only, no parameter search.',flush=True)

if __name__=='__main__':main()
