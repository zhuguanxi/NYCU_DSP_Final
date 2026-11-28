"""Preserve imported R1 models and fit fixed CPU configurations at seeds43-46."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import time
import hashlib
from importlib.metadata import version
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from report_seed_common import ML,ROOT,R1,SEEDS,R1_CONFIGS,R1_TAGS,initialize,protocol,read,write,sha,relative,receipt_valid
from r1_report_features import EVENTS,CHANNELS,FEATURES,summarize,global_means,transform,check_formulas

def main():
 initialize();check_formulas();old=read(R1/'frozen.json')
 for name,expected in old['versions'].items():assert version(name)==expected,(name,version(name),expected)
 labels=pd.read_csv(ML/'artifacts/labels_AB.csv').set_index('RecordID');y=labels.In_hospital_death
 assert len(y)==8000 and int(y.sum())==1122 and labels.source_set.value_counts().to_dict()=={'A':4000,'B':4000}
 cache=ROOT/'training_inputs';cache.mkdir(exist_ok=True)
 path=cache/'r1_AB.parquet';meanpath=cache/'r1_AB_metadata.json'
 if path.exists():
  meta=read(meanpath);assert meta['protocol_sha256']==sha(ROOT/'protocol.json') and meta['features_sha256']==sha(path)
  features=pd.read_parquet(path);means=meta['means']
 else:
  events=pd.read_parquet(ML/'artifacts/observations_AB.parquet',columns=[*EVENTS,'quality_flag'])
  events=events.loc[events.quality_flag.eq('valid')&events.variable.isin(CHANNELS),EVENTS]
  features,totals,audit=summarize(events,y.index);means=global_means(totals,y.index)
  assert audit==old['observations']
  np.testing.assert_array_equal(np.asarray(list(means.values())),np.asarray([old['imputation_means'][k] for k in means]))
  np.testing.assert_array_equal(totals['count'].to_numpy(),pd.read_parquet(ML/'artifacts/features_AB.parquet',columns=[f'{k}__count' for k in CHANNELS]).to_numpy())
  features.to_parquet(path)
  write(meanpath,dict(protocol_sha256=sha(ROOT/'protocol.json'),features_sha256=sha(path),means=means,audit=audit))
  del events,totals
 x=transform(features.astype('float64'),means);assert list(x.columns)==old['feature_schema']==FEATURES
 records=[]
 for name,params in R1_CONFIGS.items():
  prior=old['final_models'][R1_TAGS[name]];original=R1/prior['file'];assert sha(original)==prior['sha256']
  records.append(dict(model=name,seed=42,file=relative(original),model_sha256=sha(original),new_fit=False,device='cpu'))
  baseline=XGBClassifier();baseline.load_model(original);original_p=baseline.predict_proba(x)[:,1]
  for seed in SEEDS[1:]:
   receipt=ROOT/'r1_jobs'/f'{name}_{seed}.json'
   if receipt.exists():records.append(receipt_valid(receipt));continue
   params2={**params,'tree_method':'hist','device':'cpu','n_jobs':4,'random_state':seed,
    'objective':'binary:logistic','eval_metric':'logloss'}
   write(receipt.with_name(receipt.stem+'.attempt.json'),dict(model=name,seed=seed,training_stays=8000,attempts=1))
   model=XGBClassifier(**params2);start=time.monotonic();model.fit(x,y)
   destination=ROOT/'models'/f'seed_{seed}'/f'{name}.ubj';destination.parent.mkdir(parents=True,exist_ok=True)
   model.save_model(destination);p=model.predict_proba(x)[:,1]
   value=dict(model=name,seed=seed,parameters=params2,training_stays=8000,deaths=1122,new_fit=True,device='cpu',
    file=relative(destination),model_sha256=sha(destination),protocol_sha256=sha(ROOT/'protocol.json'),
    training_probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
    training_probabilities_equal_seed42=bool(np.array_equal(p,original_p)),seconds=time.monotonic()-start)
   write(receipt,value);records.append(value)
   print('DONE R1',name,seed,round(value['seconds'],2),'seconds',flush=True)
 protocol();assert len(records)==10
 write(ROOT/'r1_frozen.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),records=records,new_fits=8,
  input_features_sha256=sha(path),input_metadata_sha256=sha(meanpath),test_used=False))
 print('FROZEN both R1 configurations x five seeds; eight new final fits.',flush=True)

if __name__=='__main__':main()
