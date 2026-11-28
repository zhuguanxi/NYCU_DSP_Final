"""Evaluate all frozen MA seeds using exactly equivalent compiled routing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import joblib
import numpy as np
import pandas as pd
from ma_matched_runtime import njit,reliance
from evaluate_ma_matched_fast import compiled_routes
from report_seed_common import *
from freeze_report_seed_study import validate_frozen
from train_ma_matched import data
from ma_gbt_p12_inputs import encode,SCHEMA_FILE

@njit(cache=True,fastmath=False)
def mark_hits(x,m,left,right,feature,threshold,hits):
 for i in range(len(x)):
  if hits[i]:continue
  node=0
  while left[node]!=-1:
   f=feature[node]
   if m[i,f]!=0:hits[i]=True;break
   node=left[node] if x[i,f]<=threshold[node] else right[node]

def compiled_reliance(model,x,m):
 hits=np.zeros(len(x),dtype=np.bool_)
 for tree in model.estimators_.ravel():
  t=tree.tree_;mark_hits(x,m,t.children_left,t.children_right,t.feature,t.threshold,hits)
 return float(hits.mean())

def main():
 frozen=validate_frozen();sources={relative(Path(__file__)):sha(Path(__file__))}
 sources[relative(ML/'src/freeze_report_seed_study.py')]=sha(ML/'src/freeze_report_seed_study.py')
 schema=read(SCHEMA_FILE)['schemas']['Full']['final']
 records=read(ROOT/'ma_frozen.json')['records'];models={}
 frame,_,_,_=data();probe,m=encode(frame.iloc[:384],schema)
 random_m=np.random.default_rng(63).integers(0,2,size=m.shape,dtype=np.int8)
 for record in records:
  model=joblib.load(ML/record['file']);assert model.random_state==record['seed'] and len(model.estimators_)==300
  assert model.max_depth==3 and model.learning_rate==.05
  for mask in (m,random_m):assert compiled_reliance(model,probe,mask)==reliance(model,probe,mask)
  models[record['seed'],record['model']]=model
 write(ROOT/'ma_reliance_check.json',dict(AB_only=True,models=30,original_and_random_masks='exact',new_fits=0,source_sha256=sources))
 output=ROOT/'ma_evaluation';output.mkdir(exist_ok=True)
 manifest={};legacy=read(P60/'evaluation.json')['cases']
 for case in CASES:
  destination=output/f'{case}.npz';meta=output/f'{case}.json'
  if destination.exists():
   prior=read(meta);assert prior['frozen_sha256']==sha(ROOT/'frozen.json') and prior['source_sha256']==sources
   assert sha(destination)==prior['predictions_sha256'];manifest[case]=prior;continue
  source=ML/legacy[case]['file'];assert sha(source)==legacy[case]['sha256']
  with np.load(source) as reference:ids=reference['ids'].copy();y=reference['y'].copy()
  path,_=case_paths(case);frame=pd.read_parquet(path);np.testing.assert_array_equal(frame.index,ids)
  x,m=encode(frame,schema);names=[r[0] for r in MA_CONFIGS]
  probabilities=np.empty((5,6,len(y)));rhos=np.empty((5,6))
  with compiled_routes():
   for i,seed in enumerate(SEEDS):
    for j,name in enumerate(names):
     model=models[seed,name];probabilities[i,j]=model.predict_proba(x)[:,1]
     rhos[i,j]=compiled_reliance(model,x,m)
  with np.load(P62/'evaluation'/f'{case}.npz') as previous:
   np.testing.assert_array_equal(previous['ids'],ids);np.testing.assert_array_equal(previous['y'],y)
   oldrho=read(P62/'evaluation'/f'{case}.json')['rho']
   for j,(name,aug,_,_) in enumerate(MA_CONFIGS):
    tag=f'Full_{"gcs60" if aug else "plain"}_{name.rsplit("_",1)[1]}'
    np.testing.assert_array_equal(probabilities[0,j],previous['predictions'][list(previous['models']).index(tag)])
    assert rhos[0,j]==oldrho[tag]
  np.savez_compressed(destination,ids=ids,y=y,seeds=np.array(SEEDS),models=np.array(names),predictions=probabilities,rho=rhos)
  prior=dict(frozen_sha256=sha(ROOT/'frozen.json'),predictions_sha256=sha(destination),source_sha256=sources,
   input_sha256={relative(source):sha(source),relative(path):sha(path)},seed42_matches_P62_exactly=True)
  write(meta,prior);manifest[case]=prior;print('MA evaluation complete:',case,flush=True)
 validate_frozen();write(output/'manifest.json',dict(cases=manifest,new_model_fits=0,
  frozen_sha256=sha(ROOT/'frozen.json'),reliance_check_sha256=sha(ROOT/'ma_reliance_check.json')))
 print('All 150 MA model/cohort evaluations completed.',flush=True)

if __name__=='__main__':main()
