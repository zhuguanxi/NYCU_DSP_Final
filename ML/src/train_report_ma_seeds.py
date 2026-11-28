"""Complete six fixed Full MA/control families at seeds43-46, CPU, A/B only."""
import os
os.environ['NUMBA_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import argparse
import joblib
import numpy as np
from malearn.estimators.classification import MAGBTClassifier
from report_seed_common import ML,ROOT,P62,SEEDS,MA_CONFIGS,read,write,sha,relative,protocol,receipt_valid
from train_ma_matched import data,augmented,PARAMS,trees_equal
from ma_gbt_p12_inputs import encode
from ma_matched_parallel import parallel_splits
from ma_matched_runtime import reliance
from evaluate_ma_matched_fast import compiled_routes
from run_ma_gbt_p12 import validate_upstream

def check():
 protocol();validate_upstream()
 frame,y,_,schemas=data();frame=frame.iloc[:384];y=y[:384];checks=[]
 for name,aug,alpha,_ in MA_CONFIGS:
  raw,target,weights=augmented(frame,y,aug,'final');x,m=encode(raw,schemas['Full']['final'])
  opts={**PARAMS,'random_state':43,'n_estimators':3,'alpha':1. if alpha is None else alpha}
  mask=None if alpha is None else m
  with parallel_splits():reference=MAGBTClassifier(**opts).fit(x,target,M=mask,sample_weight=weights)
  with parallel_splits(),compiled_routes():actual=MAGBTClassifier(**opts).fit(x,target,M=mask,sample_weight=weights)
  trees_equal(reference,actual)
  np.testing.assert_array_equal(reference.predict_proba(x),actual.predict_proba(x))
  checks.append(dict(model=name,tree_and_predictions='exact'))
  print('Exact compiled-training check:',name,flush=True)
 write(ROOT/'ma_numeric_check.json',dict(AB_only=True,diagnostic_fits=12,trees_per_fit=3,checks=checks,
  protocol_sha256=sha(ROOT/'protocol.json'),formal_fits=0))

def job(item):
 seed,name,aug,alpha=item;path=ROOT/'ma_jobs'/f'{name}_{seed}.json'
 if path.exists():return receipt_valid(path)
 protocol();frame,y,_,schemas=data();xraw,target,weights=augmented(frame,y,aug,'final')
 x,m=encode(xraw,schemas['Full']['final'])
 assert weights.sum()==8000 and weights[target==1].sum()==1122
 destination=ROOT/'models'/f'seed_{seed}'/f'{name}.joblib';destination.parent.mkdir(parents=True,exist_ok=True)
 attempt=path.with_name(path.stem+'.attempt.json');count=read(attempt)['attempts']+1 if attempt.exists() else 1
 write(attempt,dict(seed=seed,model=name,attempts=count,training_stays=8000,protocol_sha256=sha(ROOT/'protocol.json')))
 print('FIT MA',name,seed,len(x),'rows',flush=True);start=time.monotonic()
 model=MAGBTClassifier(**{**PARAMS,'random_state':seed,'alpha':1. if alpha is None else alpha})
 with parallel_splits(),compiled_routes():
  model.fit(x,target,M=None if alpha is None else m,sample_weight=weights)
  p=model.predict_proba(x)[:,1]
 assert model.random_state==seed and len(model.estimators_)==300
 temp=destination.with_suffix('.joblib.tmp');joblib.dump(model,temp);temp.replace(destination)
 value=dict(model=name,seed=seed,alpha=alpha,augmentation=aug,training_stays=8000,deaths=1122,
  training_rows=len(x),total_weight=float(weights.sum()),positive_weight=float(weights[target==1].sum()),
  file=relative(destination),model_sha256=sha(destination),protocol_sha256=sha(ROOT/'protocol.json'),
  training_probability_sha256=__import__('hashlib').sha256(p.tobytes()).hexdigest(),
  seconds=time.monotonic()-start,device='cpu',new_fit=True,trees=300,max_depth=3)
 write(path,value);print('DONE MA',name,seed,round(value['seconds'],1),flush=True);return value

def train(workers):
 protocol();validate_upstream();check=read(ROOT/'ma_numeric_check.json')
 assert check['protocol_sha256']==sha(ROOT/'protocol.json') and len(check['checks'])==6
 previous=read(P62/'frozen.json');records=[]
 for name,aug,alpha,key in MA_CONFIGS:
  saved=next(r for r in previous['fits'] if r['key']==key);path=P62/'models'/f'{key}.joblib'
  assert sha(path)==saved['model_sha256']
  records.append(dict(model=name,seed=42,alpha=alpha,augmentation=aug,file=relative(path),
   model_sha256=sha(path),new_fit=False,device='cpu',trees=300,max_depth=3))
 jobs=[(seed,name,aug,alpha) for name,aug,alpha,_ in MA_CONFIGS for seed in SEEDS[1:]]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(job,item) for item in jobs]
  for future in as_completed(futures):records.append(future.result())
 protocol();assert len(records)==30
 write(ROOT/'ma_frozen.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),records=sorted(records,key=lambda r:(r['model'],r['seed'])),
  new_fits=24,diagnostic_fits=12,diagnostic_check_sha256=sha(ROOT/'ma_numeric_check.json'),test_used=False))
 print('FROZEN all six MA families x five seeds; 24 new final fits.',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('check','train'));parser.add_argument('--workers',type=int,default=6)
 args=parser.parse_args()
 if args.stage=='check':check()
 else:train(args.workers)
