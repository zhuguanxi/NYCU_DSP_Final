"""Common five-cohort evaluation; individual seeds, means and ensembles stay distinct."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from report_seed_common import *
from freeze_report_seed_study import validate_frozen
from r1_report_features import EVENTS,CHANNELS,summarize,p12_c_events,transform
from stress_missingness_c import score_metrics
from evaluate_temporal_features import TemporalPredictor
from eicu_demo_features import DEMO,INPUT_TABLES,STAY,build_features
from evaluate_eicu_demo import source_hashes
from evaluate_eicu_urine_mapping import urine_events,urine_values
from prepare import DATA

def r1_inputs():
 old=read(R1/'results/evaluation.json');folder=ROOT/'evaluation_inputs';folder.mkdir(exist_ok=True)
 metadata=folder/'r1_inputs.json'
 if metadata.exists():
  meta=read(metadata);assert meta['frozen_sha256']==sha(ROOT/'frozen.json');verify_sources(meta['files'])
  verify_sources(meta['raw_sources']);return {k:pd.read_parquet(ML/v) for k,v in meta['features'].items()},meta
 sources={};features={};audits={}
 labels=pd.read_csv(ML/'artifacts/C_evaluation/labels_C.csv').set_index('RecordID').In_hospital_death
 archive=DATA/'set-c.tar.gz';assert sha(archive)==old['input_sources']['archive_c']['sha256']
 sources[os.path.relpath(archive,ML).replace(os.sep,'/')]=sha(archive)
 frame,counts,audit=summarize(p12_c_events(archive,labels.index),labels.index)
 raw=pd.read_parquet(ML/'artifacts/C_evaluation/features_C.parquet')
 np.testing.assert_array_equal(counts['count'].to_numpy(),raw[[f'{k}__count' for k in CHANNELS]].to_numpy())
 assert audit==old['c_input_audit'];features['clean']=frame;audits['clean']=audit
 print('R1 Set C features reconstructed and original counts verified.',flush=True)
 hashes=source_hashes(DEMO)
 for table in INPUT_TABLES:
  filename=f'{table}.csv.gz';assert hashes[filename]==old['input_sources'][f'demo_{table}']['sha256']
  sources[relative(DEMO/filename)]=hashes[filename]
 events=[]
 def collect(frame,index):
  events.append(frame.rename(columns={STAY:'RecordID','minute':'elapsed_minute','value':'clean_value'}).loc[:,EVENTS])
  return pd.DataFrame(index=index)
 cohort,frames,_,_=build_features(DEMO,collect)
 assert len(cohort)==782 and int(cohort.y.sum())==91
 urine,_=urine_events(cohort.index,directory=DEMO);urine_summary=urine_values(cohort.index,urine)
 urine_input=urine.rename(columns={STAY:'RecordID','minute':'elapsed_minute','value':'clean_value'}).loc[:,EVENTS]
 for (mode,original),event in zip(frames.items(),events,strict=True):
  original.loc[:,urine_summary.columns]=urine_summary
  weight=original.static_Weight.dropna().rename('clean_value').rename_axis('RecordID').reset_index().assign(elapsed_minute=0,variable='Weight')
  event=pd.concat([event,urine_input,weight.loc[:,EVENTS]],ignore_index=True)
  frame,counts,audit=summarize(event,cohort.index);case=f'eicu_{mode}'
  assert audit==old['demo_input_audit'][case]
  np.testing.assert_array_equal(counts['count'].to_numpy(),original[[f'{k}__count' for k in CHANNELS]].to_numpy())
  features[case]=frame;audits[case]=audit
  print('R1 Demo features reconstructed:',mode,flush=True)
 files={};locations={}
 for case,frame in features.items():
  path=folder/f'r1_{case}.parquet';frame.to_parquet(path);locations[case]=relative(path);files[relative(path)]=sha(path)
 meta=dict(frozen_sha256=sha(ROOT/'frozen.json'),features=locations,files=files,raw_sources=sources,audits=audits)
 write(metadata,meta);return features,meta

def main():
 frozen=validate_frozen();ma=read(ROOT/'ma_evaluation/manifest.json')
 assert ma['frozen_sha256']==sha(ROOT/'frozen.json')
 metadata=read(P60/'evaluation.json')['cases'];outputs=ROOT/'evaluation';outputs.mkdir(exist_ok=True)
 source_files=(Path(__file__),ML/'src/r1_report_features.py',ML/'src/evaluate_temporal_features.py',
  ML/'src/eicu_demo_features.py',ML/'src/evaluate_eicu_urine_mapping.py',ML/'src/freeze_report_seed_study.py')
 sources={relative(p):sha(p) for p in source_files}
 r1features,input_meta=r1_inputs();means=read(ROOT/'training_inputs/r1_AB_metadata.json')['means']
 r1models={}
 for record in read(ROOT/'r1_frozen.json')['records']:
  model=XGBClassifier(random_state=record['seed'],device='cpu',n_jobs=4);model.load_model(ML/record['file'])
  r1models[record['model'],record['seed']]=model
 temporal={seed:joblib.load(P60/f'seed_{seed}/temporal.joblib') for seed in SEEDS}
 rounds=read(ML/'artifacts/p22_temporal_ab_pilot/fitted/manifest.json')['selected_rounds']
 assert rounds==650 and all(t.get_booster().num_boosted_rounds()==650 for t in temporal.values())
 p22_old=read(ML/'artifacts/p22_temporal_ab_pilot/evaluation/summary.json')['set_c']
 p24_old=read(ML/'artifacts/p24_eicu_temporal_urine/summary.json')['rows']
 rows=[];saved={};metrics=('ap','auroc','brier','sensitivity')
 for case in CASES:
  target=outputs/f'{case}.npz';metapath=outputs/f'{case}.json'
  if target.exists():
   prior=read(metapath);assert prior['frozen_sha256']==sha(ROOT/'frozen.json') and prior['source_sha256']==sources
   assert prior['predictions_sha256']==sha(target);rows.extend(prior['rows']);saved[case]=prior;continue
  reference=ML/metadata[case]['file'];assert sha(reference)==metadata[case]['sha256']
  with np.load(reference) as old:
   ids=old['ids'].copy();truth=old['y'].copy();oldnames=list(old['models']);oldp=old['predictions'].copy()
  expected=(782,91) if case.startswith('eicu') else (4000,585)
  assert (len(truth),int(truth.sum()))==expected
  framepath,timepath=case_paths(case);frame=pd.read_parquet(framepath);timing=pd.read_parquet(timepath)
  np.testing.assert_array_equal(frame.index,ids);np.testing.assert_array_equal(timing.index,ids)
  predictions=np.empty((len(MODELS),5,len(truth)));rhos=np.full((len(MODELS),5),np.nan)
  for name in LEGACY:predictions[MODELS.index(name)]=oldp[:,oldnames.index(name)]
  joined=frame.join(timing)
  for i,seed in enumerate(SEEDS):predictions[MODELS.index('p22'),i]=TemporalPredictor(temporal[seed],rounds).predict_proba(joined)[:,1]
  kgpath=KG/f'{case}_predictions.npz' if case.startswith('eicu') else ML/f'artifacts/p55_kedgn_missingness/ab_{dict(clean="clean",gcs_full="full_GCS",six_full="full_all_six")[case]}.npz'
  if case.startswith('eicu'):assert sha(kgpath)==read(KG/'inference.json')['predictions'][case]
  else:assert sha(kgpath)==read(ML/'artifacts/p55_kedgn_missingness/kedgn_manifest.json')['predictions'][kgpath.name]
  with np.load(kgpath) as kg:
   np.testing.assert_array_equal(kg['ids'],ids);np.testing.assert_array_equal(kg['y'],truth)
   predictions[MODELS.index('kedgn_ab')]=kg['probabilities']
  masked=() if case=='clean' or case.startswith('eicu') else ('GCS',) if case=='gcs_full' else ('GCS','Lactate','MAP','Mg','Urine','MechVent')
  features=r1features[case if case.startswith('eicu') else 'clean']
  x=transform(features.astype('float64'),means,masked);np.testing.assert_array_equal(x.index,ids)
  original_r1=pd.read_csv(R1/f'artifacts/predictions_{case}.csv',float_precision='round_trip')
  np.testing.assert_array_equal(original_r1.iloc[:,0],ids);np.testing.assert_array_equal(original_r1.y,truth)
  for name in R1_CONFIGS:
   for i,seed in enumerate(SEEDS):predictions[MODELS.index(name),i]=r1models[name,seed].predict_proba(x)[:,1]
   np.testing.assert_array_equal(predictions[MODELS.index(name),0],original_r1[R1_TAGS[name]])
  mafile=ROOT/'ma_evaluation'/f'{case}.npz';assert sha(mafile)==ma['cases'][case]['predictions_sha256']
  with np.load(mafile) as pack:
   np.testing.assert_array_equal(pack['ids'],ids);np.testing.assert_array_equal(pack['y'],truth)
   for j,name in enumerate(pack['models']):predictions[MODELS.index(name)]=pack['predictions'][:,j];rhos[MODELS.index(name)]=pack['rho'][:,j]
  assert np.isfinite(predictions).all() and ((predictions>=0)&(predictions<=1)).all()
  # Historical P22 results are seed42; Brier may have float32/float64 rounding.
  oldcase={'clean':'clean','gcs_full':'full_GCS','six_full':'full_all_six'}.get(case)
  prior=(next(r for r in p22_old if r['scenario']==oldcase and r['model']=='temporal_selected') if oldcase else
   next(r['temporal_mapped'] for r in p24_old if r['mode']==case.removeprefix('eicu_')))
  measured=score_metrics(truth,predictions[MODELS.index('p22'),0])
  np.testing.assert_allclose([measured[k] for k in metrics],[prior[k] for k in metrics],rtol=0,atol=1e-8)
  seeds=np.asarray([range(5) if name=='kedgn_ab' else SEEDS for name in MODELS])
  current=[]
  for j,name in enumerate(MODELS):
   for i,seed in enumerate(seeds[j]):
    current.append(dict(cohort=case,model=name,seed=int(seed),stays=len(truth),deaths=int(truth.sum()),
     **score_metrics(truth,predictions[j,i]),rho=float(rhos[j,i]) if np.isfinite(rhos[j,i]) else None))
  np.savez_compressed(target,ids=ids,y=truth,models=np.asarray(MODELS),seeds=seeds,predictions=predictions,rho=rhos)
  prior=dict(frozen_sha256=sha(ROOT/'frozen.json'),predictions_sha256=sha(target),source_sha256=sources,rows=current,
   seed42_r1_exact=True,seed42_p22_metrics_match=True,
   input_sha256={relative(p):sha(p) for p in (reference,framepath,timepath,kgpath,mafile)})
  write(metapath,prior);saved[case]=prior;rows.extend(current)
  print('COMMON completed all 16 models x five seeds:',case,flush=True)
 validate_frozen();verify_sources(sources);verify_sources(input_meta['files']);verify_sources(input_meta['raw_sources'])
 assert len(rows)==400
 write(outputs/'summary.json',dict(complete=True,rows=rows,cases=saved,new_model_fits=0,
  frozen_sha256=sha(ROOT/'frozen.json'),r1_input_manifest_sha256=sha(ROOT/'evaluation_inputs/r1_inputs.json'),source_sha256=sources))
 print('COMPLETE common five-seed evaluation: 400 metric rows, zero evaluation fits.',flush=True)

if __name__=='__main__':main()
