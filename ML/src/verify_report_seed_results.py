"""Independent weight, cohort, metric and publication completeness audit."""
import csv
import statistics
import subprocess
import numpy as np
from report_seed_common import *
from freeze_report_seed_study import validate_frozen
from stress_missingness_c import score_metrics

def main():
 frozen=validate_frozen();evaluation=read(ROOT/'evaluation/summary.json');public=read(ML/'reports/P63_report_five_seed_summary.json')
 assert public['complete'] and public['sample_sd_ddof']==1 and public['new_formal_fits']==32
 assert public['frozen_sha256']==sha(ROOT/'frozen.json') and public['selected_seed'] is None
 assert public['evaluation_summary_sha256']==sha(ROOT/'evaluation/summary.json')
 assert not public['test_selected'] and not public['parameters_reselected']
 assert len(public['individual_metrics'])==400 and len(public['mean_sd'])==len(public['probability_ensembles'])==80
 assert public['individual_metrics']==[{**r,'ap_drop_from_clean':next(c['ap'] for c in evaluation['rows'] if c['cohort']=='clean' and c['model']==r['model'] and c['seed']==r['seed'])-r['ap'] if not r['cohort'].startswith('eicu') else None} for r in evaluation['rows']]
 actual={(r['cohort'],r['model'],r['seed']):r for r in public['individual_metrics']}
 means={(r['cohort'],r['model']):r for r in public['mean_sd']}
 ensembles={(r['cohort'],r['model']):r for r in public['probability_ensembles']}
 seed_probability_checks={name:dict(exact_cases=0,maximum_probability_difference=0.) for name in MODELS}
 for case in CASES:
  path=ROOT/'evaluation'/f'{case}.npz';info=evaluation['cases'][case]
  assert sha(path)==info['predictions_sha256'];verify_sources(info['input_sha256']);verify_sources(info['source_sha256'])
  assert info['seed42_r1_exact'] and info['seed42_p22_metrics_match']
  with np.load(path) as p:
   assert p['predictions'].shape==(16,5,len(p['y']))
   expected=(782,91) if case.startswith('eicu') else (4000,585)
   assert (len(p['y']),int(p['y'].sum()))==expected
   assert len(set(p['ids']))==len(p['y'])
   assert list(p['models'])==list(MODELS)
   for j,name in enumerate(p['models']):
    probabilities=p['predictions'][j]
    check=seed_probability_checks[str(name)]
    check['exact_cases']+=int(np.array_equal(probabilities,np.broadcast_to(probabilities[0],probabilities.shape)))
    check['maximum_probability_difference']=max(check['maximum_probability_difference'],float(np.max(np.abs(probabilities-probabilities[0]))))
    assert list(p['seeds'][j])==list(range(5) if name=='kedgn_ab' else SEEDS)
    values={key:[] for key in ('ap','auroc','brier','sensitivity')}
    for i,seed in enumerate(p['seeds'][j]):
     row=actual[case,name,int(seed)];computed=score_metrics(p['y'],p['predictions'][j,i])
     assert (row['stays'],row['deaths'])==expected
     for key,value in computed.items():
      assert abs(row[key]-value)<1e-12;values[key].append(value)
     if str(name).startswith('ma_'):
      assert row['rho']==p['rho'][j,i] and 0<=row['rho']<=1
     else:assert row['rho'] is None
    for key,v in values.items():
     mean=means[case,name]
     assert abs(mean[key+'_mean']-statistics.mean(v))<1e-12
     assert abs(mean[key+'_sd']-statistics.stdev(v))<1e-12
    computed=score_metrics(p['y'],p['predictions'][j].mean(axis=0,dtype=np.float64))
    for key,value in computed.items():assert abs(ensembles[case,name][key]-value)<1e-12
 ma=read(ROOT/'ma_frozen.json');r1=read(ROOT/'r1_frozen.json')
 assert len(ma['records'])==30 and len(r1['records'])==10
 assert sum(r['new_fit'] for r in ma['records'])==24
 assert sum(r['new_fit'] for r in r1['records'])==8
 check=read(ROOT/'ma_numeric_check.json')
 assert check['AB_only'] and check['diagnostic_fits']==12 and all(r['tree_and_predictions']=='exact' for r in check['checks'])
 rel=read(ROOT/'ma_reliance_check.json')
 assert rel['models']==30 and rel['original_and_random_masks']=='exact' and rel['new_fits']==0
 protected=read(ROOT/'protected_report_hashes.json');verify_sources(protected)
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO,text=True,encoding='utf-8').split('\0')
 for path in protected:assert 'ML/'+path not in tracked
 for filename,count in (('P63_all_seed_results.csv',400),('P63_mean_sd_results.csv',80),
  ('P63_probability_ensembles.csv',80),('P63_seed_differences.csv',275)):
  with (ML/'reports'/filename).open(encoding='utf-8',newline='') as stream:
   reader=csv.DictReader(stream);rows=list(reader)
   assert len(rows)==count and not set(reader.fieldnames)&{'ids','y','probabilities','RecordID','patientunitstayid','uniquepid'}
 assert all(len(r['seeds'])==r['weight_sets']==5 for r in public['inventory'])
 weight_files={p for r in frozen['records'] for p in r['weight_files']}
 for path in weight_files:
  result=subprocess.run(['git','check-ignore','--quiet','--','ML/'+path],cwd=REPO)
  assert result.returncode==0,path
 verification=dict(complete=True,predictors=16,weight_sets=80,unique_weight_files=len(weight_files),
  primary_cases=5,individual_metric_rows=400,mean_sd_rows=80,ensemble_rows=80,seed_difference_rows=275,
  new_formal_fits=32,AB_numerical_diagnostic_fits=12,evaluation_fits=0,
  sample_sd_independently_recomputed=True,all_weight_hashes_verified=True,all_seed_sets_verified=True,
  all_seed42_references_preserved=True,held_report_files_preserved=len(protected),
  seed_probability_checks=seed_probability_checks,
  figure_sha256=sha(ML/'figures/P63_report_five_seed_AP.png'),
  public_summary_sha256=sha(ML/'reports/P63_report_five_seed_summary.json'))
 write(ML/'reports/P63_verification.json',verification)
 print(__import__('json').dumps(verification,indent=2),flush=True)

if __name__=='__main__':main()
