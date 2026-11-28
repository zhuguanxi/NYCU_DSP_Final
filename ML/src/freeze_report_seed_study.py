"""Freeze every report predictor before additional held-out evaluation."""
from report_seed_common import *

def validate_frozen():
 protocol();value=read(ROOT/'frozen.json')
 assert value['protocol_sha256']==sha(ROOT/'protocol.json') and not value['test_selected']
 assert len(value['records'])==80 and len({(r['model'],r['seed']) for r in value['records']})==80
 for record in value['records']:
  for path,digest in record['weight_files'].items():assert sha(ML/path)==digest,path
 assert sha(ROOT/'ma_frozen.json')==value['ma_frozen_sha256']
 assert sha(ROOT/'r1_frozen.json')==value['r1_frozen_sha256']
 verify_sources(value['evaluation_input_sha256'])
 return value

def main():
 protocol();ma=read(ROOT/'ma_frozen.json');r1=read(ROOT/'r1_frozen.json')
 for group in (ma,r1):assert group['protocol_sha256']==sha(ROOT/'protocol.json') and not group['test_used']
 assert ma['new_fits']==24 and r1['new_fits']==8
 p60=read(P60/'frozen.json');kg=read(KG/'frozen.json');records=[]
 assert p60['selected_seed'] is None and not p60['selected_on_tests']
 verify_sources(p60['protocol']['source_sha256'])
 verify_sources(read(P62/'protocol.json')['source_sha256'])
 prior_ma=read(P62/'frozen.json')
 assert prior_ma['training_only_AB'] and not prior_ma['test_accessed_for_selection']
 for name,aug,alpha,_ in MA_CONFIGS:
  policy=name.rsplit('_',1)[1]
  selected=next(r for r in prior_ma['selected'] if r['features']=='Full' and r['augmentation']==aug and r['policy']==policy)
  assert selected['alpha']==alpha
 assert not kg['test_data_used'] and kg['selection_sha256']==sha(KG/'selection.json')
 assert read(KG/'selection.json')['selected_epoch']==9
 assert all(r['epochs']==9 and r['training_stays']==8000 for r in kg['seeds'].values())
 simple={'frozen_full':'Full.joblib','frozen_elastic':'Elastic.joblib','gcs_only_full':'gcs_Full.joblib',
  'gcs_only_elastic':'gcs_Elastic.joblib','p22':'temporal.joblib'}
 for seed in SEEDS:
  hashes=p60['seeds'][str(seed)]['model_sha256'];folder=P60/f'seed_{seed}'
  for name in (*LEGACY,'p22'):
   files=[simple[name]] if name in simple else ['temporal.joblib','blind_lgb.joblib']
   if name=='p28':files+=['gcs_Elastic.joblib']
   weights={relative(folder/file):hashes[file] for file in files}
   records.append(dict(model=name,seed=seed,weight_files=weights,new_fit=False,source='P60 fixed models'))
 for seed in range(5):
  old=kg['seeds'][str(seed)];folder=KG/f'final_seed_{seed}'
  records.append(dict(model='kedgn_ab',seed=seed,new_fit=False,source='P54 fixed nine-epoch model',
   weight_files={relative(folder/'weights.pt'):old['weights_sha256'],relative(folder/'statistics.npz'):old['statistics_sha256']}))
 for group in (ma,r1):
  for r in group['records']:
   records.append(dict(model=r['model'],seed=r['seed'],new_fit=r['new_fit'],source='P63 final fit' if r['new_fit'] else 'original seed42',
    weight_files={r['file']:r['model_sha256']}))
 assert set(r['model'] for r in records)==set(MODELS)
 for name in MODELS:
  assert {r['seed'] for r in records if r['model']==name}==set(range(5) if name=='kedgn_ab' else SEEDS)
 assert sum(r['new_fit'] for r in records)==32
 legacy_cases=read(P60/'evaluation.json')
 assert legacy_cases['stress_inputs_sha256']==sha(P61/'inputs.json')
 mask_cases={r['name']:r for r in read(P61/'inputs.json')['cases']}
 p62_sources=read(P62/'evaluation/summary.json')['protected_evaluation_sources']
 evaluation_inputs={}
 for case in CASES:
  features,timing=case_paths(case)
  if not case.startswith('eicu'):
   for file in (features,timing):assert sha(file)==mask_cases[case]['files'][file.name]
  else:assert sha(features)==p62_sources[relative(features)]
  prediction=ML/legacy_cases['cases'][case]['file']
  assert sha(prediction)==legacy_cases['cases'][case]['sha256']
  for file in (features,timing,prediction):evaluation_inputs[relative(file)]=sha(file)
 value=dict(protocol_sha256=sha(ROOT/'protocol.json'),records=records,new_formal_fits=32,
  diagnostic_fits=12,ma_frozen_sha256=sha(ROOT/'ma_frozen.json'),r1_frozen_sha256=sha(ROOT/'r1_frozen.json'),
  test_selected=False,training_only_AB=True,parameters_reselected=False,evaluation_input_sha256=evaluation_inputs)
 path=ROOT/'frozen.json'
 if path.exists():assert read(path)==value
 else:write(path,value)
 validate_frozen();print('Frozen 16 predictors x five seeds; all actual weights verified.',flush=True)

if __name__=='__main__':main()
