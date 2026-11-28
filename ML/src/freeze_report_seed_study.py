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
 return value

def main():
 protocol();ma=read(ROOT/'ma_frozen.json');r1=read(ROOT/'r1_frozen.json')
 for group in (ma,r1):assert group['protocol_sha256']==sha(ROOT/'protocol.json') and not group['test_used']
 assert ma['new_fits']==24 and r1['new_fits']==8
 p60=read(P60/'frozen.json');kg=read(KG/'frozen.json');records=[]
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
 value=dict(protocol_sha256=sha(ROOT/'protocol.json'),records=records,new_formal_fits=32,
  diagnostic_fits=12,ma_frozen_sha256=sha(ROOT/'ma_frozen.json'),r1_frozen_sha256=sha(ROOT/'r1_frozen.json'),
  test_selected=False,training_only_AB=True,parameters_reselected=False)
 path=ROOT/'frozen.json'
 if path.exists():assert read(path)==value
 else:write(path,value)
 validate_frozen();print('Frozen 16 predictors x five seeds; all actual weights verified.',flush=True)

if __name__=='__main__':main()
