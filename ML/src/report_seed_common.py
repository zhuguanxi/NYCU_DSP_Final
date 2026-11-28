"""P63 fixed report scope, provenance and private checkpoint helpers."""
from pathlib import Path
import hashlib
import json

ML=Path(__file__).resolve().parents[1]
REPO=ML.parent
ROOT=ML/'artifacts/p63_report_five_seed'
P60=ML/'artifacts/p60_five_seed_stability'
P61=ML/'artifacts/p61_progressive_block_missingness'
P62=ML/'artifacts/p62_ma_gbt_matched'
R1=ML/'related_work/R1_baseline'
KG=ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121'
SEEDS=(42,43,44,45,46)
CASES=('clean','gcs_full','six_full','eicu_hourly','eicu_native')
LEGACY=('frozen_full','frozen_elastic','gcs_only_full','gcs_only_elastic','p26','p28')
MA_CONFIGS=(('ma_plain_off',False,None,'Full_plain_off_final'),
 ('ma_plain_ap',False,.1,'Full_plain_0.1_final'),
 ('ma_plain_rho95',False,1000.,'Full_plain_1000_final'),
 ('ma_gcs60_off',True,None,'Full_gcs60_off_final'),
 ('ma_gcs60_ap',True,.001,'Full_gcs60_0.001_final'),
 ('ma_gcs60_rho95',True,1000.,'Full_gcs60_1000_final'))
R1_CONFIGS={'r1_paper':dict(n_estimators=100,max_depth=5,learning_rate=.1),
 'r1_matched':dict(n_estimators=300,max_depth=3,learning_rate=.05)}
R1_TAGS={'r1_paper':'R1_XGBoost','r1_matched':'R1_XGBoost_matched'}
MODELS=(*LEGACY,'p22','kedgn_ab',*R1_CONFIGS,*(r[0] for r in MA_CONFIGS))
MAIN=('frozen_full','p22','ma_gcs60_ap','p28','kedgn_ab','r1_paper')
NAMES={'frozen_full':'原 XGBoost Full','frozen_elastic':'原 XGBoost Elastic',
 'gcs_only_full':'P11 GCS-only Full','gcs_only_elastic':'GCS-only Elastic',
 'p22':'P22 時間窗 XGBoost','p26':'P26 GCS 分流','p28':'P28 表格混合',
 'kedgn_ab':'KEDGN A+B 重訓','r1_paper':'R1 XGBoost 論文參數版','r1_matched':'R1 XGBoost 容量對齊版',
 'ma_plain_off':'同作者 GBT／無增強／無懲罰','ma_plain_ap':'P62 MA-AP Full／無增強',
 'ma_plain_rho95':'P62 MA-rho95 Full／無增強','ma_gcs60_off':'同作者 GBT／GCS60／無懲罰',
 'ma_gcs60_ap':'P62 MA-AP Full＋GCS60','ma_gcs60_rho95':'P62 MA-rho95 Full＋GCS60'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
 temp.replace(path)
def relative(path):return Path(path).resolve().relative_to(ML).as_posix()
def verify_sources(records):
 for path,digest in records.items():assert sha(ML/path)==digest,path
def protocol():
 value=read(ROOT/'protocol.json');verify_sources(value['training_source_sha256']);verify_sources(value['input_sha256'])
 assert value['seeds']==list(SEEDS) and value['models']==list(MODELS)
 return value
def receipt_valid(path):
 value=read(path);assert value['protocol_sha256']==sha(ROOT/'protocol.json')
 assert sha(ML/value['file'])==value['model_sha256']
 return value
def case_paths(case):
 folder=P60/case if case.startswith('eicu_') else P61/case
 return folder/'features.parquet',folder/'temporal.parquet'

def initialize():
 training_files=('report_seed_common.py','train_report_ma_seeds.py','train_report_r1_seeds.py','r1_report_features.py')
 scripts={relative(ML/'src'/name):sha(ML/'src'/name) for name in training_files}
 scripts.update({relative(ML/'src'/name):sha(ML/'src'/name) for name in ('ma_matched_parallel.py',
  'ma_matched_runtime.py','evaluate_ma_matched_fast.py','train_ma_matched.py')})
 inputs=[ML/'reports/P63_report_five_seed_protocol.md',P60/'frozen.json',P60/'evaluation.json',
  P62/'frozen.json',P62/'selection.json',P62/'protocol.json',P62/'parallel_numeric_check.json',
  R1/'frozen.json',R1/'r1_features.py',R1/'run.py',KG/'frozen.json',KG/'selection.json',
  ML/'artifacts/features_AB.parquet',ML/'artifacts/labels_AB.csv',ML/'artifacts/folds3.csv',
  ML/'artifacts/observations_AB.parquet',ML/'artifacts/ma_gbt_p12/encoded_schemas.json',
  ML/'artifacts/p22_temporal_ab_pilot/fitted/manifest.json',ML/'artifacts/p26_gcs_routed_temporal/manifest.json',
  ML/'artifacts/p28_robust_ensemble/manifest.json']
 value=dict(stage='P63 report five-seed completion',date='2026-11-28',seeds=SEEDS,kedgn_seeds=(0,1,2,3,4),
  models=MODELS,main_models=MAIN,training='P12 A+B only, fixed 8000 stays/1122 deaths',
  selections_repeated=False,test_selected=False,augmentation_positions_fixed=True,
  alpha_policy='reuse P62 seed42 A/B-selected alphas unchanged for every new seed',
  ma_configs=[dict(model=m,augmentation=a,alpha=p) for m,a,p,_ in MA_CONFIGS],r1_configs=R1_CONFIGS,
  new_formal_fits=dict(ma=24,r1=8,total=32),
  training_device='original CPU algorithms; GPU permission does not change fixed training implementation',
  summary='all five individual metrics, arithmetic mean, sample SD ddof=1; probability ensembles separately',
  evaluation='five original primary cohorts, corrected explicit urine mapping; no new temporal-block evaluation',
  training_source_sha256=scripts,input_sha256={relative(p):sha(p) for p in inputs})
 path=ROOT/'protocol.json'
 if path.exists():assert read(path)==json.loads(json.dumps(value))
 else:write(path,value)
 return protocol()
