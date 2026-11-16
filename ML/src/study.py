"""Small, resumable development-only ML study. Never reads C outcomes/features."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import time,json,hashlib,itertools,warnings,threading,platform
import numpy as np
import pandas as pd
import joblib,psutil
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,f1_score,precision_recall_curve,auc
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'ML/artifacts';REP=ROOT/'ML/reports';FIG=ROOT/'ML/figures'
X=pd.read_parquet(ART/'features_AB.parquet');Y=pd.read_csv(ART/'labels_AB.csv').set_index('RecordID').loc[X.index];y=Y.In_hospital_death
FD=pd.read_csv(ART/'feature_dictionary.csv').set_index('feature').loc[X.columns]
FD.index.name='feature'
SEED=42
CAT=[c for c in X.columns if FD.loc[c,'kind']=='categorical']
LEDGER=ART/'fit_ledger.jsonl'
DATA_SHA=hashlib.sha256((ART/'features_AB.parquet').read_bytes()).hexdigest()
BASE={'Dummy':{},'LR':{'C':1.,'l1_ratio':0.},'RF':{'max_depth':8,'min_samples_leaf':5},'XGBoost':{'max_depth':3,'min_child_weight':1},'LightGBM':{'num_leaves':31,'min_child_samples':20}}

def frame_md(df):
 vals=df.copy()
 for c in vals:
  if pd.api.types.is_float_dtype(vals[c]):vals[c]=vals[c].map(lambda x:f'{x:.4f}' if pd.notna(x) else '')
 return '| '+' | '.join(map(str,vals.columns))+' |\n|'+'|'.join(['---']*len(vals.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in vals.itertuples(index=False,name=None))

def folds(n=3,adult=False):
 ix=np.flatnonzero((X.static_Age>=18).to_numpy()) if adult else np.arange(len(X))
 return [(ix[a],ix[b]) for a,b in StratifiedKFold(n_splits=n,shuffle=True,random_state=SEED).split(ix,y.iloc[ix])]

FOLDS=folds()
assign=pd.DataFrame({'RecordID':X.index,'fold3':-1})
for k,(_,va) in enumerate(FOLDS):assign.loc[va,'fold3']=k
assign.to_csv(ART/'folds3.csv',index=False)

def make_preprocessor(cols,scale=False):
 num=[c for c in cols if c not in CAT];cat=[c for c in cols if c in CAT]
 nparts=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
 if scale:nparts.append(('scale',StandardScaler()))
 cp=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=-1)),('encode',OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False))])
 return ColumnTransformer([('num',Pipeline(nparts),num),('cat',cp,cat)],verbose_feature_names_out=False,sparse_threshold=0)

def estimator(model,params):
 if model=='Dummy':return DummyClassifier(strategy='prior')
 if model=='LR':return LogisticRegression(solver='saga',max_iter=2000,tol=0.001,random_state=SEED,**params)
 if model=='RF':return RandomForestClassifier(n_estimators=300,max_features='sqrt',n_jobs=4,random_state=SEED,**params)
 if model=='XGBoost':return XGBClassifier(n_estimators=300,learning_rate=0.05,tree_method='hist',n_jobs=4,random_state=SEED,eval_metric='logloss',**params)
 if model=='LightGBM':return LGBMClassifier(n_estimators=300,learning_rate=0.05,n_jobs=4,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True,**params)
 raise ValueError(model)

def log_fit(rec):
 with LEDGER.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')

def fitted(train,cols,model,params,role='prediction'):
 budget=json.loads((ART/'execution_budget.json').read_text()) if (ART/'execution_budget.json').exists() else {'max_total_fits':150}
 count=sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0
 if count>=budget['max_total_fits']:raise RuntimeError('Fit budget reached; approval required before extending')
 proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
 def sample():
  while not stop.wait(.1):peak[0]=max(peak[0],proc.memory_info().rss)
 th=threading.Thread(target=sample,daemon=True);th.start();start=time.perf_counter()
 pipe=Pipeline([('pre',make_preprocessor(cols,model=='LR')),('model',estimator(model,params))])
 with warnings.catch_warnings(record=True) as w,threadpool_limits(limits=4):
  warnings.simplefilter('always',ConvergenceWarning);pipe.fit(X.iloc[train][cols],y.iloc[train])
 stop.set();th.join();elapsed=time.perf_counter()-start
 conv=[str(z.message) for z in w if issubclass(z.category,ConvergenceWarning)]
 rec={'role':role,'model':model,'params':params,'train_n':len(train),'input_n':len(cols),'encoded_n':len(pipe['pre'].get_feature_names_out()),'fit_s':elapsed,'peak_rss_mb':peak[0]/1024**2,'convergence_warnings':conv,'feature_data_sha256':DATA_SHA}
 log_fit(rec)
 if peak[0]>8*1024**3:raise MemoryError('Observed RSS exceeds approved 8GB target')
 return pipe,rec

def basic_cols(train,family=2):
 cols=FD.index[FD.family<=family].tolist();z=X.iloc[train][cols]
 return [c for c in cols if z[c].notna().any() and z[c].nunique(dropna=False)>1]

def quality_cols(train,base):
 cache=ART/'quality_cache';cache.mkdir(exist_ok=True)
 key=hashlib.sha256((DATA_SHA+json.dumps(base)+json.dumps(np.asarray(train).tolist())).encode()).hexdigest()[:20]
 target=cache/(key+'.joblib')
 if target.exists():return joblib.load(target)
 z=X.iloc[train][base]
 numeric=[c for c in base if c not in CAT and FD.loc[c,'statistic'] not in ['count','missing','std_defined','slope_defined']]
 corr=z[numeric].corr(method='spearman',min_periods=100)
 order={'last':0,'mean':1,'max':2,'min':3,'first':4,'ever':0,'std':5,'delta':6,'slope':7,'observed_fraction':2,'admission':0}
 priority=sorted(numeric,key=lambda c:(float(z[c].isna().mean()),order.get(FD.loc[c,'statistic'],9),c))
 keep=[];removed=[]
 for c in priority:
  match=next((k for k in keep if pd.notna(corr.loc[c,k]) and abs(corr.loc[c,k])>=.9),None)
  if match is None:keep.append(c)
  else:removed.append({'removed':c,'kept':match,'rho':corr.loc[c,match]})
 # Non-summary indicators and categories remain, except exact duplicates.
 retained=[c for c in base if c not in numeric or c in keep];final=[]
 for c in retained:
  dup=next((k for k in final if z[c].equals(z[k])),None)
  if dup is None:final.append(c)
  else:removed.append({'removed':c,'kept':dup,'rho':'exact_duplicate'})
 result=(final,pd.DataFrame(removed));joblib.dump(result,target);return result

def coefficient_table(pipe,cols):
 names=pipe['pre'].get_feature_names_out();coef=pipe['model'].coef_[0]
 num=[c for c in cols if c not in CAT]
 records=[]
 for i,(name,v) in enumerate(zip(names,coef)):
  base=name if i<len(num) else next((c for c in CAT if name.startswith(c+'_')),name)
  records.append(dict(encoded_feature=name,feature=base,variable=FD.loc[base,'group'],statistic=FD.loc[base,'statistic'],coefficient=float(v),abs_coefficient=abs(float(v))))
 return pd.DataFrame(records)

def selector(train,base,key):
 pth=ART/'selectors'/f'{key}.joblib';pth.parent.mkdir(exist_ok=True)
 if pth.exists():
  cached=joblib.load(pth)
  if cached.get('feature_data_sha256')==DATA_SHA:return cached
 q,removed=quality_cols(train,base)
 pipe,rec=fitted(train,q,'LR',{'C':1.,'l1_ratio':.5},'selector')
 coef=coefficient_table(pipe,q);coef.to_csv(pth.with_suffix('.csv'),index=False);removed.to_csv(pth.with_name(pth.stem+'_removed.csv'),index=False)
 nonzero=coef.loc[coef.abs_coefficient>1e-8,'feature'].unique().tolist()
 # Dynamic rankings use numeric summaries only, not measurement pattern or static dummies.
 bio=coef[(~coef.variable.str.startswith('static_')) & (~coef.statistic.isin(['count','missing','std_defined','slope_defined']))]
 ranks=bio.groupby('variable').abs_coefficient.mean().sort_values(ascending=False)
 result={'quality':q,'elastic':nonzero,'rank':ranks.index.tolist(),'ranking_values':ranks.to_dict(),'converged':not rec['convergence_warnings'],'feature_data_sha256':DATA_SHA}
 if not result['converged']:result['elastic']=[]
 joblib.dump(result,pth);return result

def select_cols(train,family,strategy,key):
 base=basic_cols(train,family)
 if strategy=='Full':return base
 if strategy=='Quality':return quality_cols(train,base)[0]
 sel=selector(train,base,key)
 if strategy=='Elastic':return sel['elastic']
 if strategy in ['Top10','Top20']:
  groups=sel['rank'][:int(strategy[3:])]
  return [c for c in base if FD.loc[c,'group'].startswith('static_') or FD.loc[c,'group'] in groups]
 raise ValueError(strategy)

def metrics(yy,p):
 pp=(p>=.5).astype(int);pr,re,_=precision_recall_curve(yy,p)
 return dict(AP=average_precision_score(yy,p),AUROC=roc_auc_score(yy,p),Brier=brier_score_loss(yy,p),AUPRC_trapezoid=auc(re,pr),Precision_05=precision_score(yy,pp,zero_division=0),Recall_05=recall_score(yy,pp,zero_division=0),F1_05=f1_score(yy,pp,zero_division=0))

def run(model,params,family=2,strategy='Full',nfold=3,adult=False):
 description=dict(model=model,params=params,family=family,strategy=strategy,nfold=nfold,adult=adult,data_sha=hashlib.sha256((ART/'features_AB.parquet').read_bytes()).hexdigest())
 key=hashlib.sha256(json.dumps(description,sort_keys=True).encode()).hexdigest()[:16]
 folder=ART/'runs'/key;folder.mkdir(parents=True,exist_ok=True)
 resultfile=folder/'results.csv'
 if resultfile.exists():return pd.read_csv(resultfile),key
 results=[];predictions=[];selected=[]
 for fold,(tr,va) in enumerate(folds(nfold,adult)):
  # The selector is only trained on this fold's training IDs.
  selkey=f'fold{nfold}_{fold}_adult{adult}'
  cols=select_cols(tr,family,strategy,selkey)
  if not cols:
   results.append(dict(**description,fold=fold,status='selector_failure',AP=np.nan));continue
  saved=folder/f'fold{fold}.joblib'
  if saved.exists():pipe,info,cols=joblib.load(saved)
  else:
   pipe,info=fitted(tr,cols,model,params);joblib.dump((pipe,info,cols),saved)
  t=time.perf_counter();p=pipe.predict_proba(X.iloc[va][cols])[:,1];ps=time.perf_counter()-t
  assert len(p)==len(va) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  results.append(dict(model=model,params=json.dumps(params,sort_keys=True),family=family,strategy=strategy,nfold=nfold,adult=adult,fold=fold,status='ok',**metrics(y.iloc[va],p),fit_s=info['fit_s'],predict_s=ps,input_n=len(cols),encoded_n=info['encoded_n'],peak_rss_mb=info['peak_rss_mb'],converged=not info['convergence_warnings']))
  predictions.extend(dict(RecordID=int(X.index[j]),fold=fold,y=int(y.iloc[j]),score=float(v)) for j,v in zip(va,p))
  selected.extend(dict(fold=fold,feature=c,variable=FD.loc[c,'group']) for c in cols)
  print(f'{model} F{family} {strategy} fold {fold}: AP {results[-1]["AP"]:.4f} AUC {results[-1]["AUROC"]:.4f} fit {info["fit_s"]:.1f}s',flush=True)
 pd.DataFrame(results).to_csv(resultfile,index=False)
 pd.DataFrame(predictions).to_csv(folder/'oof.csv',index=False);pd.DataFrame(selected).to_csv(folder/'selected.csv',index=False)
 (folder/'config.json').write_text(json.dumps(description,indent=2),encoding='utf-8')
 return pd.DataFrame(results),key

def summary(results):
 a=pd.concat(results,ignore_index=True);numeric=['AP','AUROC','Brier','Recall_05','Precision_05','fit_s','input_n','peak_rss_mb']
 return a.groupby(['model','family','strategy','params'],dropna=False)[numeric].agg(['mean','std']).reset_index()

def simple_summary(a):
 cols=['AP','AUROC','Brier','fit_s','input_n','peak_rss_mb']
 z=a.groupby(['model','family','strategy','nfold','adult'],dropna=False)[cols].agg(['mean','std']).reset_index();z.columns=['_'.join(c).strip('_') for c in z.columns]
 return z

def write_report(name,title,text,a):
 table=simple_summary(a).sort_values('AP_mean',ascending=False)
 table.to_csv(ART/(name+'_summary.csv'),index=False)
 (REP/(name+'.md')).write_text('# '+title+'\n\n日期：2026-10-13。全部為 A＋B 開發結果，C 未評估。\n\n'+text+'\n\n'+frame_md(table[['model','family','strategy','nfold','adult','AP_mean','AP_std','AUROC_mean','Brier_mean','input_n_mean','fit_s_mean']])+'\n\n平均±SD是折間描述，不是獨立測試成績或95%信賴區間。',encoding='utf-8')

def main(stage):
 if stage=='baseline':
  allr=[];mapping={}
  for m,p in BASE.items():r,k=run(m,p);allr.append(r);mapping[m]=k
  a=pd.concat(allr);a.to_csv(ART/'baseline_folds.csv',index=False);(ART/'baseline_runs.json').write_text(json.dumps(mapping,indent=2))
  write_report('P1_基準模型','P1：四模型基準報告','F2 完整特徵；所有模型使用共同訓練內補值，LR 另標準化。分類閾值0.5只作描述。',a)
 elif stage=='tune':
  candidates={'LR':[dict(C=c,l1_ratio=r) for c,r in itertools.product([.1,1.,10.],[0.,.5])],'RF':[dict(max_depth=d,min_samples_leaf=l) for d,l in itertools.product([8,None],[5,20])],'XGBoost':[dict(max_depth=d,min_child_weight=w) for d,w in itertools.product([3,5],[1,10])],'LightGBM':[dict(num_leaves=n,min_child_samples=s) for n,s in itertools.product([15,31],[20,50])]}
  allr=[];best={};scores=[]
  for m,ps in candidates.items():
   rr=[]
   for p in ps:
    r,k=run(m,p);allr.append(r);rr.append((r.AP.mean(),p,k));scores.append(dict(model=m,params=json.dumps(p),AP=r.AP.mean(),AUROC=r.AUROC.mean(),fit_s=r.fit_s.mean(),converged=r.converged.all()))
   rr.sort(key=lambda v:v[0],reverse=True);best[m]={'params':rr[0][1],'key':rr[0][2],'AP':rr[0][0]}
  boost=max(['XGBoost','LightGBM'],key=lambda m:best[m]['AP']);best['selected_boost']=boost
  (ART/'best_models.json').write_text(json.dumps(best,indent=2));pd.DataFrame(scores).to_csv(ART/'parameter_comparison.csv',index=False)
  a=pd.concat(allr);a.to_csv(ART/'tuning_folds.csv',index=False)
  chosen=pd.concat([pd.read_csv(ART/'runs'/best[m]['key']/'results.csv') for m in candidates])
  write_report('P2_有限調參','P2：有限超參數比較',f'比較18組預定配置，每組三折；相同基準配置重用。依平均AP選取，每個方法並非全面最佳化。後續boosting選定 **{boost}**。詳細各配置見 parameter_comparison.csv。',chosen)
 elif stage=='families':
  best=json.loads((ART/'best_models.json').read_text());allr=[]
  for m in ['LR',best['selected_boost']]:
   for f in range(3):r,k=run(m,best[m]['params'],family=f);allr.append(r)
  a=pd.concat(allr);a.to_csv(ART/'family_folds.csv',index=False)
  write_report('P3_特徵家族','P3：特徵家族消融','F0基本摘要，F1增加變化，F2增加觀測模式。每種模型共用在F2選定的參數，不宣稱每組特徵已獨立最佳化。',a)
 elif stage=='selection':
  best=json.loads((ART/'best_models.json').read_text());allr=[];keys={}
  for m in ['LR',best['selected_boost']]:
   for s in ['Full','Quality','Elastic','Top10','Top20']:
    r,k=run(m,best[m]['params'],strategy=s);allr.append(r);keys[m+'_'+s]=k
  a=pd.concat(allr);a.to_csv(ART/'selection_folds.csv',index=False);(ART/'selection_runs.json').write_text(json.dumps(keys,indent=2))
  b=a[a.model==best['selected_boost']];full=b[b.strategy=='Full'].AP.mean();c=b.groupby('strategy').agg(AP=('AP','mean'),n=('input_n','mean'));eligible=c[(c.AP>=full-.01)&(c.index!='Full')].sort_values(['n','AP'],ascending=[True,False])
  chosen=eligible.index[0] if len(eligible) else 'Full'
  best['selected_strategy']=chosen;(ART/'best_models.json').write_text(json.dumps(best,indent=2))
  a['AP_delta_to_full']=a.apply(lambda r:r.AP-a[(a.model==r.model)&(a.strategy=='Full')&(a.fold==r.fold)].AP.iloc[0],axis=1);a.to_csv(ART/'selection_folds.csv',index=False)
  write_report('P4_特徵篩選','P4：特徵篩選報告',f'所有selector只fit訓練折。暫定AP容忍差0.01；以符合者中較少欄位為優先，選定boosting精簡策略 **{chosen}**。未把驗證importance排名用來重訓並在同一折評估。',a)
 elif stage=='confirm':
  best=json.loads((ART/'best_models.json').read_text());m=best['selected_boost'];s=best['selected_strategy'];allr=[]
  for st in dict.fromkeys(['Full',s]):r,k=run(m,best[m]['params'],strategy=st,nfold=5);allr.append(r)
  r,k=run(m,best[m]['params'],strategy=s,adult=True);allr.append(r);a=pd.concat(allr);a.to_csv(ART/'confirmation_folds.csv',index=False)
  write_report('P6_穩定性補充','P6：五折與成人限定敏感性','五折是同一開發集的穩定性檢查；另做Age>=18的三折敏感性，切分重新生成，不與原三折作嚴格成對推論。adult=True列和五折列請以詳細CSV區分，下面平均表包含兩種檢查。',a)
 else:raise ValueError(stage)
 budget=json.loads((ART/'execution_budget.json').read_text()) if (ART/'execution_budget.json').exists() else {'max_total_fits':150}
 assert sum(1 for _ in LEDGER.open())<=budget['max_total_fits'],'Approved fit budget exceeded'

if __name__=='__main__':
 import sys
 main(sys.argv[1])
