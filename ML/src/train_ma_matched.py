"""P62: prespecified 300-tree/depth-3 matched MA/no-penalty GBT, A/B only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import joblib
import numpy as np
from malearn.estimators.classification import MAGBTClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from ma_gbt_p12_inputs import SCHEMA_FILE,encode
from prepare import OUT,sha
from run_ma_gbt_p12 import validate_upstream,fast_scalar_splits
from train_missingness_augmentation import ab_inputs,mask_observed,SEED
from stress_missingness_c import apply_masks,CHANNELS
from ma_matched_runtime import accelerated_splits,reliance,fast_tree_reliance

ROOT=OUT/'p62_ma_gbt_matched'
PARAMS=dict(n_estimators=300,max_depth=3,learning_rate=.05,loss='log_loss',subsample=1.,random_state=42)
ALPHAS=(.001,.01,.1,1.,10.,100.,1000.)
VIEWS=('clean','full_GCS','full_all_six')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');temp.replace(p)
def full_mask(x,channels):
    indices={k:np.flatnonzero(x[f'{k}__count'].gt(0).to_numpy()) for k in channels}
    return apply_masks(x,indices,{k:{'set_c_matched_observed':0} for k in channels},'full')
@lru_cache(maxsize=1)
def data():return (*ab_inputs(),read(SCHEMA_FILE)['schemas'])
def augmented(x,y,enabled,fold):
    if not enabled:return x,y,np.ones(len(y))
    masked,changed,_=mask_observed(x,{'GCS':60},SEED+(1000 if fold=='final' else 100*int(fold)))
    import pandas as pd
    train=pd.concat([x,masked.iloc[changed]])
    weights=np.ones(len(x));weights[changed]=.5
    return train,np.concatenate([y,y[changed]]),np.concatenate([weights,np.full(len(changed),.5)])
def metrics(model,frame,y,schema):
    x,m=encode(frame,schema);p=model.predict_proba(x)[:,1]
    return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),
                sensitivity=float(np.mean(p[y==1]>=.5)),rho=reliance(model,x,m))
def key_for(name,aug,alpha,fold):return f'{name}_{"gcs60" if aug else "plain"}_{"off" if alpha is None else format(alpha,"g")}_{fold}'
def fit_job(job):
    name,aug,alpha,fold=job;key=key_for(*job);path=ROOT/'jobs'/f'{key}.json'
    if path.exists():
        receipt=read(path)
        assert receipt['protocol_sha256']==sha(ROOT/'protocol.json')
        if fold=='final':assert sha(ROOT/'models'/f'{key}.joblib')==receipt['model_sha256']
        return receipt
    frame,y,folds,schemas=data();isfinal=fold=='final';take=np.ones(len(y),dtype=bool) if isfinal else folds!=int(fold)
    schema=schemas[name][str(fold)];raw,labels,weights=augmented(frame.loc[take],y[take],aug,fold)
    x,m=encode(raw,schema);start=time.perf_counter()
    print('FIT',key,'rows',len(x),flush=True)
    model=MAGBTClassifier(alpha=1. if alpha is None else alpha,**PARAMS)
    with accelerated_splits():model.fit(x,labels,M=None if alpha is None else m,sample_weight=weights)
    result=dict(key=key,features=name,augmentation=aug,alpha=alpha,fold=str(fold),source_stays=int(take.sum()),
        training_rows=len(x),total_weight=float(weights.sum()),positive_weight=float(weights[labels==1].sum()),
        seconds=time.perf_counter()-start,protocol_sha256=sha(ROOT/'protocol.json'))
    assert weights.sum()==take.sum() and weights[labels==1].sum()==y[take].sum()
    if isfinal:
        modelpath=ROOT/'models'/f'{key}.joblib';modelpath.parent.mkdir(exist_ok=True)
        joblib.dump(model,modelpath);result['model_sha256']=sha(modelpath)
    else:
        valid=frame.loc[~take];result['validation']={
          'clean':metrics(model,valid,y[~take],schema),
          'full_GCS':metrics(model,full_mask(valid,['GCS']),y[~take],schema),
          'full_all_six':metrics(model,full_mask(valid,CHANNELS),y[~take],schema)}
    write(path,result);print('DONE',key,round(result['seconds'],2),flush=True);return result
def sources():
    from train_missingness_augmentation import AB_FEATURES,AB_LABELS,AB_FOLDS
    root=Path(__file__).resolve().parents[1]
    paths=[AB_FEATURES,AB_LABELS,AB_FOLDS,SCHEMA_FILE,*[root/'src'/s for s in ('train_ma_matched.py','ma_matched_runtime.py','ma_gbt_p12_inputs.py','run_ma_gbt_p12.py','train_missingness_augmentation.py','stress_missingness_c.py','prepare.py')],root/'reports/P62_ma_gbt_matched_protocol.md']
    return {p.relative_to(root).as_posix():sha(p) for p in paths}
def verify_sources(protocol):
    ml=Path(__file__).resolve().parents[1]
    for p,digest in protocol['source_sha256'].items():assert sha(ml/p)==digest,p
def trees_equal(a,b):
    for t,u in zip(a.estimators_.ravel(),b.estimators_.ravel(),strict=True):
        for f in ('feature','threshold','value','children_left','children_right'):np.testing.assert_array_equal(getattr(t.tree_,f),getattr(u.tree_,f))
def check():
    validate_upstream();frame,y,folds,schemas=data();probe=frame.loc[folds!=0].iloc[:384];labels=y[folds!=0][:384];schema=schemas['Full']['0']
    x,m=encode(probe,schema);receipt=[]
    for alpha,aug in [(None,False),(.1,False),(10.,True),(1000.,True)]:
        raw,labels2,weights=augmented(probe,labels,aug,0);xa,ma=encode(raw,schema)
        opts={**PARAMS,'n_estimators':3};mask=None if alpha is None else ma
        with fast_scalar_splits():slow=MAGBTClassifier(alpha=1. if alpha is None else alpha,**opts).fit(xa,labels2,M=mask,sample_weight=weights)
        with accelerated_splits():fast=MAGBTClassifier(alpha=1. if alpha is None else alpha,**opts).fit(xa,labels2,M=mask,sample_weight=weights)
        trees_equal(slow,fast);np.testing.assert_array_equal(slow.predict_proba(x),fast.predict_proba(x))
        for t in slow.estimators_.ravel():
            np.testing.assert_array_equal(t.compute_missingness_reliance(x,m,reduce=False),fast_tree_reliance(t,x,m,reduce=False))
        if alpha is None:
            with accelerated_splits():zero=MAGBTClassifier(alpha=1.,**opts).fit(xa,labels2,M=np.zeros_like(ma),sample_weight=weights)
            trees_equal(fast,zero)
        receipt.append(dict(alpha=alpha,augmentation=aug,tree_structure='exact',probabilities='exact',reliance='exact'))
        print('CHECK',alpha,aug,'exact',flush=True)
    ROOT.mkdir(exist_ok=True);write(ROOT/'numeric_check.json',dict(checks=receipt,only_AB=True,new_scientific_fits=0))
    # Time a full-size A/B fold without reading any held-out cohort.
    take=folds!=0;x,m=encode(frame.loc[take],schema);start=time.perf_counter()
    with accelerated_splits():model=MAGBTClassifier(alpha=.1,**{**PARAMS,'n_estimators':10}).fit(x,y[take],M=m)
    print('A/B throughput: 10 depth-3 trees',round(time.perf_counter()-start,2),'seconds',flush=True)
def train(workers):
    validate_upstream();ROOT.mkdir(exist_ok=True)
    assert (ROOT/'numeric_check.json').exists(),'Run check before any scientific fit'
    protocol=dict(name='P62 matched MA-GBT control',training='P12 A/B only',parameters=PARAMS,alphas=ALPHAS,
       feature_sets=['Full','Elastic'],augmentation='same weighted GCS60 copies in both groups',seed=42,
       selections=['maximum mean validation AP','minimum rho within 95% best validation AUROC'],
       no_penalty_control='same author MAGBTClassifier, M=None; tested equal to all-zero penalty matrix',
       source_sha256=sources(),numeric_check_sha256=sha(ROOT/'numeric_check.json'),device='cpu',workers=workers)
    path=ROOT/'protocol.json'
    if path.exists():assert read(path)==json.loads(json.dumps(protocol))
    else:write(path,protocol)
    verify_sources(protocol)
    jobs=[(name,aug,a,str(f)) for name in ('Full','Elastic') for aug in (False,True) for a in (None,*ALPHAS) for f in range(3)]
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(fit_job,job) for job in jobs]
        for f in as_completed(pending):results.append(f.result())
        candidates=[];selected=[]
        for name in ('Full','Elastic'):
            for aug in (False,True):
                rows=[]
                for alpha in (None,*ALPHAS):
                    group=[r for r in results if (r['features'],r['augmentation'],r['alpha'])==(name,aug,alpha)]
                    assert len(group)==3
                    row=dict(features=name,augmentation=aug,alpha=alpha,views={v:{m:float(np.mean([r['validation'][v][m] for r in group])) for m in group[0]['validation'][v]} for v in VIEWS})
                    rows.append(row);candidates.append(row)
                ma=[r for r in rows if r['alpha'] is not None]
                best=max(ma,key=lambda r:(r['views']['clean']['ap'],-r['views']['clean']['rho'],-r['alpha']))
                floor=.95*max(r['views']['clean']['auroc'] for r in ma)
                trade=min([r for r in ma if r['views']['clean']['auroc']>=floor],key=lambda r:(r['views']['clean']['rho'],-r['views']['clean']['auroc'],r['alpha']))
                for policy,row in [('off',rows[0]),('ap',best),('rho95',trade)]:
                    selected.append(dict(features=name,augmentation=aug,policy=policy,alpha=row['alpha'],validation=row['views']))
        selection=dict(candidates=candidates,selected=selected,protocol_sha256=sha(path),selected_on='A/B only',cv_fits=len(results))
        selectionpath=ROOT/'selection.json'
        if selectionpath.exists():assert read(selectionpath)==selection
        else:write(selectionpath,selection)
        finaljobs=list(dict.fromkeys((r['features'],r['augmentation'],r['alpha'],'final') for r in selected))
        finals=list(pool.map(fit_job,finaljobs))
    verify_sources(protocol)
    write(ROOT/'frozen.json',dict(protocol_sha256=sha(path),selection_sha256=sha(selectionpath),cv_fits=len(results),
       final_fits=len(finals),fits=finals,selected=selected,training_only_AB=True,test_accessed_for_selection=False))
    print('FROZEN',len(results),'CV fits,',len(finals),'unique final fits',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','train']);p.add_argument('--workers',type=int,default=6,choices=[1,2,3,4,6]);a=p.parse_args()
    if a.mode=='check':check()
    else:train(a.workers)
