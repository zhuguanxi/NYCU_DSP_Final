"""Checked equivalent CPU batching; separate receipts and atomic shared cache."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['NUMBA_NUM_THREADS']='2'
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from malearn.estimators.classification import MAGBTClassifier
from malearn.estimators.regression import MADTRegressor
import train_ma_matched as t
from ma_matched_parallel import parallel_splits
from ma_matched_runtime import accelerated_splits
BASE=t.ROOT;ASSIST=BASE/'assist_parallel'
def check():
    # Check every candidate, not just whichever feature wins, including ties.
    rng=np.random.default_rng(141)
    for n in (8,65,384):
        X=np.asarray(rng.normal(size=(n,32)),dtype=np.float32);X[:,0]=1;X[:,1]=np.round(X[:,1]);X[:,2]=1e-8*X[:,2]
        y=rng.normal(size=n);w=rng.choice([.5,1.],size=n);M=rng.integers(2,size=X.shape,dtype=np.int8)
        tree=MADTRegressor(alpha=10.);tree.n_features_in_=X.shape[1]
        with accelerated_splits():a=list(tree._get_split_candidates(X,y,M,w,0,np.random.RandomState(42)))
        with parallel_splits():b=list(tree._get_split_candidates(X,y,M,w,0,np.random.RandomState(42)))
        assert a==b,(n,max(abs(x[0]-z[0]) for x,z in zip(a,b) if np.isfinite(x[0])))
    frame,y,folds,schemas=t.data();probe=frame.loc[folds!=0].iloc[:384];truth=y[folds!=0][:384];schema=schemas['Full']['0']
    for alpha,aug in [(None,False),(.1,False),(10.,True),(1000.,True)]:
        raw,target,weights=t.augmented(probe,truth,aug,0);X,M=t.encode(raw,schema)
        params={**t.PARAMS,'n_estimators':3};mask=None if alpha is None else M
        with accelerated_splits():a=MAGBTClassifier(alpha=1. if alpha is None else alpha,**params).fit(X,target,M=mask,sample_weight=weights)
        with parallel_splits():b=MAGBTClassifier(alpha=1. if alpha is None else alpha,**params).fit(X,target,M=mask,sample_weight=weights)
        t.trees_equal(a,b);np.testing.assert_array_equal(a.predict_proba(X),b.predict_proba(X))
    take=folds!=0;X,M=t.encode(frame.loc[take],schema);start=time.perf_counter()
    with accelerated_splits():reference=MAGBTClassifier(alpha=.1,**{**t.PARAMS,'n_estimators':10}).fit(X,y[take],M=M)
    reference_seconds=time.perf_counter()-start;start=time.perf_counter()
    with parallel_splits():model=MAGBTClassifier(alpha=.1,**{**t.PARAMS,'n_estimators':10}).fit(X,y[take],M=M)
    seconds=time.perf_counter()-start
    t.trees_equal(reference,model);np.testing.assert_array_equal(reference.predict_proba(X),model.predict_proba(X))
    result=dict(all_candidate_scores='exact',four_weighted_model_checks='exact',AB_only=True,
       diagnostic_fits=10,ten_tree_seconds=seconds,reference_ten_tree_seconds=reference_seconds,
       full_fold_tree_and_probability_check='exact',parallel_source_sha256=t.sha(Path(__file__).with_name('ma_matched_parallel.py')))
    t.write(BASE/'parallel_numeric_check.json',result);print('PARALLEL CHECK PASSED: scalar',reference_seconds,'parallel',seconds,'seconds /10 trees',flush=True)
def execute(job):
    t.ROOT=ASSIST;t.accelerated_splits=parallel_splits
    source=BASE/'jobs'/f'{t.key_for(*job)}.json';dest=ASSIST/'jobs'/source.name
    if source.exists() and not dest.exists():
        dest.parent.mkdir(exist_ok=True)
        try:os.link(source,dest)
        except FileExistsError:pass
    return t.fit_job(job)
def main():
    protocol=t.read(BASE/'protocol.json');t.verify_sources(protocol)
    assert t.read(BASE/'parallel_numeric_check.json')['all_candidate_scores']=='exact'
    ASSIST.mkdir(exist_ok=True);p=ASSIST/'protocol.json'
    if p.exists():assert t.sha(p)==t.sha(BASE/'protocol.json')
    else:shutil.copyfile(BASE/'protocol.json',p)
    jobs=[(name,aug,a,str(f)) for name,aug in [('Elastic',False),('Full',True),('Full',False),('Elastic',True)]
          for a in reversed((None,*t.ALPHAS)) for f in reversed(range(3))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            r=future.result();src=ASSIST/'jobs'/f'{r["key"]}.json';dest=BASE/'jobs'/src.name
            try:os.link(src,dest)
            except FileExistsError:
                old=t.read(dest)
                for field in ('key','features','augmentation','alpha','fold','validation','protocol_sha256'):
                    assert old[field]==r[field],(r['key'],field)
    t.write(BASE/'parallel_execution.json',dict(cpu_only=True,workers=4,jobs=96,source_sha256=t.sha(Path(__file__)),
       check_sha256=t.sha(BASE/'parallel_numeric_check.json'),all_candidate_scores_exact=True))
    print('PARALLEL ASSIST COMPLETE',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args()
    if a.mode=='check':check()
    else:main()
