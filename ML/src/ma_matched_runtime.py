"""CPU-only acceleration of the pinned author's split loop, checked against it."""
import os
from pathlib import Path
ML = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ML/'artifacts/p62_ma_matched_runtime/numba_cache')
import sys
sys.path.insert(0,str(ML/'artifacts/p62_ma_matched_runtime/site'))
import weakref
from contextlib import contextmanager
import numpy as np
from numba import njit
from malearn.estimators.regression import MADTRegressor
from malearn.estimators.base import BaseMADT

@njit(cache=True,fastmath=False)
def scan(x,y,w,high,high2,nhigh):
    low=0.0;low2=0.0;nlow=0.0;best=-np.inf;chosen=-1
    for i in range(len(y)-1):
        yi=y[i];wi=w[i]
        nlow+=wi;low+=yi*wi;low2+=(yi**2)*wi
        nhigh-=wi;high-=yi*wi;high2-=(yi**2)*wi
        if nlow==0.0: continue
        if nhigh==0.0: break
        a=float(x[i]);b=float(x[i+1])
        if abs(a-b)<=1e-8+1e-5*abs(b): continue
        vl=low2/nlow-low**2/nlow**2
        vh=high2/nhigh-high**2/nhigh**2
        score=-(nhigh*vh+nlow*vl)
        if score>best: best=score;chosen=i
    return best,chosen

def fast_tree_reliance(self,X,M,sample_mask=None,reduce=True,max_depth=None):
    if max_depth is None: max_depth=self.max_depth
    hits=np.zeros_like(M)
    rows=np.arange(len(X)) if sample_mask is None else np.flatnonzero(sample_mask)
    todo=[(self.root_,rows)]
    while todo:
        node,rows=todo.pop()
        if not len(rows) or node.depth>max_depth or node.left_subtree is None: continue
        f=node.feature;hits[rows,f]=M[rows,f]
        left=X[rows,f].astype(np.float64)<=node.threshold
        todo.append((node.left_subtree,rows[left]));todo.append((node.right_subtree,rows[~left]))
    return float(hits.max(axis=1).mean()) if reduce else hits

def reliance(model,X,M):
    hits=np.zeros_like(M)
    for tree in model.estimators_.ravel(): hits|=fast_tree_reliance(tree,X,M,reduce=False)
    return float(hits.max(axis=1).mean())

@contextmanager
def accelerated_splits():
    original=MADTRegressor._best_split;original_rho=BaseMADT.compute_missingness_reliance
    cache={}
    def best(self,X,y,feature,sample_weight,M=None):
        # Root orders recur across boosting stages; identity checks prevent stale reuse.
        if len(X)>=4000:
            key=id(X);entry=cache.get(key)
            if entry is None or entry[0]() is not X:
                for old in list(cache):
                    if cache[old][0]() is None: del cache[old]
                if len(cache)>4: cache.clear()
                entry=(weakref.ref(X),{});cache[key]=entry
            order=entry[1].get(feature)
            if order is None: order=np.argsort(X[:,feature]);entry[1][feature]=order
        else: order=np.argsort(X[:,feature])
        xs=np.ascontiguousarray(X[order,feature]);ys=np.ascontiguousarray(y[order]);ws=np.ascontiguousarray(sample_weight[order])
        value,index=scan(xs,ys,ws,float(np.sum(ys*ws)),float(np.sum(ys**2*ws)),float(np.sum(sample_weight)))
        if index<0:return -np.inf,None,None
        if M is not None:value-=self.alpha*np.mean(sample_weight*M[:,feature])
        return value,feature,0.5*(xs[index]+xs[index+1])
    MADTRegressor._best_split=best;BaseMADT.compute_missingness_reliance=fast_tree_reliance
    try:yield
    finally:MADTRegressor._best_split=original;BaseMADT.compute_missingness_reliance=original_rho
