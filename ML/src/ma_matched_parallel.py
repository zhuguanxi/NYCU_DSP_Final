"""Parallelize independent features while preserving each sequential split scan."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref
import numpy as np
from numba import njit,prange
from malearn.estimators.regression import MADTRegressor
from ma_matched_runtime import accelerated_splits,scan

@njit(cache=True,parallel=True,fastmath=False)
def all_scores(xs,ys,ws,totals,totals2,total_weight):
    values=np.empty(xs.shape[0],dtype=np.float64);indices=np.empty(xs.shape[0],dtype=np.int64)
    for f in prange(xs.shape[0]):
        if xs[f,0]==xs[f,-1]:values[f]=-np.inf;indices[f]=-1
        else:values[f],indices[f]=scan(xs[f],ys[f],ws[f],totals[f],totals2[f],total_weight)
    return values,indices

@contextmanager
def parallel_splits():
    original=MADTRegressor._get_split_candidates;had_own='_get_split_candidates' in MADTRegressor.__dict__;cache={}
    sorter=ThreadPoolExecutor(max_workers=2)
    def candidates(self,X,y,M,sample_weight,depth,random_state):
        features=np.arange(self.n_features_in_);random_state.shuffle(features)
        entry=cache.get(id(X)) if len(X)>=4000 else None
        if entry is None or entry[0]() is not X or not np.array_equal(entry[1],sample_weight):
            pieces=np.array_split(X.T,2,axis=0)
            pending=[sorter.submit(np.argsort,piece,axis=1) for piece in pieces]
            order=np.vstack([f.result() for f in pending])
            xs=np.ascontiguousarray(np.take_along_axis(X.T,order,axis=1));ws=np.ascontiguousarray(sample_weight[order])
            entry=(weakref.ref(X),sample_weight.copy(),order,xs,ws)
            if len(X)>=4000:
                for dead in list(cache):
                    if cache[dead][0]() is None:del cache[dead]
                if len(cache)>4:cache.clear()
                cache[id(X)]=entry
        _,_,order,xs,ws=entry;ys=np.ascontiguousarray(y[order])
        values,indices=all_scores(xs,ys,ws,np.sum(ys*ws,axis=1),np.sum(ys**2*ws,axis=1),float(np.sum(sample_weight)))
        if M is not None:values-=self.alpha*np.mean(sample_weight[:,None]*M,axis=0)
        for f in features:
            i=indices[f]
            if i<0:yield -np.inf,None,None
            else:yield values[f],f,.5*(xs[f,i]+xs[f,i+1])
    with accelerated_splits():
        MADTRegressor._get_split_candidates=candidates
        try:yield
        finally:
            if had_own:MADTRegressor._get_split_candidates=original
            else:del MADTRegressor._get_split_candidates
            sorter.shutdown(wait=True)
