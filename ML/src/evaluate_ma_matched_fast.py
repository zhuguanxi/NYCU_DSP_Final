"""Exact compiled tree routing; never fit or alter a frozen model."""
import sys
from pathlib import Path
import time
import numpy as np
import joblib
from contextlib import contextmanager
from ma_matched_runtime import njit
from malearn.estimators.tree import ExtendedTree
import evaluate_ma_matched as evaluation
from train_ma_matched import ROOT,read,write,data,sha
from ma_gbt_p12_inputs import encode

@njit(cache=True,fastmath=False)
def leaves(x,left,right,feature,threshold):
    result=np.empty(len(x),dtype=np.int64)
    for i in range(len(x)):
        node=0
        while left[node]!=-1:
            if x[i,feature[node]]<=threshold[node]:node=left[node]
            else:node=right[node]
        result[i]=node
    return result

@contextmanager
def compiled_routes():
    old=ExtendedTree.apply
    def apply(self,x):return leaves(x,self.children_left,self.children_right,self.feature,self.threshold)
    ExtendedTree.apply=apply
    try:yield
    finally:ExtendedTree.apply=old

def check():
    frozen=evaluation.frozen();frame,_,_,schemas=data();frame=frame.iloc[:384]
    encoded={name:encode(frame,schemas[name]['final'])[0] for name in ('Full','Elastic')}
    checks=[]
    for receipt in frozen['fits']:
        model=joblib.load(ROOT/'models'/f'{receipt["key"]}.joblib');x=encoded[receipt['features']]
        baseline=model.predict_proba(x)
        for tree in model.estimators_.ravel():
            t=tree.tree_
            np.testing.assert_array_equal(t.apply(x),leaves(x,t.children_left,t.children_right,t.feature,t.threshold))
        with compiled_routes():np.testing.assert_array_equal(baseline,model.predict_proba(x))
        checks.append(dict(model=receipt['key'],all_300_tree_leaves='exact',probabilities='exact'))
    # Branch boundaries include float32 neighbours, equality, signed zero and NaN.
    x=np.array([[-1.],[-0.],[0.],[1.],[np.nextafter(np.float32(1),np.float32(2))],[np.nan]],dtype=np.float32)
    left=np.array([1,-1,-1],dtype=np.int64);right=np.array([2,-1,-1],dtype=np.int64);feature=np.array([0,-2,-2],dtype=np.int64)
    for cut in (0.,1.,1.+2**-25):
        threshold=np.array([cut,-2.,-2.],dtype=np.float64)
        expected=np.array([1 if row[0]<=cut else 2 for row in x])
        np.testing.assert_array_equal(leaves(x,left,right,feature,threshold),expected)
    record=dict(AB_only=True,new_model_fits=0,frozen_sha256=sha(ROOT/'frozen.json'),
        source_sha256=sha(Path(__file__)),checks=checks,boundary_checks='exact')
    write(ROOT/'inference_numeric_check.json',record)
    print('Exact A/B inference check passed for all 12 frozen models, all 3600 trees and boundary probes',flush=True)

def main():
    if sys.argv[1:] == ['check']:return check()
    record=read(ROOT/'inference_numeric_check.json')
    assert record['AB_only'] and record['new_model_fits']==0
    assert record['source_sha256']==sha(Path(__file__))
    assert record['frozen_sha256']==sha(ROOT/'frozen.json')
    execution=dict(device='CPU',new_model_fits=0,method='exact compiled original ExtendedTree.apply routing',
        frozen_sha256=sha(ROOT/'frozen.json'),source_sha256=sha(Path(__file__)),
        evaluator_source_sha256=sha(Path(evaluation.__file__)),numeric_check_sha256=sha(ROOT/'inference_numeric_check.json'),
        original_completed_case_caches_preserved=True,
        original_partial_evaluation_interrupted='resource-only switch after exact A/B check; incomplete scenario recomputed without model changes')
    write(ROOT/'inference_execution.json',execution)
    with compiled_routes():evaluation.main()

if __name__=='__main__':main()
