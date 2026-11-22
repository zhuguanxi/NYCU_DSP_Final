"""Replay the fixed P33 OOF CNN and select a missingness-dependent rule on A/B."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='3'

import numpy as np
import torch

ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
from train_hourly_sequence import fit,masked,score,validate_inputs,digest,read_json,save_json,SEED,THREADS

ROOT=ML/'artifacts/p56_observation_mixture'
P33=ML/'artifacts/p33_hourly_sequence'
P26=ML/'artifacts/p26_gcs_routed_temporal'
WEIGHTS=(0.,.15,.25,.5)


def risks(anchor,routed,sequence,observed,rule):
    reference=np.where(observed,anchor,routed if rule['missing_base']=='p26' else anchor)
    weight=np.where(observed,rule['observed_sequence'],rule['missing_sequence'])
    return reference+weight*(sequence-reference)


def main():
    ROOT.mkdir(exist_ok=True)
    output=ROOT/'selection.json'
    if output.exists(): raise FileExistsError(output)
    manifest=read_json(P33/'inputs/manifest.json')
    path=P33/'inputs/ab.npz'
    if digest(path)!=manifest['bundle_sha256']: raise ValueError('P33 input bundle changed')
    with np.load(path,allow_pickle=False) as source: data={k:source[k] for k in source.files}
    validate_inputs(data,manifest)
    x=data['cube']; y=data['y']; folds=data['folds']; views=manifest['views']
    present=np.isfinite(x[:,manifest['groups'].index('GCS')]).any(axis=1)
    observed=np.stack([present&~hidden[:,manifest['groups'].index('GCS')] for hidden in data['val_hidden']])
    route=read_json(P26/'manifest.json')
    rule=route['selected']
    with np.load(P26/'ab_candidate_predictions.npz',allow_pickle=False) as old:
        routed=np.stack([np.where(observed[i],old[f'{rule["primary"]}_{view}'],
                    (1-rule['weight'])*old[f'{rule["primary"]}_{view}']+rule['weight']*old[f'{rule["fallback"]}_{view}'])
                         for i,view in enumerate(views)])
    checked=score(y,folds,views,routed,manifest['weights'])
    expected=route['a_b_metrics'][route['winner']]
    for key in ('clean','full_gcs','shift','mapped'):
        np.testing.assert_allclose(checked[key],expected[key],rtol=0,atol=1e-12)
    probabilities=np.full(data['anchor'].shape,np.nan)
    for fold in range(3):
        held=folds==fold
        checkpoint=ROOT/f'fold_{fold}_oof.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as old: values=old['probabilities']
        else:
            tensors=[(torch.from_numpy(masked(x[held],hidden[held])),torch.from_numpy(static[held]))
                     for hidden,static in zip(data['val_hidden'],data['val_static'])]
            _,prediction=fit(data,~held,fold,'plain',8,tensors,seed=SEED)
            values=prediction[8]
            np.savez_compressed(checkpoint,probabilities=values)
        probabilities[:,held]=values
        print(f'Replayed fixed P33 CNN OOF fold {fold}',flush=True)
    baseline=score(y,folds,views,.75*data['anchor']+.25*probabilities,manifest['weights'])
    fitted=read_json(P33/'fitted/manifest.json')
    expected=fitted['metrics'][fitted['winner']]
    for key,value in baseline.items():
        np.testing.assert_allclose(value,expected[key],rtol=0,atol=1e-12)
    candidates={}
    recipes={}
    for base in ('p28','p26'):
        for available in WEIGHTS:
            for absent in WEIGHTS:
                name=f'{base}:observed_{available:.2f}:missing_{absent:.2f}'
                recipe=dict(missing_base=base,observed_sequence=available,missing_sequence=absent)
                predicted=risks(data['anchor'],routed,probabilities,observed,recipe)
                candidates[name]=score(y,folds,views,predicted,manifest['weights'])
                recipes[name]=recipe
    eligible=[name for name,values in candidates.items() if all(values[k]>=baseline[k]-.003 for k in ('clean','full_gcs'))]
    best=max(eligible,key=lambda k:(candidates[k]['objective'],candidates[k]['clean'],candidates[k]['full_gcs']))
    accepted=candidates[best]['objective']>=baseline['objective']+.001
    selection=dict(dataset='P12 A/B',stays=len(y),deaths=int(y.sum()),folds=3,cnn_replay_fits=3,
                   baseline=baseline,candidates=candidates,eligible=eligible,best_candidate=best,
                   accepted=bool(accepted),selected=recipes[best] if accepted else None,
                   gain=candidates[best]['objective']-baseline['objective'],minimum_gain=.001,
                   guard_tolerance=.003,test_data_used=False,seed=SEED,device='cpu',
                   source_sha256={str(p):digest(p) for p in (path,P33/'inputs/manifest.json',P33/'fitted/manifest.json',
                                     P26/'manifest.json',P26/'ab_candidate_predictions.npz',Path(__file__))})
    np.savez_compressed(ROOT/'oof.npz',anchor=data['anchor'],routed=routed,sequence=probabilities,
                        observed=observed,y=y,folds=folds)
    save_json(output,selection)
    print(json.dumps({'accepted':bool(accepted),'best':best,'gain':selection['gain'],'baseline':baseline,
                      'candidate':candidates[best]},indent=2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    main()
