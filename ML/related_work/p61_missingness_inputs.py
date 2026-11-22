"""Label-blind, nested hour deletion; rebuild all affected feature representations."""
import json
from pathlib import Path
import sys
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import numpy as np
import pandas as pd
from pilot_temporal_features import time_summary
from prepare_hourly_sequence import hourly_cube,static_values
from stress_missingness_c import load_cohort
from prepare import OUT,sha,stats_spec

ROOT=OUT/'p61_progressive_block_missingness'
SIX=('GCS','Lactate','MAP','Mg','Urine','MechVent')
MASK_SEED=2026112261

def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def hour_masks(n,design,length,repeat):
    if design=='clean': return np.zeros((n,48),dtype=bool)
    if design=='full': return np.ones((n,48),dtype=bool)
    rng=np.random.default_rng(MASK_SEED+repeat)
    if design=='random':
        rank=np.argsort(rng.random((n,48)),axis=1,kind='stable')
        result=np.zeros((n,48),dtype=bool)
        np.put_along_axis(result,rank[:,:length],True,axis=1)
    else:
        start=np.floor(rng.random(n)*(49-length)).astype(int)
        hours=np.arange(48)[None,:]
        result=(hours>=start[:,None])&(hours<start[:,None]+length)
    assert (result.sum(1)==length).all()
    return result

def rebuild(raw,events,channels):
    output=raw.copy()
    for channel in channels:
        rows=events.loc[events.variable.eq(channel)].sort_values(['RecordID','elapsed_minute','_order'],kind='stable')
        groups=rows.groupby('RecordID',sort=False)
        values=groups.clean_value
        count=values.size().reindex(raw.index,fill_value=0)
        spec=stats_spec(channel)
        for stat in spec:
            if stat in ('min','max','mean','first','last','std'): series=getattr(values,stat)()
            elif stat=='ever': series=values.max()
            elif stat=='observed_fraction': series=values.mean()
            elif stat=='delta': series=values.last()-values.first()
            elif stat=='slope':
                piece=rows.assign(t=rows.elapsed_minute/60)
                piece['tc']=piece.t-piece.groupby('RecordID').t.transform('mean')
                piece['xc']=piece.clean_value-piece.groupby('RecordID').clean_value.transform('mean')
                numerator=(piece.tc*piece.xc).groupby(piece.RecordID).sum()
                denominator=(piece.tc**2).groupby(piece.RecordID).sum()
                series=numerator/denominator.where(denominator>0)
            else: raise ValueError(stat)
            output[f'{channel}__{stat}']=series.reindex(raw.index)
        output[f'{channel}__count']=count
        output[f'{channel}__missing']=count.eq(0).astype(int)
        for stat in ('std','slope'):
            if stat in spec: output[f'{channel}__{stat}_defined']=np.isfinite(output[f'{channel}__{stat}']).astype(int)
    return output

def main():
    ROOT.mkdir(exist_ok=True)
    destination=ROOT/'inputs.json'
    if destination.exists(): raise FileExistsError(destination)
    raw,y,_=load_cohort()
    event_path=OUT/'related_work/common_sequence_inputs/p12_c_events.parquet'
    events=pd.read_parquet(event_path).assign(_order=lambda f:np.arange(len(f)))
    rebuilt=rebuild(raw,events,SIX)
    columns=[c for c in raw if c.split('__')[0] in SIX]
    np.testing.assert_allclose(raw[columns],rebuilt[columns],atol=1e-8,rtol=1e-10,equal_nan=True)
    for design in ('random','block'):
        for repeat in range(3):
            masks=[hour_masks(len(raw),design,length,repeat) for length in (12,24,36)]
            assert not (masks[0]&~masks[1]).any() and not (masks[1]&~masks[2]).any()
    specs=[dict(name='clean',target='none',channels=[],design='clean',hours=0,repeat=-1)]
    for target,channels in [('gcs',('GCS',)),('six',SIX)]:
        specs.append(dict(name=f'{target}_full',target=target,channels=list(channels),design='full',hours=48,repeat=-1))
        for design in ('random','block'):
            for length in (12,24,36):
                for repeat in range(3):
                    specs.append(dict(name=f'{target}_{design}_{length}_r{repeat}',target=target,channels=list(channels),
                                      design=design,hours=length,repeat=repeat))
    row=raw.index.get_indexer(events.RecordID)
    assert (row>=0).all()
    event_hours=(events.elapsed_minute.to_numpy()//60).clip(0,47).astype(int)
    manifest=[]
    for case in specs:
        folder=ROOT/case['name'];folder.mkdir(exist_ok=True)
        hours=hour_masks(len(raw),case['design'],case['hours'],case['repeat'])
        targeted=events.variable.isin(case['channels']).to_numpy()
        removed=targeted&hours[row,event_hours]
        retained=events.loc[~removed]
        frame=raw if not case['channels'] else rebuild(raw,retained,case['channels'])
        untouched=[c for c in raw if c.split('__')[0] not in case['channels']]
        pd.testing.assert_frame_equal(raw[untouched],frame[untouched])
        frame.to_parquet(folder/'features.parquet')
        timing=time_summary(retained,raw.index);timing.to_parquet(folder/'temporal.parquet')
        np.savez_compressed(folder/'input.npz',ids=raw.index.to_numpy(),y=y,
                            cube=hourly_cube(retained,raw.index),static=static_values(frame),hours=hours)
        exposure=dict(target_observations=int(targeted.sum()),removed_observations=int(removed.sum()),
                      affected_stays=int(np.unique(row[removed]).size),
                      actual_removed_fraction=float(removed.sum()/targeted.sum()) if targeted.any() else 0.)
        manifest.append(dict(**case,**exposure,files={f:sha(folder/f) for f in ('features.parquet','temporal.parquet','input.npz')}))
        print(f'Prepared {case["name"]}: removed {int(removed.sum())} observations',flush=True)
    sources=[event_path,OUT/'C_evaluation/features_C.parquet',OUT/'C_evaluation/labels_C.csv',
             ML/'reports/P61_progressive_block_protocol.md']
    write(destination,dict(stays=len(raw),deaths=int(y.sum()),mask_seed=MASK_SEED,repeats=3,cases=manifest,
        source_sha256={str(p.relative_to(ML)):sha(p) for p in sources},
        leakage_audit='Masks use fixed row order and RNG only; no outcome-dependent decisions; all target summaries rebuilt'))
    print('Prepared 39 fixed scenarios with nested, outcome-blind clock masks.',flush=True)

if __name__=='__main__': main()
